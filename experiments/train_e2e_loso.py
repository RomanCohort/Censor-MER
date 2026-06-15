"""
End-to-End LOSO Training on Pre-extracted Frames
=================================================
Train Censor backbone + fusion head directly on frames (no cached features).
Backbone will learn MER-specific features during training.

Usage:
  python experiments/train_e2e_loso.py --dataset casme2 --num_classes 5
  python experiments/train_e2e_loso.py --dataset casme2 --num_classes 4  # 4-class
  python experiments/train_e2e_loso.py --dataset casme2 --quick_test  # 3 folds only

Output:
  results/train_e2e_loso_<dataset>.json
  checkpoints/censor_best_fold<N>.pt
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Configuration
# =============================================================================

DATA_PATHS = {
    'casme2': '/root/autodl-tmp/data/CASME2',
    'smic': '/root/SMIC_all_cropped',
    'samm': '/root/data/SAMM/SAMM',
}

TRAIN_CONFIG = {
    'epochs': 30,
    'batch_size': 4,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'patience': 10,
    'num_workers': 2,
    'seed': 42,
    'gradient_clip': 1.0,
}

# =============================================================================
# Dataset
# =============================================================================

class PreextractedFramesDataset(Dataset):
    """Load pre-extracted frames from .npz file."""

    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.frames = data['frames']    # (N, C, T, H, W) float32
        self.labels = data['labels']    # (N,) int64
        self.subjects = list(data['subjects'])  # (N,) str

        print(f"[PreextractedFramesDataset] Loaded {len(self.labels)} samples")
        print(f"  frames: {self.frames.shape}")
        print(f"  subjects: {len(set(self.subjects))} unique")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.frames[idx]).float()  # (C, T, H, W)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, self.subjects[idx]


def build_loso_splits(dataset):
    """Build LOSO splits from dataset."""
    subjects = sorted(set(dataset.subjects))
    # CASME2 excluded subjects (if needed)
    CASME2_EXCLUDED = ['sub13', 'sub22']  # Adjust based on your protocol
    subjects = [s for s in subjects if s not in CASME2_EXCLUDED]

    subj_to_idx = defaultdict(list)
    for i in range(len(dataset)):
        subj = dataset.subjects[i]
        if subj in subjects:
            subj_to_idx[subj].append(i)

    splits = []
    for subj in subjects:
        test_idx = subj_to_idx[subj]
        train_idx = [i for s, idxs in subj_to_idx.items() if s != subj for i in idxs]
        splits.append((train_idx, test_idx, subj))

    return splits, subjects


# =============================================================================
# Simple Censor Model for E2E Training
# =============================================================================

class SimpleCensorE2E(nn.Module):
    """Simplified Censor for end-to-end training: backbone + linear head."""

    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()

        # Load backbone components
        from model.backbones import FastSubcorticalPathway, SlowCorticalPathway
        from config.defaults import FAST_PATHWAY_CONFIG, SLOW_PATHWAY_CONFIG

        self.fast_path = FastSubcorticalPathway(FAST_PATHWAY_CONFIG, pretrained=pretrained)
        self.slow_path = SlowCorticalPathway(SLOW_PATHWAY_CONFIG, pretrained=pretrained)

        # Feature dimensions
        fast_dim = 512
        slow_dim = 768
        fused_dim = fast_dim + slow_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # Simple preprocessing (no saliency/rPPG for speed)
        self.use_simple_preprocess = True

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) RGB video
        Returns:
            logits: (B, num_classes)
        """
        B, C, T, H, W = x.shape

        # Simple preprocessing: frame difference as motion
        diff = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        flow_x = diff.mean(dim=1, keepdim=True)
        flow_y = diff.std(dim=1, keepdim=True)
        flow_x = F.pad(flow_x, (0, 0, 0, 0, 0, 1))
        flow_y = F.pad(flow_y, (0, 0, 0, 0, 0, 1))
        flow_stack = torch.cat([flow_x, flow_y], dim=1)  # (B, 2, T, H, W)

        # Fast pathway (motion)
        fast_feat = self.fast_path(flow_stack)  # (B, 512)

        # Slow pathway (RGB + zeros for rPPG channel)
        zeros = torch.zeros(B, 3, T, H, W, device=x.device)
        slow_input = torch.cat([x, zeros], dim=1)  # (B, 6, T, H, W)
        slow_feat, _ = self.slow_path(slow_input)  # (B, 768)

        # Concatenate and classify
        fused = torch.cat([fast_feat, slow_feat], dim=1)  # (B, 1280)
        logits = self.classifier(fused)

        return logits


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y, _ in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Mixed precision
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.2%}'})

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


def run_loso_training(args, device):
    """Run LOSO training with full model."""

    print("\n" + "=" * 70)
    print("End-to-End LOSO Training")
    print("=" * 70)

    # Load dataset
    data_root = DATA_PATHS[args.dataset]
    npz_path = Path(data_root) / 'preextracted.npz'

    if not npz_path.exists():
        print(f"[ERROR] No preextracted.npz at {npz_path}")
        print("  Run: python experiments/preextract_frames.py --dataset " + args.dataset)
        return None

    dataset = PreextractedFramesDataset(str(npz_path))
    splits, subjects = build_loso_splits(dataset)

    if args.quick_test:
        splits = splits[:3]
        print(f"[Quick Test] Using only {len(splits)} folds")

    results = {
        'config': {
            'dataset': args.dataset,
            'num_classes': args.num_classes,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
        },
        'folds': [],
        'summary': {},
    }

    fold_accs = []

    for fold_idx, (train_idx, test_idx, test_subject) in enumerate(splits):
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx+1}/{len(splits)}: Test subject = {test_subject}")
        print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
        print("=" * 70)

        # Create data loaders
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

        # Create model
        model = SimpleCensorE2E(num_classes=args.num_classes, pretrained=True).to(device)

        # Optimizer with different LR for backbone vs head
        backbone_params = list(model.fast_path.parameters()) + list(model.slow_path.parameters())
        head_params = model.classifier.parameters()

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': args.lr},           # Higher LR for head
        ], weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler = GradScaler()

        # Training loop
        best_acc = 0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                      optimizer, scaler, device)
            val_acc = evaluate(model, test_loader, device)
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                # Save best model
                if args.save_checkpoints:
                    ckpt_dir = Path('checkpoints')
                    ckpt_dir.mkdir(exist_ok=True)
                    torch.save(model.state_dict(), ckpt_dir / f'censor_best_fold{fold_idx+1}.pt')
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                      f"Loss={train_loss:.4f}, Train={train_acc:.2%}, Val={val_acc:.2%} "
                      f"(Best={best_acc:.2%} @ {best_epoch})")

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        fold_accs.append(best_acc)
        results['folds'].append({
            'fold': fold_idx + 1,
            'test_subject': test_subject,
            'best_accuracy': best_acc,
            'best_epoch': best_epoch,
            'train_samples': len(train_idx),
            'test_samples': len(test_idx),
        })

        print(f"\nFold {fold_idx+1} Result: {best_acc:.2%} (epoch {best_epoch})")
        print(f"Running Mean: {np.mean(fold_accs):.2%} ± {np.std(fold_accs):.2%}")

    # Summary
    results['summary'] = {
        'mean_accuracy': float(np.mean(fold_accs)),
        'std_accuracy': float(np.std(fold_accs)),
        'num_folds': len(fold_accs),
        'all_fold_accs': [float(a) for a in fold_accs],
    }

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Mean Accuracy: {np.mean(fold_accs):.2%} ± {np.std(fold_accs):.2%}")
    print(f"Per-fold: {[f'{a:.2%}' for a in fold_accs]}")

    # Save results
    output_path = Path('results') / f'train_e2e_loso_{args.dataset}.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {output_path}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='End-to-End LOSO Training')
    parser.add_argument('--dataset', type=str, default='casme2', choices=['casme2', 'smic', 'samm'])
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--quick_test', action='store_true', help='Run only 3 folds')
    parser.add_argument('--save_checkpoints', action='store_true', help='Save best model per fold')
    args = parser.parse_args()

    print("=" * 70)
    print("End-to-End LOSO Training")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset}, Num classes: {args.num_classes}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    # Set seed
    torch.manual_seed(TRAIN_CONFIG['seed'])
    np.random.seed(TRAIN_CONFIG['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run training
    run_loso_training(args, device)


if __name__ == '__main__':
    main()
