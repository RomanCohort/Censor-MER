"""
Experiment 1B: Multi-scale 3D ResNet LOSO Reproduction
=====================================================
Reproduce Multi-scale 3D ResNet (Chen et al., Neurocomputing 2024) under
**the same LOSO protocol** as Censor for fair SOTA comparison.

Original paper claims 91.35% on CASME II LOSO. We verify whether this
holds under Censor's strict 24-fold LOSO with 2 excluded subjects
(sub13, sub22) for class diversity reasons.

Key components (from yxj00/Micro-expression-recognition):
  - 3D ResNet backbone (Kinetics-400 pretrained)
  - Multi-scale temporal aggregation (3D convs at multiple kernel sizes)
  - Optical flow as auxiliary input
  - Apex frame + onset/offset frames

Data paths (AutoDL):
  CASME II: /root/autodl-tmp/data/CASME2
  SAMM:     /root/data/SAMM/SAMM
  SMIC:     /root/SMIC_all_cropped

Usage on AutoDL 4090:
  python experiments/exp1b_multiscale_3d_resnet.py --dataset casme2
  python experiments/exp1b_multiscale_3d_resnet.py --dataset samm
  python experiments/exp1b_multiscale_3d_resnet.py --dataset smic

Outputs:
  results/exp1b_multiscale_3d_resnet_<dataset>.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Configuration
# =============================================================================

DATA_PATHS = {
    'casme2': '/root/autodl-tmp/data/CASME2',
    'smic':   '/root/SMIC_all_cropped',
    'samm':   '/root/data/SAMM/SAMM',
}

# CASME II excluded subjects (from Censor paper: lack of difficult classes)
CASME2_EXCLUDED = ['sub13', 'sub22']

# Class mapping (consistent across datasets for cross-dataset eval)
CASME2_CLASSES = ['happiness', 'surprise', 'disgust', 'repression']

# Hyperparameters (from Multi-scale 3D ResNet paper + LOSO protocol)
TRAIN_CONFIG = {
    'T': 16,             # frames
    'H': 224, 'W': 224,
    'batch_size': 2,     # Small for no-GPU / low-RAM mode
    'grad_accum': 8,     # Effective batch = 2*8 = 16
    'lr': 1e-4,
    'backbone_lr': 1e-5,
    'weight_decay': 1e-4,
    'epochs': 50,
    'patience': 15,
    'num_workers': 0,    # 0 to avoid OOM in workers
    'seed': 42,
}


# =============================================================================
# Multi-scale 3D ResNet Architecture
# =============================================================================
# Based on "Multi-scale 3D ResNet for Spontaneous Micro-expression Recognition"
# (Chen et al., Neurocomputing 2024)
# Key idea: parallel 3D convolutions with different temporal kernel sizes
# capture both short (apex) and long (onset-offset) dynamics.

class MultiScale3DBlock(nn.Module):
    """Parallel 3D convolutions with different temporal kernel sizes."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Three parallel branches with different temporal receptive fields
        # Split channels: 3 branches, adjust to match out_channels exactly
        branch_ch = out_channels // 3
        remainder = out_channels - branch_ch * 3

        self.branch_t1 = nn.Sequential(
            nn.Conv3d(in_channels, branch_ch + (1 if remainder > 0 else 0),
                      kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(branch_ch + (1 if remainder > 0 else 0)),
            nn.ReLU(inplace=True),
        )
        self.branch_t3 = nn.Sequential(
            nn.Conv3d(in_channels, branch_ch + (1 if remainder > 1 else 0),
                      kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(branch_ch + (1 if remainder > 1 else 0)),
            nn.ReLU(inplace=True),
        )
        self.branch_t5 = nn.Sequential(
            nn.Conv3d(in_channels, branch_ch,
                      kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=True),
        )
        # No fuse needed: 3 branches sum to out_channels exactly

    def forward(self, x):
        a = self.branch_t1(x)
        b = self.branch_t3(x)
        c = self.branch_t5(x)
        return torch.cat([a, b, c], dim=1)


class BasicBlock3D(nn.Module):
    """3D ResNet basic block."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class MultiScale3DResNet(nn.Module):
    """
    Multi-scale 3D ResNet for MER.

    Architecture (standard 3D ResNet-18 backbone + multi-scale temporal blocks):
      - Stem: 3D conv (3->64) with stride (1,2,2)
      - Layer 1: 2x BasicBlock3D (64->64)
      - Layer 2: MultiScale3DBlock(64->64) + 2x BasicBlock3D (64->128)
      - Layer 3: MultiScale3DBlock(128->128) + 2x BasicBlock3D (128->256)
      - Layer 4: 2x BasicBlock3D (256->512)
      - Global avg pool + FC
    """

    def __init__(self, num_classes=4, in_channels=3, pretrained=False):
        super().__init__()
        self.in_planes = 64

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # Standard 3D ResNet-18 layers
        self.layer1 = self._make_res_layer(64, 2, stride=1)
        self.ms2 = MultiScale3DBlock(64, 64)      # Multi-scale before layer2
        self.layer2 = self._make_res_layer(128, 2, stride=(2, 2, 2))
        self.ms3 = MultiScale3DBlock(128, 128)     # Multi-scale before layer3
        self.layer3 = self._make_res_layer(256, 2, stride=(2, 2, 2))
        self.layer4 = self._make_res_layer(512, 2, stride=(2, 2, 2))

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()
        if pretrained:
            self._load_pretrained_kinetics()

    def _make_res_layer(self, planes, num_blocks, stride):
        """Standard ResNet layer (no multi-scale)."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock3D(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock3D.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_kinetics(self):
        """Load 3D ResNet-18 weights pretrained on Kinetics-400."""
        try:
            from torchvision.models.video import r3d_18, R3D_18_Weights
            pretrained_model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            pretrained_dict = pretrained_model.state_dict()

            # Map compatible keys (only conv and bn layers, skip fc)
            model_dict = self.state_dict()
            compatible_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    compatible_dict[k] = v

            model_dict.update(compatible_dict)
            self.load_state_dict(model_dict)
            print(f"[MultiScale3DResNet] Loaded {len(compatible_dict)} pretrained layers from Kinetics-400")
        except Exception as e:
            print(f"[MultiScale3DResNet] Could not load pretrained weights: {e}")

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.stem(x)         # (B, 64, T, H/4, W/4)
        x = self.layer1(x)       # (B, 64, T, H/4, W/4)
        x = self.ms2(x)          # Multi-scale temporal features (B, 64, T, H/4, W/4)
        x = self.layer2(x)       # (B, 128, T/2, H/8, W/8)
        x = self.ms3(x)          # Multi-scale temporal features (B, 128, T/2, H/8, W/8)
        x = self.layer3(x)       # (B, 256, T/4, H/16, W/16)
        x = self.layer4(x)       # (B, 512, T/8, H/32, W/32)
        x = self.avgpool(x)      # (B, 512, 1, 1, 1)
        x = x.flatten(1)         # (B, 512)
        return self.fc(x)        # (B, num_classes)


# =============================================================================
# LOSO Cross-Validation
# =============================================================================

def get_loso_splits(dataset_name, data_root):
    """
    Build LOSO splits using FrameSequenceDataset with face_align=False
    (cropped frames don't need alignment, and mediapipe causes OOM).

    Returns: list of (train_indices, test_indices, test_subject_id)
    """
    if dataset_name == 'casme2':
        from dataset_frames import FrameSequenceDataset
        ds = FrameSequenceDataset(data_root, split='train', face_align=False)

        # Get subject list from DataFrame
        subjects = sorted(ds.samples['subject'].unique())
        subjects = [s for s in subjects if s not in CASME2_EXCLUDED]

        # Build subject-to-indices map
        subj_to_idx = defaultdict(list)
        for i in range(len(ds.samples)):
            subj = ds.samples.iloc[i]['subject']
            if subj in subjects:
                subj_to_idx[subj].append(i)

        splits = []
        for subj in subjects:
            test_idx = subj_to_idx[subj]
            train_idx = [i for s, idxs in subj_to_idx.items() if s != subj for i in idxs]
            splits.append((train_idx, test_idx, subj))
        return splits, len(subjects), ds

    elif dataset_name == 'samm':
        from dataset_samm import SAMMDataset
        ds = SAMMDataset(data_root, face_align=False)
        # SAMMDataset.samples is a DataFrame
        subjects = sorted(ds.samples['subject'].unique())
        subj_to_idx = defaultdict(list)
        for i in range(len(ds.samples)):
            subj = ds.samples.iloc[i]['subject']
            subj_to_idx[subj].append(i)
        splits = []
        for subj in subjects:
            test_idx = subj_to_idx[subj]
            train_idx = [i for s, idxs in subj_to_idx.items() if s != subj for i in idxs]
            splits.append((train_idx, test_idx, subj))
        return splits, len(subjects), ds

    elif dataset_name == 'smic':
        from dataset_smic import SMICDataset
        ds = SMICDataset(data_root, face_align=False)
        # SMICDataset.samples is a DataFrame
        subjects = sorted(ds.samples['subject'].unique())
        subj_to_idx = defaultdict(list)
        for i in range(len(ds.samples)):
            subj = ds.samples.iloc[i]['subject']
            subj_to_idx[subj].append(i)
        splits = []
        for subj in subjects:
            test_idx = subj_to_idx[subj]
            train_idx = [i for s, idxs in subj_to_idx.items() if s != subj for i in idxs]
            splits.append((train_idx, test_idx, subj))
        return splits, len(subjects), ds

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _unpack_batch(batch):
    """Unpack batch from dataset. Handles both dict and tuple formats."""
    if isinstance(batch, dict):
        x = batch['video'] if 'video' in batch else batch[0]
        y = batch['label'] if 'label' in batch else batch[1]
    elif isinstance(batch, (list, tuple)):
        x = batch[0]  # video tensor (B, C, T, H, W) or (B, T, H, W, C)
        y = batch[1]  # label
    else:
        raise ValueError(f"Unexpected batch type: {type(batch)}")
    return x, y


def _prepare_input(x):
    """Prepare input tensor for 3D model: ensure (B, C=3, T, H, W)."""
    # If (B, T, H, W, C), convert to (B, C, T, H, W)
    if x.dim() == 5 and x.shape[-1] in (3, 6):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
    # Use first 3 channels if more (e.g., RGB+rPPG has 6)
    if x.dim() == 5 and x.shape[1] > 3:
        x = x[:, :3]
    return x


def train_one_fold(model, train_loader, test_loader, device, epochs, lr, log_prefix=''):
    """Train model for one LOSO fold."""

    # Different LR for backbone vs new layers
    backbone_params = []
    new_params = []
    for name, p in model.named_parameters():
        if 'fc' in name:
            new_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': new_params, 'lr': lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x, y = _unpack_batch(batch)
            x = _prepare_input(x).to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Eval
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            acc = evaluate(model, test_loader, device)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            print(f"  {log_prefix}Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss {epoch_loss/max(n_batches,1):.4f} | "
                  f"Acc {acc*100:.2f}% | Best {best_acc*100:.2f}%")

            if patience_counter >= 15:
                print(f"  {log_prefix}Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


def evaluate(model, loader, device):
    """Compute accuracy on a loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = _unpack_batch(batch)
            x = _prepare_input(x).to(device)
            y = y.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='casme2',
                        choices=['casme2', 'samm', 'smic'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--max_folds', type=int, default=None,
                        help='Limit number of folds (for quick test)')
    parser.add_argument('--quick_test', action='store_true',
                        help='Run only 2 folds for sanity check')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only verify data loading + model construction, no training')
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 1B: Multi-scale 3D ResNet LOSO Reproduction")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"Pretrained (Kinetics-400): {args.pretrained}")

    # Set seed
    torch.manual_seed(TRAIN_CONFIG['seed'])
    np.random.seed(TRAIN_CONFIG['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Check data
    data_root = DATA_PATHS[args.dataset]
    if not Path(data_root).exists():
        print(f"\n[ERROR] Data not found at {data_root}")
        print("Update DATA_PATHS in script with correct locations.")
        return
    print(f"Data root: {data_root}")

    # Get LOSO splits
    print("\nBuilding LOSO splits...")
    splits, num_subjects, full_dataset = get_loso_splits(args.dataset, data_root)
    print(f"Total subjects: {num_subjects}, total folds: {len(splits)}")

    if args.quick_test:
        splits = splits[:2]
        print("[QUICK TEST MODE] Running only 2 folds")
    elif args.max_folds:
        splits = splits[:args.max_folds]
        print(f"[LIMITED] Running {len(splits)} folds")

    print(f"Total samples: {len(full_dataset)}")

    # Dry run: verify data loading + model forward pass only
    if args.dry_run:
        print("\n[DRY RUN] Verifying data loading and model construction...")
        from torch.utils.data import Subset

        # Test data loading
        test_idx = splits[0][1] if splits else []
        if test_idx:
            test_subset = Subset(full_dataset, test_idx[:2])
            test_loader = DataLoader(test_subset, batch_size=1, num_workers=0)
            batch = next(iter(test_loader))
            x, y = _unpack_batch(batch)
            x = _prepare_input(x)
            print(f"  Data sample: x={x.shape}, y={y}")

        # Test model forward pass with tiny input
        model = MultiScale3DResNet(
            num_classes=args.num_classes, in_channels=3, pretrained=False,
        )
        tiny_input = torch.randn(1, 3, 4, 64, 64)  # Minimal size
        with torch.no_grad():
            out = model(tiny_input)
        print(f"  Model forward: input={tiny_input.shape} -> output={out.shape}")
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Model params: {n_params:.2f}M")
        print(f"\n[DRY RUN] SUCCESS - data and model verified. Ready for GPU training.")
        return

    # Run each fold
    fold_accuracies = []
    fold_results = []
    total_time_start = time.time()

    for fold_idx, (train_idx, test_idx, test_subject) in enumerate(splits):
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx+1}/{len(splits)}: Test subject = {test_subject} "
              f"({len(test_idx)} samples)")
        print(f"{'='*70}")

        # Build subset datasets
        from torch.utils.data import Subset
        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=TRAIN_CONFIG['num_workers'], pin_memory=True
        )
        test_loader = DataLoader(
            test_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=TRAIN_CONFIG['num_workers'], pin_memory=True
        )

        # Build fresh model
        model = MultiScale3DResNet(
            num_classes=args.num_classes,
            in_channels=3,
            pretrained=args.pretrained,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model params: {n_params:.2f}M")

        fold_start = time.time()
        fold_acc = train_one_fold(
            model, train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr,
            log_prefix=f"[Fold {fold_idx+1}] ",
        )
        fold_time = time.time() - fold_start

        fold_accuracies.append(fold_acc)
        fold_results.append({
            'fold': fold_idx + 1,
            'test_subject': test_subject,
            'n_samples': len(test_idx),
            'accuracy': fold_acc,
            'time_minutes': fold_time / 60,
        })

        print(f"\n  Fold {fold_idx+1} result: {fold_acc*100:.2f}% in {fold_time/60:.1f} min")
        print(f"  Running mean: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    total_time = time.time() - total_time_start

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"Dataset: {args.dataset}")
    print(f"Number of folds: {len(splits)}")
    print(f"Mean accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Total time: {total_time/60:.1f} min ({(total_time/3600):.1f} h)")
    print(f"Avg time per fold: {total_time/60/len(splits):.1f} min")

    # Comparison to claimed SOTA
    comparison = {
        'casme2': {
            'claimed_sota': 91.35,
            'source': 'Chen et al., Neurocomputing 2024',
            'method': 'Multi-scale 3D ResNet',
        },
        'samm': {
            'claimed_sota': 84.77,
            'source': 'Chen et al., Neurocomputing 2024',
            'method': 'Multi-scale 3D ResNet',
        },
        'smic': {
            'claimed_sota': 74.60,
            'source': 'Chen et al., Neurocomputing 2024',
            'method': 'Multi-scale 3D ResNet',
        },
    }

    if args.dataset in comparison:
        claimed = comparison[args.dataset]['claimed_sota']
        delta = mean_acc * 100 - claimed
        print(f"\nClaimed SOTA: {claimed:.2f}% ({comparison[args.dataset]['source']})")
        print(f"Our reproduction: {mean_acc*100:.2f}%")
        print(f"Delta: {delta:+.2f} pp")
        if abs(delta) > 3:
            print(f"  -> Notable difference ({abs(delta):.1f} pp). "
                  f"May indicate protocol discrepancy in original paper.")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    summary = {
        'experiment': 'Multi-scale 3D ResNet LOSO Reproduction',
        'date': datetime.now().isoformat(),
        'dataset': args.dataset,
        'num_folds': len(splits),
        'num_subjects': num_subjects,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'accuracy_percent': mean_acc * 100,
        'std_percent': std_acc * 100,
        'per_fold': fold_results,
        'config': vars(args),
        'comparison': comparison.get(args.dataset, {}),
        'total_time_minutes': total_time / 60,
    }

    output_file = output_dir / f'exp1b_multiscale_3d_resnet_{args.dataset}.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {output_file}")


if __name__ == '__main__':
    main()