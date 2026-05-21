"""
Censor -- Cross-Dataset Pretrain + Finetune Pipeline
=====================================================
Strategy:
  1. Pretrain on CASME2 (largest AU annotations, onset/apex/offset)
  2. Finetune on SMIC-HS + SAMM (transfer biomimetic features)

Key design:
  - Unified 4-class label space: happiness(0), surprise(1), disgust(2), repression(3)
  - Load pretrained backbone, replace MoE head for new dataset
  - Freeze backbone first N epochs, then unfreeze with lower LR
  - LOSO cross-validation on target dataset

Usage:
  # Step 1: Pretrain on CASME2
  python train_cross.py --stage pretrain --data_root /root/autodl-tmp/data/CASME2

  # Step 2: Finetune on SMIC
  python train_cross.py --stage finetune \
    --data_root /root/autodl-tmp/data/SMIC_all_cropped \
    --dataset smic \
    --pretrained /root/autodl-tmp/checkpoints/pretrain_best.pth

  # Step 3: Finetune on SAMM
  python train_cross.py --stage finetune \
    --data_root /root/autodl-tmp/data/SAMM \
    --dataset samm \
    --pretrained /root/autodl-tmp/checkpoints/pretrain_best.pth
"""

import os
import sys
import csv
import random
import argparse
import time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from main import Censor
from config.defaults import MOE_CONFIG
from dataset_frames import FrameSequenceDataset, get_casme2_dataloaders, get_loso_subjects
from dataset_smic import SMICDataset, get_smic_dataloaders, get_smic_subjects
from dataset_samm import SAMMDataset, get_samm_dataloaders, get_samm_subjects
from dataset_samm import SAMMDataset, get_samm_dataloaders, get_samm_subjects
from train_frames import (
    FocalLoss, compute_me_loss, compute_au_loss,
    MetricsTracker, compute_landmark_loss
)


# =============================================================================
# Unified Label Mapping (4-class)
# =============================================================================
# All datasets map to: happiness(0), surprise(1), disgust(2), repression(3)
# Samples that don't fit these 4 classes are excluded.

UNIFIED_CLASSES = 4
UNIFIED_NAMES = ['happiness', 'surprise', 'disgust', 'repression']

# SMIC -> unified: positive=happiness, negative=disgust, surprise=surprise
SMIC_TO_UNIFIED = {
    'positive': 0,   # happiness
    'negative': 2,   # disgust
    'surprise': 1,   # surprise
}

# SAMM -> unified (SAMM has more fine-grained labels)
SAMM_TO_UNIFIED = {
    'Happiness': 0,
    'Surprise': 1,
    'Disgust': 2,
    'Contempt': 2,      # merge into disgust
    'Fear': 2,          # merge into disgust
    'Sadness': 2,       # merge into disgust
    'Anger': 2,         # merge into disgust
    'Repression': 3,
    # 'Other' excluded
}


# =============================================================================
# Model Utilities for Transfer Learning
# =============================================================================

def load_pretrained_backbone(model, pretrained_path):
    """
    Load pretrained weights into model, handling head mismatch.

    Strategy: load all matching keys, skip MoE head (different num_classes).
    This preserves: preprocessing, dual-pathway backbones, attention, fusion, AU decoder.
    """
    ckpt = torch.load(pretrained_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)

    model_dict = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in state_dict.items():
        if key in model_dict:
            if value.shape == model_dict[key].shape:
                model_dict[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key}: shape mismatch {value.shape} vs {model_dict[key].shape}")
        else:
            skipped_keys.append(f"{key}: not in model")

    model.load_state_dict(model_dict)

    print(f"[Transfer] Loaded {len(loaded_keys)} / {len(state_dict)} parameters")
    if skipped_keys:
        print(f"[Transfer] Skipped {len(skipped_keys)} keys:")
        for k in skipped_keys[:10]:
            print(f"  - {k}")
        if len(skipped_keys) > 10:
            print(f"  ... and {len(skipped_keys) - 10} more")

    return model


def freeze_backbone(model):
    """Freeze dual-pathway backbones + preprocessing."""
    frozen_count = 0
    for name, param in model.named_parameters():
        # Freeze: preprocessing, fast_pathway, slow_pathway
        if any(k in name for k in ['saliency', 'rppg', 'flow',
                                     'fast_pathway', 'slow_pathway']):
            param.requires_grad = False
            frozen_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Freeze] Frozen {frozen_count} parameter tensors, "
          f"trainable: {trainable:,} / {total:,}")


def unfreeze_backbone(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Unfreeze] All parameters trainable: {trainable:,}")


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(model, loader, optimizer, scaler, device, args, epoch):
    model.train()
    tracker = MetricsTracker()

    for batch_idx, (videos, me_labels, au_labels) in enumerate(loader):
        videos = videos.to(device)
        me_labels = me_labels.to(device)
        au_labels = au_labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(videos)

            me_loss = compute_me_loss(outputs['me_logits'], me_labels)
            au_loss = compute_au_loss(outputs['au_intensities'], au_labels)
            lm_loss = compute_landmark_loss(outputs['au_opd'])
            moe_loss = outputs['moe_aux_loss']

            total_loss = (me_loss
                          + args.au_loss_weight * au_loss
                          + args.landmark_loss_weight * lm_loss
                          + args.moe_loss_weight * moe_loss)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        tracker.update_me(outputs['me_logits'].detach(), me_labels)
        tracker.update_au(outputs['au_intensities'].detach(), au_labels)
        tracker.losses['me'].append(me_loss.item())
        tracker.losses['au'].append(au_loss.item())
        tracker.losses['total'].append(total_loss.item())

        if batch_idx % args.log_interval == 0:
            print(f"  [{batch_idx}/{len(loader)}] "
                  f"me: {me_loss.item():.4f} | au: {au_loss.item():.4f} | "
                  f"total: {total_loss.item():.4f}")

    return tracker


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    tracker = MetricsTracker()

    for videos, me_labels, au_labels in loader:
        videos = videos.to(device)
        me_labels = me_labels.to(device)
        au_labels = au_labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(videos)

        me_loss = compute_me_loss(outputs['me_logits'], me_labels)
        au_loss = compute_au_loss(outputs['au_intensities'], au_labels)

        tracker.update_me(outputs['me_logits'], me_labels)
        tracker.update_au(outputs['au_intensities'], au_labels)
        tracker.losses['me'].append(me_loss.item())
        tracker.losses['au'].append(au_loss.item())

    return tracker


def get_dataloaders(args):
    """Get dataloaders for the specified dataset."""
    face_align = not args.no_face_align

    if args.dataset == 'casme2':
        return get_casme2_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            T=args.T, H=args.H, W=args.W,
            num_workers=args.num_workers,
            face_align=face_align,
            loso_fold=args.loso_fold if args.loso else None,
        )
    elif args.dataset == 'smic':
        return get_smic_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            T=args.T, H=args.H, W=args.W,
            num_workers=args.num_workers,
            face_align=face_align,
            loso_fold=args.loso_fold if args.loso else None,
        )
    elif args.dataset == 'samm':
        return get_samm_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            T=args.T, H=args.H, W=args.W,
            num_workers=args.num_workers,
            face_align=face_align,
            loso_fold=args.loso_fold if args.loso else None,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def run_training(args):
    """Main training loop for pretrain or finetune."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build model
    model = Censor(fast_preprocess=True, diff_mode=args.diff_mode, verbose=False)

    # Load pretrained weights for finetune
    if args.stage == 'finetune' and args.pretrained:
        print(f"\n[Finetune] Loading pretrained weights from: {args.pretrained}")
        model = load_pretrained_backbone(model, args.pretrained)

        # Freeze backbone for finetune
        if args.freeze_backbone:
            freeze_backbone(model)
    elif args.stage == 'pretrain' and args.freeze_backbone:
        freeze_backbone(model)

    model = model.to(device)

    # Data
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, val_loader = get_dataloaders(args)
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Optimizer with different LR for backbone vs head
    if args.stage == 'finetune':
        # Lower LR for pretrained backbone, higher for new head
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ['fast_pathway', 'slow_pathway',
                                         'saliency', 'rppg', 'flow',
                                         'amygdala', 'ffa', 'casa',
                                         'fusion', 'sparse_control']):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.lr / 10},  # 10x lower for backbone
            {'params': head_params, 'lr': args.lr},
        ], weight_decay=args.weight_decay)
        print(f"[Finetune] Backbone LR: {args.lr/10:.2e}, Head LR: {args.lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()

    # Output dir
    stage_dir = f"{args.stage}_{args.dataset}"
    output_dir = os.path.join(args.output_dir, stage_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Unfreeze backbone after freeze_epochs
        if args.freeze_backbone and epoch == args.freeze_epochs + 1:
            unfreeze_backbone(model)
            # Re-create optimizer with differential LR
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if any(k in name for k in ['fast_pathway', 'slow_pathway',
                                             'saliency', 'rppg', 'flow',
                                             'amygdala', 'ffa', 'casa',
                                             'fusion', 'sparse_control']):
                    backbone_params.append(param)
                else:
                    head_params.append(param)

            optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': args.lr / 10},
                {'params': head_params, 'lr': args.lr},
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-6
            )
            scaler = torch.cuda.amp.GradScaler()

        # Train
        train_tracker = train_one_epoch(model, train_loader, optimizer, scaler, device, args, epoch)

        # Validate
        val_tracker = validate(model, val_loader, device, args)

        # Scheduler step
        scheduler.step()

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[-1]['lr']

        # Print epoch summary
        print(f"\n  [Train] me: {np.mean(train_tracker.losses['me']):.4f} | "
              f"au: {np.mean(train_tracker.losses['au']):.4f}")
        print(f"  [Val]   me: {np.mean(val_tracker.losses['me']):.4f} | "
              f"au: {np.mean(val_tracker.losses['au']):.4f}")
        print(f"  [Val]   ME Acc: {val_tracker.me_accuracy:.4f} | "
              f"ME F1: {val_tracker.me_f1:.4f} | AU F1: {val_tracker.au_f1:.4f}")
        print(f"  LR: {lr:.2e} | Epoch time: {epoch_time:.1f}s")

        # Save best
        if val_tracker.me_accuracy > best_val_acc:
            best_val_acc = val_tracker.me_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {
                    'me_accuracy': val_tracker.me_accuracy,
                    'me_f1': val_tracker.me_f1,
                    'au_f1': val_tracker.au_f1,
                },
                'args': vars(args),
            }, os.path.join(output_dir, f'{args.stage}_best.pth'))
            print(f"  ** New best: {best_val_acc:.4f} **")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': {
                    'me_accuracy': val_tracker.me_accuracy,
                    'me_f1': val_tracker.me_f1,
                    'au_f1': val_tracker.au_f1,
                },
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f" {args.stage.upper()} on {args.dataset} completed")
    print(f" Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f" Best Val Acc: {best_val_acc:.4f}")
    print(f" Checkpoints: {output_dir}")
    print(f"{'='*60}")

    return best_val_acc


def run_loso(args):
    """Run Leave-One-Subject-Out cross-validation."""
    # Get subjects
    if args.dataset == 'casme2':
        subjects = get_loso_subjects(args.data_root)
    elif args.dataset == 'smic':
        subjects = get_smic_subjects(args.data_root)
    elif args.dataset == 'samm':
        subjects = get_samm_subjects(args.data_root)
    else:
        raise ValueError(f"LOSO not supported for {args.dataset}")

    if subjects is None:
        print("[Error] Cannot get subjects. Check data_root and labels.csv")
        return

    num_folds = len(subjects)
    print(f"\nLOSO: {num_folds} subjects = {num_folds} folds")
    print(f"Subjects: {subjects}")

    folds_to_run = [args.loso_fold] if args.loso_fold is not None else list(range(num_folds))

    fold_results = {}
    for fold_idx in folds_to_run:
        fold_args = argparse.Namespace(**vars(args))
        fold_args.loso_fold = fold_idx
        fold_args.loso = True

        acc = run_training(fold_args)
        fold_results[fold_idx] = {'acc': acc, 'subject': subjects[fold_idx]}

    # Print LOSO summary
    print(f"\n{'='*60}")
    print(f" LOSO Cross-Validation Summary ({args.dataset})")
    print(f"{'='*60}")
    accs = [r['acc'] for r in fold_results.values()]
    for fold_idx, res in sorted(fold_results.items()):
        print(f"  Fold {fold_idx} ({res['subject']}): Acc={res['acc']:.4f}")
    print(f"\n  Mean Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"{'='*60}")

    # Save summary
    stage_dir = f"{args.stage}_{args.dataset}"
    summary_path = os.path.join(args.output_dir, stage_dir, 'loso_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'subject', 'accuracy'])
        writer.writeheader()
        for fold_idx, res in sorted(fold_results.items()):
            writer.writerow({'fold': fold_idx, 'subject': res['subject'], 'accuracy': res['acc']})
    print(f"LOSO summary saved to: {summary_path}")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Censor Cross-Dataset Training')

    # Stage
    parser.add_argument('--stage', type=str, default='pretrain',
                        choices=['pretrain', 'finetune'],
                        help='pretrain on CASME2 or finetune on SMIC/SAMM')
    parser.add_argument('--dataset', type=str, default='casme2',
                        choices=['casme2', 'smic', 'samm'],
                        help='Target dataset')

    # Data
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/data/CASME2')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint (for finetune)')

    # Frame sequence params
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--H', type=int, default=224)
    parser.add_argument('--W', type=int, default=224)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)

    # Loss weights
    parser.add_argument('--au_loss_weight', type=float, default=0.1)
    parser.add_argument('--moe_loss_weight', type=float, default=0.01)
    parser.add_argument('--landmark_loss_weight', type=float, default=0.05)

    # Freeze
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone for initial epochs')
    parser.add_argument('--freeze_epochs', type=int, default=30,
                        help='Epochs to freeze backbone before unfreezing')

    # LOSO
    parser.add_argument('--loso', action='store_true',
                        help='Enable Leave-One-Subject-Out cross-validation')
    parser.add_argument('--loso_fold', type=int, default=None,
                        help='Run specific LOSO fold')

    # Face alignment
    parser.add_argument('--no_face_align', action='store_true',
                        help='Disable face alignment')

    # Misc
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Stage: {args.stage}")
    print(f"Dataset: {args.dataset}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Face alignment: {not args.no_face_align}")

    if args.loso:
        run_loso(args)
    else:
        run_training(args)
