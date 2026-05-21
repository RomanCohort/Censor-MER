"""
Censor -- Frame Sequence Training Script
=========================================
Training pipeline using preprocessed frame sequences (CASME2 cropped/).

Usage:
    python train_frames.py --data_root /root/autodl-tmp/data/CASME2 --epochs 50 --batch_size 4

    # Quick test with synthetic data:
    python train_frames.py --synthetic_data --epochs 2 --batch_size 2
"""

import os
import sys
import csv
import argparse
import time
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from main import Censor
from config.defaults import AU_DECODER_CONFIG, MOE_CONFIG, DATA_CONFIG
from dataset_frames import FrameSequenceDataset, get_casme2_dataloaders


# =============================================================================
# Synthetic Dataset (for testing the training pipeline)
# =============================================================================

class SyntheticMERDataset(Dataset):
    ME_CATEGORIES = ["Happiness", "Surprise", "Disgust", "Repression"]

    def __init__(self, num_samples=100, T=16, H=224, W=224):
        self.num_samples = num_samples
        self.T = T
        self.H = H
        self.W = W

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mean = torch.tensor(DATA_CONFIG['normalize_mean']).view(3, 1, 1, 1)
        std = torch.tensor(DATA_CONFIG['normalize_std']).view(3, 1, 1, 1)
        video = torch.randn(3, self.T, self.H, self.W) * 0.1 + 0.5
        video = (video - mean) / std
        me_label = torch.randint(0, 4, (1,)).item()
        au_label = (torch.rand(28) > 0.7).float()
        return video, me_label, au_label


# =============================================================================
# Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha  # class weights

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Pulls features of same-class samples together, pushes different-class apart.
    Particularly effective for small datasets where class boundaries are unclear.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (B, D) normalized feature vectors
            labels: (B,) class labels
        """
        device = features.device
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device)

        # Normalize features
        features = nn.functional.normalize(features, dim=1)

        # Similarity matrix: (B, B)
        sim = torch.matmul(features, features.T) / self.temperature

        # Mask: same-class pairs (positive pairs)
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float()  # (B, B)

        # Remove self-similarity from positive mask
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        # For numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        # Log-sum-exp over all negatives + positives (denominator)
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-likelihood over positive pairs
        pos_per_sample = mask.sum(dim=1)
        # Only compute for samples with at least one positive pair
        has_pos = pos_per_sample > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=device)

        mean_log_prob = (mask * log_prob).sum(dim=1) / (pos_per_sample + 1e-8)
        loss = -mean_log_prob[has_pos].mean()
        return loss


def compute_me_loss(me_logits, me_labels):
    # Focal Loss with class weights for CASME2 4-class (exclude others)
    # 0:happiness(32), 1:surprise(28), 2:disgust(63), 3:repression(27)
    freq = torch.tensor([32., 28., 63., 27.], device=me_logits.device)
    weights = 1.0 / freq
    weights = weights / weights.sum() * len(weights)
    return FocalLoss(alpha=weights, gamma=2.0)(me_logits, me_labels)


def compute_au_loss(au_intensities, au_labels):
    au_mean = au_intensities.mean(dim=1).float()  # (B, 28) - force fp32 for AMP safety
    au_labels = au_labels.float()
    with torch.cuda.amp.autocast(enabled=False):
        return nn.BCELoss()(au_mean, au_labels)


def compute_landmark_loss(au_intensities, au_labels):
    temporal_diff = au_intensities[:, 1:, :] - au_intensities[:, :-1, :]
    smoothness_loss = torch.mean(temporal_diff ** 2)
    au_labels_expanded = au_labels.unsqueeze(1).expand(-1, au_intensities.shape[1], -1)
    active_mask = (au_labels_expanded > 0.5).float()
    peak_consistency = -torch.mean(active_mask * torch.log(au_intensities + 1e-8))
    return smoothness_loss + 0.1 * peak_consistency


# =============================================================================
# Metrics
# =============================================================================

class MetricsTracker:
    ME_CATEGORIES = ["Happiness", "Surprise", "Disgust", "Repression"]

    def __init__(self):
        self.reset()

    def reset(self):
        self.me_correct = 0
        self.me_total = 0
        self.au_true_pos = 0
        self.au_pred_pos = 0
        self.au_label_pos = 0
        self.losses = defaultdict(list)
        self.class_correct = defaultdict(int)
        self.class_total = defaultdict(int)

    def update_me(self, me_logits, me_labels):
        preds = me_logits.argmax(dim=1)
        self.me_correct += (preds == me_labels).sum().item()
        self.me_total += me_labels.shape[0]
        for i in range(len(preds)):
            self.class_total[me_labels[i].item()] += 1
            if preds[i] == me_labels[i]:
                self.class_correct[me_labels[i].item()] += 1

    def update_au(self, au_intensities, au_labels):
        au_mean = au_intensities.mean(dim=1)
        au_preds = (au_mean > 0.5).float()
        self.au_label_pos += au_labels.sum().item()
        self.au_pred_pos += au_preds.sum().item()
        self.au_true_pos += (au_preds * au_labels).sum().item()

    def update_loss(self, name, value):
        self.losses[name].append(value)

    @property
    def me_accuracy(self):
        return self.me_correct / max(self.me_total, 1)

    @property
    def me_f1(self):
        f1_scores = []
        for cls in range(4):
            total = self.class_total.get(cls, 0)
            correct = self.class_correct.get(cls, 0)
            prec = correct / max(total, 1)
            rec = correct / max(self.me_total, 1) * self.me_total / max(total, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            f1_scores.append(f1)
        return np.mean(f1_scores)

    @property
    def au_f1(self):
        precision = self.au_true_pos / max(self.au_pred_pos, 1)
        recall = self.au_true_pos / max(self.au_label_pos, 1)
        return 2 * precision * recall / max(precision + recall, 1e-8)

    @property
    def au_hamming(self):
        return 1.0 - (self.au_true_pos / max(self.au_pred_pos + self.au_label_pos - self.au_true_pos, 1))

    def report(self, prefix="Train"):
        avg_losses = {k: np.mean(v) for k, v in self.losses.items()}
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
        print(f"  [{prefix}] {loss_str}")
        print(f"  [{prefix}] ME Acc: {self.me_accuracy:.4f} | ME F1: {self.me_f1:.4f} | AU F1: {self.au_f1:.4f}")

    def to_csv_row(self):
        row = {
            'me_acc': self.me_accuracy,
            'me_f1': self.me_f1,
            'au_f1': self.au_f1,
            'au_hamming': self.au_hamming,
        }
        for k, v in self.losses.items():
            row[f'loss_{k}'] = np.mean(v)
        return row


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters with exponential decay.
    At validation time, use EMA parameters for more stable predictions.
    Typically improves accuracy by 1-3% on small datasets.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        """Replace model params with EMA params for validation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original params after validation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# =============================================================================
# CSV Logger
# =============================================================================

class CSVLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.fieldnames = []
        self.epoch = 0

    def log(self, metrics_dict):
        self.epoch += 1
        row = {'epoch': self.epoch}
        row.update(metrics_dict)
        if self.epoch == 1:
            self.fieldnames = list(row.keys())
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        else:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row)


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args

        # Optimizer
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # LR scheduler: warmup + cosine
        warmup_epochs = max(3, args.epochs // 20)
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, end_factor=1.0,
                    total_iters=warmup_epochs
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=args.epochs - warmup_epochs,
                    eta_min=args.lr * 0.01
                ),
            ],
            milestones=[warmup_epochs]
        )

        # Mixed precision
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            print("[Trainer] AMP disabled (no CUDA). Training on CPU.")

        # Loss weights
        self.loss_weights = {
            'me': 1.0,
            'au': args.au_loss_weight,
            'moe': args.moe_loss_weight,
            'landmark': args.landmark_loss_weight,
            'supcon': args.supcon_weight,
        }

        # EMA
        self.ema = None
        if args.ema_decay > 0:
            self.ema = EMA(model, decay=args.ema_decay)
            print(f"[Trainer] EMA enabled (decay={args.ema_decay})")

        # SupCon Loss
        self.supcon_loss = SupConLoss(temperature=0.07)

        # CSV logger
        os.makedirs(args.output_dir, exist_ok=True)
        self.csv_logger = CSVLogger(os.path.join(args.output_dir, 'metrics.csv'))

    def train_epoch(self, train_loader):
        self.model.train()
        metrics = MetricsTracker()

        for batch_idx, (videos, me_labels, au_labels) in enumerate(train_loader):
            videos = videos.to(self.device)
            me_labels = me_labels.to(self.device)
            au_labels = au_labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(videos)
                    total_loss = self._compute_loss(outputs, me_labels, au_labels)
            else:
                outputs = self.model(videos)
                total_loss = self._compute_loss(outputs, me_labels, au_labels)

            # BioMoE feedback
            if hasattr(self.model, 'moe') and hasattr(self.model.moe, 'apply_feedback'):
                with torch.no_grad():
                    preds = outputs['me_logits'].argmax(dim=1)
                    correct = (preds == me_labels).float()
                    fb = correct.mean().item()
                    if hasattr(self.model.moe, 'mode') and self.model.moe.mode == 'hybrid':
                        self.model.moe.apply_feedback(fb)

            # Backward
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # EMA update
            if self.ema is not None:
                self.ema.update(self.model)

            # Update metrics
            with torch.no_grad():
                metrics.update_me(outputs['me_logits'], me_labels)
                metrics.update_au(outputs['au_intensities'], au_labels)
                loss_me = compute_me_loss(outputs['me_logits'], me_labels)
                loss_au = compute_au_loss(outputs['au_intensities'], au_labels)
                loss_landmark = compute_landmark_loss(outputs['au_intensities'], au_labels)
                loss_moe = outputs['moe_aux_loss']

                metrics.update_loss('me', loss_me.item())
                metrics.update_loss('au', loss_au.item())
                metrics.update_loss('moe', loss_moe.item())
                metrics.update_loss('landmark', loss_landmark.item())
                metrics.update_loss('total', total_loss.item())

            if (batch_idx + 1) % self.args.log_interval == 0:
                metrics.report(prefix=f"Train Batch {batch_idx+1}/{len(train_loader)}")

        return metrics

    def _compute_loss(self, outputs, me_labels, au_labels):
        loss_me = compute_me_loss(outputs['me_logits'], me_labels)
        loss_au = compute_au_loss(outputs['au_intensities'], au_labels)
        loss_landmark = compute_landmark_loss(outputs['au_intensities'], au_labels)
        loss_moe = outputs['moe_aux_loss']

        # SupCon loss on fused features
        loss_supcon = torch.tensor(0.0, device=me_labels.device)
        if self.loss_weights['supcon'] > 0 and 'adapted_feat' in outputs:
            loss_supcon = self.supcon_loss(outputs['adapted_feat'], me_labels)

        total_loss = (
            self.loss_weights['me'] * loss_me +
            self.loss_weights['au'] * loss_au +
            self.loss_weights['moe'] * loss_moe +
            self.loss_weights['landmark'] * loss_landmark +
            self.loss_weights['supcon'] * loss_supcon
        )
        return total_loss

    @torch.no_grad()
    def validate(self, val_loader):
        # Apply EMA parameters for validation
        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        metrics = MetricsTracker()

        for videos, me_labels, au_labels in val_loader:
            videos = videos.to(self.device)
            me_labels = me_labels.to(self.device)
            au_labels = au_labels.to(self.device)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(videos)
            else:
                outputs = self.model(videos)

            metrics.update_me(outputs['me_logits'], me_labels)
            metrics.update_au(outputs['au_intensities'], au_labels)
            loss_me = compute_me_loss(outputs['me_logits'], me_labels)
            loss_au = compute_au_loss(outputs['au_intensities'], au_labels)
            metrics.update_loss('me', loss_me.item())
            metrics.update_loss('au', loss_au.item())

        # Restore original parameters
        if self.ema is not None:
            self.ema.restore(self.model)

        return metrics

    def save_checkpoint(self, epoch, metrics, path, is_best=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': {
                'me_accuracy': metrics.me_accuracy,
                'me_f1': metrics.me_f1,
                'au_f1': metrics.au_f1,
            },
            'args': vars(self.args),
        }
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        torch.save(checkpoint, path)
        status = "BEST" if is_best else f"epoch {epoch}"
        print(f"[Trainer] Checkpoint saved ({status}): {path}")

    def train(self, train_loader, val_loader, args):
        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            # Unfreeze backbones after freeze_epochs
            if args.freeze_backbone and epoch == args.freeze_epochs + 1:
                print(f"\n[Unfreeze] Unfreezing backbones at epoch {epoch}")
                for name, param in self.model.named_parameters():
                    if 'fast_pathway' in name or 'slow_pathway' in name:
                        param.requires_grad = True
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"[Unfreeze] Trainable params: {trainable:,}")

            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*50}")

            # Training
            train_metrics = self.train_epoch(train_loader)
            train_metrics.report(prefix="Train")

            # Validation
            if epoch % args.val_every == 0:
                val_metrics = self.validate(val_loader)
                val_metrics.report(prefix="Val")

                # Log to CSV
                row = {f'train_{k}': v for k, v in train_metrics.to_csv_row().items()}
                row.update({f'val_{k}': v for k, v in val_metrics.to_csv_row().items()})
                row['lr'] = self.optimizer.param_groups[0]['lr']
                self.csv_logger.log(row)

                # Save best checkpoint
                is_best = val_metrics.me_accuracy > best_val_acc
                if is_best:
                    best_val_acc = val_metrics.me_accuracy
                    best_path = os.path.join(args.output_dir, 'best_model.pth')
                    self.save_checkpoint(epoch, val_metrics, best_path, is_best=True)
            else:
                row = {f'train_{k}': v for k, v in train_metrics.to_csv_row().items()}
                row['lr'] = self.optimizer.param_groups[0]['lr']
                self.csv_logger.log(row)

            # Save periodic checkpoint
            if epoch % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
                self.save_checkpoint(epoch, train_metrics, ckpt_path)

            # Update LR
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            epoch_time = time.time() - epoch_start
            print(f"  LR: {current_lr:.2e} | Epoch time: {epoch_time:.1f}s")

        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Checkpoints saved to: {args.output_dir}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Censor MER Training (Frame Sequences)')

    # Data
    parser.add_argument('--synthetic_data', action='store_true',
                        help='Use synthetic data for debugging')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/data/CASME2',
                        help='Root directory of CASME2 data')

    # Frame sequence params
    parser.add_argument('--T', type=int, default=16, help='Number of frames to sample')
    parser.add_argument('--H', type=int, default=224, help='Spatial height')
    parser.add_argument('--W', type=int, default=224, help='Spatial width')

    # Training
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)

    # Loss weights
    parser.add_argument('--au_loss_weight', type=float, default=0.1,
                        help='AU loss weight (reduced -- AU labels are noisy)')
    parser.add_argument('--moe_loss_weight', type=float, default=0.01)
    parser.add_argument('--landmark_loss_weight', type=float, default=0.05)
    parser.add_argument('--supcon_weight', type=float, default=0.1,
                        help='Supervised contrastive loss weight (0 to disable)')

    # EMA
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay (0 to disable, 0.999 typical)')

    # Validation / Save
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

    # Face alignment
    parser.add_argument('--no_face_align', action='store_true',
                        help='Disable gaze stabilization reflex (face alignment)')

    # Onset-Apex difference mode (lateral inhibition)
    parser.add_argument('--diff_mode', action='store_true',
                        help='Use onset-apex difference instead of rPPG (lateral inhibition)')

    # Backbone freezing (for small datasets)
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze dual-pathway backbones (only train head)')
    parser.add_argument('--freeze_epochs', type=int, default=30,
                        help='Epochs to freeze backbone before unfreezing')

    # LOSO cross-validation
    parser.add_argument('--loso', action='store_true',
                        help='Enable Leave-One-Subject-Out cross-validation')
    parser.add_argument('--loso_fold', type=int, default=None,
                        help='Run specific LOSO fold (0-based). If not set with --loso, runs all folds.')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_fold(args, fold_idx, loso_subjects=None):
    """Train a single LOSO fold or standard train/val split."""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = Censor(fast_preprocess=True, diff_mode=args.diff_mode, verbose=False)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Freeze backbones if requested (small dataset regularization)
    if args.freeze_backbone:
        print(f"[Freeze] Freezing dual-pathway backbones for {args.freeze_epochs} epochs")
        for name, param in model.named_parameters():
            if 'fast_pathway' in name or 'slow_pathway' in name:
                param.requires_grad = False
        trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Freeze] Trainable params: {trainable_frozen:,} / {total_params:,}")

    fold_label = f"fold_{fold_idx}" if fold_idx is not None else "standard"
    output_dir = os.path.join(args.output_dir, fold_label)

    print(f"\n{'='*60}")
    print(f" Training: {fold_label}")
    print(f" Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"{'='*60}")

    # Data
    face_align = not args.no_face_align
    loso_fold = fold_idx if args.loso else None

    train_loader, val_loader = get_casme2_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        T=args.T, H=args.H, W=args.W,
        num_workers=args.num_workers,
        face_align=face_align,
        loso_fold=loso_fold,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Override output_dir for this fold
    fold_args = argparse.Namespace(**vars(args))
    fold_args.output_dir = output_dir

    # Trainer
    trainer = Trainer(model, device, fold_args)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"Resumed from epoch {ckpt['epoch']}")

    # Train
    trainer.train(train_loader, val_loader, fold_args)

    # Return best accuracy
    best_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location='cpu')
        best_acc = ckpt['metrics']['me_accuracy']
        best_f1 = ckpt['metrics']['me_f1']
        print(f"\n[{fold_label}] Best Val Acc: {best_acc:.4f}, F1: {best_f1:.4f}")
        return best_acc, best_f1
    return 0.0, 0.0


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Random seed: {args.seed}")
    print(f"Face alignment: {'disabled' if args.no_face_align else 'enabled (gaze stabilization reflex)'}")
    print(f"LOSO: {'enabled' if args.loso else 'disabled (random split)'}")

    if args.synthetic_data:
        # Synthetic data test
        print("\nUsing synthetic data for testing...")
        model = Censor(fast_preprocess=True, verbose=False)
        train_dataset = SyntheticMERDataset(num_samples=100, T=args.T, H=args.H, W=args.W)
        val_dataset = SyntheticMERDataset(num_samples=20, T=args.T, H=args.H, W=args.W)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)

        trainer = Trainer(model, device, args)
        trainer.train(train_loader, val_loader, args)

    elif args.loso:
        # LOSO cross-validation
        from dataset_frames import get_loso_subjects

        subjects = get_loso_subjects(args.data_root)
        if subjects is None:
            print("[Error] Cannot get subjects. Make sure labels.csv exists.")
            exit(1)

        num_folds = len(subjects)
        print(f"\nLOSO: {num_folds} subjects = {num_folds} folds")
        print(f"Subjects: {subjects}")

        # Determine which folds to run
        if args.loso_fold is not None:
            folds_to_run = [args.loso_fold]
        else:
            folds_to_run = list(range(num_folds))

        fold_results = {}
        for fold_idx in folds_to_run:
            acc, f1 = train_single_fold(args, fold_idx, loso_subjects=subjects)
            fold_results[fold_idx] = {'acc': acc, 'f1': f1, 'subject': subjects[fold_idx]}

        # Print LOSO summary
        print(f"\n{'='*60}")
        print(f" LOSO Cross-Validation Summary")
        print(f"{'='*60}")
        accs = []
        f1s = []
        for fold_idx, res in sorted(fold_results.items()):
            print(f"  Fold {fold_idx} ({res['subject']}): Acc={res['acc']:.4f}, F1={res['f1']:.4f}")
            accs.append(res['acc'])
            f1s.append(res['f1'])

        print(f"\n  Mean Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  Mean F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"{'='*60}")

        # Save LOSO summary
        summary_path = os.path.join(args.output_dir, 'loso_summary.csv')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['fold', 'subject', 'accuracy', 'f1'])
            writer.writeheader()
            for fold_idx, res in sorted(fold_results.items()):
                writer.writerow({
                    'fold': fold_idx,
                    'subject': res['subject'],
                    'accuracy': res['acc'],
                    'f1': res['f1'],
                })
        print(f"LOSO summary saved to: {summary_path}")

    else:
        # Standard train/val split
        acc, f1 = train_single_fold(args, fold_idx=None)
        print(f"\nFinal: Acc={acc:.4f}, F1={f1:.4f}")
