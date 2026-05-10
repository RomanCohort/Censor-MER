"""
Censor -- Training Script
=========================
Full training pipeline for the Biomimetic Dual-Pathway MER system.

Loss functions:
    1. L_me: Cross-entropy for 7-class micro-expression classification
    2. L_au: Binary cross-entropy for 28-class AU multi-label recognition
    3. L_aux: MoE load balancing loss (prevents expert collapse)
    4. L_opd: Temporal smoothness + peak consistency
    5. L_total = L_me + alpha * L_au + beta * L_aux + gamma * L_opd

Metrics:
    - Micro-expression: Accuracy, F1-score (per-class + weighted)
    - Action Units: F1-score (macro), Hamming loss

Usage:
    # Real dataset training (after data preparation):
    python train.py --dataset casme2 --data_root ./data/CASME_II --epochs 50 --batch_size 2

    # Debug with synthetic data:
    python train.py --synthetic_data --epochs 2 --batch_size 2

    # Synthetic data generator (for pipeline testing):
    python train.py --synthetic_data --epochs 50 --batch_size 2 --lr 1e-4
"""

import os
import sys
import csv
import glob
import argparse
import time
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent))
from main import Censor
from config.defaults import AU_DECODER_CONFIG, MOE_CONFIG, DATA_CONFIG


# =============================================================================
# Synthetic Dataset (for testing the training pipeline)
# =============================================================================

class SyntheticMERDataset(Dataset):
    """
    Synthetic micro-expression dataset for pipeline testing.

    Generates random video tensors with corresponding labels.
    Replace with actual dataset loader for production training.
    """

    ME_CATEGORIES = ["Happiness", "Sadness", "Surprise", "Fear", "Anger", "Disgust", "Contempt"]

    def __init__(self, num_samples=100, T=16, H=224, W=224):
        self.num_samples = num_samples
        self.T = T
        self.H = H
        self.W = W

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Video: (C, T, H, W), normalized with ImageNet stats
        mean = torch.tensor(DATA_CONFIG['normalize_mean']).view(-1, 1, 1, 1)
        std = torch.tensor(DATA_CONFIG['normalize_std']).view(-1, 1, 1, 1)
        video = torch.randn(3, self.T, self.H, self.W) * 0.1 + 0.5
        video = (video - mean) / std

        # ME label (random)
        me_label = torch.randint(0, len(self.ME_CATEGORIES), (1,)).item()

        # AU labels (28 random binary, ~30% activation)
        au_label = (torch.rand(28) > 0.7).float()

        return video, me_label, au_label


# =============================================================================
# Loss Functions
# =============================================================================

def compute_me_loss(me_logits, me_labels):
    """Cross-entropy loss for 7-class micro-expression classification."""
    return nn.CrossEntropyLoss()(me_logits, me_labels)


def compute_au_loss(au_intensities, au_labels):
    """Binary cross-entropy loss for multi-label AU recognition."""
    # Average over temporal dimension, then BCE over AU channels
    au_mean = au_intensities.mean(dim=1)  # (B, 28)
    return nn.BCELoss()(au_mean, au_labels)


def compute_landmark_loss(au_intensities, au_labels):
    """Onset-Peak-Decay (OPD) auxiliary loss for temporal smoothness."""
    # Temporal smoothness loss (L2 on temporal differences)
    temporal_diff = au_intensities[:, 1:, :] - au_intensities[:, :-1, :]
    smoothness_loss = torch.mean(temporal_diff ** 2)

    # Peak consistency: if AU is active, encourage high-activation frames
    au_labels_expanded = au_labels.unsqueeze(1).expand(-1, au_intensities.shape[1], -1)
    active_mask = (au_labels_expanded > 0.5).float()
    peak_consistency = -torch.mean(
        active_mask * torch.log(au_intensities + 1e-8)
    )

    return smoothness_loss + 0.1 * peak_consistency


# =============================================================================
# Metrics
# =============================================================================

class MetricsTracker:
    """Tracks training and validation metrics."""

    ME_CATEGORIES = ["Happiness", "Sadness", "Surprise", "Fear", "Anger", "Disgust", "Contempt"]

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
        """Update micro-expression accuracy."""
        preds = me_logits.argmax(dim=1)
        self.me_correct += (preds == me_labels).sum().item()
        self.me_total += me_labels.shape[0]
        for i in range(len(preds)):
            self.class_total[me_labels[i].item()] += 1
            if preds[i] == me_labels[i]:
                self.class_correct[me_labels[i].item()] += 1

    def update_au(self, au_intensities, au_labels):
        """Update AU F1 metrics."""
        au_mean = au_intensities.mean(dim=1)  # (B, 28)
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
        """Compute per-class F1 and return weighted average."""
        f1_scores = []
        for cls in range(7):
            total = self.class_total.get(cls, 0)
            correct = self.class_correct.get(cls, 0)
            # Precision / Recall for this class
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
        """Fraction of incorrect AU predictions."""
        return 1.0 - (self.au_true_pos / max(self.au_pred_pos + self.au_label_pos - self.au_true_pos, 1))

    def report(self, prefix="Train"):
        """Print current metrics."""
        avg_losses = {k: np.mean(v) for k, v in self.losses.items()}
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
        print(f"  [{prefix}] {loss_str}")
        print(f"  [{prefix}] ME Acc: {self.me_accuracy:.4f} | ME F1: {self.me_f1:.4f} | AU F1: {self.au_f1:.4f}")

    def to_csv_row(self):
        """Export metrics as a dict for CSV logging."""
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
# CSV Logger
# =============================================================================

class CSVLogger:
    """Appends metrics to a CSV file after each epoch."""

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
# Training Loop
# =============================================================================

class Trainer:
    """
    Trainer for the Censor model.

    Handles:
        - Training loop with mixed precision (AMP) — auto-detect device
        - Multi-task loss optimization
        - Gradient clipping
        - Learning rate scheduling (warmup + cosine)
        - Checkpointing (best + periodic)
        - CSV logging
        - Validation
    """

    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args

        # Optimizer with weight decay
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

        # Warmup + Cosine LR scheduler
        warmup_epochs = max(3, args.epochs // 20)  # 3 epochs or 5% of training
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                # Linear warmup
                optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, end_factor=1.0,
                    total_iters=warmup_epochs
                ),
                # Cosine annealing
                optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=args.epochs - warmup_epochs,
                    eta_min=args.lr * 0.01
                ),
            ],
            milestones=[warmup_epochs]
        )

        # Mixed precision — use torch.amp (PyTorch 1.10+)
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
            print("[Trainer] AMP disabled (no CUDA). Training on CPU.")

        # Loss weights
        self.loss_weights = {
            'me': 1.0,
            'au': args.au_loss_weight,
            'moe': args.moe_loss_weight,
            'landmark': args.landmark_loss_weight,
        }

        # CSV logger
        os.makedirs(args.output_dir, exist_ok=True)
        self.csv_logger = CSVLogger(os.path.join(args.output_dir, 'metrics.csv'))

    def train_epoch(self, train_loader):
        """Run one training epoch."""
        self.model.train()
        metrics = MetricsTracker()

        for batch_idx, (videos, me_labels, au_labels) in enumerate(train_loader):
            videos = videos.to(self.device)
            me_labels = me_labels.to(self.device)
            au_labels = au_labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(videos)
                    total_loss = self._compute_loss(outputs, me_labels, au_labels)
            else:
                outputs = self.model(videos)
                total_loss = self._compute_loss(outputs, me_labels, au_labels)

            # === FEEDBACK: Apply correction-based feedback to BioMoE ===
            if hasattr(self.model, 'moe') and hasattr(self.model.moe, 'apply_feedback'):
                with torch.no_grad():
                    preds = outputs['me_logits'].argmax(dim=1)
                    correct = (preds == me_labels).float()
                    # Batch average feedback: 1.0 if correct, 0.0 if wrong
                    fb = correct.mean().item()
                    # Apply as feedback (only in hybrid mode)
                    if hasattr(self.model.moe, 'mode') and self.model.moe.mode == 'hybrid':
                        self.model.moe.apply_feedback(fb)
            # === END FEEDBACK ===

            # Backward pass with gradient clipping
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
        """Compute weighted multi-task loss."""
        loss_me = compute_me_loss(outputs['me_logits'], me_labels)
        loss_au = compute_au_loss(outputs['au_intensities'], au_labels)
        loss_landmark = compute_landmark_loss(outputs['au_intensities'], au_labels)
        loss_moe = outputs['moe_aux_loss']

        total_loss = (
            self.loss_weights['me'] * loss_me +
            self.loss_weights['au'] * loss_au +
            self.loss_weights['moe'] * loss_moe +
            self.loss_weights['landmark'] * loss_landmark
        )
        return total_loss

    @torch.no_grad()
    def validate(self, val_loader):
        """Run validation."""
        self.model.eval()
        metrics = MetricsTracker()

        for videos, me_labels, au_labels in val_loader:
            videos = videos.to(self.device)
            me_labels = me_labels.to(self.device)
            au_labels = au_labels.to(self.device)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(videos)
            else:
                outputs = self.model(videos)

            metrics.update_me(outputs['me_logits'], me_labels)
            metrics.update_au(outputs['au_intensities'], au_labels)
            loss_me = compute_me_loss(outputs['me_logits'], me_labels)
            loss_au = compute_au_loss(outputs['au_intensities'], au_labels)
            metrics.update_loss('me', loss_me.item())
            metrics.update_loss('au', loss_au.item())

        return metrics

    def save_checkpoint(self, epoch, metrics, path, is_best=False):
        """Save model checkpoint."""
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
        torch.save(checkpoint, path)
        status = "BEST" if is_best else f"epoch {epoch}"
        print(f"[Trainer] Checkpoint saved ({status}): {path}")

    def train(self, train_dataset, val_dataset, args):
        """Full training pipeline."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,  # No CUDA available
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False,
        )

        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

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
                # Log training metrics only
                row = {f'train_{k}': v for k, v in train_metrics.to_csv_row().items()}
                row['lr'] = self.optimizer.param_groups[0]['lr']
                self.csv_logger.log(row)

            # Save periodic checkpoint
            if epoch % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
                val_m = val_loader is not None and self.validate(val_loader) if epoch % args.val_every == 0 else train_metrics
                self.save_checkpoint(epoch, val_m, ckpt_path)

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
    parser = argparse.ArgumentParser(description='Censor MER Training')

    # Data
    parser.add_argument('--synthetic_data', action='store_true',
                        help='Use synthetic data for debugging')
    parser.add_argument('--dataset', type=str, default='casme2',
                        choices=['casme2', 'samm', 'smic', 'mmew', 'casme3'],
                        help='Dataset name (used to find data subfolder)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory of dataset')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)

    # Loss weights
    parser.add_argument('--au_loss_weight', type=float, default=0.5)
    parser.add_argument('--moe_loss_weight', type=float, default=0.01)
    parser.add_argument('--landmark_loss_weight', type=float, default=0.1)

    # Validation / Save
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")

    # Model
    print("Building Censor model...")
    model = Censor()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Data
    if args.synthetic_data:
        print("Using synthetic data for testing...")
        print("  (Use --dataset + --data_root for real data)")
        train_dataset = SyntheticMERDataset(num_samples=100)
        val_dataset = SyntheticMERDataset(num_samples=20)
    else:
        from dataset import MERDataset

        data_dir = os.path.join(args.data_root, args.dataset)
        print(f"\nLoading dataset: {args.dataset}")
        print(f"Data directory: {data_dir}")

        if not os.path.exists(os.path.join(data_dir, 'videos')):
            print(f"[ERROR] 'videos/' folder not found in {data_dir}")
            print(f"\nDataset structure should be:")
            print(f"  {data_dir}/")
            print(f"    videos/")
            print(f"      video1.avi")
            print(f"      video2.avi")
            print(f"      ...")
            print(f"    labels.csv  (columns: video_path,subject,me_label,au_01,...,au_28)")
            print(f"\nTo prepare the dataset, run: python prepare_data.py --dataset {args.dataset}")
            sys.exit(1)

        # Full dataset (train split for augmentation)
        full_dataset = MERDataset(data_dir, split='train')

        # Random split: 80% train, 20% val
        val_ratio = 0.2
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size

        rng = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=rng
        )

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

    # Trainer
    trainer = Trainer(model, device, args)
    trainer.train(train_dataset, val_dataset, args)