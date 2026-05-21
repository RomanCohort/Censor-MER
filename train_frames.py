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
    ME_CATEGORIES = ["Happiness", "Sadness", "Surprise", "Fear", "Anger", "Disgust", "Contempt"]

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
        me_label = torch.randint(0, len(self.ME_CATEGORIES), (1,)).item()
        au_label = (torch.rand(28) > 0.7).float()
        return video, me_label, au_label


# =============================================================================
# Loss Functions
# =============================================================================

def compute_me_loss(me_logits, me_labels):
    return nn.CrossEntropyLoss()(me_logits, me_labels)


def compute_au_loss(au_intensities, au_labels):
    au_mean = au_intensities.mean(dim=1)  # (B, 28)
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
        for cls in range(7):
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

        total_loss = (
            self.loss_weights['me'] * loss_me +
            self.loss_weights['au'] * loss_au +
            self.loss_weights['moe'] * loss_moe +
            self.loss_weights['landmark'] * loss_landmark
        )
        return total_loss

    @torch.no_grad()
    def validate(self, val_loader):
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

    def train(self, train_loader, val_loader, args):
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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
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
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

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


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"Random seed: {args.seed}")

    # Model
    print("\nBuilding Censor model...")
    model = Censor(fast_preprocess=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Data
    if args.synthetic_data:
        print("\nUsing synthetic data for testing...")
        train_dataset = SyntheticMERDataset(num_samples=100, T=args.T, H=args.H, W=args.W)
        val_dataset = SyntheticMERDataset(num_samples=20, T=args.T, H=args.H, W=args.W)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
    else:
        print(f"\nLoading CASME2 frame sequences from: {args.data_root}")
        train_loader, val_loader = get_casme2_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            T=args.T, H=args.H, W=args.W,
            num_workers=args.num_workers,
        )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Resume
    trainer = Trainer(model, device, args)
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"Resumed from epoch {ckpt['epoch']}")

    # Train
    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    trainer.train(train_loader, val_loader, args)
