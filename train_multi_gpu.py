"""
Censor -- Multi-GPU Distributed Training
========================================
High-performance training pipeline with:
- PyTorch DistributedDataParallel (DDP)
- Mixed Precision Training (AMP)
- Gradient accumulation for large models
- Multi-node support
- Resume from checkpoint

Usage:
    # Single node, multi-GPU:
    python -m torch.distributed.launch --nproc_per_node=4 train_multi_gpu.py --epochs 50

    # Multi-node (run on each node):
    python -m torch.distributed.launch --nproc_per_node=4 --master_addr=192.168.1.1 train_multi_gpu.py

    # Resume training:
    python train_multi_gpu.py --resume ./checkpoints/checkpoint_epoch_10.pth
"""

import os
import sys
import csv
import glob
import argparse
import time
import random
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))
from main import Censor
from train import (
    SyntheticMERDataset,
    MetricsTracker,
    CSVLogger,
    compute_me_loss,
    compute_au_loss,
    compute_landmark_loss,
)

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("[WARN] pynvml not available, GPU monitoring disabled")


# =============================================================================
# Utility Functions
# =============================================================================

def setup(rank, world_size, master_addr='127.0.0.1', master_port=29500):
    """Initialize distributed process group."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup():
    """Cleanup distributed process group."""
    dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes."""
    with torch.no_grad():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / world_size
    return tensor


def is_main_process(rank):
    """Check if this is the main process."""
    return rank == 0


def save_on_main(obj, path):
    """Save only on main process."""
    if is_main_process(dist.get_rank()):
        torch.save(obj, path)


# =============================================================================
# GPU Monitoring
# =============================================================================

class GPUMonitor:
    """Monitor GPU stats using pynvml."""

    def __init__(self):
        if not PYNVML_AVAILABLE:
            self.available = False
            return

        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                        for i in range(self.device_count)]
            self.available = True
        except Exception as e:
            print(f"[WARN] NVML init failed: {e}")
            self.available = False

    def get_stats(self, device_id=None):
        """Get GPU stats for specified device or all devices."""
        if not self.available:
            return {}

        stats = {}
        devices = [device_id] if device_id is not None else range(self.device_count)

        for i in devices:
            try:
                handle = self.handles[i]
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                stats[i] = {
                    'memory_used_mb': mem_info.used / 1024**2,
                    'memory_total_mb': mem_info.total / 1024**2,
                    'memory_used_pct': 100 * mem_info.used / max(mem_info.total, 1),
                    'utilization_pct': util.gpu,
                    'temperature_c': temp,
                }
            except Exception:
                pass

        return stats

    def __del__(self):
        if self.available:
            pynvml.nvmlShutdown()


# =============================================================================
# Distributed Trainer
# =============================================================================

class DistributedTrainer:
    """
    Distributed trainer using DDP.

    Handles:
        - Multi-GPU gradient synchronization
        - Mixed precision training (AMP)
        - Gradient accumulation
        - Learning rate scaling
        - State logging
    """

    def __init__(self, model, rank, world_size, args):
        self.rank = rank
        self.world_size = world_size
        self.args = args

        # Wrap model with DDP
        self.model = DDP(
            model.to(args.device),
            device_ids=[rank],
            find_unused_parameters=True
        )

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

        # Mixed precision scaler
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Loss weights
        self.loss_weights = {
            'me': 1.0,
            'au': args.au_loss_weight,
            'moe': args.moe_loss_weight,
            'landmark': args.landmark_loss_weight,
        }

        # GPU monitor
        self.gpu_monitor = GPUMonitor() if PYNVML_AVAILABLE else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Metrics tracker (for main process)
        self.metrics_tracker = MetricsTracker()

    def load_state(self, checkpoint_path):
        """Load training state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            return

        checkpoint = torch.load(
            checkpoint_path,
            map_location=f'cuda:{self.rank}'
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)

        if is_main_process(self.rank):
            print(f"[Trainer] Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def save_checkpoint(self, path, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': vars(self.args),
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if is_main_process(self.rank):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(checkpoint, path)

            # Also save as best model
            if is_best:
                best_path = os.path.join(
                    os.path.dirname(path), 'best_model.pth'
                )
                torch.save(checkpoint, best_path)

    def train_step(self, videos, me_labels, au_labels):
        """Single training step."""
        self.model.train()

        # Forward pass with AMP
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                outputs = self.model(videos)
                total_loss = self._compute_loss(outputs, me_labels, au_labels)
        else:
            outputs = self.model(videos)
            total_loss = self._compute_loss(outputs, me_labels, au_labels)

        # Handle gradient accumulation
        loss_scale = self.args.gradient_accumulation_steps
        total_loss = total_loss / loss_scale

        # Backward pass
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # Gradient clipping
        if (self.global_step + 1) % loss_scale == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.args.max_grad_norm
            )

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        self.global_step += 1

        # Compute individual losses for logging
        with torch.no_grad():
            loss_me = compute_me_loss(outputs['me_logits'], me_labels).item()
            loss_au = compute_au_loss(outputs['au_intensities'], au_labels).item()
            loss_moe = outputs['moe_aux_loss'].item()

        return {
            'total_loss': total_loss.item() * loss_scale,
            'loss_me': loss_me,
            'loss_au': loss_au,
            'loss_moe': loss_moe,
            'outputs': outputs,
        }

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
        """Validation pass."""
        self.model.eval()
        metrics = MetricsTracker()

        for videos, me_labels, au_labels in val_loader:
            videos = videos.to(self.args.device)
            me_labels = me_labels.to(self.args.device)
            au_labels = au_labels.to(self.args.device)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(videos)
            else:
                outputs = self.model(videos)

            metrics.update_me(outputs['me_logits'], me_labels)
            metrics.update_au(outputs['au_intensities'], au_labels)

        # Reduce metrics across processes
        metrics.me_correct = reduce_tensor(
            torch.tensor(metrics.me_correct, device=self.args.device),
            self.world_size
        ).item()
        metrics.me_total = reduce_tensor(
            torch.tensor(metrics.me_total, device=self.args.device),
            self.world_size
        ).item()

        return metrics

    def get_gpu_stats(self):
        """Get current GPU stats."""
        if self.gpu_monitor:
            return self.gpu_monitor.get_stats(self.rank)
        return {}


def run_training(rank, world_size, args):
    """Main training loop on a single process."""

    # Setup distributed
    setup(rank, world_size, args.master_addr, args.master_port)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Set seed
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Attach args device
    args.device = device

    if is_main_process(rank):
        print(f"\n{'='*50}")
        print(f"Starting Training on Rank {rank}/{world_size}")
        print(f"{'='*50}")

    # Create model
    model = Censor()

    # Create trainer
    trainer = DistributedTrainer(model, rank, world_size, args)

    # Load checkpoint if resuming
    if args.resume:
        trainer.load_state(args.resume)

    # Dataset
    if args.synthetic_data:
        train_dataset = SyntheticMERDataset(num_samples=args.train_samples)
        val_dataset = SyntheticMERDataset(num_samples=args.val_samples)
    else:
        from dataset import MERDataset
        data_dir = os.path.join(args.data_root, args.dataset)
        full_dataset = MERDataset(data_dir, split='train')

        val_ratio = 0.2
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size

        rng = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=rng
        )

    # Distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # CSV logger (main process only)
    if is_main_process(rank):
        csv_logger = CSVLogger(os.path.join(args.output_dir, 'metrics.csv'))
    else:
        csv_logger = None

    # Training loop
    best_val_acc = 0.0

    for epoch in range(trainer.current_epoch + 1, args.epochs + 1):
        trainer.current_epoch = epoch
        train_sampler.set_epoch(epoch)

        epoch_start = time.time()

        if is_main_process(rank):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*50}")

        # Training
        metrics = MetricsTracker()
        trainer.model.train()

        for batch_idx, (videos, me_labels, au_labels) in enumerate(train_loader):
            videos = videos.to(device)
            me_labels = me_labels.to(device)
            au_labels = au_labels.to(device)

            results = trainer.train_step(videos, me_labels, au_labels)

            # Update metrics
            outputs = results['outputs']
            metrics.update_me(outputs['me_logits'], me_labels)
            metrics.update_au(outputs['au_intensities'], au_labels)
            metrics.update_loss('me', results['loss_me'])
            metrics.update_loss('au', results['loss_au'])
            metrics.update_loss('moe', results['loss_moe'])
            metrics.update_loss('total', results['total_loss'])

            if is_main_process(rank) and (batch_idx + 1) % args.log_interval == 0:
                metrics.report(prefix=f"Train Batch {batch_idx+1}/{len(train_loader)}")

        # Validation
        if epoch % args.val_every == 0:
            val_metrics = trainer.validate(val_loader)

            # Save if best
            if is_main_process(rank) and val_metrics.me_accuracy > best_val_acc:
                best_val_acc = val_metrics.me_accuracy
                is_best = True
            else:
                is_best = False

            if is_main_process(rank):
                val_metrics.report(prefix="Val")
        else:
            val_metrics = None
            is_best = False

        # Save checkpoint
        if is_main_process(rank):
            if epoch % args.save_every == 0:
                ckpt_path = os.path.join(
                    args.output_dir,
                    f'checkpoint_epoch_{epoch}.pth'
                )
                trainer.save_checkpoint(ckpt_path, is_best=is_best)

            # Log to CSV
            row = {f'train_{k}': v for k, v in metrics.to_csv_row().items()}
            if val_metrics:
                row.update({f'val_{k}': v for k, v in val_metrics.to_csv_row().items()})
            row['lr'] = trainer.optimizer.param_groups[0]['lr']
            row['gpu_memory_mb'] = trainer.get_gpu_stats().get(rank, {}).get('memory_used_mb', 0)
            csv_logger.log(row)

        # Sync
        dist.barrier()

        epoch_time = time.time() - epoch_start

        if is_main_process(rank):
            current_lr = trainer.optimizer.param_groups[0]['lr']
            gpu_stats = trainer.get_gpu_stats()
            gpu_mem = gpu_stats.get(rank, {}).get('memory_used_mb', 0)

            print(f"  LR: {current_lr:.2e} | Epoch time: {epoch_time:.1f}s | GPU Mem: {gpu_mem:.0f}MB")

    # Cleanup
    cleanup()

    if is_main_process(rank):
        print(f"\nTraining completed! Best accuracy: {best_val_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Censor Multi-GPU Training')

    # Distributed
    parser.add_argument('--nproc_per_node', type=int, default=1,
                        help='Number of processes per node')
    parser.add_argument('--nnodes', type=int, default=1,
                        help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='Rank of current node')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1',
                        help='Master node address')
    parser.add_argument('--master_port', type=int, default=29500,
                        help='Master node port')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Data
    parser.add_argument('--synthetic_data', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--dataset', type=str, default='casme2',
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    parser.add_argument('--train_samples', type=int, default=100,
                        help='Synthetic train samples')
    parser.add_argument('--val_samples', type=int, default=20,
                        help='Synthetic val samples')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')

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
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    world_size = args.nproc_per_node * args.nnodes

    if world_size > 1:
        # Distributed training
        mp.spawn(
            run_training,
            args=(world_size, args),
            nprocs=args.nproc_per_node,
            join=True
        )
    else:
        # Single GPU/CPU training
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        run_training(0, 1, args)


if __name__ == '__main__':
    main()