"""
Censor -- Multi-Dataset Joint Pretrain + Fine-tune
====================================================
SOTA training pipeline for micro-expression recognition.

Strategy:
  Phase 1: Joint pretrain on CASME2 + SMIC + SAMM (unified 5-class)
  Phase 2: Fine-tune on target dataset (CASME2 LOSO)

Key SOTA techniques:
  - ArcFace angular margin loss
  - Manifold MixUp in feature space
  - Supervised Contrastive (SupCon) loss
  - LOSO (Leave-One-Subject-Out) evaluation
  - Differential LR (backbone 10x smaller than head)
  - Warmup + Cosine annealing scheduler
  - Gradient accumulation for effective larger batch

Usage:
  # Phase 1: Joint pretrain (all 3 datasets)
  python train_cross.py --phase pretrain \
      --casme_root /root/autodl-tmp/data/CASME2 \
      --smic_root /root/autodl-tmp/data/SMIC \
      --samm_root /root/autodl-tmp/data/SAMM \
      --use_arcface --arcface_margin 0.2 \
      --mixup_alpha 0.2 --supcon_weight 0.1

  # Phase 2: Fine-tune on CASME2 with LOSO
  python train_cross.py --phase finetune \
      --casme_root /root/autodl-tmp/data/CASME2 \
      --pretrained pretrained_joint_best.pth \
      --loso --use_arcface --lr 1e-4

  # Phase 2 alt: Fine-tune on SMIC
  python train_cross.py --phase finetune \
      --smic_root /root/autodl-tmp/data/SMIC \
      --pretrained pretrained_joint_best.pth \
      --use_arcface --lr 1e-4
"""

import os
import sys
import csv
import argparse
import time
import random
import math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent))

from config.defaults import (
    INPUT_CONFIG, FAST_PATHWAY_CONFIG, SLOW_PATHWAY_CONFIG,
    AMYGDALA_CONFIG, FFA_CONFIG, CASA_CONFIG, FUSION_CONFIG,
    AU_DECODER_CONFIG, MOE_CONFIG, RADAR_CONFIG,
)
from main import Censor
from dataset_frames import CASME2FrameDataset, SMICFrameDataset, SAMMFrameDataset
from train_frames import compute_me_loss, compute_au_loss, compute_landmark_loss


# =============================================================================
# Unified 5-class emotion mapping (CASME2 + SMIC + SAMM compatible)
# =============================================================================
# CASME2: happiness(0), surprise(1), disgust(2), fear(3), repression(4), others(-1)
# SMIC:   happiness(0), surprise(1), disgust(2), fear(3), others(-1)
# SAMM:   happiness(0), surprise(1), disgust(2), fear(3), anger(4), contempt(5), others(-1)
# Unified 5-class: happiness(0), surprise(1), disgust(2), fear(3), anger(4)
# "others", "repression", "contempt" are excluded from training

UNIFIED_EMOTION_MAP = {
    'happiness': 0,
    'surprise': 1,
    'disgust': 2,
    'fear': 3,
    'anger': 4,
    # Excluded:
    'sadness': -1,
    'contempt': -1,
    'repression': -1,
    'others': -1,
    'other': -1,
}

NUM_UNIFIED_CLASSES = 5


# =============================================================================
# Loss Functions
# =============================================================================

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device)

        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float()
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_per_sample = mask.sum(dim=1)
        has_pos = pos_per_sample > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=device)

        mean_log_prob = (mask * log_prob).sum(dim=1) / (pos_per_sample + 1e-8)
        loss = -mean_log_prob[has_pos].mean()
        return loss


class ArcFaceLoss(nn.Module):
    """ArcFace: Additive Angular Margin Loss (Deng et al., 2019)."""
    def __init__(self, in_features, out_features, margin=0.2, scale=30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(features, weight)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(1.0 - cosine ** 2)
        cos_theta_plus_m = cosine * self.cos_m - sine * self.sin_m

        one_hot = F.one_hot(labels, self.out_features).float()

        output = torch.where(
            cosine > self.threshold,
            cos_theta_plus_m,
            cosine - self.mm
        )

        final_cosine = one_hot * output + (1.0 - one_hot) * cosine
        logits = self.scale * final_cosine
        loss = F.cross_entropy(logits, labels)
        return loss


# =============================================================================
# Manifold MixUp
# =============================================================================

def manifold_mixup(feat, y_me, y_au, alpha=0.2):
    """Feature-space MixUp. Preserves micro-expression signals better than pixel-level."""
    if alpha <= 0:
        return feat, y_me, y_me, y_au, y_au, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = feat.size(0)
    index = torch.randperm(batch_size, device=feat.device)
    mixed_feat = lam * feat + (1 - lam) * feat[index]
    y_me_a, y_me_b = y_me, y_me[index]
    y_au_a, y_au_b = y_au, y_au[index]
    return mixed_feat, y_me_a, y_me_b, y_au_a, y_au_b, lam


# =============================================================================
# Multi-Dataset Trainer
# =============================================================================

class CrossDatasetTrainer:
    """
    Multi-dataset joint pretrain + fine-tune trainer.

    Phase 1 (pretrain): CASME2 + SMIC + SAMM joint training, unified 5-class
    Phase 2 (finetune): Single dataset fine-tuning, LOSO or random split
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.phase = args.phase  # 'pretrain' or 'finetune'

        # Build model
        print(f"\n[Censor] Building model (phase={self.phase})...")
        self.model = Censor(
            fast_preprocess=True,
            diff_mode=True,
            verbose=False,
            enable_sparse_control=False,
        ).to(self.device)

        # Adjust MoE head for unified 5-class if pretraining
        if self.phase == 'pretrain':
            self._adjust_moe_for_classes(NUM_UNIFIED_CLASSES)

        # ArcFace
        self.use_arcface = args.use_arcface
        if self.use_arcface:
            num_cls = NUM_UNIFIED_CLASSES if self.phase == 'pretrain' else 4
            self.arcface_loss = ArcFaceLoss(
                in_features=1024,
                out_features=num_cls,
                margin=args.arcface_margin,
                scale=30.0,
            ).to(self.device)
            print(f"[Trainer] ArcFace enabled (margin={args.arcface_margin}, classes={num_cls})")

        # SupCon
        self.supcon_loss = SupConLoss(temperature=0.07)

        # Loss weights
        self.loss_weights = {
            'me': 1.0,
            'au': 0.1,
            'moe': 0.01,
            'landmark': 0.05,
            'supcon': args.supcon_weight,
            'arcface': args.arcface_weight,
        }

        # Load pretrained checkpoint for finetune
        if self.phase == 'finetune' and args.pretrained:
            self._load_pretrained(args.pretrained)

        # Optimizer with differential LR
        self._setup_optimizer()

        # Datasets
        self._setup_datasets()

        # Scheduler
        self._setup_scheduler()

        # Best tracking
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.patience_counter = 0

        # CSV logger
        csv_path = args.log_dir or './logs'
        os.makedirs(csv_path, exist_ok=True)
        phase_tag = 'pretrain' if self.phase == 'pretrain' else 'finetune'
        self.csv_file = open(os.path.join(csv_path, f'{phase_tag}_log.csv'), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'epoch', 'train_loss', 'val_loss', 'val_acc', 'val_f1',
            'lr', 'phase'
        ])

    def _adjust_moe_for_classes(self, num_classes):
        """Adjust MoE head output for different number of classes."""
        moe = self.model.moe
        # Reinitialize the expert heads for new class count
        for expert in moe.experts:
            old_linear = expert[-1]  # Last layer is Linear
            new_linear = nn.Linear(old_linear.in_features, num_classes).to(self.device)
            nn.init.xavier_uniform_(new_linear.weight)
            expert[-1] = new_linear
        # Reinitialize gating network
        old_gate = moe.gate
        moe.gate = nn.Sequential(
            nn.Linear(moe.input_dim, moe.gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(moe.gating_hidden_dim, moe.num_experts),
        ).to(self.device)
        print(f"[Censor] Adjusted MoE for {num_classes} classes")

    def _load_pretrained(self, pretrained_path):
        """Load pretrained weights for fine-tuning."""
        print(f"[Trainer] Loading pretrained: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Filter out mismatched keys (MoE head may have different num_classes)
        model_dict = self.model.state_dict()
        loaded_keys = []
        skipped_keys = []

        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    model_dict[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
            # Skip keys not in current model (e.g., arcface_loss.weight)

        self.model.load_state_dict(model_dict)
        print(f"[Trainer] Loaded {len(loaded_keys)} params, skipped {len(skipped_keys)} shape mismatches")
        if skipped_keys:
            for s in skipped_keys[:5]:
                print(f"  Skip: {s}")

    def _setup_optimizer(self):
        """Differential LR: backbone 10x smaller than head."""
        backbone_params = []
        head_params = []

        # Identify backbone vs head modules
        backbone_names = {'fast_pathway', 'slow_pathway', 'saliency', 'rppg', 'flow'}
        head_names = {'amygdala', 'ffa', 'casa', 'fusion', 'au_decoder', 'moe', 'radar'}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(bn in name for bn in backbone_names):
                backbone_params.append(param)
            else:
                head_params.append(param)

        lr = self.args.lr
        backbone_lr = lr * self.args.backbone_lr_factor

        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
            {'params': head_params, 'lr': lr, 'name': 'head'},
        ]

        # Add ArcFace params to head group
        if self.use_arcface:
            param_groups.append({
                'params': self.arcface_loss.parameters(),
                'lr': lr, 'name': 'arcface'
            })

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
        )

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        backbone_count = sum(p.numel() for p in backbone_params)
        head_count = sum(p.numel() for p in head_params)
        print(f"[Trainer] Total trainable: {total_params:,}")
        print(f"[Trainer]   Backbone: {backbone_count:,} (lr={backbone_lr:.2e})")
        print(f"[Trainer]   Head:     {head_count:,} (lr={lr:.2e})")

    def _setup_datasets(self):
        """Setup datasets for current phase."""
        args = self.args

        if self.phase == 'pretrain':
            # Joint pretrain: load all available datasets
            datasets = []

            if args.casme_root and os.path.exists(args.casme_root):
                ds = CASME2FrameDataset(
                    root_dir=args.casme_root,
                    T=args.T, H=args.H, W=args.W,
                    augment=True,
                    emotion_map=UNIFIED_EMOTION_MAP,
                    exclude_negative=True,  # Exclude "others" class
                )
                datasets.append(ds)
                print(f"[Data] CASME2: {len(ds)} samples")

            if args.smic_root and os.path.exists(args.smic_root):
                ds = SMICFrameDataset(
                    root_dir=args.smic_root,
                    T=args.T, H=args.H, W=args.W,
                    augment=True,
                    emotion_map=UNIFIED_EMOTION_MAP,
                    exclude_negative=True,
                )
                datasets.append(ds)
                print(f"[Data] SMIC: {len(ds)} samples")

            if args.samm_root and os.path.exists(args.samm_root):
                ds = SAMMFrameDataset(
                    root_dir=args.samm_root,
                    T=args.T, H=args.H, W=args.W,
                    augment=True,
                    emotion_map=UNIFIED_EMOTION_MAP,
                    exclude_negative=True,
                )
                datasets.append(ds)
                print(f"[Data] SAMM: {len(ds)} samples")

            if not datasets:
                raise ValueError("No datasets found! Check --casme_root, --smic_root, --samm_root")

            self.train_dataset = ConcatDataset(datasets)
            total = len(self.train_dataset)
            # 90/10 split for pretrain validation
            val_size = max(1, int(total * 0.1))
            train_size = total - val_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            print(f"[Data] Joint pretrain: {train_size} train, {val_size} val")

        elif self.phase == 'finetune':
            # Fine-tune on single target dataset
            target = args.target_dataset

            if target == 'casme2':
                DatasetClass = CASME2FrameDataset
                root = args.casme_root
            elif target == 'smic':
                DatasetClass = SMICFrameDataset
                root = args.smic_root
            elif target == 'samm':
                DatasetClass = SAMMFrameDataset
                root = args.samm_root
            else:
                raise ValueError(f"Unknown target dataset: {target}")

            if not root or not os.path.exists(root):
                raise ValueError(f"Target dataset root not found: {root}")

            full_dataset = DatasetClass(
                root_dir=root,
                T=args.T, H=args.H, W=args.W,
                augment=True,
            )

            if args.loso:
                # LOSO: Leave-One-Subject-Out
                self.loso_subjects = full_dataset.subjects if hasattr(full_dataset, 'subjects') else []
                self.full_dataset = full_dataset
                self.train_dataset = None  # Will be set per-fold
                self.val_dataset = None
                print(f"[Data] LOSO mode: {len(self.loso_subjects)} subjects")
            else:
                # Random 80/20 split
                total = len(full_dataset)
                val_size = max(1, int(total * 0.2))
                train_size = total - val_size
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    full_dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
                print(f"[Data] Fine-tune: {train_size} train, {val_size} val")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        ) if self.train_dataset else None

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        ) if self.val_dataset else None

    def _setup_scheduler(self):
        """Warmup + Cosine annealing scheduler."""
        total_steps = len(self.train_loader) * self.args.epochs
        warmup_steps = min(self.args.warmup_epochs * len(self.train_loader), total_steps // 10)

        # Linear warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=warmup_steps,
        )

        # Cosine annealing
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.args.lr * 0.01,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        self.scheduler_step_per_batch = True
        print(f"[Trainer] Scheduler: {warmup_steps} warmup steps + cosine (total {total_steps})")

    def _compute_loss(self, outputs, me_labels, au_labels,
                      mixup_outputs=None, mixup_lam=1.0):
        """Compute total loss with ArcFace, SupCon, MixUp support."""
        # ME classification loss
        if self.use_arcface and 'adapted_feat' in outputs:
            loss_me = self.arcface_loss(outputs['adapted_feat'], me_labels)
        else:
            loss_me = compute_me_loss(outputs['me_logits'], me_labels,
                                      label_smoothing=self.args.label_smoothing)

        loss_au = compute_au_loss(outputs['au_intensities'], au_labels)
        loss_landmark = compute_landmark_loss(outputs['au_intensities'], au_labels)
        loss_moe = outputs['moe_aux_loss']

        # Manifold MixUp
        if mixup_outputs is not None and mixup_lam < 1.0:
            me_labels_b = mixup_outputs['me_labels_b']
            au_labels_b = mixup_outputs['au_labels_b']
            loss_me_b = compute_me_loss(mixup_outputs['me_logits'], me_labels_b,
                                        label_smoothing=self.args.label_smoothing)
            loss_au_b = compute_au_loss(mixup_outputs['au_intensities'], au_labels_b)
            loss_me = mixup_lam * loss_me + (1 - mixup_lam) * loss_me_b
            loss_au = mixup_lam * loss_au + (1 - mixup_lam) * loss_au_b

        # SupCon loss
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

    def train_epoch(self):
        """Train one epoch with Manifold MixUp."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        accum_steps = self.args.grad_accum_steps

        for batch_idx, batch in enumerate(self.train_loader):
            frames = batch['frames'].to(self.device)
            me_labels = batch['me_label'].to(self.device)
            au_labels = batch.get('au_label', torch.zeros(frames.size(0), 28)).to(self.device)

            # Forward
            outputs = self.model(frames)

            # Manifold MixUp on adapted_feat
            mixup_outputs = None
            mixup_lam = 1.0
            if self.args.mixup_alpha > 0 and 'adapted_feat' in outputs:
                mixed_feat, me_a, me_b, au_a, au_b, lam = manifold_mixup(
                    outputs['adapted_feat'], me_labels, au_labels,
                    alpha=self.args.mixup_alpha
                )
                if lam < 1.0:
                    mixup_lam = lam
                    me_logits_mix, _, _ = self.model.moe(mixed_feat)
                    au_intensities_mix, _ = self.model.au_decoder(mixed_feat)
                    mixup_outputs = {
                        'me_logits': me_logits_mix,
                        'au_intensities': au_intensities_mix,
                        'me_labels_b': me_b,
                        'au_labels_b': au_b,
                    }

            # Compute loss
            loss = self._compute_loss(outputs, me_labels, au_labels,
                                      mixup_outputs, mixup_lam)
            loss = loss / accum_steps

            # Backward
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler_step_per_batch:
                    self.scheduler.step()

            # Metrics
            total_loss += loss.item() * accum_steps
            preds = outputs['me_logits'].argmax(dim=1)
            correct += (preds == me_labels).sum().item()
            total += me_labels.size(0)

        avg_loss = total_loss / max(1, len(self.train_loader))
        acc = correct / max(1, total)
        return avg_loss, acc

    @torch.no_grad()
    def validate(self, loader=None):
        """Validate on given loader."""
        self.model.eval()
        if loader is None:
            loader = self.val_loader

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch in loader:
            frames = batch['frames'].to(self.device)
            me_labels = batch['me_label'].to(self.device)
            au_labels = batch.get('au_label', torch.zeros(frames.size(0), 28)).to(self.device)

            outputs = self.model(frames)
            loss = self._compute_loss(outputs, me_labels, au_labels)

            total_loss += loss.item()
            preds = outputs['me_logits'].argmax(dim=1)
            correct += (preds == me_labels).sum().item()
            total += me_labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(me_labels.cpu().numpy())

        avg_loss = total_loss / max(1, len(loader))
        acc = correct / max(1, total)

        # Compute F1
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return avg_loss, acc, f1

    def train(self):
        """Main training loop."""
        args = self.args

        if args.loso and self.phase == 'finetune':
            self._train_loso()
        else:
            self._train_standard()

    def _train_standard(self):
        """Standard training loop (pretrain or finetune with random split)."""
        args = self.args
        print(f"\n{'='*60}")
        print(f" Phase: {self.phase.upper()}")
        print(f" Epochs: {args.epochs}")
        print(f" Batch size: {args.batch_size}")
        print(f" LR: {args.lr} (backbone {args.lr * args.backbone_lr_factor:.2e})")
        print(f" ArcFace: {self.use_arcface}")
        print(f" MixUp alpha: {args.mixup_alpha}")
        print(f" SupCon weight: {args.supcon_weight}")
        print(f"{'='*60}\n")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_f1 = self.validate()

            # LR logging
            current_lr = self.optimizer.param_groups[0]['lr']

            elapsed = time.time() - t0

            print(f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.2e}")

            # CSV log
            self.csv_writer.writerow([
                epoch, f'{train_loss:.4f}', f'{val_loss:.4f}',
                f'{val_acc:.4f}', f'{val_f1:.4f}', f'{current_lr:.2e}',
                self.phase
            ])
            self.csv_file.flush()

            # Save best
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_f1 = val_f1
                self._save_checkpoint('best', epoch, val_acc, val_f1)
                self.patience_counter = 0
                print(f"  -> New best: acc={val_acc:.4f} f1={val_f1:.4f}")
            else:
                self.patience_counter += 1

            # Save periodic checkpoint
            if epoch % args.save_every == 0:
                self._save_checkpoint(f'epoch_{epoch}', epoch, val_acc, val_f1)

            # Early stopping
            if self.patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

        print(f"\nBest {self.phase} result: acc={self.best_acc:.4f} f1={self.best_f1:.4f}")
        self.csv_file.close()

    def _train_loso(self):
        """LOSO (Leave-One-Subject-Out) cross-validation."""
        args = self.args
        subjects = self.loso_subjects
        num_folds = len(subjects)

        print(f"\n{'='*60}")
        print(f" LOSO Fine-tune: {num_folds} folds")
        print(f"{'='*60}\n")

        all_fold_accs = []
        all_fold_f1s = []

        for fold_idx, test_subject in enumerate(subjects):
            print(f"\n--- Fold {fold_idx+1}/{num_folds}: test_subject={test_subject} ---")

            # Split dataset by subject
            train_indices = []
            val_indices = []
            for i in range(len(self.full_dataset)):
                sample = self.full_dataset[i]
                if hasattr(self.full_dataset, 'get_subject'):
                    subj = self.full_dataset.get_subject(i)
                else:
                    subj = sample.get('subject', '')
                if subj == test_subject:
                    val_indices.append(i)
                else:
                    train_indices.append(i)

            if len(val_indices) == 0 or len(train_indices) == 0:
                print(f"  Skipping fold (train={len(train_indices)}, val={len(val_indices)})")
                continue

            train_subset = torch.utils.data.Subset(self.full_dataset, train_indices)
            val_subset = torch.utils.data.Subset(self.full_dataset, val_indices)

            train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True)

            # Reset model for each fold (reload pretrained)
            if args.pretrained:
                self._load_pretrained(args.pretrained)
            else:
                # Reinitialize head only
                for module in [self.model.moe, self.model.au_decoder, self.model.fusion]:
                    for layer in module.modules():
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()

            # Reset optimizer
            self._setup_optimizer()
            self._setup_scheduler()

            # Train fold
            best_fold_acc = 0.0
            best_fold_f1 = 0.0
            patience_counter = 0

            for epoch in range(1, args.epochs + 1):
                self.model.train()
                total_loss = 0.0
                correct = 0
                total = 0

                for batch in train_loader:
                    frames = batch['frames'].to(self.device)
                    me_labels = batch['me_label'].to(self.device)
                    au_labels = batch.get('au_label',
                                         torch.zeros(frames.size(0), 28)).to(self.device)

                    outputs = self.model(frames)

                    # Manifold MixUp
                    mixup_outputs = None
                    mixup_lam = 1.0
                    if args.mixup_alpha > 0 and 'adapted_feat' in outputs:
                        mixed_feat, me_a, me_b, au_a, au_b, lam = manifold_mixup(
                            outputs['adapted_feat'], me_labels, au_labels,
                            alpha=args.mixup_alpha
                        )
                        if lam < 1.0:
                            mixup_lam = lam
                            me_logits_mix, _, _ = self.model.moe(mixed_feat)
                            au_intensities_mix, _ = self.model.au_decoder(mixed_feat)
                            mixup_outputs = {
                                'me_logits': me_logits_mix,
                                'au_intensities': au_intensities_mix,
                                'me_labels_b': me_b,
                                'au_labels_b': au_b,
                            }

                    loss = self._compute_loss(outputs, me_labels, au_labels,
                                              mixup_outputs, mixup_lam)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    if self.scheduler_step_per_batch:
                        self.scheduler.step()

                    total_loss += loss.item()
                    preds = outputs['me_logits'].argmax(dim=1)
                    correct += (preds == me_labels).sum().item()
                    total += me_labels.size(0)

                train_acc = correct / max(1, total)

                # Validate
                val_loss, val_acc, val_f1 = self.validate(val_loader)

                if epoch % 5 == 0 or epoch == 1:
                    print(f"  Epoch {epoch}/{args.epochs} | "
                          f"Train Acc: {train_acc:.4f} | "
                          f"Val Acc: {val_acc:.4f} F1: {val_f1:.4f}")

                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc
                    best_fold_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

            all_fold_accs.append(best_fold_acc)
            all_fold_f1s.append(best_fold_f1)
            print(f"  Fold result: acc={best_fold_acc:.4f} f1={best_fold_f1:.4f}")

        # Summary
        if all_fold_accs:
            mean_acc = np.mean(all_fold_accs)
            std_acc = np.std(all_fold_accs)
            mean_f1 = np.mean(all_fold_f1s)
            std_f1 = np.std(all_fold_f1s)
            print(f"\n{'='*60}")
            print(f" LOSO Results ({len(all_fold_accs)} folds)")
            print(f"{'='*60}")
            print(f"  Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
            print(f"  F1 Score: {mean_f1:.4f} +/- {std_f1:.4f}")
            print(f"  Per-fold: {[f'{a:.3f}' for a in all_fold_accs]}")
            print(f"{'='*60}")

            # Save LOSO results
            results_path = os.path.join(args.log_dir or './logs', 'loso_results.txt')
            with open(results_path, 'w') as f:
                f.write(f"LOSO Results ({len(all_fold_accs)} folds)\n")
                f.write(f"Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}\n")
                f.write(f"F1 Score: {mean_f1:.4f} +/- {std_f1:.4f}\n")
                for i, (acc, f1) in enumerate(zip(all_fold_accs, all_fold_f1s)):
                    f.write(f"Fold {i}: acc={acc:.4f} f1={f1:.4f}\n")

        self.csv_file.close()

    def _save_checkpoint(self, tag, epoch, val_acc, val_f1):
        """Save model checkpoint."""
        save_dir = self.args.save_dir or './checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        phase_tag = 'pretrain' if self.phase == 'pretrain' else 'finetune'
        path = os.path.join(save_dir, f'{phase_tag}_{tag}.pth')

        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
            'phase': self.phase,
        }
        if self.use_arcface:
            state['arcface_state_dict'] = self.arcface_loss.state_dict()

        torch.save(state, path)
        print(f"  Saved: {path}")


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Censor Multi-Dataset Training')

    # Phase
    parser.add_argument('--phase', type=str, default='pretrain',
                        choices=['pretrain', 'finetune'],
                        help='Training phase: pretrain (joint) or finetune (single)')

    # Dataset roots
    parser.add_argument('--casme_root', type=str, default=None,
                        help='CASME2 dataset root directory')
    parser.add_argument('--smic_root', type=str, default=None,
                        help='SMIC dataset root directory')
    parser.add_argument('--samm_root', type=str, default=None,
                        help='SAMM dataset root directory')
    parser.add_argument('--target_dataset', type=str, default='casme2',
                        choices=['casme2', 'smic', 'samm'],
                        help='Target dataset for finetune phase')

    # Model
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained checkpoint path for finetune')
    parser.add_argument('--no_fast_preprocess', action='store_true',
                        help='Use TV-L1 optical flow (slow) instead of frame difference')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--backbone_lr_factor', type=float, default=0.1,
                        help='Backbone LR = lr * this factor (0.1 = 10x smaller)')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--grad_accum_steps', type=int, default=2,
                        help='Gradient accumulation steps (effective batch = batch_size * this)')
    parser.add_argument('--save_every', type=int, default=10)

    # Data
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--H', type=int, default=224)
    parser.add_argument('--W', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--loso', action='store_true',
                        help='Use Leave-One-Subject-Out evaluation')

    # Loss
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--use_arcface', action='store_true',
                        help='Use ArcFace angular margin loss')
    parser.add_argument('--arcface_margin', type=float, default=0.2,
                        help='ArcFace angular margin (0.2 typical, 0.5 aggressive)')
    parser.add_argument('--arcface_weight', type=float, default=1.0,
                        help='Weight for ArcFace loss')
    parser.add_argument('--supcon_weight', type=float, default=0.1,
                        help='Weight for SupCon loss (0 to disable)')

    # Augmentation
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Manifold MixUp alpha (0 to disable)')

    # Output
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Default save/log dirs
    if args.save_dir is None:
        args.save_dir = f'./checkpoints/{args.phase}'
    if args.log_dir is None:
        args.log_dir = f'./logs/{args.phase}'

    # Train
    trainer = CrossDatasetTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
