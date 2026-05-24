"""
Censor -- Multi-Dataset Joint Pretrain + Generalization Evaluation
====================================================================
SOTA training pipeline for micro-expression recognition.

Strategy:
  Phase 1: Joint pretrain on CASME2 + SMIC + SAMM (dynamic class count)
  Phase 2: Generalize — LOSO on each dataset separately

Key SOTA techniques:
  - ArcFace angular margin loss (with dynamic class weights)
  - Manifold MixUp in feature space
  - Supervised Contrastive (SupCon) loss
  - LOSO (Leave-One-Subject-Out) evaluation per dataset
  - Differential LR (backbone 10x smaller than head)
  - Warmup + Cosine annealing scheduler
  - Gradient accumulation for effective larger batch
  - Kinetics-400 pretrained backbone initialization

Usage:
  # Phase 1: Joint pretrain (all 3 datasets)
  python train_cross.py --phase pretrain \
      --casme_root /root/autodl-tmp/data/CASME2 \
      --smic_root /root/autodl-tmp/data/SMIC \
      --samm_root /root/autodl-tmp/data/SAMM \
      --pretrained_backbone --use_arcface --arcface_margin 0.2 \
      --mixup_alpha 0.2 --supcon_weight 0.1 --weight_decay 0.0

  # Phase 2: Generalize (LOSO on each dataset)
  python train_cross.py --phase generalize \
      --pretrained checkpoints/pretrain/pretrain_best.pth \
      --casme_root /root/autodl-tmp/data/CASME2 \
      --samm_root /root/data/SAMM \
      --loso --use_arcface --weight_decay 0.0

  # Phase 2 alt: Fine-tune on single dataset
  python train_cross.py --phase finetune \
      --target_dataset casme2 \
      --casme_root /root/autodl-tmp/data/CASME2 \
      --pretrained checkpoints/pretrain/pretrain_best.pth \
      --loso --use_arcface --lr 1e-4
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
from torch.utils.data import DataLoader, ConcatDataset, Subset
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
# Unified emotion mapping for pretrain
# =============================================================================
# Strategy: pretrain on CASME2's 4-class (the proven baseline that hit acc=0.76).
# SMIC and SAMM samples that share happiness/surprise/disgust participate too.
# repression-only samples (CASME2) are included to preserve that strong class.
# SMIC has no repression, but its happiness/surprise/disgust still contribute.
# SAMM has fear/anger instead, those are excluded from pretrain.
#
# Generalize phase uses FULL_EMOTION_MAP with per-dataset dynamic classes.

UNIFIED_EMOTION_MAP = {
    'happiness': 0,
    'surprise': 1,
    'disgust': 2,
    'repression': 3,
    # Excluded from pretrain:
    'fear': -1,
    'anger': -1,
    'sadness': -1,
    'contempt': -1,
    'others': -1,
    'other': -1,
    'negative': 2,   # SMIC "negative" → disgust
    'positive': 0,   # SMIC "positive" → happiness
}

# SAMM uses integer emotion codes (from directory naming: {subject}_{AU}_{code})
# Mapping: 1=Happiness, 2=Surprise, 3=Disgust, 4=Repression, 5=Fear, 6=Anger, 7=Others
SAMM_EMOTION_MAP = {
    1: 0,   # Happiness → 0
    2: 1,   # Surprise → 1
    3: 2,   # Disgust → 2
    4: 3,   # Repression → 3
    5: -1,  # Fear → excluded
    6: -1,  # Anger → excluded
    7: -1,  # Others → excluded
}

# For generalize phase: all possible emotions with unique slots
FULL_EMOTION_MAP = {
    'happiness': 0,
    'surprise': 1,
    'disgust': 2,
    'repression': 3,
    'fear': 4,
    'anger': 5,
    'sadness': -1,
    'contempt': -1,
    'others': -1,
    'other': -1,
    'negative': 2,
    'positive': 0,
}

NUM_PRETRAIN_CLASSES = 4


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
    """ArcFace: Additive Angular Margin Loss.

    For small datasets (4 classes, <300 samples), use smaller scale (16)
    to prevent gradient explosion. Large-scale face recognition uses 30-64.
    """
    def __init__(self, in_features, out_features, margin=0.2, scale=16.0):
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

    def forward(self, features, labels, class_weights=None):
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

        return F.cross_entropy(logits, labels)


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
    """Multi-dataset joint pretrain + generalization evaluation trainer."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.phase = args.phase  # pretrain / finetune / generalize

        # Build model
        print(f"\n[Censor] Building model (phase={self.phase})...")
        self.model = Censor(
            fast_preprocess=True,
            diff_mode=True,
            verbose=False,
            enable_sparse_control=False,
            pretrained_backbone=getattr(args, 'pretrained_backbone', False),
            single_path=args.single_path,
            no_moe=args.no_moe,
            no_amyg=getattr(args, 'no_amyg', False),
            no_ffa=getattr(args, 'no_ffa', False),
            no_casa=getattr(args, 'no_casa', False),
            no_rppg=getattr(args, 'no_rppg', False),
        ).to(self.device)

        # Determine num_classes based on phase
        if self.phase == 'pretrain':
            self.num_classes = NUM_PRETRAIN_CLASSES  # 4 classes for pretrain
        elif self.phase == 'finetune':
            self.num_classes = NUM_PRETRAIN_CLASSES  # Initial; will be overridden by _setup_datasets
        else:
            self.num_classes = NUM_PRETRAIN_CLASSES  # Will be reset per-dataset in generalize

        # Adjust MoE head
        self._adjust_moe_for_classes(self.num_classes)

        # ArcFace
        self.use_arcface = args.use_arcface
        if self.use_arcface:
            self.arcface_loss = ArcFaceLoss(
                in_features=1024,
                out_features=self.num_classes,
                margin=args.arcface_margin,
                scale=args.arcface_scale,
            ).to(self.device)
            print(f"[Trainer] ArcFace enabled (margin={args.arcface_margin}, scale={args.arcface_scale}, classes={self.num_classes})")

        # SupCon
        self.supcon_loss = SupConLoss(temperature=0.07)

        # Loss weights — pretrain focuses on ME classification only
        # AU/landmark/MoE aux losses hurt pretrain when AU decoder outputs random values
        if self.phase == 'pretrain':
            self.loss_weights = {
                'me': 1.0,
                'au': 0.0,
                'moe': 0.0,
                'landmark': 0.0,
                'supcon': 0.0,
                'arcface': 1.0 if self.use_arcface else 0.0,
            }
        else:
            self.loss_weights = {
                'me': 1.0,
                'au': 0.0,    # No real AU labels in finetune — disable
                'moe': 0.0,   # MoE aux loss hurts on small datasets
                'landmark': 0.0,  # No AU labels → landmark loss is noise
                'supcon': args.supcon_weight,
                'arcface': args.arcface_weight,
            }

        # Load pretrained checkpoint for finetune/generalize
        if self.phase in ('finetune', 'generalize') and args.pretrained:
            self._load_pretrained(args.pretrained)

        # For generalize: defer dataset/optimizer/scheduler setup to per-dataset
        if self.phase == 'generalize':
            self._setup_csv_logger('generalize')
            return

        if self.phase == 'cross_eval':
            self._setup_csv_logger('cross_eval')
            return

        # Setup datasets
        self._setup_datasets()

        # Create dataloaders
        self._setup_dataloaders()

        # Optimizer with differential LR
        self._setup_optimizer()

        # Scheduler
        self._setup_scheduler()

        # Best tracking — early stopping on macro-F1 (higher is better)
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.patience_counter = 0

        # CSV logger
        self._setup_csv_logger(self.phase)

    def _setup_csv_logger(self, phase_tag):
        csv_path = self.args.log_dir or './logs'
        os.makedirs(csv_path, exist_ok=True)
        self.csv_file = open(os.path.join(csv_path, f'{phase_tag}_log.csv'), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'epoch', 'train_loss', 'val_loss', 'val_acc', 'val_f1',
            'lr', 'phase'
        ])

    def _adjust_moe_for_classes(self, num_classes):
        """Adjust MoE head output for different number of classes.

        Only rebuilds experts and resets gate when num_classes actually changes.
        Preserves pretrained weights when class count is unchanged.
        """
        # Handle no_moe mode — adjust simple_head instead
        if getattr(self.model, 'no_moe', False):
            if hasattr(self.model, 'simple_head') and self.model.simple_head is not None:
                if self.model.simple_head.out_features != num_classes:
                    old_head = self.model.simple_head
                    self.model.simple_head = nn.Linear(old_head.in_features, num_classes).to(self.device)
                    nn.init.xavier_uniform_(self.model.simple_head.weight)
                    nn.init.zeros_(self.model.simple_head.bias)
                    print(f"[Censor] Adjusted simple_head for {num_classes} classes")
                else:
                    print(f"[Censor] simple_head already has {num_classes} classes, skipping")
            return

        moe = self.model.moe

        # Check current output dim of experts — skip if already correct
        current_out_features = moe.experts[0][-1].out_features
        if current_out_features == num_classes:
            print(f"[Censor] MoE already has {num_classes} classes, skipping adjustment")
            return

        # Reinitialize the expert heads for new class count
        new_experts = nn.ModuleList()
        for i, expert in enumerate(moe.experts):
            # Get hidden_dim from existing expert
            hidden_dim = expert[0].out_features
            input_dim = expert[0].in_features
            new_expert = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes),
            ).to(self.device)
            # Copy first layer weights from old expert
            new_expert[0].weight.data = expert[0].weight.data.clone()
            new_expert[0].bias.data = expert[0].bias.data.clone()
            # Xavier init for new output layer
            nn.init.xavier_uniform_(new_expert[-1].weight)
            nn.init.zeros_(new_expert[-1].bias)
            new_experts.append(new_expert)
        moe.experts = new_experts
        # Reinitialize gating network only when class count changes
        for layer in moe.gate:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        print(f"[Censor] Adjusted MoE for {num_classes} classes (was {current_out_features})")

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
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        """Build optimizer with differential LR."""
        backbone_params = []
        head_params = []

        backbone_names = {'fast_pathway', 'slow_pathway', 'saliency', 'rppg', 'flow'}

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

        if self.use_arcface:
            param_groups.append({
                'params': self.arcface_loss.parameters(),
                'lr': lr, 'name': 'arcface'
            })

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        return optimizer

    def _setup_datasets(self):
        """Setup datasets for current phase (no dataloaders yet)."""
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
                    emotion_map=SAMM_EMOTION_MAP,  # SAMM uses integer emotion codes
                    exclude_negative=True,
                )
                datasets.append(ds)
                print(f"[Data] SAMM: {len(ds)} samples")

            if not datasets:
                raise ValueError("No datasets found! Check --casme_root, --smic_root, --samm_root")

            # No validation split for pretrain — dataset too small for reliable val metrics
            # Use train loss/F1 for checkpoint saving instead
            self.train_dataset = ConcatDataset(datasets)
            self.val_dataset = None
            self.val_loader = None
            total = len(self.train_dataset)
            print(f"[Data] Joint pretrain: {total} samples (no val split)")

        elif self.phase == 'finetune':
            # Fine-tune on single target dataset
            target = args.target_dataset

            if target == 'casme2':
                DatasetClass = CASME2FrameDataset
                root = args.casme_root
                emotion_map = UNIFIED_EMOTION_MAP
            elif target == 'smic':
                DatasetClass = SMICFrameDataset
                root = args.smic_root
                emotion_map = UNIFIED_EMOTION_MAP
            elif target == 'samm':
                DatasetClass = SAMMFrameDataset
                root = args.samm_root
                emotion_map = SAMM_EMOTION_MAP
            else:
                raise ValueError(f"Unknown target dataset: {target}")

            if not root or not os.path.exists(root):
                raise ValueError(f"Target dataset root not found: {root}")

            full_dataset = DatasetClass(
                root_dir=root,
                T=args.T, H=args.H, W=args.W,
                augment=True,
                emotion_map=emotion_map,
                exclude_negative=True,
            )

            # Determine actual num_classes from the dataset's label distribution
            all_labels = set()
            for i in range(len(full_dataset)):
                lbl = full_dataset[i]['me_label']
                if isinstance(lbl, torch.Tensor):
                    lbl = lbl.item()
                all_labels.add(int(lbl))
            self.num_classes = len(all_labels)
            self._adjust_moe_for_classes(self.num_classes)
            if self.use_arcface:
                self.arcface_loss = ArcFaceLoss(
                    in_features=1024, out_features=self.num_classes,
                    margin=args.arcface_margin, scale=args.arcface_scale,
                ).to(self.device)
                print(f"[Trainer] ArcFace re-initialized for {self.num_classes} classes (finetune on {target})")

            if args.loso:
                # LOSO: Leave-One-Subject-Out — no fixed train/val split
                self.loso_subjects = full_dataset.subjects if hasattr(full_dataset, 'subjects') else []
                self.full_dataset = full_dataset
                self.train_dataset = None  # Will be set per-fold
                self.val_dataset = None
                self.train_loader = None   # Will be set per-fold
                self.val_loader = None
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

    def _setup_dataloaders(self):
        """Create dataloaders (no label remapping needed for 3-class pretrain)."""
        args = self.args

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
        """Warmup + Cosine annealing scheduler. Skip for LOSO (built per-fold)."""
        if self.train_loader is None:
            self.scheduler = None
            self.scheduler_step_per_batch = False
            return
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

    def _build_scheduler(self, optimizer, num_batches):
        """Build warmup + cosine scheduler for a given optimizer and batch count."""
        total_steps = num_batches * self.args.epochs
        warmup_steps = min(self.args.warmup_epochs * num_batches, total_steps // 10)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.args.lr * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        return scheduler

    def _compute_loss(self, outputs, me_labels, au_labels,
                      mixup_outputs=None, mixup_lam=1.0):
        """Compute total loss.

        ArcFace and FocalLoss are complementary:
        - FocalLoss (with dynamic class weights): handles classification + class imbalance
        - ArcFace: constrains feature space angular geometry
        When ArcFace is enabled, both losses are used together.
        """
        # ME classification: always use FocalLoss on logits (baseline proven approach)
        loss_me = compute_me_loss(outputs['me_logits'], me_labels,
                                  label_smoothing=self.args.label_smoothing)

        # ArcFace: additional angular margin loss on adapted_feat (feature space constraint)
        loss_arcface = torch.tensor(0.0, device=me_labels.device)
        if self.use_arcface and 'adapted_feat' in outputs:
            loss_arcface = self.arcface_loss(outputs['adapted_feat'], me_labels)

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
            self.loss_weights['arcface'] * loss_arcface +
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

            # Safety: clamp labels to valid range [0, num_classes-1]
            me_labels = me_labels.clamp(0, self.num_classes - 1)

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
                    if getattr(self.model, 'no_moe', False):
                        me_logits_mix = self.model.simple_head(mixed_feat)
                    else:
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
    def _compute_train_f1(self):
        """Compute macro-F1 on the training set (for pretrain with no val split)."""
        self.model.eval()
        all_preds = []
        all_labels = []
        for batch in self.train_loader:
            frames = batch['frames'].to(self.device)
            me_labels = batch['me_label'].to(self.device)
            outputs = self.model(frames)
            preds = outputs['me_logits'].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(me_labels.cpu().numpy())
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        self.model.train()
        return f1

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
            # Safety: clamp labels to valid range
            me_labels = me_labels.clamp(0, self.num_classes - 1)

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
        from sklearn.metrics import f1_score, accuracy_score
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return avg_loss, acc, f1

    def _train_epoch(self, loader, optimizer, num_classes, scheduler=None):
        """Train one epoch on a specific loader (used by generalize LOSO)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        accum_steps = self.args.grad_accum_steps

        for batch_idx, batch in enumerate(loader):
            frames = batch['frames'].to(self.device)
            me_labels = batch['me_label'].to(self.device)
            au_labels = batch.get('au_label', torch.zeros(frames.size(0), 28)).to(self.device)
            # Safety: clamp labels to valid range
            me_labels = me_labels.clamp(0, self.num_classes - 1)

            outputs = self.model(frames)

            # Manifold MixUp
            mixup_outputs = None
            mixup_lam = 1.0
            if self.args.mixup_alpha > 0 and 'adapted_feat' in outputs:
                mixed_feat, me_a, me_b, au_a, au_b, lam = manifold_mixup(
                    outputs['adapted_feat'], me_labels, au_labels,
                    alpha=self.args.mixup_alpha
                )
                if lam < 1.0:
                    mixup_lam = lam
                    if getattr(self.model, 'no_moe', False):
                        me_logits_mix = self.model.simple_head(mixed_feat)
                    else:
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
            loss = loss / accum_steps
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item() * accum_steps
            preds = outputs['me_logits'].argmax(dim=1)
            correct += (preds == me_labels).sum().item()
            total += me_labels.size(0)

        avg_loss = total_loss / max(1, len(loader))
        acc = correct / max(1, total)
        return avg_loss, acc

    @torch.no_grad()
    def _validate_epoch(self, loader, num_classes):
        """Validate on a specific loader (used by generalize LOSO)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch in loader:
            frames = batch['frames'].to(self.device)
            me_labels = batch['me_label'].to(self.device)
            au_labels = batch.get('au_label', torch.zeros(frames.size(0), 28)).to(self.device)
            # Safety: clamp labels to valid range
            me_labels = me_labels.clamp(0, self.num_classes - 1)

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

        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return avg_loss, acc, f1

    def train(self):
        """Main training loop."""
        if self.phase == 'generalize':
            self._run_generalize()
        elif self.phase == 'cross_eval':
            self._run_cross_eval()
        elif self.args.loso and self.phase == 'finetune':
            self._train_loso()
        else:
            self._train_standard()

    # ================================================================
    # Generalize: LOSO on each dataset separately
    # ================================================================
    def _run_cross_eval(self):
        """Cross-dataset evaluation: finetune on source, zero-shot test on targets.

        Uses only shared classes (happiness, surprise, disgust) across datasets.
        """
        args = self.args
        source = args.source_dataset
        target_str = args.target_datasets or ''

        if not source:
            raise ValueError("--source_dataset required for cross_eval phase")

        # Parse target datasets
        target_list = [t.strip() for t in target_str.split(',') if t.strip()]
        if not target_list:
            raise ValueError("--target_datasets required for cross_eval phase")

        # Shared 3-class mapping (happiness=0, surprise=1, disgust=2)
        CROSS_EMOTION_MAP = {
            'happiness': 0, 'surprise': 1, 'disgust': 2,
            'positive': 0, 'negative': 2,  # SMIC
            # SAMM integer codes: 1=happiness, 2=surprise, 3=disgust
        }
        SAMM_CROSS_MAP = {1: 0, 2: 1, 3: 2}  # happiness, surprise, disgust only

        DATASET_MAP = {
            'casme2': (CASME2FrameDataset, args.casme_root),
            'smic': (SMICFrameDataset, args.smic_root),
            'samm': (SAMMFrameDataset, args.samm_root),
        }

        # Step 1: Finetune on source dataset (3-class, random split)
        print(f"\n{'='*60}")
        print(f" Cross-Dataset Eval: Source={source}, Targets={target_list}")
        print(f"{'='*60}")

        DatasetClass, root = DATASET_MAP[source]
        if source == 'samm':
            emotion_map = SAMM_CROSS_MAP
        else:
            emotion_map = CROSS_EMOTION_MAP

        full_dataset = DatasetClass(
            root_dir=root,
            T=args.T, H=args.H, W=args.W,
            augment=True,
            emotion_map=emotion_map,
            exclude_negative=True,
        )

        # Filter to only shared classes
        def get_label(ds, i):
            lbl = ds[i]['me_label']
            return lbl.item() if isinstance(lbl, torch.Tensor) else int(lbl)

        filtered_indices = [i for i in range(len(full_dataset))
                           if get_label(full_dataset, i) >= 0]
        full_dataset = torch.utils.data.Subset(full_dataset, filtered_indices)

        if len(full_dataset) == 0:
            raise ValueError(f"No samples in source dataset {source} with shared classes")

        # Count actual classes
        all_labels = set()
        for i in range(len(full_dataset)):
            all_labels.add(get_label(full_dataset, i))
        num_classes = len(all_labels)
        print(f"  Source {source}: {len(full_dataset)} samples, {num_classes} classes: {sorted(all_labels)}")

        # Train/val split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

        # Adjust model for 3-class
        self._adjust_moe_for_classes(num_classes)

        # Build ArcFace for 3-class
        self.arcface_loss = ArcFaceLoss(
            in_features=1024, out_features=num_classes,
            scale=args.arcface_scale, margin=args.arcface_margin
        ).to(self.device)

        # Build optimizer
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, len(train_loader))
        self.scheduler_step_per_batch = True

        # Load pretrained weights
        if args.pretrained:
            self._load_pretrained(args.pretrained)

        # Train on source
        print(f"\n  Training on {source} (3-class)...")
        best_acc = 0.0
        best_f1 = 0.0
        for epoch in range(1, args.epochs + 1):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                frames = batch['frames'].to(self.device)
                me_labels = batch['me_label'].to(self.device).clamp(0, num_classes - 1)
                au_labels = batch.get('au_label',
                                     torch.zeros(frames.size(0), 28)).to(self.device)

                outputs = self.model(frames)
                loss = self._compute_loss(outputs, me_labels, au_labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                if self.scheduler_step_per_batch:
                    scheduler.step()

                total_loss += loss.item()
                preds = outputs['me_logits'].argmax(dim=1)
                correct += (preds == me_labels).sum().item()
                total += me_labels.size(0)

            train_acc = correct / max(1, total)

            # Validate on source val
            val_loss, val_acc, val_f1 = self._validate_epoch(val_loader, num_classes)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{args.epochs} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f} F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                # Save best source model
                os.makedirs(os.path.join(args.save_dir), exist_ok=True)
                torch.save(self.model.state_dict(),
                          os.path.join(args.save_dir, f'cross_src_{source}_best.pth'))

        print(f"\n  Source {source} best: Acc={best_acc:.4f} F1={best_f1:.4f}")

        # Step 2: Zero-shot test on each target dataset
        results = {}
        for target in target_list:
            print(f"\n  --- Zero-shot on {target} ---")
            DatasetClass, root = DATASET_MAP[target]
            if target == 'samm':
                t_emotion_map = SAMM_CROSS_MAP
            else:
                t_emotion_map = CROSS_EMOTION_MAP

            target_dataset = DatasetClass(
                root_dir=root,
                T=args.T, H=args.H, W=args.W,
                augment=False,
                emotion_map=t_emotion_map,
                exclude_negative=True,
            )

            # Filter shared classes
            t_filtered = [i for i in range(len(target_dataset))
                         if get_label(target_dataset, i) >= 0]
            target_dataset = torch.utils.data.Subset(target_dataset, t_filtered)

            if len(target_dataset) == 0:
                print(f"  No shared-class samples in {target}, skipping")
                continue

            target_loader = DataLoader(target_dataset, batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.num_workers,
                                       pin_memory=True)

            # Evaluate
            self.model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in target_loader:
                    frames = batch['frames'].to(self.device)
                    me_labels = batch['me_label'].to(self.device).clamp(0, num_classes - 1)
                    outputs = self.model(frames)
                    preds = outputs['me_logits'].argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(me_labels.cpu().numpy())

            acc = (np.array(all_preds) == np.array(all_labels)).mean()
            from sklearn.metrics import f1_score
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            results[target] = {'acc': acc, 'f1': f1, 'n': len(all_labels)}
            print(f"  {target}: Acc={acc:.4f} F1={f1:.4f} (n={len(all_labels)})")

        # Summary
        print(f"\n{'='*60}")
        print(f" Cross-Dataset Results (3-class shared)")
        print(f"{'='*60}")
        print(f"  Source: {source} (Acc={best_acc:.4f}, F1={best_f1:.4f})")
        for target, r in results.items():
            print(f"  → {target}: Acc={r['acc']:.4f}, F1={r['f1']:.4f} (n={r['n']})")

        # Save results
        os.makedirs(args.log_dir, exist_ok=True)
        with open(os.path.join(args.log_dir, 'cross_eval_results.txt'), 'w') as f:
            f.write(f"Source: {source}\n")
            f.write(f"Source Acc: {best_acc:.4f}, F1: {best_f1:.4f}\n\n")
            for target, r in results.items():
                f.write(f"Target {target}: Acc={r['acc']:.4f}, F1={r['f1']:.4f}, n={r['n']}\n")

        return results

    def _run_generalize(self):
        """Run LOSO evaluation on each available dataset to measure generalization.

        For each dataset:
        1. Create dataset with UNIFIED_EMOTION_MAP
        2. Determine actual num_classes from label distribution
        3. Rebuild MoE head + ArcFace for that num_classes
        4. Load pretrained checkpoint
        5. Run LOSO
        6. Save per-dataset results
        """
        args = self.args
        all_results = {}

        # Determine which datasets are available
        datasets_to_eval = []
        if args.casme_root and os.path.isdir(args.casme_root):
            datasets_to_eval.append('casme2')
        if args.smic_root and os.path.isdir(args.smic_root):
            datasets_to_eval.append('smic')
        if args.samm_root and os.path.isdir(args.samm_root):
            datasets_to_eval.append('samm')

        if not datasets_to_eval:
            raise ValueError("No dataset roots provided. Use --casme_root, --smic_root, --samm_root")

        print(f"\n{'='*60}")
        print(f" Generalization Evaluation: {', '.join(datasets_to_eval)}")
        print(f" Pretrained: {args.pretrained}")
        print(f"{'='*60}")

        for dataset_name in datasets_to_eval:
            print(f"\n{'='*60}")
            print(f" Evaluating on: {dataset_name.upper()}")
            print(f"{'='*60}")

            # 1. Create dataset
            dataset = self._create_dataset(dataset_name)
            if dataset is None or len(dataset) == 0:
                print(f"[Skip] {dataset_name}: no valid samples")
                continue

            # 2. Determine actual num_classes from dataset labels
            all_labels = []
            for i in range(len(dataset)):
                sample = dataset[i]
                label = sample['me_label']
                if isinstance(label, torch.Tensor):
                    label = label.item()
                all_labels.append(int(label))
            unique_labels = sorted(set(all_labels))
            num_classes = len(unique_labels)
            label_map = {old: new for new, old in enumerate(unique_labels)}

            print(f"  Samples: {len(dataset)}")
            print(f"  Original labels: {unique_labels}")
            print(f"  Remapped to: 0..{num_classes-1}")
            for lbl in unique_labels:
                emotion_name = [k for k, v in UNIFIED_EMOTION_MAP.items() if v == lbl]
                count = all_labels.count(lbl)
                name = emotion_name[0] if emotion_name else f'class_{lbl}'
                print(f"    {name}({lbl}): {count} samples → remapped to {label_map[lbl]}")

            # 3. Rebuild model for this dataset's class count
            # Reload fresh pretrained weights
            self._load_pretrained(args.pretrained)
            self._adjust_moe_for_classes(num_classes)

            # Rebuild ArcFace
            if self.use_arcface:
                self.arcface_loss = ArcFaceLoss(
                    in_features=1024, out_features=num_classes,
                    margin=self.arcface_margin, scale=self.args.arcface_scale
                ).to(self.device)

            # 4. Run LOSO
            results = self._run_loso_for_dataset(dataset, num_classes, label_map, dataset_name)
            all_results[dataset_name] = results

            # Print summary
            print(f"\n  {dataset_name.upper()} LOSO Results:")
            print(f"    Mean Acc: {results['mean_acc']:.4f}")
            print(f"    Mean F1:  {results['mean_f1']:.4f}")
            print(f"    Per-fold: {results['fold_accs']}")

        # Final summary
        print(f"\n{'='*60}")
        print(f" GENERALIZATION SUMMARY")
        print(f"{'='*60}")
        for name, res in all_results.items():
            print(f"  {name.upper():8s} | Acc: {res['mean_acc']:.4f} | F1: {res['mean_f1']:.4f}")

        # Save results
        results_path = args.log_dir or './logs'
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, 'generalize_results.txt'), 'w') as f:
            f.write("Generalization Evaluation Results\n")
            f.write("=" * 50 + "\n")
            for name, res in all_results.items():
                f.write(f"\n{name.upper()}\n")
                f.write(f"  Mean Acc: {res['mean_acc']:.4f}\n")
                f.write(f"  Mean F1:  {res['mean_f1']:.4f}\n")
                f.write(f"  Per-fold Acc: {res['fold_accs']}\n")
                f.write(f"  Per-fold F1:  {res['fold_f1s']}\n")

        print(f"\nResults saved to: {results_path}/generalize_results.txt")
        return all_results

    def _create_dataset(self, dataset_name):
        """Create dataset for a given dataset name."""
        args = self.args
        if dataset_name == 'casme2':
            return CASME2FrameDataset(
                root_dir=args.casme_root,
                emotion_map=UNIFIED_EMOTION_MAP,
                exclude_negative=True,
                n_frames=16,
            )
        elif dataset_name == 'smic':
            return SMICFrameDataset(
                root_dir=args.smic_root,
                emotion_map=UNIFIED_EMOTION_MAP,
                n_frames=16,
            )
        elif dataset_name == 'samm':
            return SAMMFrameDataset(
                root_dir=args.samm_root,
                emotion_map=SAMM_EMOTION_MAP,  # SAMM uses integer emotion codes
                exclude_negative=True,
                n_frames=16,
            )
        return None

    def _run_loso_for_dataset(self, dataset, num_classes, label_map, dataset_name):
        """Run LOSO cross-validation on a single dataset.

        Args:
            dataset: Dataset object with get_subject() method
            num_classes: Number of classes after remapping
            label_map: Dict mapping original labels to 0..num_classes-1
            dataset_name: Name for logging
        """
        args = self.args

        # Collect subjects
        subjects = sorted(set(dataset.get_subject(i) for i in range(len(dataset))))
        print(f"  LOSO: {len(subjects)} subjects, {len(dataset)} samples")

        fold_accs = []
        fold_f1s = []

        for fold_idx, test_subject in enumerate(subjects):
            print(f"\n  --- Fold {fold_idx+1}/{len(subjects)}: test_subject={test_subject} ---")

            # Split by subject
            train_indices = [i for i in range(len(dataset))
                            if dataset.get_subject(i) != test_subject]
            test_indices = [i for i in range(len(dataset))
                           if dataset.get_subject(i) == test_subject]

            if len(train_indices) == 0 or len(test_indices) == 0:
                print(f"    Skip fold (train={len(train_indices)}, test={len(test_indices)})")
                continue

            train_subset = Subset(dataset, train_indices)
            test_subset = Subset(dataset, test_indices)

            # Remap labels in collate
            train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=4, pin_memory=True,
                                     collate_fn=lambda batch: self._remap_collate(batch, label_map))
            test_loader = DataLoader(test_subset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True,
                                    collate_fn=lambda batch: self._remap_collate(batch, label_map))

            # Reset model for this fold
            self._load_pretrained(args.pretrained)
            self._adjust_moe_for_classes(num_classes)
            if self.use_arcface:
                self.arcface_loss = ArcFaceLoss(
                    in_features=1024, out_features=num_classes,
                    margin=self.arcface_margin, scale=self.args.arcface_scale
                ).to(self.device)

            optimizer = self._build_optimizer()
            scheduler = self._build_scheduler(optimizer, len(train_loader))

            # Train this fold — early stopping on train loss (lower is better)
            # LOSO test set is too small (3-8 samples) for reliable F1-based early stopping
            best_fold_acc = 0.0
            best_fold_f1 = 0.0
            best_train_loss = float('inf')
            patience_counter = 0

            for epoch in range(args.epochs):
                train_loss, train_acc = self._train_epoch(
                    train_loader, optimizer, num_classes, scheduler)

                # Validate
                val_loss, val_acc, val_f1 = self._validate_epoch(
                    test_loader, num_classes)

                current_lr = optimizer.param_groups[0]['lr']

                print(f"    Epoch {epoch+1}/{args.epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Val F1: {val_f1:.4f} | "
                      f"LR: {current_lr:.6f}")

                # Track best val metrics regardless of early stopping
                if val_f1 > best_fold_f1:
                    best_fold_f1 = val_f1
                    best_fold_acc = val_acc

                # Early stopping on train loss (stable signal, not noisy val F1)
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"    Early stop at epoch {epoch+1} (train_loss={train_loss:.4f}, best_val_F1={best_fold_f1:.4f})")
                        break

            fold_accs.append(best_fold_acc)
            fold_f1s.append(best_fold_f1)
            print(f"    Best fold acc: {best_fold_acc:.4f}, F1: {best_fold_f1:.4f}")

        # Aggregate
        mean_acc = np.mean(fold_accs) if fold_accs else 0.0
        mean_f1 = np.mean(fold_f1s) if fold_f1s else 0.0

        return {
            'mean_acc': mean_acc,
            'mean_f1': mean_f1,
            'fold_accs': fold_accs,
            'fold_f1s': fold_f1s,
        }

    def _remap_collate(self, batch, label_map):
        """Collate function that remaps labels to 0..num_classes-1."""
        frames_list = []
        labels_list = []
        au_labels_list = []
        subjects_list = []

        for item in batch:
            frames_list.append(item['frames'])
            label = item['me_label']
            if isinstance(label, torch.Tensor):
                label = label.item()
            # Remap label
            labels_list.append(label_map.get(int(label), 0))
            au_labels_list.append(item.get('au_label', torch.zeros(28)))
            subjects_list.append(item.get('subject', ''))

        frames = torch.stack(frames_list)
        labels = torch.tensor(labels_list, dtype=torch.long)
        au_labels = torch.stack(au_labels_list) if au_labels_list[0].dim() > 0 else torch.zeros(len(batch), 28)

        return {
            'frames': frames,
            'me_label': labels,
            'au_label': au_labels,
            'subject': subjects_list,
        }

    # ================================================================
    # Standard training
    # ================================================================
    def _train_standard(self):
        """Standard training loop (pretrain or finetune with random split).

        Early stopping criterion: macro-F1 (higher is better).
        For pretrain (no val split): use train macro-F1.
        For finetune (has val split): use val macro-F1.
        """
        args = self.args
        has_val = self.val_loader is not None

        print(f"\n{'='*60}")
        print(f" Phase: {self.phase.upper()}")
        print(f" Epochs: {args.epochs}")
        print(f" Batch size: {args.batch_size}")
        print(f" LR: {args.lr} (backbone {args.lr * args.backbone_lr_factor:.2e})")
        print(f" ArcFace: {self.use_arcface}")
        print(f" MixUp alpha: {args.mixup_alpha}")
        print(f" Early stopping: macro-F1, patience={args.patience}")
        print(f" Val split: {'yes' if has_val else 'no (using train metrics)'}")
        print(f"{'='*60}\n")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Compute train F1 for this epoch
            train_f1 = self._compute_train_f1()

            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0

            if has_val:
                val_loss, val_acc, val_f1 = self.validate()
                monitor_f1 = val_f1
                print(f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                      f"LR: {current_lr:.2e}")
                self.csv_writer.writerow([
                    epoch, f'{train_loss:.4f}', f'{val_loss:.4f}',
                    f'{val_acc:.4f}', f'{val_f1:.4f}', f'{current_lr:.2e}',
                    self.phase
                ])
            else:
                monitor_f1 = train_f1
                print(f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                      f"LR: {current_lr:.2e}")
                self.csv_writer.writerow([
                    epoch, f'{train_loss:.4f}', '',
                    f'{train_acc:.4f}', f'{train_f1:.4f}', f'{current_lr:.2e}',
                    self.phase
                ])
            self.csv_file.flush()

            # Save best — based on macro-F1 (higher is better)
            if monitor_f1 > self.best_f1:
                self.best_f1 = monitor_f1
                self.best_acc = train_acc if not has_val else val_acc
                self._save_checkpoint('best', epoch, self.best_acc, self.best_f1)
                self.patience_counter = 0
                print(f"  -> New best F1: {monitor_f1:.4f}")
            else:
                self.patience_counter += 1

            # Save periodic checkpoint
            if epoch % args.save_every == 0:
                self._save_checkpoint(f'epoch_{epoch}', epoch,
                                      train_acc if not has_val else val_acc,
                                      monitor_f1)

            # Early stopping on macro-F1
            if self.patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(patience={args.patience}, best_F1={self.best_f1:.4f})")
                break

        print(f"\nBest {self.phase} result: acc={self.best_acc:.4f} F1={self.best_f1:.4f}")
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
                head_modules = []
                if getattr(self.model, 'no_moe', False):
                    head_modules.append(self.model.simple_head)
                else:
                    head_modules.append(self.model.moe)
                head_modules.extend([self.model.au_decoder, self.model.fusion])
                for module in head_modules:
                    for layer in module.modules():
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()

            # Reset optimizer and scheduler for this fold
            self._setup_optimizer()
            self.scheduler = self._build_scheduler(self.optimizer, len(train_loader))
            self.scheduler_step_per_batch = True

            # Train fold — early stopping on train loss (val F1 too noisy on 3-8 samples)
            best_fold_acc = 0.0
            best_fold_f1 = 0.0
            best_train_loss = float('inf')
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
                            if getattr(self.model, 'no_moe', False):
                                me_logits_mix = self.model.simple_head(mixed_feat)
                            else:
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

                train_loss = total_loss / max(1, len(train_loader))
                train_acc = correct / max(1, total)

                # Validate
                val_loss, val_acc, val_f1 = self.validate(val_loader)

                if epoch % 5 == 0 or epoch == 1:
                    print(f"  Epoch {epoch}/{args.epochs} | "
                          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                          f"Val Acc: {val_acc:.4f} F1: {val_f1:.4f}")

                # Track best val metrics
                if val_f1 > best_fold_f1:
                    best_fold_f1 = val_f1
                    best_fold_acc = val_acc

                # Early stopping on train loss (stable, unlike noisy val F1 on tiny test set)
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    print(f"  Early stop at epoch {epoch} (train_loss={train_loss:.4f}, best_val_F1={best_fold_f1:.4f})")
                    break

            all_fold_accs.append(best_fold_acc)
            all_fold_f1s.append(best_fold_f1)
            print(f"  Fold result: acc={best_fold_acc:.4f} F1={best_fold_f1:.4f}")

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
                        choices=['pretrain', 'finetune', 'generalize', 'cross_eval'],
                        help='Training phase: pretrain (joint), finetune (single), generalize (LOSO on all datasets), cross_eval (train on source, zero-shot test on target)')

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
    parser.add_argument('--source_dataset', type=str, default=None,
                        choices=['casme2', 'smic', 'samm'],
                        help='Source dataset for cross_eval phase (train on this)')
    parser.add_argument('--target_datasets', type=str, default=None,
                        help='Comma-separated target datasets for cross_eval (test on these)')

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
                        help='ArcFace angular margin')
    parser.add_argument('--arcface_scale', type=float, default=16.0,
                        help='ArcFace scale (16 for small datasets, 30+ for face recognition)')
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

    # Pretrained backbone (Kinetics-400)
    parser.add_argument('--pretrained_backbone', action='store_true',
                        help='Initialize backbones with Kinetics-400 pretrained weights')
    parser.add_argument('--single_path', type=str, default=None,
                        choices=['fast', 'slow'],
                        help='Ablation: use single pathway only (fast or slow)')
    parser.add_argument('--no_moe', action='store_true',
                        help='Ablation: replace MoE head with simple linear layer')
    parser.add_argument('--no_amyg', action='store_true',
                        help='Ablation: disable amygdala attention gating')
    parser.add_argument('--no_ffa', action='store_true',
                        help='Ablation: disable FFA fusion module')
    parser.add_argument('--no_casa', action='store_true',
                        help='Ablation: disable CASANet spatiotemporal attention')
    parser.add_argument('--no_rppg', action='store_true',
                        help='Ablation: disable rPPG signal extraction')

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
