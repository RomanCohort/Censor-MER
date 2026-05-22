#!/bin/bash
# Censor Full Training Pipeline on AutoDL
# ==========================================
# Phase 1: Joint pretrain on CASME2 + SMIC + SAMM (4-class, ~444 samples)
# Phase 2: Finetune on each dataset (random 8:2 split)
# Phase 3: LOSO on CASME2 (26-fold, publishable numbers)
#
# Usage:
#   bash scripts/train_full.sh
#
# Prerequisites:
#   - CASME2 at /root/autodl-tmp/data/CASME2 (with cropped/ and xlsx)
#   - SMIC at /root/SMIC_all_cropped
#   - SAMM at /root/data/SAMM

set -e

# ============================
# Configuration
# ============================
CASME_ROOT="/root/autodl-tmp/data/CASME2"
SMIC_ROOT="/root/SMIC_all_cropped"
SAMM_ROOT="/root/data/SAMM"

SAVE_DIR="./checkpoints"
LOG_DIR="./logs"

# ============================
# Phase 1: Joint Pretrain
# ============================
echo "=========================================="
echo " Phase 1: Joint Pretrain (CASME2+SMIC+SAMM)"
echo "=========================================="

python train_cross.py \
    --phase pretrain \
    --casme_root $CASME_ROOT \
    --smic_root $SMIC_ROOT \
    --samm_root $SAMM_ROOT \
    --pretrained_backbone \
    --epochs 80 \
    --batch_size 16 \
    --lr 3e-4 \
    --backbone_lr_factor 0.1 \
    --weight_decay 0.0 \
    --warmup_epochs 5 \
    --patience 50 \
    --grad_accum_steps 1 \
    --use_arcface \
    --arcface_margin 0.2 \
    --arcface_scale 16 \
    --label_smoothing 0.1 \
    --save_dir $SAVE_DIR/pretrain \
    --log_dir $LOG_DIR/pretrain

echo ""
echo "Phase 1 complete. Best checkpoint: $SAVE_DIR/pretrain/pretrain_best.pth"
echo ""

# ============================
# Phase 2: Finetune per dataset (random split, quick results)
# ============================
echo "=========================================="
echo " Phase 2: Finetune per dataset"
echo "=========================================="

# --- CASME2 ---
echo "--- CASME2 finetune ---"
python train_cross.py \
    --phase finetune \
    --target_dataset casme2 \
    --pretrained $SAVE_DIR/pretrain/pretrain_best.pth \
    --casme_root $CASME_ROOT \
    --smic_root $SMIC_ROOT \
    --samm_root $SAMM_ROOT \
    --pretrained_backbone \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --backbone_lr_factor 0.1 \
    --weight_decay 0.0 \
    --warmup_epochs 3 \
    --patience 50 \
    --grad_accum_steps 2 \
    --use_arcface \
    --arcface_margin 0.3 \
    --arcface_scale 16 \
    --label_smoothing 0.1 \
    --save_dir $SAVE_DIR/ft_casme2 \
    --log_dir $LOG_DIR/ft_casme2

echo ""

# --- SMIC ---
echo "--- SMIC finetune ---"
python train_cross.py \
    --phase finetune \
    --target_dataset smic \
    --pretrained $SAVE_DIR/pretrain/pretrain_best.pth \
    --casme_root $CASME_ROOT \
    --smic_root $SMIC_ROOT \
    --samm_root $SAMM_ROOT \
    --pretrained_backbone \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --backbone_lr_factor 0.1 \
    --weight_decay 0.0 \
    --warmup_epochs 3 \
    --patience 50 \
    --grad_accum_steps 2 \
    --use_arcface \
    --arcface_margin 0.3 \
    --arcface_scale 16 \
    --label_smoothing 0.1 \
    --save_dir $SAVE_DIR/ft_smic \
    --log_dir $LOG_DIR/ft_smic

echo ""

# --- SAMM ---
echo "--- SAMM finetune ---"
python train_cross.py \
    --phase finetune \
    --target_dataset samm \
    --pretrained $SAVE_DIR/pretrain/pretrain_best.pth \
    --casme_root $CASME_ROOT \
    --smic_root $SMIC_ROOT \
    --samm_root $SAMM_ROOT \
    --pretrained_backbone \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --backbone_lr_factor 0.1 \
    --weight_decay 0.0 \
    --warmup_epochs 3 \
    --patience 50 \
    --grad_accum_steps 2 \
    --use_arcface \
    --arcface_margin 0.3 \
    --arcface_scale 16 \
    --label_smoothing 0.1 \
    --save_dir $SAVE_DIR/ft_samm \
    --log_dir $LOG_DIR/ft_samm

echo ""
echo "Phase 2 complete."
echo ""

# ============================
# Phase 3: LOSO on CASME2 (publishable results)
# ============================
echo "=========================================="
echo " Phase 3: LOSO on CASME2 (26-fold)"
echo "=========================================="

python train_cross.py \
    --phase finetune \
    --target_dataset casme2 \
    --pretrained $SAVE_DIR/pretrain/pretrain_best.pth \
    --casme_root $CASME_ROOT \
    --smic_root $SMIC_ROOT \
    --samm_root $SAMM_ROOT \
    --loso \
    --pretrained_backbone \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --backbone_lr_factor 0.1 \
    --weight_decay 0.0 \
    --warmup_epochs 3 \
    --patience 50 \
    --grad_accum_steps 2 \
    --use_arcface \
    --arcface_margin 0.3 \
    --arcface_scale 16 \
    --label_smoothing 0.1 \
    --save_dir $SAVE_DIR/loso_casme2 \
    --log_dir $LOG_DIR/loso_casme2

echo ""
echo "=========================================="
echo " All Training Complete!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "  Pretrain:         $LOG_DIR/pretrain/"
echo "  CASME2 finetune:  $LOG_DIR/ft_casme2/"
echo "  SMIC finetune:    $LOG_DIR/ft_smic/"
echo "  SAMM finetune:    $LOG_DIR/ft_samm/"
echo "  CASME2 LOSO:      $LOG_DIR/loso_casme2/"
