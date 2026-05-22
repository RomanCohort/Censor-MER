#!/bin/bash
# Censor SOTA Training Pipeline on AutoDL
# ==========================================
# Phase 1: Joint pretrain on CASME2 + SMIC + SAMM
# Phase 2: Generalize — LOSO on each dataset separately
#
# Usage:
#   bash scripts/train_sota.sh
#
# Prerequisites:
#   - CASME2 at /root/autodl-tmp/data/CASME2 (with cropped/ and labels.csv)
#   - SMIC at /root/autodl-tmp/data/SMIC
#   - SAMM at /root/data/SAMM

set -e

# ============================
# Configuration
# ============================
CASME_ROOT="/root/autodl-tmp/data/CASME2"
SMIC_ROOT="/root/autodl-tmp/data/SMIC"
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
    --batch_size 8 \
    --lr 3e-4 \
    --backbone_lr_factor 0.1 \
    --weight_decay 0.0 \
    --warmup_epochs 5 \
    --patience 20 \
    --grad_accum_steps 2 \
    --use_arcface \
    --arcface_margin 0.2 \
    --mixup_alpha 0.2 \
    --supcon_weight 0.1 \
    --label_smoothing 0.1 \
    --save_dir $SAVE_DIR/pretrain \
    --log_dir $LOG_DIR/pretrain

echo ""
echo "Phase 1 complete. Best checkpoint: $SAVE_DIR/pretrain/pretrain_best.pth"
echo ""

# ============================
# Phase 2: Generalize (LOSO on each dataset)
# ============================
echo "=========================================="
echo " Phase 2: Generalize (LOSO per dataset)"
echo "=========================================="

python train_cross.py \
    --phase generalize \
    --pretrained $SAVE_DIR/pretrain/pretrain_best.pth \
    --casme_root $CASME_ROOT \
    --smic_root $SMIC_ROOT \
    --samm_root $SAMM_ROOT \
    --loso \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --backbone_lr_factor 0.1 \
    --weight_decay 0.0 \
    --warmup_epochs 3 \
    --patience 15 \
    --grad_accum_steps 2 \
    --use_arcface \
    --arcface_margin 0.3 \
    --mixup_alpha 0.1 \
    --supcon_weight 0.05 \
    --label_smoothing 0.05 \
    --save_dir $SAVE_DIR/generalize \
    --log_dir $LOG_DIR/generalize

echo ""
echo "=========================================="
echo " Training Complete!"
echo "=========================================="
echo "Results saved to: $LOG_DIR/generalize/generalize_results.txt"