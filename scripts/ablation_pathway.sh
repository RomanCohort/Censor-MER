#!/bin/bash
# Ablation Study: Single-path vs Dual-path
# ==========================================
# Tests whether the dual-pathway architecture actually helps
# by comparing Fast-only, Slow-only, and Full dual-path models.
#
# Usage:
#   bash scripts/ablation_pathway.sh
#
# All three runs use the same CASME2 LOSO protocol for fair comparison.

set -e

CASME_ROOT="/root/autodl-tmp/data/CASME2"
SMIC_ROOT="/root/SMIC_all_cropped"
SAMM_ROOT="/root/data/SAMM"
SAVE_DIR="./checkpoints"
LOG_DIR="./logs"

# ============================
# 1. Fast Path Only (3D ResNet18)
# ============================
echo "=========================================="
echo " Ablation 1: Fast Path Only (3D ResNet18)"
echo "=========================================="

python train_cross.py \
    --phase pretrain \
    --casme_root $CASME_ROOT \
    --smic_root $SMIC_ROOT \
    --samm_root $SAMM_ROOT \
    --pretrained_backbone \
    --single_path fast \
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
    --save_every 999 \
    --save_dir $SAVE_DIR/pretrain_fast \
    --log_dir $LOG_DIR/pretrain_fast

echo ""
echo "Fast pretrain done. Running LOSO..."

python train_cross.py \
    --phase finetune \
    --target_dataset casme2 \
    --pretrained $SAVE_DIR/pretrain_fast/pretrain_best.pth \
    --casme_root $CASME_ROOT \
    --smic_root $SMIC_ROOT \
    --samm_root $SAMM_ROOT \
    --loso \
    --pretrained_backbone \
    --single_path fast \
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
    --save_every 999 \
    --save_dir $SAVE_DIR/loso_fast \
    --log_dir $LOG_DIR/loso_fast

echo ""
echo "Fast LOSO complete."
echo ""

# ============================
# 2. Slow Path Only (3D Swin Transformer)
# ============================
echo "=========================================="
echo " Ablation 2: Slow Path Only (3D Swin)"
echo "=========================================="

python train_cross.py \
    --phase pretrain \
    --casme_root $CASME_ROOT \
    --smic_root $SMIC_ROOT \
    --samm_root $SAMM_ROOT \
    --pretrained_backbone \
    --single_path slow \
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
    --save_every 999 \
    --save_dir $SAVE_DIR/pretrain_slow \
    --log_dir $LOG_DIR/pretrain_slow

echo ""
echo "Slow pretrain done. Running LOSO..."

python train_cross.py \
    --phase finetune \
    --target_dataset casme2 \
    --pretrained $SAVE_DIR/pretrain_slow/pretrain_best.pth \
    --casme_root $CASME_ROOT \
    --smic_root $SMIC_ROOT \
    --samm_root $SAMM_ROOT \
    --loso \
    --pretrained_backbone \
    --single_path slow \
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
    --save_every 999 \
    --save_dir $SAVE_DIR/loso_slow \
    --log_dir $LOG_DIR/loso_slow

echo ""
echo "Slow LOSO complete."
echo ""

# ============================
# 3. Dual Path (Full Model) — uses existing checkpoint
# ============================
echo "=========================================="
echo " Ablation 3: Dual Path (Full Model)"
echo "=========================================="

if [ -f "$SAVE_DIR/pretrain/pretrain_best.pth" ]; then
    echo "Using existing pretrain checkpoint: $SAVE_DIR/pretrain/pretrain_best.pth"
else
    echo "No existing pretrain checkpoint, running pretrain first..."
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
        --save_every 999 \
        --save_dir $SAVE_DIR/pretrain \
        --log_dir $LOG_DIR/pretrain
fi

echo ""
echo "Running Dual-Path LOSO..."

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
    --save_every 999 \
    --save_dir $SAVE_DIR/loso_dual \
    --log_dir $LOG_DIR/loso_dual

echo ""
echo "=========================================="
echo " Ablation Study Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Fast-only LOSO:  $LOG_DIR/loso_fast/"
echo "  Slow-only LOSO:  $LOG_DIR/loso_slow/"
echo "  Dual-path LOSO:  $LOG_DIR/loso_dual/"
