#!/bin/bash
# Cross-Dataset Generalization Experiment
# ==========================================
# Train on each dataset (3-class shared), zero-shot test on others.
# Shared classes: happiness, surprise, disgust (present in all 3 datasets)
#
# Usage:
#   bash scripts/cross_dataset.sh

set -e

CASME_ROOT="/root/autodl-tmp/data/CASME2"
SMIC_ROOT="/root/SMIC_all_cropped"
SAMM_ROOT="/root/data/SAMM"

SAVE_DIR="./checkpoints"
LOG_DIR="./logs"

# ============================
# CASME2 → SMIC, SAMM
# ============================
echo "=========================================="
echo " CASME2 → SMIC + SAMM"
echo "=========================================="
python train_cross.py \
    --phase cross_eval \
    --source_dataset casme2 \
    --target_datasets "smic,samm" \
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
    --patience 20 \
    --grad_accum_steps 2 \
    --use_arcface \
    --arcface_margin 0.3 \
    --arcface_scale 16 \
    --label_smoothing 0.1 \
    --save_dir $SAVE_DIR/cross_casme2 \
    --log_dir $LOG_DIR/cross_casme2

echo ""

# ============================
# SMIC → CASME2, SAMM
# ============================
echo "=========================================="
echo " SMIC → CASME2 + SAMM"
echo "=========================================="
python train_cross.py \
    --phase cross_eval \
    --source_dataset smic \
    --target_datasets "casme2,samm" \
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
    --patience 20 \
    --grad_accum_steps 2 \
    --use_arcface \
    --arcface_margin 0.3 \
    --arcface_scale 16 \
    --label_smoothing 0.1 \
    --save_dir $SAVE_DIR/cross_smic \
    --log_dir $LOG_DIR/cross_smic

echo ""

# ============================
# SAMM → CASME2, SMIC
# ============================
echo "=========================================="
echo " SAMM → CASME2 + SMIC"
echo "=========================================="
python train_cross.py \
    --phase cross_eval \
    --source_dataset samm \
    --target_datasets "casme2,smic" \
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
    --patience 20 \
    --grad_accum_steps 2 \
    --use_arcface \
    --arcface_margin 0.3 \
    --arcface_scale 16 \
    --label_smoothing 0.1 \
    --save_dir $SAVE_DIR/cross_samm \
    --log_dir $LOG_DIR/cross_samm

echo ""
echo "=========================================="
echo " Cross-Dataset Evaluation Complete!"
echo "=========================================="
