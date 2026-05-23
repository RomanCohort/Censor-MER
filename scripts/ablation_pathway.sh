#!/bin/bash
# Ablation Study: Pathway & MoE
# ==========================================
# Tests:
#   A1. Fast-only (3D ResNet18)
#   A2. Slow-only (3D Swin Transformer)
#   A3. Dual-path, No MoE (simple linear head)
#   A4. Full model (dual-path + MoE)
#
# All use CASME2 LOSO for fair comparison.

set -e

CASME_ROOT="/root/autodl-tmp/data/CASME2"
SMIC_ROOT="/root/SMIC_all_cropped"
SAMM_ROOT="/root/data/SAMM"
SAVE_DIR="./checkpoints"
LOG_DIR="./logs"

run_loso() {
    local TAG=$1
    local EXTRA=$2

    echo "=========================================="
    echo " Ablation: $TAG"
    echo "=========================================="

    # Pretrain
    python train_cross.py \
        --phase pretrain \
        --casme_root $CASME_ROOT \
        --smic_root $SMIC_ROOT \
        --samm_root $SAMM_ROOT \
        --pretrained_backbone \
        $EXTRA \
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
        --save_dir $SAVE_DIR/pretrain_${TAG} \
        --log_dir $LOG_DIR/pretrain_${TAG}

    echo ""
    echo "Pretrain done for $TAG. Running LOSO..."

    # LOSO
    python train_cross.py \
        --phase finetune \
        --target_dataset casme2 \
        --pretrained $SAVE_DIR/pretrain_${TAG}/pretrain_best.pth \
        --casme_root $CASME_ROOT \
        --smic_root $SMIC_ROOT \
        --samm_root $SAMM_ROOT \
        --loso \
        --pretrained_backbone \
        $EXTRA \
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
        --save_dir $SAVE_DIR/loso_${TAG} \
        --log_dir $LOG_DIR/loso_${TAG}

    echo ""
    echo "LOSO complete for $TAG."
    echo ""
}

# ============================
# A1: Fast-only
# ============================
run_loso "fast_only" "--single_path fast"

# ============================
# A2: Slow-only
# ============================
run_loso "slow_only" "--single_path slow"

# ============================
# A3: Dual-path, No MoE
# ============================
run_loso "dual_nomoe" "--no_moe"

# ============================
# A4: Full model (dual + MoE)
# ============================
if [ -f "$SAVE_DIR/pretrain/pretrain_best.pth" ]; then
    echo "=========================================="
    echo " Ablation: Full model (using existing pretrain)"
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
        --save_every 999 \
        --save_dir $SAVE_DIR/loso_full \
        --log_dir $LOG_DIR/loso_full

    echo ""
    echo "Full model LOSO complete."
    echo ""
else
    echo "No existing pretrain checkpoint, running pretrain first..."
    run_loso "full" ""
fi

# ============================
# Summary
# ============================
echo "=========================================="
echo " Ablation Study Complete!"
echo "=========================================="
echo ""
echo "Results (CASME2 LOSO):"
echo "  A1 Fast-only:     $LOG_DIR/loso_fast_only/"
echo "  A2 Slow-only:      $LOG_DIR/loso_slow_only/"
echo "  A3 Dual+NoMoE:    $LOG_DIR/loso_dual_nomoe/"
echo "  A4 Full (Dual+MoE): $LOG_DIR/loso_full/"