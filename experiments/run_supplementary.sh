#!/bin/bash
# =============================================================================
# CENSOR Supplementary Experiments - FULL RUN on AutoDL RTX 4090
# =============================================================================
# Step 1: Pre-extract frames (once, ~10min per dataset)
# Step 2: Train all experiments (~20h total)
#
# Usage:
#   nohup bash experiments/run_supplementary.sh > run_all.log 2>&1 &
#   tail -f run_all.log
# =============================================================================

set -e

echo "=========================================="
echo "CENSOR Supplementary Experiments"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

mkdir -p results

# =============================================================================
# Step 1: Pre-extract frames (eliminates CPU JPEG decoding bottleneck)
# =============================================================================
echo ""
echo "=========================================="
echo "Step 1: Pre-extracting frames"
echo "=========================================="

python experiments/preextract_frames.py --dataset casme2
python experiments/preextract_frames.py --dataset samm
python experiments/preextract_frames.py --dataset smic

echo "Pre-extraction done: $(date)"

# =============================================================================
# Step 2: Run all experiments
# =============================================================================

# P0: Multi-scale 3D ResNet LOSO (Most Important! ~4h)
echo ""
echo "=========================================="
echo "[1/7] Multi-scale 3D ResNet - CASME II"
echo "=========================================="
python experiments/exp1b_multiscale_3d_resnet.py \
    --dataset casme2 --epochs 50 --pretrained --batch_size 8 \
    2>&1 | tee results/exp1b_casme2.log
echo "[1/7] Done: $(date)"

echo ""
echo "=========================================="
echo "[2/7] Multi-scale 3D ResNet - SAMM"
echo "=========================================="
python experiments/exp1b_multiscale_3d_resnet.py \
    --dataset samm --epochs 50 --pretrained --batch_size 8 \
    2>&1 | tee results/exp1b_samm.log
echo "[2/7] Done: $(date)"

echo ""
echo "=========================================="
echo "[3/7] Multi-scale 3D ResNet - SMIC"
echo "=========================================="
python experiments/exp1b_multiscale_3d_resnet.py \
    --dataset smic --epochs 50 --pretrained --num_classes 3 --batch_size 8 \
    2>&1 | tee results/exp1b_smic.log
echo "[3/7] Done: $(date)"

# P0: Cross-Dataset Ablation (~6h)
echo ""
echo "=========================================="
echo "[4/7] Cross-Dataset Ablation - SAMM"
echo "=========================================="
python experiments/exp7_deep_ablation.py \
    --experiment cross_dataset --dataset samm --epochs 50 --batch_size 8 \
    2>&1 | tee results/exp7_cross_samm.log
echo "[4/7] Done: $(date)"

echo ""
echo "=========================================="
echo "[5/7] Cross-Dataset Ablation - SMIC"
echo "=========================================="
python experiments/exp7_deep_ablation.py \
    --experiment cross_dataset --dataset smic --epochs 50 --num_classes 3 --batch_size 8 \
    2>&1 | tee results/exp7_cross_smic.log
echo "[5/7] Done: $(date)"

# P1: MoE Alternative Fusion (~4h)
echo ""
echo "=========================================="
echo "[6/7] MoE Alternative Fusion - CASME II"
echo "=========================================="
python experiments/exp7_deep_ablation.py \
    --experiment moe_alternatives --dataset casme2 --epochs 50 --batch_size 8 \
    2>&1 | tee results/exp7_moe_casme2.log
echo "[6/7] Done: $(date)"

# P1: Failure Case Analysis (~4h)
echo ""
echo "=========================================="
echo "[7/7] Failure Case Analysis - CASME II"
echo "=========================================="
python experiments/exp8_failure_analysis.py \
    --dataset casme2 --batch_size 8 \
    2>&1 | tee results/exp8_casme2.log
echo "[7/7] Done: $(date)"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "End: $(date)"
echo "=========================================="

ls -la results/*.json 2>/dev/null || echo "No results yet"

echo ""
echo "KEY COMPARISON:"
echo "  Multi-scale 3D ResNet claimed: 91.35% (CASME II)"
echo "  Our Censor: 87.74%"
echo "  If exp1b < 89%, our result is competitive under strict LOSO"