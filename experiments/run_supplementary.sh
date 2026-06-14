#!/bin/bash
# =============================================================================
# Run All Supplementary Experiments on AutoDL (RTX 4090)
# =============================================================================
# Estimated time: ~20 hours total
# Estimated cost: ~40 RMB (2 RMB/h for RTX 4090)
#
# Usage:
#   cd /root/your_censor_repo
#   git pull
#   nohup bash experiments/run_supplementary.sh > supplementary_experiments.log 2>&1 &
#   tail -f supplementary_experiments.log
# =============================================================================

set -e

echo "=========================================="
echo "CENSOR Supplementary Experiments"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# Check data paths
CASME2="/root/autodl-tmp/data/CASME2"
SAMM="/root/data/SAMM/SAMM"
SMIC="/root/SMIC_all_cropped"

for path in "$CASME2" "$SAMM" "$SMIC"; do
    if [ -d "$path" ]; then
        echo "[OK] Data found: $path"
    else
        echo "[WARNING] Data NOT found: $path"
    fi
done

# Create results directory
mkdir -p results

# =============================================================================
# P0: Multi-scale 3D ResNet LOSO Reproduction (Most Important!)
# =============================================================================
# Expected time: ~4h on 4090
# If this shows 85-88% instead of claimed 91.35%, our 87.74% is competitive!

echo ""
echo "=========================================="
echo "P0: Multi-scale 3D ResNet LOSO Reproduction"
echo "=========================================="
echo "Start time: $(date)"

python experiments/exp1b_multiscale_3d_resnet.py \
    --dataset casme2 \
    --epochs 50 \
    --pretrained \
    --batch_size 8 \
    2>&1 | tee -a results/exp1b_casme2.log

echo "CASME2 done: $(date)"

python experiments/exp1b_multiscale_3d_resnet.py \
    --dataset samm \
    --epochs 50 \
    --pretrained \
    --batch_size 8 \
    2>&1 | tee -a results/exp1b_samm.log

echo "SAMM done: $(date)"

python experiments/exp1b_multiscale_3d_resnet.py \
    --dataset smic \
    --epochs 50 \
    --pretrained \
    --num_classes 3 \
    --batch_size 8 \
    2>&1 | tee -a results/exp1b_smic.log

echo "SMIC done: $(date)"

# =============================================================================
# P0: Cross-Dataset Ablation on SAMM and SMIC
# =============================================================================
# Expected time: ~6h on 4090

echo ""
echo "=========================================="
echo "P0: Cross-Dataset Ablation (SAMM)"
echo "=========================================="
echo "Start time: $(date)"

python experiments/exp7_deep_ablation.py \
    --experiment cross_dataset \
    --dataset samm \
    --epochs 50 \
    --batch_size 8 \
    2>&1 | tee -a results/exp7_cross_samm.log

echo "SAMM ablation done: $(date)"

python experiments/exp7_deep_ablation.py \
    --experiment cross_dataset \
    --dataset smic \
    --epochs 50 \
    --num_classes 3 \
    --batch_size 8 \
    2>&1 | tee -a results/exp7_cross_smic.log

echo "SMIC ablation done: $(date)"

# =============================================================================
# P1: MoE Alternative Fusion Comparison
# =============================================================================
# Expected time: ~4h on 4090

echo ""
echo "=========================================="
echo "P1: MoE Alternative Fusion"
echo "=========================================="
echo "Start time: $(date)"

python experiments/exp7_deep_ablation.py \
    --experiment moe_alternatives \
    --dataset casme2 \
    --epochs 50 \
    --batch_size 8 \
    2>&1 | tee -a results/exp7_moe_casme2.log

echo "MoE alternatives done: $(date)"

# =============================================================================
# P1: Failure Case Analysis
# =============================================================================
# Expected time: ~4h on 4090 (requires training + inference)

echo ""
echo "=========================================="
echo "P1: Failure Case Analysis"
echo "=========================================="
echo "Start time: $(date)"

python experiments/exp8_failure_analysis.py \
    --dataset casme2 \
    --batch_size 8 \
    2>&1 | tee -a results/exp8_casme2.log

echo "CASME2 failure analysis done: $(date)"

python experiments/exp8_failure_analysis.py \
    --dataset samm \
    --batch_size 8 \
    2>&1 | tee -a results/exp8_samm.log

echo "SAMM failure analysis done: $(date)"

# =============================================================================
# P2: rPPG Feature Analysis
# =============================================================================
# Expected time: ~30 min (no training, feature extraction only)

echo ""
echo "=========================================="
echo "P2: rPPG Feature Analysis"
echo "=========================================="
echo "Start time: $(date)"

python experiments/exp7_deep_ablation.py \
    --experiment rppg_analysis \
    --dataset casme2 \
    2>&1 | tee -a results/exp7_rppg_casme2.log

echo "rPPG analysis done: $(date)"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "End time: $(date)"
echo "=========================================="

echo ""
echo "Results files:"
ls -la results/exp1b_*.json 2>/dev/null || echo "  No exp1b results"
ls -la results/exp7_*.json 2>/dev/null || echo "  No exp7 results"
ls -la results/exp8_*.json 2>/dev/null || echo "  No exp8 results"

echo ""
echo "Key comparison (Multi-scale 3D ResNet vs Censor):"
echo "  CASME II claimed: 91.35%"
echo "  Our Censor: 87.74%"
echo "  Check exp1b results for reproduced accuracy"

echo ""
echo "Next steps:"
echo "  1. Compare exp1b results with original SOTA claims"
echo "  2. If exp1b shows <89%, our 87.74% is competitive under strict LOSO"
echo "  3. Update paper with new cross-dataset ablation data"
echo "  4. Add confusion matrix and failure analysis to Discussion"