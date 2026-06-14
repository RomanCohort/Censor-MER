#!/bin/bash
# Quick test: run all experiments with --quick_test --epochs 2 to verify flow
# No GPU needed, ~10 min total

set -e

echo "=========================================="
echo "Quick Test: All Supplementary Experiments"
echo "Date: $(date)"
echo "=========================================="

mkdir -p results

# P0: Multi-scale 3D ResNet LOSO (CASME II)
echo "\n[1/6] exp1b CASME II..."
python experiments/exp1b_multiscale_3d_resnet.py --dataset casme2 --pretrained --quick_test --epochs 2 --batch_size 2
echo "[1/6] OK"

# P0: Cross-dataset ablation (SAMM)
echo "\n[2/6] exp7 cross-dataset SAMM..."
python experiments/exp7_deep_ablation.py --experiment cross_dataset --dataset samm --epochs 2 --batch_size 2 --quick_test
echo "[2/6] OK"

# P0: Cross-dataset ablation (SMIC)
echo "\n[3/6] exp7 cross-dataset SMIC..."
python experiments/exp7_deep_ablation.py --experiment cross_dataset --dataset smic --epochs 2 --batch_size 2 --num_classes 3 --quick_test
echo "[3/6] OK"

# P1: MoE alternatives
echo "\n[4/6] exp7 MoE alternatives..."
python experiments/exp7_deep_ablation.py --experiment moe_alternatives --dataset casme2 --epochs 2 --batch_size 2 --quick_test
echo "[4/6] OK"

# P1: Failure analysis
echo "\n[5/6] exp8 failure analysis CASME II..."
python experiments/exp8_failure_analysis.py --dataset casme2 --batch_size 2 --quick_test
echo "[5/6] OK"

# P2: rPPG analysis
echo "\n[6/6] exp7 rPPG analysis..."
python experiments/exp7_deep_ablation.py --experiment rppg_analysis --dataset casme2
echo "[6/6] OK (may need checkpoint)"

echo "\n=========================================="
echo "ALL QUICK TESTS COMPLETE"
echo "=========================================="

ls -la results/*.json 2>/dev/null || echo "No results yet"