#!/bin/bash
# Quick test: verify all experiments can load data + build models
# No GPU needed, ~5 min total
# For actual training, run on GPU with full params

set -e

echo "=========================================="
echo "Quick Test: All Supplementary Experiments"
echo "Date: $(date)"
echo "=========================================="

mkdir -p results

# P0: Multi-scale 3D ResNet LOSO (CASME II)
echo ""
echo "[1/6] exp1b CASME II (dry run)..."
python experiments/exp1b_multiscale_3d_resnet.py --dataset casme2 --pretrained --dry_run
echo "[1/6] OK"

# P0: Multi-scale 3D ResNet LOSO (SAMM)
echo ""
echo "[2/6] exp1b SAMM (dry run)..."
python experiments/exp1b_multiscale_3d_resnet.py --dataset samm --pretrained --dry_run
echo "[2/6] OK"

# P0: Multi-scale 3D ResNet LOSO (SMIC)
echo ""
echo "[3/6] exp1b SMIC (dry run)..."
python experiments/exp1b_multiscale_3d_resnet.py --dataset smic --pretrained --num_classes 3 --dry_run
echo "[3/6] OK"

# P0: Cross-dataset ablation - just verify dataset loads
echo ""
echo "[4/6] Verify SAMM dataset loading..."
python -c "
import sys; sys.path.insert(0, '.')
from experiments.exp7_deep_ablation import get_dataset, get_loso_splits
ds = get_dataset('samm', '/root/data/SAMM/SAMM')
splits, subjects = get_loso_splits(ds, 'samm')
print(f'  SAMM: {len(ds)} samples, {len(subjects)} subjects, {len(splits)} folds')
"
echo "[4/6] OK"

echo ""
echo "[4b] Verify SMIC dataset loading..."
python -c "
import sys; sys.path.insert(0, '.')
from experiments.exp7_deep_ablation import get_dataset, get_loso_splits
ds = get_dataset('smic', '/root/SMIC_all_cropped')
splits, subjects = get_loso_splits(ds, 'smic')
print(f'  SMIC: {len(ds)} samples, {len(subjects)} subjects, {len(splits)} folds')
"
echo "[4b] OK"

# P1: MoE alternatives - verify fusion modules
echo ""
echo "[5/6] Verify MoE alternative fusion modules..."
python -c "
import torch, sys; sys.path.insert(0, '.')
from experiments.exp7_deep_ablation import ConcatFusion, AttentionFusion, FeatureEnsemble
for cls_name, cls in [('ConcatFusion', ConcatFusion), ('AttentionFusion', AttentionFusion), ('FeatureEnsemble', FeatureEnsemble)]:
    m = cls(num_classes=4)
    fast = torch.randn(2, 512)
    slow = torch.randn(2, 768)
    out = m(fast, slow)
    print(f'  {cls_name}: OK ({out.shape})')
"
echo "[5/6] OK"

# P1: Failure analysis - verify CASME II LOSO splits
echo ""
echo "[6/6] Verify failure analysis LOSO splits..."
python -c "
import sys; sys.path.insert(0, '.')
from experiments.exp8_failure_analysis import get_dataset, get_loso_splits
ds = get_dataset('casme2', '/root/autodl-tmp/data/CASME2')
splits, subjects = get_loso_splits(ds, 'casme2')
print(f'  CASME II: {len(ds)} samples, {len(subjects)} subjects, {len(splits)} folds')
print(f'  Fold 1 test subject: {splits[0][2]}, {len(splits[0][1])} samples')
"
echo "[6/6] OK"

echo ""
echo "=========================================="
echo "ALL QUICK TESTS PASSED"
echo "=========================================="
echo ""
echo "To run actual experiments on GPU:"
echo "  python experiments/exp1b_multiscale_3d_resnet.py --dataset casme2 --pretrained --batch_size 8"
echo "  bash experiments/run_supplementary.sh"