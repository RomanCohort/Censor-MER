# Reproducibility Guide for IEEE TAC Paper

## Paper Information

**Title**: Component Contribution Analysis in Biomimetic Micro-Expression Recognition: A Comprehensive Ablation Study

**Target Venue**: IEEE Transactions on Affective Computing

---

## Dataset

### CASME II (Required)

**Download**: http://casme.psych.ac.cn/casme2-en

**Required Files**:
- `CASME2_RAW.zip` - Raw video data
- `CASME2_Labeling_Emotion.xlsx` - Emotion labels

**Preprocessing**:
```bash
python scripts/convert_casme2.py --data_root ./data/CASME_II
```

This will:
1. Extract frames from videos (200fps)
2. Align faces using MTCNN
3. Normalize to 224x224
4. Generate optical flow using TV-L1
5. Extract rPPG signals using CHROM

---

## Environment Setup

### Requirements

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- Python >= 3.9
- PyTorch >= 2.0.0
- CUDA >= 11.7 (recommended)
- Transformers >= 4.30.0

### Hardware Requirements

- **GPU**: NVIDIA GPU with >= 16GB VRAM (recommended)
- **CPU**: >= 8 cores
- **Memory**: >= 32GB RAM
- **Storage**: >= 100GB for dataset + cache

---

## Training Commands

### Full Model Training (LOSO)

```bash
python train.py --dataset casme2 \
    --data_root ./data/CASME_II \
    --epochs 50 \
    --batch_size 8 \
    --gradient_accumulation 2 \
    --lr 1e-4 \
    --backbone_lr 1e-5 \
    --label_smoothing 0.1 \
    --early_stopping_patience 20 \
    --protocol loso \
    --output_dir ./checkpoints/censor_full
```

### Ablation Experiments

```bash
# 1. Fast-only (3D ResNet-18)
python train.py --config configs/ablation/fast_only.yaml

# 2. Slow-only (3D Swin-T)
python train.py --config configs/ablation/slow_only.yaml

# 3. Dual-no-MoE (Linear head)
python train.py --config configs/ablation/dual_no_moe.yaml

# 4. No-CASANet
python train.py --config configs/ablation/no_casanet.yaml

# 5. No-rPPG
python train.py --config configs/ablation/no_rppg.yaml

# 6. Full Model
python train.py --config configs/ablation/full.yaml
```

---

## Model Architecture

### Parameter Count (68.35M total)

| Module | Parameters |
|--------|------------|
| SaliencyDetector | 0.12M |
| rPPGExtractor | — (non-parametric) |
| TVL1OpticalFlow | — (non-parametric) |
| FastPath (3D ResNet-18) | 12.85M |
| SlowPath (3D Swin-T) | 31.40M |
| AmygdalaGate | 0.08M |
| FFA Fusion | 1.64M |
| CASANet | 2.12M |
| TSFmicroFusion | 4.38M |
| AU Decoder | 8.45M |
| MoE Head | 7.31M |

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | |
| Learning rate | 1e-4 | Backbone: 1e-5 |
| Batch size | 8 | With gradient accumulation ×2 |
| Epochs | 50 | |
| Early stopping | patience 20 | |
| Label smoothing | 0.1 | |
| Dropout | 0.0 | Removed for small datasets |
| L2 regularization | 0.0 | Removed for small datasets |
| Temporal frames | 16 | |
| Spatial size | 224×224 | |

---

## Evaluation Protocol

### LOSO Cross-Validation (24 folds)

```bash
python evaluate.py --protocol loso \
    --checkpoint ./checkpoints/censor_full \
    --output_dir ./results/loso_results.csv
```

**Expected Results**:
- Accuracy: 87.74% ± 12.76%
- F1-score: 83.34% ± 17.63%

### Statistical Tests

We use Welch's t-test (unequal variance) with Bonferroni correction:

```python
from scipy.stats import ttest_ind

# Compare Full vs No-rPPG
t_stat, p_value = ttest_ind(full_results, no_rppg_results, equal_var=False)
p_adj = p_value * num_comparisons  # Bonferroni
```

---

## Cross-Dataset Transfer

```bash
# CASME II → SMIC
python train_cross.py --source casme2 --target smic

# CASME II → SAMM
python train_cross.py --source casme2 --target samm
```

---

## Code Repository

**GitHub**: https://github.com/RomanCohort/Censor-MER

**Key Files**:
- `train.py` - Main training script
- `main.py` - Censor model definition
- `config/defaults.py` - Hyperparameter configs
- `model/` - Module implementations
- `scripts/convert_casme2.py` - Data preprocessing

---

## Pretrained Checkpoints

Available at: `./checkpoints/`

| Model | Path | Notes |
|-------|------|-------|
| Censor Full | `checkpoints/censor_full/` | Best LOSO model |
| Censor G-SNN | `checkpoints/censor_g_snn/` | Generation variant |

---

## Expected Runtime

| Task | Time (GPU) |
|------|------------|
| Data preprocessing | ~2 hours |
| Single LOSO fold | ~30 min |
| Full LOSO (24 folds) | ~12 hours |
| All ablation experiments | ~72 hours |

---

## Contact

For reproduction issues, contact: [author email]

---

## Citation

```bibtex
@article{censor2026mer,
  title={Component Contribution Analysis in Biomimetic Micro-Expression Recognition: A Comprehensive Ablation Study},
  author={[Author Name]},
  journal={IEEE Transactions on Affective Computing},
  year={2026}
}
```