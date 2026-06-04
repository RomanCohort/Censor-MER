# Censor MER Code Directory

This directory contains the complete implementation for the paper:

**"Component Contribution Analysis in Biomimetic Micro-Expression Recognition: A Comprehensive Ablation Study"**

## Structure

```
D:\censor\
├── train.py              # Main training script (LOSO)
├── main.py               # Censor model definition
├── evaluate.py           # Evaluation script
├── requirements.txt      # Python dependencies
├── config/
│   └── defaults.py       # Hyperparameter configs
├── model/
│   ├── backbones.py      # 3D ResNet-18, 3D Swin-T
│   ├── moe_head.py       # Mixture-of-Experts gating
│   ├── attention.py      # CASANet temporal attention
│   ├── preprocessing.py  # rPPG CHROM extraction
│   └── fusion.py         # Dual-pathway fusion
├── scripts/
│   └── convert_casme2.py # Data preprocessing
└── checkpoints/          # Model weights
```

## Quick Start

1. Download CASME II dataset
2. Preprocess: `python scripts/convert_casme2.py`
3. Train: `python train.py --dataset casme2`
4. Evaluate: `python evaluate.py --protocol loso`

## Ablation Configs

See `configs/ablation/` for 6 experimental variants:
- `fast_only.yaml`
- `slow_only.yaml`
- `dual_no_moe.yaml`
- `no_casanet.yaml`
- `no_rppg.yaml`
- `full.yaml`
