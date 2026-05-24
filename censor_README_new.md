# Censor: Biomimetic Dual-Pathway Micro-Expression Recognition System

> **仿生双通道微表情识别系统** — A PyTorch implementation of a biomimetic dual-pathway architecture for micro-expression recognition (MER), simulating the fusiform-amygdala circuit in the human visual pathway.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Stage 1: Biomimetic Preprocessing](#stage-1-biomimetic-preprocessing)
  - [Stage 2: Dual-Pathway Backbones](#stage-2-dual-pathway-backbones)
  - [Stage 3: Fusiform-Amygdala Attention Circuit](#stage-3-fusiform-amygdala-attention-circuit)
  - [Stage 4: Spatio-Temporal Fusion](#stage-4-spatio-temporal-fusion)
  - [Stage 5: Dynamic AU Decoder](#stage-5-dynamic-au-decoder)
  - [Stage 6: Mixture-of-Experts Head](#stage-6-mixture-of-experts-head)
  - [Stage 7: Emotion Reporter](#stage-7-emotion-reporter)
- [Mathematical Formulation](#mathematical-formulation)
- [Benchmark Datasets](#benchmark-datasets)
- [State-of-the-Art Comparison](#state-of-the-art-comparison)
- [Installation](#installation)
  - [System Requirements](#system-requirements)
  - [Environment Setup](#environment-setup)
  - [Dependency Details](#dependency-details)
- [Quick Start](#quick-start)
- [Training](#training)
  - [Basic Training](#basic-training)
  - [Advanced Training Options](#advanced-training-options)
  - [Checkpointing & Resume](#checkpointing--resume)
- [Usage Guide](#usage-guide)
  - [Python API](#python-api)
  - [Configuration](#configuration)
  - [Output Interpretation](#output-interpretation)
- [Citation](#citation)

---

## Overview

Micro-expressions (MEs) are brief, involuntary facial expressions that occur when a person suppresses or conceals their true emotions. They last between **40–200 ms** and are characterized by subtle muscle activations measurable via Action Units (AUs) in the Facial Action Coding System (FACS).

**Censor** proposes a biomimetic dual-pathway architecture that mirrors the human visual system's fast subcortical and slow cortical pathways:

```
Input Video (B×3×16×224×224)
  │
  ├── [Stage 1] Biomimetic Preprocessing
  │   ├── SaliencyDetector (Foveal sampling via Gaussian pyramid)
  │   ├── rPPGExtractor (Remote photoplethysmography blood-flow)
  │   └── TVL1OpticalFlow (OpenCV DualTVL1 optical flow)
  │
  ├── [Stage 2] Dual-Pathway Backbones
  │   ├── FastPath: 3D ResNet-18 (optical flow) → 512-D
  │   └── SlowPath: 3D Swin-Transformer (RGB+rPPG) → 768-D + spatial map
  │
  ├── [Stage 3] Fusiform-Amygdala Attention
  │   ├── Amygdala: attention prior map from fast pathway
  │   ├── FFA: SE-style cross-pathway gating
  │   └── CASANet: apex frame detection via triangular attention
  │
  ├── [Stage 4] TSFmicroFusion (Bidirectional cross-attention, 1024-D)
  │
  ├── [Stage 5] Dynamic AU Decoder (BiLSTM, 28 AUs + OPD landmarks)
  │
  ├── [Stage 6] MoE Head (3 experts + PersonalizedRadar TTA)
  │
  └── [Stage 7] Emotion Reporter (template + LLM-based reports)
```

| Metric | Value |
|--------|-------|
| **Total Parameters** | 68,353,230 |
| **Architecture** | Dual-pathway: 3D ResNet-18 + 3D Swin-Transformer |
| **Preprocessing** | Gaussian saliency + rPPG + OpenCV TV-L1 |
| **Attention** | Amygdala (FC) + FFA (SE) + CASANet (triangular MHA) |
| **Fusion** | Bidirectional cross-attention in 1024-D space |
| **AU Decoding** | BiLSTM (2 layers, 512 hidden) → 28 sigmoid outputs |
| **MoE** | 3 experts, top-2 gating, load-balancing auxiliary loss |
| **TTA** | PersonalizedRadar (5-step SGD identity adapter) |

---

## Architecture

### Stage 1: Biomimetic Preprocessing

#### SaliencyDetector — Foveal Sampling

Simulates human retinal fovea (highest cone density at 1–2° visual angle) via **Gaussian pyramid** with center-biased spatial prior:

$$S(x,y) = \sum_{l=0}^{L-1} w_l \cdot G_\sigma(x,y) \cdot I_l(x,y)$$

where $I_l$ is the $l$-th pyramid level, $G_\sigma$ is the center-biased Gaussian prior, and $w_l = 2^{-l}$ are level weights. Output: `(B, 1, T, H, W)`.

#### rPPGExtractor — Remote Photoplethysmography

Captures blood oxygen saturation fluctuations (0.5–4.0 Hz cardiac range) via **chrominance decomposition** and **temporal bandpass filtering**:

$$\text{rPPG}(t) = \sum_{c \in \{R,G,B\}} \alpha_c \cdot I_c(t)$$

$$\text{rPPG}_{\text{filtered}}(t) = \sum_{\tau=-K}^{K} h(\tau) \cdot \text{rPPG}(t-\tau)$$

where $\alpha_c$ are learned chrominance projection weights and $h$ is a learned FIR bandpass filter. Output: `(B, 3, T, H, W)`.

#### TVL1OpticalFlow — OpenCV DualTVL1

Computes real TV-L1 optical flow via OpenCV's `createOptFlow_DualTVL1`. The TV-L1 energy functional:

$$\min_u \int\left(|\nabla u| + \lambda \cdot |I_1(x+u) - I_0(x)|\right) dx$$

solved via primal-dual algorithm. Output: `(B, 2, T, H, W)`.

### Stage 2: Dual-Pathway Backbones

#### FastSubcorticalPathway — 3D ResNet-18

Processes **optical flow** input through a shallow 3D ResNet-18 variant (3 stages, 64→128→256 channels). Large temporal strides (2²,2²) simulate fast subcortical processing. Output: `(B, 512)`.

#### SlowCorticalPathway — 3D Swin-Transformer

Processes **RGB + rPPG** (6 channels) through a full 3D Swin-Transformer with 4 stages and **shifted-window multi-head self-attention (W-MSA)**:

| Stage | Blocks | Dim | Merge Stride | Resolution |
|-------|--------|-----|--------------|------------|
| 1 | 2 | 96 | (2,2,2) | T/2, H/2, W/2 |
| 2 | 2 | 192 | (2,2,2) | T/4, H/4, W/4 |
| 3 | 6 | 384 | (2,2,2) | T/8, H/8, W/8 |
| 4 | 2 | 768 | (1,1,1) | T/16, H/32, W/32 |

Relative position bias (3D meshgrid) enables accurate spatial relationship modeling. Output: **pooled `(B, 768)`** + **spatial map `(B, 768, T/16, H/32, W/32)`**.

### Stage 3: Fusiform-Amygdala Attention Circuit

#### Amygdala — Attention Prior Map

Fully-connected layers generate a spatial attention prior map from fast pathway features:

$$\text{APM} = \sigma\left(\text{FC}_{512\rightarrow256\rightarrow196}(\text{fast\_feat})\right).view(B,1,14,14)$$

The sigmoid-activated APM guides spatial attention toward facial regions of interest.

#### FFA — Feature Fusion Attention

SE-style squeeze-excitation gating for cross-pathway feature recalibration:

$$z = \sigma\left(\text{FC}_{1280\rightarrow80}(\text{concat}[f_{\text{fast}}, f_{\text{slow}}])\right)$$

$$f_{\text{fast}}^* = z_{[:512]} \odot f_{\text{fast}}, \quad f_{\text{slow}}^* = z_{[512:]} \odot f_{\text{slow}}$$

#### CASANet — Apex Frame Detection

Inverted-triangle learnable spatial mask (7×7) + temporal MultiHeadAttention for **apex frame detection** — identifying the peak intensity frame in a micro-expression sequence:

$$\text{apex\_score}_t = \text{softmax}\left(\text{MHA}(Q_t, K, V)\right) \in \mathbb{R}^T$$

The triangular prior $M_{i,j} = \exp\left(-\frac{(j-i)^2}{2\sigma_i^2}\right)$ simulates the natural onset→apex→decay pattern of micro-expressions. Input: spatial map `(B, 768, 1, 7, 7)` from Slow pathway Stage 4 (skipping global pool). Output: attended features + apex scores `(B, 1)`.

### Stage 4: Spatio-Temporal Fusion

**TSFmicroFusion** — Bidirectional cross-attention in 1024-D fused space:

$$\text{F}_{f2s} = \text{Attention}\left(Q_f \cdot W_Q, K_s \cdot W_K, V_s \cdot W_V\right) \cdot W_O$$

$$\text{F}_{s2f} = \text{Attention}\left(Q_s \cdot W_Q, K_f \cdot W_K, V_f \cdot W_V\right) \cdot W_O$$

$$f_{\text{fused}} = \alpha \cdot \text{FFN}(\text{F}_{f2s}) + (1-\alpha) \cdot \text{FFN}(\text{F}_{s2f}), \quad \alpha = \sigma(W_\alpha[f_{\text{fast}}; f_{\text{slow}}])$$

Output: `(B, 1024)`.

### Stage 5: Dynamic AU Decoder

**DynamicAUDecoder** — BiLSTM for temporal Action Unit sequence modeling:

$$\mathbf{h}_t = \text{BiLSTM}(f_{\text{fused}}, \mathbf{h}_{t-1}), \quad \mathbf{h}_T = [\mathbf{h}_t^{f}; \mathbf{h}_t^{b}]$$

$$\text{AU}_{b,t} = \sigma\left(\text{Linear}(\mathbf{h}_t)\right) \in \mathbb{R}^{28} \quad \text{(sigmoid multi-label)}$$

$$\text{OPD}_{b,u} = \left[t_{\text{onset}}, t_{\text{peak}}, t_{\text{decay}}\right] \in \mathbb{R}^3 \quad \text{(onset-peak-decay landmarks)}$$

Outputs: AU intensities `(B, T, 28)` + OPD landmarks `(B, 28, 3)`.

### Stage 6: Mixture-of-Experts Head

**MoEGatingNetwork** — Noisy top-k gating with 3 expert MLPs:

$$g = \text{softmax}\left(\text{top-}k\left(W_g \cdot f_{\text{fused}}\right)\right)$$

$$\text{ME\_logits} = \sum_{i=1}^{3} g_i \cdot \text{Expert}_i(f_{\text{fused}})$$

**Auxiliary load-balancing loss** prevents expert collapse:

$$\mathcal{L}_{\text{moe}} = \lambda \sum_{i=1}^{3} \left(\bar{f}_i - \frac{1}{3}\right)^2, \quad \bar{f}_i = \frac{1}{B}\sum_b g_i^{(b)}$$

**PersonalizedRadar** — Test-time adaptation via 5-step inner-loop SGD on support frames with an identity-initialized residual adapter.

### Stage 7: Emotion Reporter

Template-based clinical report generation with AU parsing, ME classification, rPPG physiological cues, and OPD temporal landmarks. Optional HuggingFace OPT-125M for free-text generation (falls back gracefully when offline).

---

## Mathematical Formulation

### Total Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{me}} + \alpha \mathcal{L}_{\text{au}} + \beta \mathcal{L}_{\text{moe}} + \gamma \mathcal{L}_{\text{opd}}$$

| Loss | Type | Description |
|------|------|-------------|
| $\mathcal{L}_{\text{me}}$ | Cross-Entropy | 7-class micro-expression classification |
| $\mathcal{L}_{\text{au}}$ | Binary Cross-Entropy | 28-class AU multi-label recognition |
| $\mathcal{L}_{\text{moe}}$ | Load-balancing auxiliary | Prevents expert collapse |
| $\mathcal{L}_{\text{opd}}$ | L2 smoothness + peak consistency | Onset-peak-decay temporal pattern |

### Architecture Dimensions

```
Input:             (B, 3, T=16, H=224, W=224)
     │
SaliencyMap:       (B, 1, 16, 224, 224)
rPPGHeatmap:       (B, 3, 16, 224, 224)
FlowStack:         (B, 2, 16, 224, 224)
     │
FastFeat:          (B, 512)
SlowFeat:          (B, 768) + SlowSpatial (B, 768, 1, 7, 7)
     │
FastGated:         (B, 512)
SlowGated:         (B, 768)
     │
FusedFeat:         (B, 1024)
     │
AUIntensities:     (B, 16, 28)  ← sigmoid multi-label
AUOPD:            (B, 28, 3)    ← onset/peak/decay
MELogits:         (B, 7)       ← 7-class CE
ExpertGates:      (B, 3)       ← top-2 softmax
```

---

## Benchmark Datasets

| Dataset | Samples | Subjects | Frame Rate | Resolution | Classes | Source |
|---------|---------|----------|-----------|-----------|---------|--------|
| **CASME II** | 247 | 26 | 200 fps | 640×480 | 5–7 | [CAS Official](http://casme.psych.ac.cn/casme/c2) |
| **SAMM** | 159 | 32 | 200 fps | 2040×1088 | 7–8 | [MMU](https://www.mmu.ac.uk) |
| **SMIC-HS** | 164 | 16 | 100 fps | 640×480 | 3 | [Oulu](https://www.oulu.fi) |
| **MMEW** | 300 (+900 macro) | 36 | 90 fps | 1920×1080 | 7 | [IEEE TPAMI 2022](https://github.com/benxianyeteam/MMEW-Dataset) |
| **CAS(ME)³** | ~300+ | — | 30 fps | Various | 4+ | [CAS Official](http://melab.psych.ac.cn) |
| **iMER Benchmark** | 5 datasets | — | — | — | incremental | [arXiv:2501.19111](https://arxiv.org/abs/2501.19111) |

> **Note:** Most datasets require a signed license agreement for access. CASME II, SAMM, and SMIC are accessible via their respective institutional websites. MMEW and iMER benchmark code are publicly available on GitHub.

### MEGC (Micro-Expression Grand Challenge) Results

| Year | Venue | Top Method | Approach |
|------|-------|-----------|----------|
| 2022 | ACM MM | USTC-IAT-United | Optical flow + TPS interpolation |
| 2023 | ACM MM | CAS-IA + BUST | **VideoMAE + Optical Flow** (34 teams) |
| 2024 | ACM MM | USTC + HIT | Deep learning + cross-cultural generalization |
| 2025 | ACM MM | TBD | Incremental learning + multimodal |

---

## State-of-the-Art Comparison

### Accuracy Comparison on Standard Benchmarks

| Method | Backbone | CASME II | SAMM | SMIC | CAS(ME)² |
|--------|----------|---------|------|------|------|--------|
| **Hybrid Attention-3DNet** (JJCIT 2025) | 3D CNN + SE | 93.79% | 93.61% | 93.42% | **93.95%** |
| **ROI-ArcFace** (IEEE 2025) | CNN + ROI | **93.96%** | 86.15% | 81.17% | — |
| **STRNet** (Int. J. SCC 2025) | Region-based | — | — | — | UF1=0.9792 |
| **GAM-MER** (Heliyon 2024) | Graph Attn + Transf | 91.57% | 91.25% | 86.22% | — |
| **MCCA-VNet** (PMC 2024) | ViT + XCiT + CBAM | — | — | — | UF1=0.868 |
| **SelfME** (IEEE 2024) | Transformer | 90.78% | — | 69.70% | — |
| **μ-BERT** (ACM MM 2024) | BERT-style | 90.34% | — | 85.80% | — |
| **Dual-Branch Cross-Attn** (2024) | Swin + MobileViT | — | — | — | 81.6% |
| **Multi-scale 3D ResNet** (J. Image 2024) | 3D-ResNet50 | 91.35% | 84.77% | 74.6% | — |
| **LAENet** (OA 2024) | Lightweight 3D CNN | 79.19% | — | — | — |
| **OFF-ApexNet** (baseline) | CNN | 87.64% | 54.09% | 68.17% | — |
| **LBP-TOP** (baseline) | Handcrafted | 70.26% | 39.54% | 20.00% | — |

### Latest SOTA Methods (2024-2025)

| Method | Venue | Key Innovation | Code |
|--------|-------|---------------|------|
| **VideoMAE + OF** | ACM MM 2023 | Self-supervised pre-training | [GitHub](https://github.com/VisionVR/VideoMAE) |
| **Prompt-MER** | IEEE 2024 | Visual prompt tuning | [GitHub](https://github.com/VisionVR/Prompt-MER) |
| **iMER Benchmark** | arXiv 2025 | Incremental learning framework | [GitHub](https://github.com/ZhengQinLai/IMER-benchmark) |
| **DualPrompt** | CVPR 2024 | Dual prompt learning | Available in iMER |

### Censor's Design Rationale vs SOTA

| SOTA Feature | Censor Implementation | Advantage |
|---|---|---|
| Optical flow | OpenCV DualTVL1 (real TV-L1) | Most accurate classical flow |
| Dual-pathway | 3D ResNet-18 + 3D Swin-T | Biologically motivated |
| Cross-attention fusion | Bidirectional TSFmicroFusion (1024-D) | Full feature interaction |
| AU multi-label | BiLSTM → 28 sigmoid outputs | Temporal dynamics |
| MoE routing | Noisy top-2 with load-balancing | Expert specialization |
| Apex detection | CASANet triangular attention | Mimics micro-expression pattern |
| rPPG physiology | Chrominance + bandpass filtering | Physiological correlates |
| Test-time adaptation | PersonalizedRadar (SGD identity) | Per-subject personalization |

---

## Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GTX 1080 (8GB) | NVIDIA RTX 3090/4090 (24GB) |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 50 GB SSD | 100 GB SSD |
| **OS** | Ubuntu 20.04 / Windows 10+ | Ubuntu 22.04 / Windows 11 |

> **Note:** GPU with CUDA compute capability ≥ 7.0 recommended for optimal performance.

### Environment Setup

#### Option 1: Conda (Recommended)

```bash
# Create and activate conda environment
conda create -n censor python=3.10 -y
conda activate censor

# Install PyTorch with CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install Censor dependencies
cd D:\censor
pip install -r requirements.txt
```

#### Option 2: Virtual Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 3: Docker (Optional)

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### Dependency Details

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Core deep learning framework |
| `torchvision` | ≥0.15.0 | Vision utilities |
| `numpy` | ≥1.24.0 | Numerical operations |
| `scipy` | ≥1.10.0 | Signal processing |
| `einops` | ≥0.6.0 | Tensor operations |
| `opencv-python` | ≥4.7.0 | Optical flow extraction |
| `opencv-contrib-python` | ≥4.7.0 | Extended OpenCV features |
| `Pillow` | ≥9.0.0 | Image handling |
| `transformers` | ≥4.30.0 | LLM for report generation |
| `accelerate` | ≥0.20.0 | Training acceleration |

#### Optional Dependencies

```bash
# For GPU profiling
pip install nvidia-ml-py3

# For remote monitoring
pip install tensorboard

# For distributed training
pip install torchrun  # included in PyTorch
```

---

## Quick Start

### 1. Forward Pass Test (Synthetic Data)

```bash
cd D:\censor
python main.py
```

Expected output:
```
============================================================
 Censor -- Biomimetic Dual-Pathway MER System
============================================================
[Censor] Initializing Preprocessing...
[Censor] Initializing Dual-Pathway Backbones...
...

Input video: torch.Size([2, 3, 16, 224, 224])

...

--- Stage 1: Preprocessing ---
...
--- Stage 7: Emotion Reporter ---
...

============================================================
 Final Output Summary
============================================================
  ME Logits:       torch.Size([2, 7])
  AU Intensities:  torch.Size([2, 16, 28])
  AU OPD:          torch.Size([2, 28, 3])
  Apex Scores:    torch.Size([2, 1])
  Expert Gates:   torch.Size([2, 3])
  MoE Aux Loss:   0.001234
  Adapted Feat:   torch.Size([2, 1024])
  Reports:       2 templates
============================================================
```

### 2. Training with Synthetic Data

```bash
python train.py --epochs 5 --batch_size 2 --synthetic_data
```

### 3. Training with Real Dataset

```bash
# Download CASME II dataset and place in ./data/CASME_II
# Follow dataset preparation instructions in prepare_data.py

python train.py --epochs 50 --batch_size 4 --data_root ./data/CASME_II
```

---

## Training

### Basic Training

```bash
# Full training with default settings
python train.py \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --data_root ./data/CASME_II \
    --output_dir ./checkpoints
```

### Advanced Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 2 | Batch size per GPU |
| `--lr` | 1e-4 | Initial learning rate |
| `--weight_decay` | 1e-4 | Weight decay for optimizer |
| `--au_loss_weight` | 0.5 | AU loss weight α |
| `--moe_loss_weight` | 0.01 | MoE load-balancing weight β |
| `--landmark_weight` | 0.1 | OPD landmark loss weight γ |
| `--output_dir` | ./checkpoints | Checkpoint output directory |
| `--val_every` | 1 | Validation frequency (epochs) |
| `--save_every` | 5 | Checkpoint save frequency |
| `--use_amp` | True | Automatic mixed precision |
| `--resume` | None | Resume from checkpoint path |

### Full Training Example

```bash
# Multi-GPU training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --use_amp \
    --data_root ./data/CASME_II \
    --output_dir ./checkpoints/censor_exp1
```

### Checkpointing & Resume

```bash
# Resume training from checkpoint
python train.py \
    --resume ./checkpoints/censor_epoch50.pt \
    --epochs 100
```

---

## Usage Guide

### Python API

#### Basic Usage

```python
import torch
from model import Censor

# Initialize model
model = Censor()
model.eval()  # or model.train() for training

# Prepare input
video = torch.randn(1, 3, 16, 224, 224)  # (B, C, T, H, W)

# Forward pass
with torch.no_grad():
    outputs = model(video)

# Access results
me_logits = outputs['me_logits']           # (B, 7) micro-expression scores
au_intensities = outputs['au_intensities']  # (B, T, 28) AU intensities
apex_scores = outputs['apex_scores']      # (B, 1) apex frame confidence
reports = outputs['template_report']   # List[str] clinical reports
```

#### Using with DataLoader

```python
from torch.utils.data import DataLoader
from dataset import MicroExpressionDataset

# Create dataset
dataset = MicroExpressionDataset(
    root='./data/CASME_II',
    split='test',
    T=16,
    H=224,
    W=224
)

loader = DataLoader(dataset, batch_size=4, shuffle=False)

# Process batch
results = []
with torch.no_grad():
    for videos, labels in loader:
        outputs = model(videos)
        results.append({
            'me_logits': outputs['me_logits'],
            'au_intensities': outputs['au_intensities'],
            'labels': labels
        })
```

#### Test-Time Adaptation

```python
from model import Censor, PersonalizedRadar

# Load base model
model = Censor()
checkpoint = torch.load('./checkpoints/best.pt')
model.load_state_dict(checkpoint['model_state'])

# Load support set for personalization
support_videos = torch.randn(8, 3, 16, 224, 224)
support_labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

# Apply PersonalizedRadar adaptation
radar = PersonalizedRadar(RADAR_CONFIG)
adapted_model = radar.adapt(model, support_videos, support_labels)

# Evaluate on query
query_video = torch.randn(1, 3, 16, 224, 224)
outputs = adapted_model(query_video)
```

### Configuration

#### Customizing Hyperparameters

Edit `config/defaults.py` or pass via command line:

```python
from config.defaults import FAST_PATHWAY_CONFIG, SLOW_PATHWAY_CONFIG

# Modify configuration
FAST_PATHWAY_CONFIG['output_dim'] = 256  # Reduce fast pathway dimension
SLOW_PATHWAY_CONFIG['stages'][3]['dim'] = 512  # Reduce slow pathway

# Initialize with custom config
from model import Censor
model = Censor()  # Uses modified config
```

#### Creating Custom Config File

```python
# config/custom.py
from config.defaults import *

MY_CONFIG = {
    'input': {'batch_size': 4, 'temporal': 24},
    'fast_pathway': {'output_dim': 256},
    'slow_pathway': {'output_dim': 512},
    'fusion': {'fused_dim': 768},
    'moe': {'num_experts': 4, 'top_k': 2},
}
```

### Output Interpretation

#### Micro-Expression Classification

```python
# Get predicted class
me_probs = torch.softmax(outputs['me_logits'], dim=-1)
predicted_class = me_probs.argmax(dim=-1)

# Class labels
CLASS_LABELS = ['Happiness', 'Sadness', 'Surprise', 'Fear', 'Anger', 'Disgust', 'Contempt']

for i, pred in enumerate(predicted_class):
    print(f"Sample {i}: {CLASS_LABELS[pred]} (confidence: {me_probs[i, pred]:.2%})")
```

#### Action Unit Analysis

```python
# Get active AUs (threshold > 0.3)
au_thresh = 0.3
active_aus = (outputs['au_intensities'] > au_thresh).nonzero(as_tuple=True)

AU_NAMES = {
    1: "Inner Brow Raiser", 4: "Brow Lowerer", 5: "Upper Lid Raiser",
    6: "Cheek Raiser", 7: "Lid Tightener", 9: "Nose Wrinkler",
    10: "Upper Lip Raiser", 12: "Lip Corner Puller", 14: "Dimpler",
    15: "Lip Corner Depressor", 17: "Chin Raiser", 20: "Lip Stretcher",
    23: "Lip Tightener", 25: "Lips Part", 26: "Jaw Drop"
}

# Print active AUs per frame
for b, t, au in zip(*active_aus):
    print(f"Frame {t}: AU{au} ({AU_NAMES.get(au, 'Unknown')})")
```

#### Apex Frame Detection

```python
# Get apex frame
apex_frame = outputs['apex_scores'].argmax(dim=1)
print(f"Detected apex frame: {apex_frame.item()}")
```

#### Clinical Report Generation

```python
# Get structured report
for report in outputs['template_report']:
    print(report)
    print("-" * 40)
```

---

## Project Structure

```
censor/
├── main.py                 # Censor orchestrator + forward pass test
├── train.py                # Training pipeline (multi-task loss, AMP, checkpointing)
├── dataset.py              # Dataset definitions
├── prepare_data.py         # Data preparation utilities
├── requirements.txt      # Dependencies
├── config/
│   └── defaults.py       # Central hyperparameter dictionary
├── model/
│   ├── __init__.py     # Re-exports all classes
│   ├── preprocessing.py  # SaliencyDetector, rPPGExtractor, TVL1OpticalFlow
│   ├── backbones.py    # FastSubcorticalPathway, SlowCorticalPathway
│   ├── attention.py   # Amygdala, FFA, CASANet
│   ├── fusion.py    # TSFmicroFusion
│   ├── decoders.py   # DynamicAUDecoder
│   ├── moe_head.py  # MoEGatingNetwork, PersonalizedRadar
│   └── llm_report.py # EmotionReporter
└── data/
    └── iMER/           # iMER benchmark code
        ├── backbone/   # Various backbone implementations
        ├── models/    # Continual learning models
        └── utils/     # Data utilities
```

---

## Citation

```bibtex
@article{censor2025,
  title={Censor: A Biomimetic Dual-Pathway Micro-Expression Recognition System with Fusiform-Amygdala Circuit and Mixture-of-Experts},
  author={},
  journal={},
  year={2025}
}
```

---

## References

- [CASME II Database](http://casme.psych.ac.cn/casme/c2)
- [SAMM Micro-Expression Database](https://www.mmu.ac.uk)
- [SMIC Database](https://www.oulu.fi)
- [MMEW Dataset](https://github.com/benxianyeteam/MMEW-Dataset)
- [iMER Benchmark](https://github.com/ZhengQinLai/IMER-benchmark) — arXiv:2501.19111
- [MEGC2024 ACM MM](https://researchportal.hw.ac.uk/en/publications/megc2024-acm-multimedia-2024-facial-micro-expression-grand-challe/)
- [Video-Based Facial Micro-Expression Analysis: A Survey](https://ar5iv.labs.arxiv.org/html/2201.12728) — IEEE TPAMI Survey
- [Hybrid Attention-3DNet](https://ictcsreg.psut.edu.jo/paper/export/254) — JJCIT 2025
- [GAM-MER](https://www.sciencedirect.com/science/article/pii/S2405844024119950) — Heliyon 2024
- [MCCA-VNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644463/) — ViT + CBAM 2024