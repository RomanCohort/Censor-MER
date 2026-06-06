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
- [Biomimetic Enhancements](#biomimetic-enhancements)
  - [Dynamic Topology Networks (DTN)](#1-dynamic-topology-networks-dtn)
  - [Meta-Plasticity Memory](#2-meta-plasticity-memory)
  - [Biological MoE (BioMoE)](#3-biological-moe-biomoe)
  - [EnhancedMoE Wrapper](#4-enhancedmoe-wrapper)
- [Mathematical Formulation](#mathematical-formulation)
- [Benchmark Datasets](#benchmark-datasets)
- [State-of-the-Art Comparison](#state-of-the-art-comparison)
- [Training](#training)
- [Quick Start](#quick-start)
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
AUOPD:             (B, 28, 3)    ← onset/peak/decay
MELogits:          (B, 7)       ← 7-class CE
ExpertGates:       (B, 3)       ← top-2 softmax
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

---

## State-of-the-Art Comparison

### Accuracy Comparison on Standard Benchmarks

**Note**: Some recent 2025 publications claim higher accuracy (93-94%) including Hybrid Attention-3DNet (JJCIT 2025: 93.79%) and ROI-ArcFace (IEEE 2025: 93.96%). These are included in the annotated bibliography but require verification through reproducible code and peer-reviewed confirmation. We position Censor against established, verified baselines while acknowledging higher claims in recent literature.

| Method | Backbone | CASME II | SAMM | SMIC | CAS(ME)² | Verification |
|--------|----------|---------|------|------|----------|--------------|
| **Multi-scale 3D ResNet** (Neurocomputing 2024) | 3D-ResNet50 | 91.35% | 84.77% | 74.6% | — | ✅ DOI verified |
| **GAM-MER** (Heliyon 2024) | Graph Attn + Transf | 91.57% | 91.25% | 86.22% | — | ⚠️ Paper verified |
| **SelfME** (Pattern Recognition Letters 2024) | Transformer | 90.78% | — | 69.70% | — | ✅ Verified |
| **μ-BERT** (ACM MM 2024) | BERT-style | 90.34% | — | 85.80% | — | ✅ Verified |
| **MCCA-VNet** (Eng. Applications of AI 2024) | ViT + XCiT + CBAM | — | — | — | UF1=0.868 | ⚠️ Paper verified |
| **Dual-Branch Cross-Attn** (2024) | Swin + MobileViT | — | — | — | 81.6% | ⚠️ Verify source |
| **OFF-ApexNet** (baseline) | CNN | 87.64% | 54.09% | 68.17% | — | ✅ Established baseline |
| **LAENet** (OA 2024) | Lightweight 3D CNN | 79.19% | — | — | — | ⚠️ Verify source |
| **LBP-TOP** (baseline) | Handcrafted | 70.26% | 39.54% | 20.00% | — | ✅ Established baseline |

**Unverified Claims** (excluded pending verification):
- Hybrid Attention-3DNet (JJCIT 2025): Claims 93.79% CASME II, 93.61% SAMM, 93.42% SMIC, 93.95% CAS(ME)²
- ROI-ArcFace (IEEE 2025): Claims 93.96% CASME II, 86.15% SAMM, 81.17% SMIC
- STRNet (Int. J. SCC 2025): Claims UF1=0.9792 on CAS(ME)²

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

## Training

```bash
# Install dependencies
pip install -r requirements.txt

# Full training with synthetic data (debug)
python train.py --epochs 50 --batch_size 16 --lr 1e-4 --synthetic_data

# Training with real dataset
python train.py --epochs 50 --batch_size 16 --lr 1e-4 --data_root ./data/CASME_II

# Training arguments
--epochs           Number of training epochs (default: 50)
--batch_size       Batch size (default: 2, small for GPU memory)
--lr               Learning rate (default: 1e-4)
--weight_decay     Weight decay (default: 1e-4)
--au_loss_weight   AU loss weight α (default: 0.5)
--moe_loss_weight  MoE load-balancing weight β (default: 0.01)
--landmark_weight  OPD landmark loss weight γ (default: 0.1)
--output_dir       Checkpoint output directory (default: ./checkpoints)
--val_every        Validation frequency in epochs (default: 1)
```

### Loss Weights

Default configuration: $\alpha=0.5$, $\beta=0.01$, $\gamma=0.1$

### Multi-Task Loss

```python
total_loss = (
    1.0 * loss_me        +   # Cross-entropy, 7-class
    0.5 * loss_au        +   # BCE, 28 AUs multi-label
    0.01 * loss_moe      +   # Load-balancing auxiliary
    0.1 * loss_landmark     # OPD smoothness + peak consistency
)
```

---

## Quick Start

```bash
# Forward pass test (synthetic data)
python main.py

# Expected output:
#   Total parameters: 68,353,230
#   ME Logits:       torch.Size([2, 7])
#   AU Intensities:  torch.Size([2, 16, 28])
#   AU OPD:          torch.Size([2, 28, 3])
#   Apex Scores:     torch.Size([2, 1])
#   Expert Gates:    torch.Size([2, 3])
#   MoE Aux Loss:    ~0.001
#   Reports:         2 templates
```

---

## Project Structure

```
censor/
├── main.py                 # Censor orchestrator + forward pass test
├── train.py                # Training pipeline (multi-task loss, AMP, checkpointing)
├── requirements.txt
├── config/
│   └── defaults.py         # Central hyperparameter dictionary
└── model/
    ├── __init__.py         # Re-exports all classes
    ├── preprocessing.py    # SaliencyDetector, rPPGExtractor, TVL1OpticalFlow
    ├── backbones.py        # FastSubcorticalPathway, SlowCorticalPathway
    ├── attention.py        # Amygdala, FFA, CASANet
    ├── fusion.py           # TSFmicroFusion
    ├── decoders.py         # DynamicAUDecoder
    ├── moe_head.py         # MoEGatingNetwork, PersonalizedRadar
    ├── llm_report.py       # EmotionReporter
    ├── biomimetic_enhance.py  # DTN + Meta-Plasticity
    └── biomoe.py            # Biological gating (BioMoE)
```

---

## Biomimetic Enhancements

### 1. Dynamic Topology Networks (DTN)

Inspired by cytoskeleton mechanosensitive channels. Feature edges are modulated by **tension** computed from input gradient:

```
tension = ||feature_gradient|| 
gate = sigmoid(gain × tension - threshold)
output = input × gate
```

**File**: `model/biomimetic_enhance.py`, `dynamic_topology_networks.md`

### 2. Meta-Plasticity Memory

Inspired by DNA methylation. Dual-track memory system:

- **KV Cache** (short-term): Session-level context
- **Methylation** (long-term): LoRA weight consolidation triggered by emotion intensity

```
if emotion_score > strong_threshold:
    consolidate LoRA weights with timestamp
```

**File**: `model/biomimetic_enhance.py`, `meta_plasticity_memory.md`

### 3. Biological MoE (BioMoE)

Inspired by neuronal membrane potential. Gating depends on **both input AND historical feedback**:

```
gate = f(input) + membrane_bias + emotion_gain × mood
feedback = (prediction == ground_truth)  # 1.0 or 0.0
membrane += feedback × learning_rate
```

**Modes**:
- `standard`: Original MoE (stateless)
- `bio`: Full biological gating (membrane + emotion)
- `hybrid`: Original experts + biological gating (recommended)

**Training Integration**: In `train.py`, feedback is applied automatically:

```python
# Automatic feedback during training
preds = outputs['me_logits'].argmax(dim=1)
correct = (preds == labels).float()  # 1=correct, 0=wrong
fb = correct.mean()
moe.apply_feedback(fb)  # Update membrane potential
```

**Effect**: Early training → conservative routing (errors), Late training → confident routing (success)

**File**: `model/biomoe.py`, `model/enhanced_moe.py`, `test_biomoe.py`

### 4. EnhancedMoE Wrapper

Replace original MoE with feedback-enabled version:

```python
from model.enhanced_moe import EnhancedMoE

# Hybrid mode (recommended for training)
moe = EnhancedMoE(mode="hybrid", enable_membrane=True, enable_emotion=True)

# Forward pass
output, gates, aux_loss, info = moe(x)

# External feedback (inference-time)
moe.apply_feedback(1.0)   # User confirmed correct
moe.apply_feedback(0.0)  # Prediction was wrong
moe.apply_feedback(-1.0) # User explicitly corrected

# Get state
state = moe.get_state()
# {positive_count, negative_count, accuracy, ...}
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
- [Confluencia Platform Reference](D:\IGEM集成方案\readme\TOTALREADME_katex_fixed.md)