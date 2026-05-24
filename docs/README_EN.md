# Censor: Biomimetic Dual-Pathway Micro-Expression Recognition System

> **仿生双通道微表情识别系统** — A PyTorch implementation of a biomimetic dual-pathway architecture for micro-expression recognition (MER), simulating the fusiform-amygdala circuit in the human visual pathway.

---

## Table of Contents / 目录

- [Overview / 概述](#overview--概述)
- [Architecture / 系统架构](#architecture--系统架构)
- [Mathematical Formulation / 数学公式](#mathematical-formulation--数学公式)
- [Benchmark Datasets / 基准数据集](#benchmark-datasets--基准数据集)
- [State-of-the-Art Comparison / 性能对比](#state-of-the-art-comparison--性能对比)
- [Training / 训练](#training--训练)
- [Quick Start / 快速开始](#quick-start--快速开始)
- [Project Structure / 项目结构](#project-structure--项目结构)
- [Biomimetic Enhancements / 仿生增强](#biomimetic-enhancements--仿生增强)
- [Citation / 引用](#citation--引用)

---

## Overview / 概述

Micro-expressions (MEs) are brief, involuntary facial expressions that occur when a person suppresses or conceals their true emotions. They last between **40–200 ms** and are characterized by subtle muscle activations measurable via Action Units (AUs) in the Facial Action Coding System (FACS).

微表情（ME）是一种短暂的、不自主的面部表情，当人们试图抑制或隐藏真实情绪时会发生。微表情持续时间仅为**40-200毫秒**，其特征是通过面部动作编码系统（FACS）中的动作单元（AU）衡量的细微肌肉活动。

**Censor** proposes a biomimetic dual-pathway architecture that mirrors the human visual system's fast subcortical and slow cortical pathways:

**Censor** 提出了一种仿生双通道架构，模拟人类视觉系统的快速皮层下通路和慢速皮层通路：

```
Input Video (B×3×16×224×224)
  │
  ├── [Stage 1] Biomimetic Preprocessing / 仿生预处理
  │   ├── SaliencyDetector (Foveal sampling via Gaussian pyramid) / 显著性检测（高斯金字塔中心凹采样）
  │   ├── rPPGExtractor (Remote photoplethysmography blood-flow) / rPPG提取（远程光电容积脉搏波）
  │   └── TVL1OpticalFlow (OpenCV DualTVL1 optical flow) / TVL1光流（OpenCV）
  │
  ├── [Stage 2] Dual-Pathway Backbones / 双通道骨干网络
  │   ├── FastPath: 3D ResNet-18 (optical flow) → 512-D / 快通道：3D ResNet-18 → 512维
  │   └── SlowPath: 3D Swin-Transformer (RGB+rPPG) → 768-D + spatial map / 慢通道：3D Swin → 768维+空间图
  │
  ├── [Stage 3] Fusiform-Amygdala Attention Circuit / 梭状回-杏仁核注意力回路
  │   ├── Amygdala: attention prior map from fast pathway / 杏仁核：快通道注意力先验图
  │   ├── FFA: SE-style cross-pathway gating / FFA：SE风格跨通道门控
  │   └── CASANet: apex frame detection via triangular attention / CASANet：三角注意力apex检测
  │
  ├── [Stage 4] Spatio-Temporal Fusion / 时空融合 (Bidirectional cross-attention, 1024-D)
  │
  ├── [Stage 5] Dynamic AU Decoder / 动态AU解码器 (BiLSTM, 28 AUs + OPD landmarks)
  │
  ├── [Stage 6] Mixture-of-Experts Head / 多专家头 (3 experts + PersonalizedRadar TTA)
  │
  └── [Stage 7] Emotion Reporter / 情绪报告器 (template + LLM-based reports)
```

| Metric / 指标 | Value / 数值 |
|--------------|--------------|
| **Total Parameters / 总参数量** | 68,353,230 |
| **Architecture / 架构** | Dual-pathway: 3D ResNet-18 + 3D Swin-Transformer |
| **Preprocessing / 预处理** | Gaussian saliency + rPPG + OpenCV TV-L1 |
| **Attention / 注意力** | Amygdala (FC) + FFA (SE) + CASANet (triangular MHA) |
| **Fusion / 融��** | Bidirectional cross-attention in 1024-D space |
| **AU Decoding / AU解码** | BiLSTM (2 layers, 512 hidden) → 28 sigmoid outputs |
| **MoE / 多专家** | 3 experts, top-2 gating, load-balancing auxiliary loss |
| **TTA / 测试时适配** | PersonalizedRadar (5-step SGD identity adapter) |

---

## Architecture / 系统架构

### Stage 1: Biomimetic Preprocessing / 仿生预处理

#### SaliencyDetector — Foveal Sampling / 显著性检测器 — 中心凹采样

Simulates human retinal fovea (highest cone density at 1–2° visual angle) via **Gaussian pyramid** with center-biased spatial prior:

模拟人类视网膜中心凹（1-2度视觉角处锥细胞密度最高），使用**高斯金字塔**和中心偏向的空间先验：

$$S(x,y) = \sum_{l=0}^{L-1} w_l \cdot G_\sigma(x,y) \cdot I_l(x,y)$$

其中 $I_l$ 是第 $l$ 层金字塔，$G_\sigma$ 是中心偏向的高斯先验，$w_l = 2^{-l}$ 是层级权重。/ where $I_l$ is the $l$-th pyramid level, $G_\sigma$ is the center-biased Gaussian prior, and $w_l = 2^{-l}$ are level weights.

Output / 输出：`(B, 1, T, H, W)`

#### rPPGExtractor — Remote Photoplethysmography / rPPG提取器 — 远程光电容积脉搏波

Captures blood oxygen saturation fluctuations (0.5–4.0 Hz cardiac range) via **chrominance decomposition** and **temporal bandpass filtering**:

通过**色度分解**和**时间带通滤波**捕捉血氧饱和度波动（0.5-4.0Hz心脏范围）：

$$\text{rPPG}(t) = \sum_{c \in \{R,G,B\}} \alpha_c \cdot I_c(t)$$

$$\text{rPPG}_{\text{filtered}}(t) = \sum_{\tau=-K}^{K} h(\tau) \cdot \text{rPPG}(t-\tau)$$

其中 $\alpha_c$ 是学习的色度投影权重，$h$ 是学习的FIR带通滤波器。/ where $\alpha_c$ are learned chrominance projection weights and $h$ is a learned FIR bandpass filter.

Output / 输出：`(B, 3, T, H, W)`

#### TVL1OpticalFlow — OpenCV DualTVL1

Computes real TV-L1 optical flow via OpenCV's `createOptFlow_DualTVL1`. The TV-L1 energy functional:

通过OpenCV的`createOptFlow_DualTVL1`计算真实TV-L1光流。TV-L1能量泛函：

$$\min_u \int\left(|\nabla u| + \lambda \cdot |I_1(x+u) - I_0(x)|\right) dx$$

solved via primal-dual algorithm. / 通过原始-对偶算法求解。

Output / 输出：`(B, 2, T, H, W)`

### Stage 2: Dual-Pathway Backbones / 双通道骨干网络

#### FastSubcorticalPathway — 3D ResNet-18 / 快皮层下通路 — 3D ResNet-18

Processes **optical flow** input through a shallow 3D ResNet-18 variant (3 stages, 64→128→256 channels). Large temporal strides (2²,2²) simulate fast subcortical processing.

通过浅层3D ResNet-18变体（3阶段，64→128→256通道）处理**光流**输入。大时间步长（2²,2²）模拟快速皮层下处理。

Output / 输出：`(B, 512)`

#### SlowCorticalPathway — 3D Swin-Transformer / 慢皮层通路 — 3D Swin-Transformer

Processes **RGB + rPPG** (6 channels) through a full 3D Swin-Transformer with 4 stages and **shifted-window multi-head self-attention (W-MSA)**:

通过4阶段完整3D Swin-Transformer处理**RGB + rPPG**（6通道），使用**移位窗口多头自注意力（W-MSA）**：

| Stage / 阶段 | Blocks / 块数 | Dim / 维度 | Merge Stride / 合并步长 | Resolution / 分辨率 |
|--------------|--------------|-----------|----------------------|----------------------|
| 1 | 2 | 96 | (2,2,2) | T/2, H/2, W/2 |
| 2 | 2 | 192 | (2,2,2) | T/4, H/4, W/4 |
| 3 | 6 | 384 | (2,2,2) | T/8, H/8, W/8 |
| 4 | 2 | 768 | (1,1,1) | T/16, H/32, W/32 |

Relative position bias (3D meshgrid) enables accurate spatial relationship modeling.

相对位置偏置（3D网格）实现准确的空间关系建模。

Output / 输出：**pooled `(B, 768)`** + **spatial map `(B, 768, T/16, H/32, W/32)`**

### Stage 3: Fusiform-Amygdala Attention Circuit / 梭状回-杏仁核注意力回路

#### Amygdala — Attention Prior Map / 杏仁核 — 注意力先验图

Fully-connected layers generate a spatial attention prior map from fast pathway features:

全连接层从快通道特征生成空间注意力先验图：

$$\text{APM} = \sigma\left(\text{FC}_{512\rightarrow256\rightarrow196}(\text{fast\_feat})\right).view(B,1,14,14)$$

sigmoid激活的APM指导空间注意力朝向面部感兴趣区域。/ The sigmoid-activated APM guides spatial attention toward facial regions of interest.

#### FFA — Feature Fusion Attention / FFA — 特征融合注意力

SE-style squeeze-excitation gating for cross-pathway feature recalibration:

SE风格的压缩-激励门控，用于跨通道特征重校准：

$$z = \sigma\left(\text{FC}_{1280\rightarrow80}(\text{concat}[f_{\text{fast}}, f_{\text{slow}}])\right)$$

$$f_{\text{fast}}^* = z_{[:512]} \odot f_{\text{fast}}, \quad f_{\text{slow}}^* = z_{[512:]} \odot f_{\text{slow}}$$

#### CASANet — Apex Frame Detection / CASANet — Apex帧检测

Inverted-triangle learnable spatial mask (7×7) + temporal MultiHeadAttention for **apex frame detection** — identifying the peak intensity frame in a micro-expression sequence:

逆三角形可学习空间掩码（7×7）+ 时间多头注意力，用于**apex帧检测**——识别微表情序列中峰值强度帧：

$$\text{apex\_score}_t = \text{softmax}\left(\text{MHA}(Q_t, K, V)\right) \in \mathbb{R}^T$$

三角先验 $M_{i,j} = \exp\left(-\frac{(j-i)^2}{2\sigma_i^2}\right)$ 模拟微表情自然的onset→apex→decay模式。/ The triangular prior simulates the natural onset→apex→decay pattern of micro-expressions.

Input / 输入：spatial map `(B, 768, 1, 7, 7)` from Slow pathway Stage 4 / 慢通道阶段4的空间图

Output / 输出：attended features + apex scores `(B, 1)` / 注意力特征 + apex分数

### Stage 4: Spatio-Temporal Fusion / 时空融合

**TSFmicroFusion** — Bidirectional cross-attention in 1024-D fused space:

**TSFmicroFusion** — 1024维双向交叉注意力：

$$\text{F}_{f2s} = \text{Attention}\left(Q_f \cdot W_Q, K_s \cdot W_K, V_s \cdot W_V\right) \cdot W_O$$

$$\text{F}_{s2f} = \text{Attention}\left(Q_s \cdot W_Q, K_f \cdot W_K, V_f \cdot W_V\right) \cdot W_O$$

$$f_{\text{fused}} = \alpha \cdot \text{FFN}(\text{F}_{f2s}) + (1-\alpha) \cdot \text{FFN}(\text{F}_{s2f}), \quad \alpha = \sigma(W_\alpha[f_{\text{fast}}; f_{\text{slow}}])$$

Output / 输出：`(B, 1024)`

### Stage 5: Dynamic AU Decoder / 动态AU解码器

**DynamicAUDecoder** — BiLSTM for temporal Action Unit sequence modeling:

**DynamicAUDecoder** — BiLSTM进行时间动作单元序列建模：

$$\mathbf{h}_t = \text{BiLSTM}(f_{\text{fused}}, \mathbf{h}_{t-1}), \quad \mathbf{h}_T = [\mathbf{h}_t^{f}; \mathbf{h}_t^{b}]$$

$$\text{AU}_{b,t} = \sigma\left(\text{Linear}(\mathbf{h}_t)\right) \in \mathbb{R}^{28} \quad \text{(sigmoid multi-label)}$$

$$\text{OPD}_{b,u} = \left[t_{\text{onset}}, t_{\text{peak}}, t_{\text{decay}}\right] \in \mathbb{R}^3 \quad \text{(onset-peak-decay landmarks)}$$

Output / 输出：AU intensities `(B, T, 28)` + OPD landmarks `(B, 28, 3)` / AU强度 + OPD路标

### Stage 6: Mixture-of-Experts Head / 多专家头

**MoEGatingNetwork** — Noisy top-k gating with 3 expert MLPs:

**MoEGatingNetwork** — 带3个专家MLP的噪声top-k门控：

$$g = \text{softmax}\left(\text{top-}k\left(W_g \cdot f_{\text{fused}}\right)\right)$$

$$\text{ME\_logits} = \sum_{i=1}^{3} g_i \cdot \text{Expert}_i(f_{\text{fused}})$$

**Auxiliary load-balancing loss** prevents expert collapse:

**辅助负载均衡损失**防止专家��塌��

$$\mathcal{L}_{\text{moe}} = \lambda \sum_{i=1}^{3} \left(\bar{f}_i - \frac{1}{3}\right)^2, \quad \bar{f}_i = \frac{1}{B}\sum_b g_i^{(b)}$$

**PersonalizedRadar** — Test-time adaptation via 5-step inner-loop SGD on support frames with an identity-initialized residual adapter.

**个性化Radar** — 通过5步内循环SGD在支持帧上进行测试时适配，配备identity初始化的残差适配器。

### Stage 7: Emotion Reporter / 情绪报告器

Template-based clinical report generation with AU parsing, ME classification, rPPG physiological cues, and OPD temporal landmarks. Optional HuggingFace OPT-125M for free-text generation (falls back gracefully when offline).

基于模板的临床报告生成，包含AU解析、微表情分类、rPPG生理线索和OPD时间路标。可选HuggingFace OPT-125M用于自由文本生成（离线时优雅降级）。

---

## Mathematical Formulation / 数学公式

### Total Loss Function / 总损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{me}} + \alpha \mathcal{L}_{\text{au}} + \beta \mathcal{L}_{\text{moe}} + \gamma \mathcal{L}_{\text{opd}}$$

| Loss / 损失 | Type / 类型 | Description / 描述 |
|-------------|-------------|----------------------|
| $\mathcal{L}_{\text{me}}$ | Cross-Entropy / 交叉熵 | 7-class micro-expression classification / 7类微表情分类 |
| $\mathcal{L}_{\text{au}}$ | Binary Cross-Entropy / 二值交叉熵 | 28-class AU multi-label recognition / 28类AU多标签识别 |
| $\mathcal{L}_{\text{moe}}$ | Load-balancing auxiliary / 负载均衡辅助 | Prevents expert collapse / 防止专家坍塌 |
| $\mathcal{L}_{\text{opd}}$ | L2 smoothness + peak consistency / L2平滑 + 峰值一致性 | Onset-peak-decay temporal pattern / Onset-peak-decay时间模式 |

### Architecture Dimensions / 架构维度

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
AUIntensities:     (B, 16, 28)  ← sigmoid multi-label / ← sigmoid多标签
AUOPD:             (B, 28, 3)    ← onset/peak/decay / ← onset/peak/decay
MELogits:          (B, 7)       ← 7-class CE / ← 7类CE
ExpertGates:       (B, 3)       ← top-2 softmax / ← top-2 softmax
```

---

## Benchmark Datasets / 基准数据集

| Dataset / 数据集 | Samples / 样本数 | Subjects / 被试 | Frame Rate / 帧率 | Resolution / 分辨率 | Classes / 类别 | Source / 来源 |
|-----------------|------------------|-----------------|------------------|---------------------|----------------|--------------|
| **CASME II** | 247 | 26 | 200 fps | 640×480 | 5–7 | [CAS Official](http://casme.psych.ac.cn/casme/c2) |
| **SAMM** | 159 | 32 | 200 fps | 2040×1088 | 7–8 | [MMU](https://www.mmu.ac.uk) |
| **SMIC-HS** | 164 | 16 | 100 fps | 640×480 | 3 | [Oulu](https://www.oulu.fi) |
| **MMEW** | 300 (+900 macro) | 36 | 90 fps | 1920×1080 | 7 | [IEEE TPAMI 2022](https://github.com/benxianyeteam/MMEW-Dataset) |
| **CAS(ME)³** | ~300+ | — | 30 fps | Various | 4+ | [CAS Official](http://melab.psych.ac.cn) |
| **iMER Benchmark** | 5 datasets | — | — | — | incremental | [arXiv:2501.19111](https://arxiv.org/abs/2501.19111) |

> **Note / 注意：** Most datasets require a signed license agreement for access. / 大多数数据集需要签署许可协议才能访问。

### MEGC (Micro-Expression Grand Challenge) Results / MEGC结果

| Year / 年份 | Venue / 会议 | Top Method / 冠军方法 | Approach / 方法 |
|-------------|--------------|----------------------|-----------------|
| 2022 | ACM MM | USTC-IAT-United | Optical flow + TPS interpolation / 光流 + TPS插值 |
| 2023 | ACM MM | CAS-IA + BUST | **VideoMAE + Optical Flow** / **VideoMAE + 光流** |
| 2024 | ACM MM | USTC + HIT | Deep learning + cross-cultural generalization / 深度学习 + 跨文化泛化 |

---

## State-of-the-Art Comparison / 性能对比

### Accuracy Comparison on Standard Benchmarks / 标准基准准确率对比

| Method / 方法 | Backbone / 骨干网络 | CASME II | SAMM | SMIC |
|---------------|-------------------|----------|------|------|
| **Hybrid Attention-3DNet** (JJCIT 2025) | 3D CNN + SE | 93.79% | 93.61% | 93.42% |
| **ROI-ArcFace** (IEEE 2025) | CNN + ROI | **93.96%** | 86.15% | 81.17% |
| **GAM-MER** (Heliyon 2024) | Graph Attn + Transf | 91.57% | 91.25% | 86.22% |
| **SelfME** (IEEE 2024) | Transformer | 90.78% | — | 69.70% |
| **μ-BERT** (ACM MM 2024) | BERT-style | 90.34% | — | 85.80% |
| **Multi-scale 3D ResNet** (J. Image 2024) | 3D-ResNet50 | 91.35% | 84.77% | 74.6% |
| **OFF-ApexNet** (baseline) | CNN | 87.64% | 54.09% | 68.17% |
| **LBP-TOP** (baseline) | Handcrafted | 70.26% | 39.54% | 20.00% |

### Censor's Design Rationale vs SOTA / Censor设计原理对比SOTA

| SOTA Feature / SOTA特性 | Censor Implementation / Censor实现 | Advantage / 优势 |
|------------------------|-----------------------------------|-------------------|
| Optical flow / 光流 | OpenCV DualTVL1 (real TV-L1) | Most accurate classical flow / 最准确的经典光流 |
| Dual-pathway / 双通道 | 3D ResNet-18 + 3D Swin-T | Biologically motivated / 生物学驱动 |
| Cross-attention fusion / 交叉注意力融合 | Bidirectional TSFmicroFusion (1024-D) | Full feature interaction / 完整特征交互 |
| AU multi-label / AU多标签 | BiLSTM → 28 sigmoid outputs | Temporal dynamics / 时间动态 |
| MoE routing / MoE路由 | Noisy top-2 with load-balancing | Expert specialization / 专家专业化 |
| Apex detection / Apex检测 | CASANet triangular attention | Mimics micro-expression pattern / 模拟微表情模式 |
| rPPG physiology / rPPG生理 | Chrominance + bandpass filtering | Physiological correlates / 生理相关 |
| Test-time adaptation / 测试时适配 | PersonalizedRadar (SGD identity) | Per-subject personalization / 被试个性化 |

---

## Training / 训练

```bash
# Install dependencies / 安装依赖
pip install -r requirements.txt

# Full training with synthetic data (debug) / 使用合成数据训练（调试）
python train.py --epochs 50 --batch_size 16 --lr 1e-4 --synthetic_data

# Training with real dataset / 使用真实数据集训练
python train.py --epochs 50 --batch_size 16 --lr 1e-4 --data_root ./data/CASME_II
```

| Argument / 参数 | Default / 默认值 | Description / 描述 |
|-----------------|------------------|--------------------|
| `--epochs` | 50 | Number of training epochs / 训练轮数 |
| `--batch_size` | 2 | Batch size / 批大小 |
| `--lr` | 1e-4 | Learning rate / 学习率 |
| `--weight_decay` | 1e-4 | Weight decay / 权重衰减 |
| `--au_loss_weight` | 0.5 | AU loss weight α / AU损失权重α |
| `--moe_loss_weight` | 0.01 | MoE load-balancing weight β / MoE负载均衡权重β |
| `--landmark_weight` | 0.1 | OPD landmark loss weight γ / OPD路标损失权重γ |
| `--output_dir` | ./checkpoints | Checkpoint output directory / 检查点输出目录 |

### Loss Weights / 损失权重

Default configuration / 默认配置：$\alpha=0.5$, $\beta=0.01$, $\gamma=0.1$

```python
total_loss = (
    1.0 * loss_me        +   # Cross-entropy, 7-class / 交叉熵，7类
    0.5 * loss_au        +   # BCE, 28 AUs multi-label / BCE，28 AU多标签
    0.01 * loss_moe      +   # Load-balancing auxiliary / 负载均衡辅助
    0.1 * loss_landmark     # OPD smoothness + peak consistency / OPD平滑+峰值一致性
)
```

---

## Quick Start / 快速开始

```bash
# Forward pass test (synthetic data) / 前向传播测试（合成数据）
python main.py
```

Expected output / 预期输出：
```
============================================================
 Censor -- 仿生双通道微表情识别系统
 Censor -- Biomimetic Dual-Pathway Micro-Expression Recognition
============================================================
  Total parameters: 68,353,230
  ME Logits:       torch.Size([2, 7])
  AU Intensities:  torch.Size([2, 16, 28])
  AU OPD:          torch.Size([2, 28, 3])
  Apex Scores:     torch.Size([2, 1])
  Expert Gates:    torch.Size([2, 3])
  MoE Aux Loss:    ~0.001
  Reports:         2 templates
```

---

## Project Structure / 项目结构

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

## Biomimetic Enhancements / 仿生增强

### 1. Dynamic Topology Networks (DTN) / 动态拓扑网络

Inspired by cytoskeleton mechanosensitive channels. Feature edges are modulated by **tension** computed from input gradient:

基于细胞骨架机械力敏感通道。特征边由输入梯度计算的**张力**调节：

```
tension = ||feature_gradient|| 
gate = sigmoid(gain × tension - threshold)
output = input × gate
```

**File / 文件**: `model/biomimetic_enhance.py`, `dynamic_topology_networks.md`

### 2. Meta-Plasticity Memory / 元学习可塑性记忆

Inspired by DNA methylation. Dual-track memory system:

基于DNA甲基化。双轨记忆系统：

- **KV Cache** (short-term): Session-level context / KV缓存（短期）：会话级上下文
- **Methylation** (long-term): LoRA weight consolidation triggered by emotion intensity / 甲基化（长期）：由情绪强度触发的LoRA权重整合

```
if emotion_score > strong_threshold:
    consolidate LoRA weights with timestamp
```

**File / 文件**: `model/biomimetic_enhance.py`, `meta_plasticity_memory.md`

### 3. Biological MoE (BioMoE) / 生物多专家

Inspired by neuronal membrane potential. Gating depends on **both input AND historical feedback**:

基于神经元膜电位。门控取决于**输入和历史反馈**：

```
gate = f(input) + membrane_bias + emotion_gain × mood
feedback = (prediction == ground_truth)
membrane += feedback × learning_rate
```

**Modes / 模式**:
- `standard`: Original MoE (stateless) / 原始MoE（无状态）
- `bio`: Full biological gating (membrane + emotion) / 完全生物门控
- `hybrid`: Original experts + biological gating (recommended) / 原始专家 + 生物门控（推荐）

**Training Integration / 训练集成**: In `train.py`, feedback is applied automatically:

```python
# Automatic feedback during training / 训练期间自动反馈
preds = outputs['me_logits'].argmax(dim=1)
correct = (preds == labels).float()  # 1=correct, 0=wrong
fb = correct.mean()
moe.apply_feedback(fb)  # Update membrane potential / 更新膜电位
```

**Effect / 效果**: Early training → conservative routing (errors), Late training → confident routing (success) / 早期训练→保守路由（错误），后期训练→自信路由（成功）

**File / 文件**: `model/biomoe.py`, `model/enhanced_moe.py`

### 4. EnhancedMoE Wrapper / 增强MoE封装

Replace original MoE with feedback-enabled version:

用支持反馈的版本替换原始MoE：

```python
from model.enhanced_moe import EnhancedMoE

# Hybrid mode (recommended for training) / 混合模式（推荐用于训练）
moe = EnhancedMoE(mode="hybrid", enable_membrane=True, enable_emotion=True)

# Forward pass / 前向传播
output, gates, aux_loss, info = moe(x)

# External feedback (inference-time) / 外部反馈（推理时）
moe.apply_feedback(1.0)   # User confirmed correct / 用户确认正确
moe.apply_feedback(0.0)  # Prediction was wrong / 预测错误
moe.apply_feedback(-1.0) # User explicitly corrected / 用户明确纠正

# Get state / 获取状态
state = moe.get_state()
# {positive_count, negative_count, accuracy, ...}
```

---

## Citation / 引用

```bibtex
@article{censor2025,
  title={Censor: A Biomimetic Dual-Pathway Micro-Expression Recognition System with Fusiform-Amygdala Circuit and Mixture-of-Experts},
  author={},
  journal={},
  year={2025}
}
```

---

## References / 参考资料

- [CASME II Database](http://casme.psych.ac.cn/casme/c2)
- [SAMM Micro-Expression Database](https://www.mmu.ac.uk)
- [SMIC Database](https://www.oulu.fi)
- [MMEW Dataset](https://github.com/benxianyeteam/MMEW-Dataset)
- [iMER Benchmark](https://github.com/ZhengQinLai/IMER-benchmark) — arXiv:2501.19111
- [Video-Based Facial Micro-Expression Analysis: A Survey](https://ar5iv.labs.arxiv.org/html/2201.12728) — IEEE TPAMI Survey
- [MEGC2024 ACM MM](https://researchportal.hw.ac.uk/en/publications/megc2024-acm-multimedia-2024-facial-micro-expression-grand-challe/)
- [Hybrid Attention-3DNet](https://ictcsreg.psut.edu.jo/paper/export/254) — JJCIT 2025
- [GAM-MER](https://www.sciencedirect.com/science/article/pii/S2405844024119950) — Heliyon 2024
- [MCCA-VNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644463/) — ViT + CBAM 2024