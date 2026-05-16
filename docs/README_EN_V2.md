# Censor: Biomimetic Dual-Pathway Micro-Expression Recognition & Generation System

> **仿生双通道微表情识别与图像生成系统** — A PyTorch implementation of a biomimetic dual-pathway architecture for micro-expression recognition and facial image generation, simulating the fusiform-amygdala circuit in the human visual pathway.

---

## Table of Contents / 目录

- [Overview / 概述](#overview--概述)
- [New Features (v2.0)](#new-features-v20)
- [Architecture / 系统架构](#architecture--系统架构)
- [Image Generation Pipeline / 图像生成管线](#image-generation-pipeline--图像生成管线)
- [LLM Integration / LLM集成](#llm-integration--llm集成)
- [Benchmark Datasets / 基准数据集](#benchmark-datasets--基准数据集)
- [Quick Start / 快速开始](#quick-start--快速开始)
- [Project Structure / 项目结构](#project-structure--项目结构)
- [Training / 训练](#training--训练)
- [Citation / 引用](#citation--引用)

---

## Overview / 概述

Micro-expressions (MEs) are brief, involuntary facial expressions that occur when a person suppresses or conceals their true emotions. They last between **40–200 ms** and are characterized by subtle muscle activations measurable via Action Units (AUs) in the Facial Action Coding System (FACS).

微表情（ME）是一种短暂的、不自主的面部表情，当人们试图抑制或隐藏真实情绪时会发生。微表情持续时间仅为**40-200毫秒**，其特征是通过面部动作编码系统（FACS）中的动作单元（AU）衡量的细微肌肉活动。

**Censor** proposes a biomimetic dual-pathway architecture that mirrors the human visual system's fast subcortical and slow cortical pathways, plus an enhanced image generation pipeline:

---

## New Features (v2.0)

### 🔥 Image Generation (NEW!)

| Module | Description | Parameters |
|--------|------------|-----------|
| **EnhancedBiomimeticImageGenerator** | Unified generation with all enhancements | 121.7M |
| **3D Face Prior** | 3DMM-based face estimation + normal map | ~2M |
| **SH Lighting** | 9-band Spherical Harmonics lighting | ~0.5M |
| **Text Guidance** | CLIP-based text conditioning | ~0.3M |
| **ID Preservation** | ArcFace-style identity preservation | ~1M |

### 🤖 LLM Enhancement (UPDATED)

- **Primary**: DeepSeek API (cloud)
- **Fallback**: OPT-125M (local)
- Emotion report generation with clinical descriptions

### 🎨 Comprehensive Frontend

- Streamlit-based integrated platform
- 6 tabs: Generation, Recognition, LLM Report, Training, Models, Settings

---

## Architecture / 系统架构

```
Input Video (B×3×16×224×224)
  │
  ├── [Stage 1] Biomimetic Preprocessing / 仿生预处理
  │   ├── SaliencyDetector (Foveal sampling via Gaussian pyramid)
  │   ├── rPPGExtractor (Remote photoplethysmography)
  │   └── TVL1OpticalFlow (Optical flow)
  │
  ├── [Stage 2] Dual-Pathway Backbones / 双通道骨干网络
  │   ├── FastPath: 3D ResNet-18 (optical flow) → 512-D
  │   └── SlowPath: 3D Swin-Transformer (RGB+rPPG) → 768-D
  │
  ├── [Stage 3] Fusiform-Amygdala Attention Circuit
  │   ├── Amygdala: attention prior map
  │   ├── FFA: SE-style cross-pathway gating
  │   └── CASANet: apex frame detection
  │
  ├── [Stage 4] Spatio-Temporal Fusion (1024-D)
  │
  ├── [Stage 5] Dynamic AU Decoder (28 AUs)
  │
  ├── [Stage 6] Mixture-of-Experts Head (3 experts)
  │
  └── [Stage 7] Emotion Reporter (DeepSeek LLM)
```

| Component | Specification |
|-----------|---------------|
| **Total Parameters** | ~68M (recognition), ~122M (generation) |
| **Fast Pathway** | 3D ResNet-18 → 512-D |
| **Slow Pathway** | 3D Swin-Transformer → 768-D |
| **Fusion** | Bidirectional cross-attention |
| **Attention** | Amygdala + FFA (SE) + CASANet |
| **LLM** | DeepSeek API / OPT-125M |

---

## Image Generation Pipeline / 图像生成管线

### Enhanced Generation Flow

```
Fast Features (512) + Slow Features (768)
  │
  ▼
[1] Dual-Pathway Fusion (1024)
  │
  ├─→ [2a] 3D Face Prior → Face Mesh + Normal Map
  ├─→ [2b] SH Lighting → 9-band coefficients
  ├─→ [2c] ID Encoder → Identity features
  └─→ [2d] Text Conditioning (optional)
  │
  ▼
[3] Base Image Generator (latent → 224×224 RGB)
  │
  ▼
[4] SH Lighting Renderer
  │
  ▼
[5] Visual Perception Post-Process
  │   ├── PupilController (illumination adaptation)
  │   ├── RetinalContrastNorm (local contrast)
  │   ├── MachBandEnhancer (edge sharpening)
  │   └── CenterSurroundReceptiveField (edge detection)
  │
  ▼
Output: Generated Face Image (224×224×3)
```

### Generation Modules

| Module | Function |
|--------|----------|
| **DualPathwayFusion** | SE-gated fast+slow fusion |
| **Face3DPipeline** | 3DMM estimation + normal map |
| **SHLightingPipeline** | Spherical Harmonics rendering |
| **TextGuidancePipeline** | CLIP text conditioning |
| **IDPreservationModule** | Identity consistency |
| **VisualPerceptionPostProcess** | Biomimetic post-processing |

---

## LLM Integration / LLM集成

### DeepSeek API

```python
from model.llm_report import EmotionReporter, DeepSeekClient

# Initialize with API key
client = DeepSeekClient(api_key="your-key")
report = client.generate(prompt, max_tokens=100)
```

### Environment Variables

```bash
export DEEPSEEK_API_KEY="your-api-key"
# or
export OPENAI_API_KEY="your-api-key"
```

### Fallback

If API key is not available, automatically falls back to local OPT-125M.

---

## Benchmark Datasets / 基准数据集

| Dataset | Samples | Subjects | Micro-Expressions |
|---------|---------|----------|------------------|
| CASME II | 300+ | 35 | 7 categories |
| SAMM | 400+ | 32 | 8 categories |
| SMIC-HS | 400+ | 55 | 5 categories |

---

## Quick Start / 快速开始

### Installation

```bash
pip install torch torchvision
pip install streamlit
pip install transformers  # For LLM
pip install opencv-python
```

### Run Recognition

```bash
python main.py --video path/to/video.mp4
```

### Run Image Generation

```bash
python train_image_generator.py --test
```

### Run Frontend

```bash
streamlit run frontend/app.py
```

### Python Usage

```python
import torch
from model.enhanced_image_generator import EnhancedBiomimeticImageGenerator, EnhancedConfig

# Create generator
config = EnhancedConfig()
generator = EnhancedBiomimeticImageGenerator(config)

# Generate
fast_feat = torch.randn(1, 512)
slow_feat = torch.randn(1, 768)

with torch.no_grad():
    image = generator(fast_feat, slow_feat)

print(f"Generated: {image.shape}")  # (1, 3, 224, 224)
```

---

## Project Structure / 项目结构

```
censor/
├── model/
│   ├── biomimetic_image_generator.py  # Basic generator
│   ├── enhanced_image_generator.py   # Enhanced generator (NEW!)
│   ├── face_3d_prior.py              # 3D face prior (NEW!)
│   ├── sh_lighting.py               # SH lighting (NEW!)
│   ├── text_guided_generation.py     # Text guidance (NEW!)
│   ├── identity_preservation.py    # ID preservation (NEW!)
│   ├── llm_report.py                # LLM reporter (UPDATED)
│   ├── biomoe.py                   # BioMoE
│   ├── fusion.py                    # Fusion modules
│   └── ...
├── visual_perception.py             # Visual post-processing
├── config/
│   └── defaults.py
├── frontend/
│   └── app.py                     # Streamlit frontend (NEW!)
├── train_image_generator.py         # Training script
├── docs/
│   └── README_EN.md
└── README.md
```

---

## Training / 训练

### Image Generation Training

```bash
python train_image_generator.py
```

### Loss Functions

- L2 Reconstruction: `||generated - target||²`
- Perceptual Loss: VGG feature alignment
- Illumination Smoothness: Temporal consistency
- Sparse Regularization: Anti-overfitting
- Identity Preservation: ArcFace consistency

---

## Citation / 引用

```bibtex
@misc{censor2024,
  title={Censor: Biomimetic Dual-Pathway Micro-Expression Recognition},
  author={Censor Team},
  year={2024},
  url={https://github.com/...}
}
```

---

**Last Updated**: 2024-05-16
**Version**: 2.0