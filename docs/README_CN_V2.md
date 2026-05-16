# Censor: 仿生双通道微表情识别与图像生成系统

> **仿生双通道微表情识别与图像生成系统** — 基于PyTorch实现的仿生双通道架构，用于微表情识别和人脸图像生成，模拟人类视觉通路中的梭状回-杏仁核回路。

---

## 目录

- [概述](#概述)
- [新功能 (v2.0)](#新功能-v20)
- [系统架构](#系统架构)
- [图像生成管线](#图像生成管线)
- [LLM集成](#llm集成)
- [基准数据集](#基准数据集)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [训练](#训练)
- [引用](#引用)

---

## 概述

微表情（ME）是一种短暂的、不自主的面部表情，当人们试图抑制或隐藏真实情绪时会发生。微表情持续时间仅为**40-200毫秒**，其特征是通过面部动作编码系统（FACS）中的动作单元（AU）衡量的细微肌肉活动。

**Censor** 提出了一种仿生双通道架构，模拟人类视觉系统的快速皮层下通路和慢速皮层通路，并配备增强版图像生成管线：

---

## 新功能 (v2.0)

### 🔥 图像生成（新增！）

| 模块 | 描述 | 参数量 |
|--------|------------|-----------|
| **EnhancedBiomimeticImageGenerator** | 统一增强版生成器 | 121.7M |
| **3D Face Prior** | 3DMM人脸估计+法线图 | ~2M |
| **SH Lighting** | 9带球谐光照 | ~0.5M |
| **Text Guidance** | CLIP文本引导 | ~0.3M |
| **ID Preservation** | ArcFace风格ID保持 | ~1M |

### 🤖 LLM增强（更新）

- **主要**: DeepSeek API（云端）
- **备用**: OPT-125M（本地）
- 带临床描述的情绪报告生成

### 🎨 综合前端

- 基于Streamlit的集成平台
- 6个标签页：生成、识别、LLM报告、训练、模型、设置

---

## 系统架构

```
输入视频 (B×3×16×224×224)
  │
  ├── [阶段1] 仿生预处理
  │   ├── SaliencyDetector（高斯金字塔显著性检测）
  │   ├── rPPGExtractor（远程光电容积脉搏波）
  │   └── TVL1OpticalFlow（光流）
  │
  ├── [阶段2] 双通道骨干网络
  │   ├── FastPath: 3D ResNet-18（光流）→ 512维
  │   └── SlowPath: 3D Swin-Transformer（RGB+rPPG）→ 768维
  │
  ├── [阶段3] 梭状回-杏仁核注意力回路
  │   ├── Amygdala：注意力先验图
  │   ├── FFA：SE风格跨通道门控
  │   └── CASANet：apex帧检测
  │
  ├── [阶段4] 时空融合（1024维）
  │
  ├── [阶段5] 动态AU解码器（28个AU）
  │
  ├── [阶段6] 多专家头（3个专家）
  │
  └── [阶段7] 情绪报告器（DeepSeek LLM）
```

| 组件 | 规格 |
|--------|---------------|
| **总参数量** | ~68M（识别），~122M（生成）|
| **快通道** | 3D ResNet-18 → 512维 |
| **慢通道** | 3D Swin-Transformer → 768维 |
| **融合** | 双向交叉注意力 |
| **注意力** | Amygdala + FFA (SE) + CASANet |
| **LLM** | DeepSeek API / OPT-125M |

---

## 图像生成管线

### 增强版生成流程

```
快速特征 (512) + 慢速特征 (768)
  │
  ▼
[1] 双通道融合 (1024)
  │
  ├─→ [2a] 3D人脸先验 → 人脸网格 + 法线图
  ├─→ [2b] SH光照 → 9带系数
  ├─→ [2c] ID编码器 → 身份特征
  └─→ [2d] 文本条件（可选）
  │
  ▼
[2] 基础图像生成器（latent → 224×224 RGB）
  │
  ▼
[3] SH光照渲染器
  │
  ▼
[4] 视觉后处理
  │   ├── PupilController（光照适应）
  │   ├── RetinalContrastNorm（局部对比度）
  │   ├── MachBandEnhancer（边缘锐化）
  │   └── CenterSurroundReceptiveField（边缘检测）
  │
  ▼
输出: 生成的人脸图像 (224×224×3)
```

### 生成模块

| 模块 | 功能 |
|--------|----------|
| **DualPathwayFusion** | SE门控快速+慢速融合 |
| **Face3DPipeline** | 3DMM估计 + 法线图 |
| **SHLightingPipeline** | 球谐光照渲染 |
| **TextGuidancePipeline** | CLIP文本条件 |
| **IDPreservationModule** | 身份一致性 |
| **VisualPerceptionPostProcess** | 仿生后处理 |

---

## LLM集成

### DeepSeek API

```python
from model.llm_report import EmotionReporter, DeepSeekClient

# 使用API密钥初始化
client = DeepSeekClient(api_key="your-key")
report = client.generate(prompt, max_tokens=100)
```

### 环境变量

```bash
export DEEPSEEK_API_KEY="your-api-key"
# 或
export OPENAI_API_KEY="your-api-key"
```

### 备用方案

如果API密钥不可用，自动回退到本地OPT-125M。

---

## 基准数据集

| 数据集 | 样本数 | 被试 | 微表情类别 |
|---------|---------|----------|------------------|
| CASME II | 300+ | 35 | 7类 |
| SAMM | 400+ | 32 | 8类 |
| SMIC-HS | 400+ | 55 | 5类 |

---

## 快速开始

### 安装

```bash
pip install torch torchvision
pip install streamlit
pip install transformers  # 用于LLM
pip install opencv-python
```

### 运行识别

```bash
python main.py --video path/to/video.mp4
```

### 运行图像生成

```bash
python train_image_generator.py --test
```

### 运行前端

```bash
streamlit run frontend/app.py
```

### Python使用方法

```python
import torch
from model.enhanced_image_generator import EnhancedBiomimeticImageGenerator, EnhancedConfig

# 创建生成器
config = EnhancedConfig()
generator = EnhancedBiomimeticImageGenerator(config)

# 生成
fast_feat = torch.randn(1, 512)
slow_feat = torch.randn(1, 768)

with torch.no_grad():
    image = generator(fast_feat, slow_feat)

print(f"生成图像: {image.shape}")  # (1, 3, 224, 224)
```

---

## 项目结构

```
censor/
├── model/
│   ├── biomimetic_image_generator.py  # 基础生成器
│   ├── enhanced_image_generator.py   # 增强版生成器（新增！）
│   ├── face_3d_prior.py              # 3D人脸先验（新增！）
│   ├── sh_lighting.py               # SH光照（新增！）
│   ├── text_guided_generation.py     # 文本引导（新增！）
│   ├── identity_preservation.py    # ID保持（新增！）
│   ├── llm_report.py                # LLM报告器（更新）
│   ├── biomoe.py                   # BioMoE
│   ├── fusion.py                    # 融合模块
│   └── ...
├── visual_perception.py             # 视觉后处理
├── config/
│   └── defaults.py
├── frontend/
│   └── app.py                     # Streamlit前端（新增！）
├── train_image_generator.py         # 训练脚本
├── docs/
│   └── README_CN_V2.md
└── README.md
```

---

## 训练

### 图像生成训练

```bash
python train_image_generator.py
```

### 损失函数

- L2重建损失: `||生成图 - 目标图||²`
- 感知损失: VGG特征对齐
- 光照平滑: 时间一致性
- 稀疏正则: 防过拟合
- 身份保持: ArcFace一致性

---

## 引用

```bibtex
@misc{censor2024,
  title={Censor: 仿生双通道微表情识别系统},
  author={Censor团队},
  year={2024},
  url={https://github.com/...}
}
```

---

**最后更新**: 2024-05-16
**版本**: 2.0