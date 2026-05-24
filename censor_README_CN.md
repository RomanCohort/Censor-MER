# Censor: 仿生双通道微表情识别系统

> 基于PyTorch实现的仿生双通道微表情识别（MER）架构，模拟人类视觉通路中的梭状回-杏仁核神经回路。

---

## 目录

- [概述](#概述)
- [系统架构](#系统架构)
  - [阶段1: 仿生预处理](#阶段1-仿生预处理)
  - [阶段2: 双通道骨干网络](#阶段2-双通道骨干网络)
  - [阶段3: 梭状回-杏仁核注意力回路](#阶段3-梭状回-杏仁核注意力回路)
  - [阶段4: 时空融合](#阶段4-时空融合)
  - [阶段5: 动态AU解码器](#阶段5-动态au解码器)
  - [阶段6: 多专家头](#阶段6-多专家头)
  - [阶段7: 情绪报告器](#阶段7-情绪报告器)
- [数学公式](#数学公式)
- [基准数据集](#基准数据集)
- [性能对比](#性能对比)
- [安装指南](#安装指南)
  - [系统要求](#系统要求)
  - [环境配置](#环境配置)
  - [依赖详情](#依赖详情)
- [快速开始](#快速开始)
- [训练指南](#训练指南)
  - [基础训练](#基础训练)
  - [高级训练选项](#高级训练选项)
  - [断点续训](#断点续训)
- [使用指南](#使用指南)
  - [Python API](#python-api)
  - [配置说明](#配置说明)
  - [输出解读](#输出解读)
- [项目结构](#项目结构)
- [引用](#引用)

---

## 概述

微表情（ME）是一种短暂的、不自主的面部表情，当人们试图抑制或隐藏真实情绪时会发生。微表情持续时间仅为**40-200毫秒**，其特征是通过面部动作编码系统（FACS）中的动作单元（AU）衡量的细微肌肉活动。

**Censor** 提出了一种仿生双通道架构，模拟人类视觉系统的快速皮层下通路和慢速皮层通路：

```
输入视频 (B×3×16×224×224)
  │
  ├── [阶段1] 仿生预处理
  │   ├── 显著性检测器（高斯金字塔实现中心凹采样）
  │   ├── rPPG提取器（远程光电容积脉搏波）
  │   └── TVL1光流（OpenCV DualTVL1）
  │
  ├── [阶段2] 双通道骨干网络
  │   ├── 快通道: 3D ResNet-18（光流）→ 512维
  │   └── 慢通道: 3D Swin-Transformer（RGB+rPPG）→ 768维 + 空间图
  │
  ├── [阶段3] 梭状回-杏仁核注意力
  │   ├── 杏仁核: 快通道生成的注意力先验图
  │   ├── FFA: SE风格跨通道门控
  │   └── CASANet: 三角注意力实现apex帧检测
  │
  ├── [阶段4] TSFmicroFusion（双向交叉注意力，1024维）
  │
  ├── [阶段5] 动态AU解码器（BiLSTM，28个AU + OPD路标）
  │
  ├── [阶段6] MoE头（3专家 + 个性化Radar TTA）
  │
  └── [阶段7] 情绪报告器（模板 + LLM生成报告）
```

| 指标 | 数值 |
|------|-------|
| **总参数量** | 68,353,230 |
| **架构** | 双通道: 3D ResNet-18 + 3D Swin-Transformer |
| **预处理** | 高斯显著性 + rPPG + OpenCV TV-L1 |
| **注意力** | 杏仁核(FC) + FFA(SE) + CASANet(三角MHA) |
| **融合** | 双向交叉注意力，1024维空间 |
| **AU解码** | BiLSTM（2层，512隐藏）→ 28个sigmoid输出 |
| **MoE** | 3专家，top-2门控，负载均衡辅助损失 |
| **TTA** | 个性化Radar（5步SGD identity适配器） |

---

## 系统架构

### 阶段1: 仿生预处理

#### 显著性检测器 — 中心凹采样

模拟人类视网膜中心凹（1-2度视觉角处锥细胞密度最高），使用**高斯金字塔**和中心偏向的空间先验：

$$S(x,y) = \sum_{l=0}^{L-1} w_l \cdot G_\sigma(x,y) \cdot I_l(x,y)$$

其中 $I_l$ 是第 $l$ 层金字塔，$G_\sigma$ 是中心偏向的高斯先验，$w_l = 2^{-l}$ 是层级权重。输出：`(B, 1, T, H, W)`。

#### rPPG提取器 — 远程光电容积脉搏波

通过**色度分解**和**时间带通滤波**捕捉血氧饱和度波动（0.5-4.0Hz心脏范围）：

$$\text{rPPG}(t) = \sum_{c \in \{R,G,B\}} \alpha_c \cdot I_c(t)$$

$$\text{rPPG}_{\text{filtered}}(t) = \sum_{\tau=-K}^{K} h(\tau) \cdot \text{rPPG}(t-\tau)$$

其中 $\alpha_c$ 是学习的色度投影权重，$h$ 是学习的FIR带通滤波器。输出：`(B, 3, T, H, W)`。

#### TVL1光流 — OpenCV DualTVL1

通过OpenCV的`createOptFlow_DualTVL1`计算真实TV-L1光流。TV-L1能量泛函：

$$\min_u \int\left(|\nabla u| + \lambda \cdot |I_1(x+u) - I_0(x)|\right) dx$$

通过原始-对偶算法求解。输出：`(B, 2, T, H, W)`。

### 阶段2: 双通道骨干网络

#### 快皮层下通路 — 3D ResNet-18

通过浅层3D ResNet-18变体（3阶段，64→128→256通道）处理**光流**输入。大时间步长（2²,2²）模拟快速皮层下处理。输出：`(B, 512)`。

#### 慢皮层通路 — 3D Swin-Transformer

通过4阶段完整3D Swin-Transformer处理**RGB + rPPG**（6通道），使用**移位窗口多头自注意力（W-MSA）**：

| 阶段 | 块数 | 维度 | 合并步长 | 分辨率 |
|-------|------|------|----------|--------|
| 1 | 2 | 96 | (2,2,2) | T/2, H/2, W/2 |
| 2 | 2 | 192 | (2,2,2) | T/4, H/4, W/4 |
| 3 | 6 | 384 | (2,2,2) | T/8, H/8, W/8 |
| 4 | 2 | 768 | (1,1,1) | T/16, H/32, W/32 |

相对位置偏置（3D网格）实现准确的空间关系建模。输出：**池化`(B, 768)`** + **空间图`(B, 768, T/16, H/32, W/32)`**。

### 阶段3: 梭状回-杏仁核注意力回路

#### 杏仁核 — 注意力先验图

全连接层从快通道特征生成空间注意力先验图：

$$\text{APM} = \sigma\left(\text{FC}_{512\rightarrow256\rightarrow196}(\text{fast\_feat})\right).view(B,1,14,14)$$

sigmoid激活的APM指导空间注意力朝向面部感兴趣区域。

#### FFA — 特征融合注意力

SE风格的压缩-激励门控，用于跨通道特征重校准：

$$z = \sigma\left(\text{FC}_{1280\rightarrow80}(\text{concat}[f_{\text{fast}}, f_{\text{slow}}])\right)$$

$$f_{\text{fast}}^* = z_{[:512]} \odot f_{\text{fast}}, \quad f_{\text{slow}}^* = z_{[512:]} \odot f_{\text{slow}}$$

#### CASANet — Apex帧检测

逆三角形可学习空间掩码（7×7）+ 时间多头注意力，用于**apex帧检测**——识别微表情序列中峰值强度帧：

$$\text{apex\_score}_t = \text{softmax}\left(\text{MHA}(Q_t, K, V)\right) \in \mathbb{R}^T$$

三角先验 $M_{i,j} = \exp\left(-\frac{(j-i)^2}{2\sigma_i^2}\right)$ 模拟微表情自然的onset→apex→decay模式。输入：慢通道阶段4的空间图`(B, 768, 1, 7, 7)`。输出：注意力特征 + apex分数`(B, 1)`。

### 阶段4: 时空融合

**TSFmicroFusion** — 1024维双向交叉注意力：

$$\text{F}_{f2s} = \text{Attention}\left(Q_f \cdot W_Q, K_s \cdot W_K, V_s \cdot W_V\right) \cdot W_O$$

$$\text{F}_{s2f} = \text{Attention}\left(Q_s \cdot W_Q, K_f \cdot W_K, V_f \cdot W_V\right) \cdot W_O$$

$$f_{\text{fused}} = \alpha \cdot \text{FFN}(\text{F}_{f2s}) + (1-\alpha) \cdot \text{FFN}(\text{F}_{s2f}), \quad \alpha = \sigma(W_\alpha[f_{\text{fast}}; f_{\text{slow}}])$$

输出：`(B, 1024)`。

### 阶段5: 动态AU解码器

**DynamicAUDecoder** — BiLSTM进行时间动作单元序列建模：

$$\mathbf{h}_t = \text{BiLSTM}(f_{\text{fused}}, \mathbf{h}_{t-1}), \quad \mathbf{h}_T = [\mathbf{h}_t^{f}; \mathbf{h}_t^{b}]$$

$$\text{AU}_{b,t} = \sigma\left(\text{Linear}(\mathbf{h}_t)\right) \in \mathbb{R}^{28} \quad \text{(sigmoid多标签)}$$

$$\text{OPD}_{b,u} = \left[t_{\text{onset}}, t_{\text{peak}}, t_{\text{decay}}\right] \in \mathbb{R}^3 \quad \text{(onset-peak-decay路标)}$$

输出：AU强度`(B, T, 28)` + OPD路标`(B, 28, 3)`。

### 阶段6: 多专家头

**MoEGatingNetwork** — 带3个专家MLP的噪声top-k门控：

$$g = \text{softmax}\left(\text{top-}k\left(W_g \cdot f_{\text{fused}}\right)\right)$$

$$\text{ME\_logits} = \sum_{i=1}^{3} g_i \cdot \text{Expert}_i(f_{\text{fused}})$$

**辅助负载均衡损失**防止专家坍塌：

$$\mathcal{L}_{\text{moe}} = \lambda \sum_{i=1}^{3} \left(\bar{f}_i - \frac{1}{3}\right)^2, \quad \bar{f}_i = \frac{1}{B}\sum_b g_i^{(b)}$$

**个性化Radar** — 通过5步内循环SGD在支持帧上进行测试时适配，配备identity初始化的残差适配器。

### 阶段7: 情绪报告器

基于模板的临床报告生成，包含AU解析、微表情分类、rPPG生理线索和OPD时间路标。可选HuggingFace OPT-125M用于自由文本生成（离线时优雅降级）。

---

## 数学公式

### 总损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{me}} + \alpha \mathcal{L}_{\text{au}} + \beta \mathcal{L}_{\text{moe}} + \gamma \mathcal{L}_{\text{opd}}$$

| 损失 | 类型 | 描述 |
|------|------|------|
| $\mathcal{L}_{\text{me}}$ | 交叉熵 | 7类微表情分类 |
| $\mathcal{L}_{\text{au}}$ | 二值交叉熵 | 28类AU多标签识别 |
| $\mathcal{L}_{\text{moe}}$ | 负载均衡辅助 | 防止专家坍塌 |
| $\mathcal{L}_{\text{opd}}$ | L2平滑 + 峰值一致性 | Onset-peak-decay时间模式 |

### 架构维度

```
输入:             (B, 3, T=16, H=224, W=224)
     │
显著性图:        (B, 1, 16, 224, 224)
rPPG热图:        (B, 3, 16, 224, 224)
光流栈:          (B, 2, 16, 224, 224)
     │
快通道特征:       (B, 512)
慢通道特征:       (B, 768) + 慢空间图 (B, 768, 1, 7, 7)
     │
快门控:          (B, 512)
慢门控:          (B, 768)
     │
融合特征:        (B, 1024)
     │
AU强度:          (B, 16, 28)  ← sigmoid多标签
AU OPD:          (B, 28, 3)    ← onset/peak/decay
ME logits:       (B, 7)       ← 7类CE
专家门控:        (B, 3)       ← top-2 softmax
```

---

## 基准数据集

| 数据集 | 样本数 | 被试 | 帧率 | 分辨率 | 类别 | 来源 |
|--------|--------|------|------|--------|------|------|
| **CASME II** | 247 | 26 | 200 fps | 640×480 | 5-7 | [CAS官网](http://casme.psych.ac.cn/casme/c2) |
| **SAMM** | 159 | 32 | 200 fps | 2040×1088 | 7-8 | [MMU](https://www.mmu.ac.uk) |
| **SMIC-HS** | 164 | 16 | 100 fps | 640×480 | 3 | [奥卢大学](https://www.oulu.fi) |
| **MMEW** | 300 (+900宏表情) | 36 | 90 fps | 1920×1080 | 7 | [IEEE TPAMI 2022](https://github.com/benxianyeteam/MMEW-Dataset) |
| **CAS(ME)³** | ~300+ | — | 30 fps | 多种 | 4+ | [CAS官网](http://melab.psych.ac.cn) |
| **iMER基准** | 5数据集 | — | — | — | 增量 | [arXiv:2501.19111](https://arxiv.org/abs/2501.19111) |

> **注意：** 大多数数据集需要签署许可协议才能访问。CASME II、SAMM和SMIC可通过各自机构网站访问。MMEW和iMER基准代码在GitHub上公开可用。

### MEGC（微表情挑战赛）结果

| 年份 | 会议 | 冠军方法 | 方法 |
|------|------|----------|------|
| 2022 | ACM MM | USTC-IAT-United | 光流 + TPS插值 |
| 2023 | ACM MM | CAS-IA + BUST | **VideoMAE + 光流**（34队） |
| 2024 | ACM MM | USTC + HIT | 深度学习 + 跨文化泛化 |
| 2025 | ACM MM | 待定 | 增量学习 + 多模态 |

---

## 性能对比

### 标准基准准确率对比

| 方法 | 骨干网络 | CASME II | SAMM | SMIC | CAS(ME)² |
|------|----------|---------|------|------|----------|
| **混合注意力3DNet**（JJCIT 2025） | 3D CNN + SE | 93.79% | 93.61% | 93.42% | **93.95%** |
| **ROI-ArcFace**��IEEE 2025） | CNN + ROI | **93.96%** | 86.15% | 81.17% | — |
| **STRNet**（Int. J. SCC 2025） | 区域 | — | — | — | UF1=0.9792 |
| **GAM-MER**（Heliyon 2024） | 图注意力+Transf | 91.57% | 91.25% | 86.22% | — |
| **MCCA-VNet**（PMC 2024） | ViT + XCiT + CBAM | — | — | — | UF1=0.868 |
| **SelfME**（IEEE 2024） | Transformer | 90.78% | — | 69.70% | — |
| **μ-BERT**（ACM MM 2024） | BERT风格 | 90.34% | — | 85.80% | — |
| **双分支交叉注意力**（2024） | Swin + MobileViT | — | — | — | 81.6% |
| **多尺度3D ResNet**（J. Image 2024） | 3D-ResNet50 | 91.35% | 84.77% | 74.6% | — |
| **LAENet**（OA 2024） | 轻量级3D CNN | 79.19% | — | — | — |
| **OFF-ApexNet**（基线） | CNN | 87.64% | 54.09% | 68.17% | — |
| **LBP-TOP**（基线） | 手工特征 | 70.26% | 39.54% | 20.00% | — |

### 最新SOTA方法（2024-2025）

| 方法 | 会议 | 关键创新 | 代码 |
|------|------|----------|------|
| **VideoMAE + OF** | ACM MM 2023 | 自监督预训练 | [GitHub](https://github.com/VisionVR/VideoMAE) |
| **Prompt-MER** | IEEE 2024 | 视觉提示调优 | [GitHub](https://github.com/VisionVR/Prompt-MER) |
| **iMER基准** | arXiv 2025 | 增量学习框架 | [GitHub](https://github.com/ZhengQinLai/IMER-benchmark) |
| **DualPrompt** | CVPR 2024 | 双提示学习 | 包含在iMER中 |

### Censor设计原理对比SOTA

| SOTA特性 | Censor实现 | 优势 |
|---------|------------|------|
| 光流 | OpenCV DualTVL1（真实TV-L1） | 最准确的经典光流 |
| 双通道 | 3D ResNet-18 + 3D Swin-T | 生物学驱动 |
| 交叉注意力融合 | 双向TSFmicroFusion（1024维） | 完整特征交互 |
| AU多标签 | BiLSTM → 28个sigmoid输出 | 时间动态 |
| MoE路由 | 噪声top-2 + 负载均衡 | 专家专业化 |
| Apex检测 | CASANet三角注意力 | 模拟微表情模式 |
| rPPG生理 | 色度+带通滤波 | 生理相关 |
| 测试时适配 | 个性化Radar（SGD identity） | 被试个性化 |

---

## 安装指南

### 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| **GPU** | NVIDIA GTX 1080（8GB） | NVIDIA RTX 3090/4090（24GB） |
| **内存** | 16 GB | 32 GB |
| **存储** | 50 GB SSD | 100 GB SSD |
| **系统** | Ubuntu 20.04 / Windows 10+ | Ubuntu 22.04 / Windows 11 |

> **注意：** 推荐使用CUDA计算能力≥7.0的GPU以获得最佳性能。

### 环境配置

#### 方式1：Conda（推荐）

```bash
# 创建并激活conda环境
conda create -n censor python=3.10 -y
conda activate censor

# 安装CUDA 11.8的PyTorch
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 安装Censor依赖
cd D:\censor
pip install -r requirements.txt
```

#### 方式2：虚拟环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

#### 方式3：Docker（可选）

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### 依赖详情

| 包名 | 版本 | 用途 |
|------|------|------|
| `torch` | ≥2.0.0 | 核心深度学习框架 |
| `torchvision` | ≥0.15.0 | 视觉工具 |
| `numpy` | ≥1.24.0 | 数值运算 |
| `scipy` | ≥1.10.0 | 信号处理 |
| `einops` | ≥0.6.0 | 张量运算 |
| `opencv-python` | ≥4.7.0 | 光流提取 |
| `opencv-contrib-python` | ≥4.7.0 | 扩展OpenCV功能 |
| `Pillow` | ≥9.0.0 | 图像处理 |
| `transformers` | ≥4.30.0 | LLM生成报告 |
| `accelerate` | ≥0.20.0 | 训练加速 |

#### 可选依赖

```bash
# GPU性能分析
pip install nvidia-ml-py3

# 远程监控
pip install tensorboard

# 分布式训练
pip install torchrun  # 包含在PyTorch中
```

---

## 快速开始

### 1. 前向传播测试（合成数据）

```bash
cd D:\censor
python main.py
```

预期输出：
```
============================================================
 Censor -- 仿生双通道微表情识别系统
============================================================
[Censor] 初始化预处理...
[Censor] 初始化双通道骨干网络...
...

输入视频: torch.Size([2, 3, 16, 224, 224])

...

============================================================
 最终输出摘要
============================================================
  ME Logits:       torch.Size([2, 7])
  AU强度:          torch.Size([2, 16, 28])
  AU OPD:          torch.Size([2, 28, 3])
  Apex分数:        torch.Size([2, 1])
  专家门控:        torch.Size([2, 3])
  MoE辅助损失:     0.001234
  适配特征:       torch.Size([2, 1024])
  报告:           2个模板
============================================================
```

### 2. 使用合成数据训练

```bash
python train.py --epochs 5 --batch_size 2 --synthetic_data
```

### 3. 使用真实数据集训练

```bash
# 下载CASME II数据集并放入 ./data/CASME_II
# 按照 prepare_data.py 中的数据准备说明操作

python train.py --epochs 50 --batch_size 4 --data_root ./data/CASME_II
```

---

## 训练指南

### 基础训练

```bash
# 默认设置完整训练
python train.py \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --data_root ./data/CASME_II \
    --output_dir ./checkpoints
```

### 高级训练选项

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 2 | 每GPU批大小 |
| `--lr` | 1e-4 | 初始学习率 |
| `--weight_decay` | 1e-4 | 优化器权重衰减 |
| `--au_loss_weight` | 0.5 | AU损失权重α |
| `--moe_loss_weight` | 0.01 | MoE负载均衡权重β |
| `--landmark_weight` | 0.1 | OPD路标损失权重γ |
| `--output_dir` | ./checkpoints | 检查点输出目录 |
| `--val_every` | 1 | 验证频率（轮） |
| `--save_every` | 5 | 保存频率 |
| `--use_amp` | True | 自动混合精度 |
| `--resume` | None | 从检查点恢复 |

### 完整训练示例

```bash
# 多GPU训练
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

### 断点续训

```bash
# 从检查点恢复训练
python train.py \
    --resume ./checkpoints/censor_epoch50.pt \
    --epochs 100
```

---

## 使用指南

### Python API

#### 基础用法

```python
import torch
from model import Censor

# 初始化模型
model = Censor()
model.eval()  # 或 model.train() 用于训练

# 准备输入
video = torch.randn(1, 3, 16, 224, 224)  # (B, C, T, H, W)

# 前向传播
with torch.no_grad():
    outputs = model(video)

# 访问结果
me_logits = outputs['me_logits']           # (B, 7) 微表情分数
au_intensities = outputs['au_intensities']  # (B, T, 28) AU强度
apex_scores = outputs['apex_scores']      # (B, 1) apex帧置信度
reports = outputs['template_report']   # List[str] 临床报告
```

#### 使用DataLoader

```python
from torch.utils.data import DataLoader
from dataset import MicroExpressionDataset

# 创建数据集
dataset = MicroExpressionDataset(
    root='./data/CASME_II',
    split='test',
    T=16,
    H=224,
    W=224
)

loader = DataLoader(dataset, batch_size=4, shuffle=False)

# 处理批次
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

#### 测试时适配

```python
from model import Censor, PersonalizedRadar

# 加载基础模型
model = Censor()
checkpoint = torch.load('./checkpoints/best.pt')
model.load_state_dict(checkpoint['model_state'])

# 加载支持集用于个性化
support_videos = torch.randn(8, 3, 16, 224, 224)
support_labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

# 应用个性化Radar适配
radar = PersonalizedRadar(RADAR_CONFIG)
adapted_model = radar.adapt(model, support_videos, support_labels)

# 在查询样本上评估
query_video = torch.randn(1, 3, 16, 224, 224)
outputs = adapted_model(query_video)
```

### 配置说明

#### 自定义超参数

编辑`config/defaults.py`或通过命令行传递：

```python
from config.defaults import FAST_PATHWAY_CONFIG, SLOW_PATHWAY_CONFIG

# 修改配置
FAST_PATHWAY_CONFIG['output_dim'] = 256  # 减小快通道维度
SLOW_PATHWAY_CONFIG['stages'][3]['dim'] = 512  # 减小慢通道

# 使用自定义配置初始化
from model import Censor
model = Censor()  # 使用修改后的配置
```

#### 创建自定义配置文件

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

### 输出解读

#### 微表情分类

```python
# 获取预测类别
me_probs = torch.softmax(outputs['me_logits'], dim=-1)
predicted_class = me_probs.argmax(dim=-1)

# 类别标签
CLASS_LABELS = ['高兴', '悲伤', '惊讶', '恐惧', '愤怒', '厌恶', '藐视']

for i, pred in enumerate(predicted_class):
    print(f"样本{i}: {CLASS_LABELS[pred]} (置信度: {me_probs[i, pred]:.2%})")
```

#### 动作单元分析

```python
# 获取活跃AU（阈值>0.3）
au_thresh = 0.3
active_aus = (outputs['au_intensities'] > au_thresh).nonzero(as_tuple=True)

AU_NAMES = {
    1: "内眉提肌", 4: "降眉肌", 5: "上睑提肌",
    6: "颧肌", 7: "眼轮匝肌", 9: "鼻肌",
    10: "上唇提肌", 12: "口角提肌", 14: "颏肌",
    15: "口角降肌", 17: "颏提肌", 20: "唇拉伸肌",
    23: "嘴唇压肌", 25: "唇分开", 26: "下颌下垂"
}

# 打印每帧活跃AU
for b, t, au in zip(*active_aus):
    print(f"帧{t}: AU{au} ({AU_NAMES.get(au, '未知')})")
```

#### Apex帧检测

```python
# 获取apex帧
apex_frame = outputs['apex_scores'].argmax(dim=1)
print(f"检测到的apex帧: {apex_frame.item()}")
```

#### 临床报告生成

```python
# 获取结构化报告
for report in outputs['template_report']:
    print(report)
    print("-" * 40)
```

---

## 项目结构

```
censor/
├── main.py                 # Censor编排器 + 前向传播测试
├── train.py                # 训练流程（多任务损失、AMP、检查点）
├── dataset.py             # 数据集定义
├── prepare_data.py        # 数据准备工具
├── requirements.txt     # 依赖
├── config/
│   └── defaults.py      # 中央超参数字典
├── model/
│   ├── __init__.py    # 重新导出所有类
│   ├── preprocessing.py  # 显著性检测器、rPPG提取器、TVL1光流
│   ├── backbones.py   # 快皮层下通路、慢皮层通路
│   ├── attention.py  # 杏仁核、FFA、CASANet
│   ├── fusion.py    # TSFmicroFusion
│   ├── decoders.py  # 动态AU解码器
│   ├── moe_head.py # MoE门控网络、个性化Radar
│   └── llm_report.py # 情绪报告器
└── data/
    └── iMER/          # iMER基���代���
        ├── backbone/  # 各种骨干网络实现
        ├── models/    # 持续学习模型
        └── utils/     # 数据工具
```

---

## 引用

```bibtex
@article{censor2025,
  title={Censor: 仿生双通道微表情识别系统与梭状回-杏仁核回路及多专家},
  author={},
  journal={},
  year={2025}
}
```

---

## 参考资料

- [CASME II数据库](http://casme.psych.ac.cn/casme/c2)
- [SAMM微表情数据库](https://www.mmu.ac.uk)
- [SMIC数据库](https://www.oulu.fi)
- [MMEW数据集](https://github.com/benxianyeteam/MMEW-Dataset)
- [iMER基准](https://github.com/ZhengQinLai/IMER-benchmark) — arXiv:2501.19111
- [MEGC2024 ACM MM](https://researchportal.hw.ac.uk/en/publications/megc2024-acm-multimedia-2024-facial-micro-expression-grand-challe/)
- [基于视频的微表情分析：综述](https://ar5iv.labs.arxiv.org/html/2201.12728) — IEEE TPAMI综述
- [混合注意力3DNet](https://ictcsreg.psut.edu.jo/paper/export/254) — JJCIT 2025
- [GAM-MER](https://www.sciencedirect.com/science/article/pii/S2405844024119950) — Heliyon 2024
- [MCCA-VNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644463/) — ViT + CBAM 2024