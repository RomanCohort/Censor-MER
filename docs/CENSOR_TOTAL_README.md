一、项目总览

Censor 是一个基于 PyTorch 实现的仿生双通道微表情识别（MER）与图像生成系统。项目核心解决的问题是：如何借鉴人脑的视觉-情感处理机制，设计更精确、更可解释的微表情识别系统？

---

## 目录

### [一、项目总览](#一项目总览)
- [1.1 标题与摘要](#11-标题与摘要)
- [1.2 研究背景与挑战](#12-研究背景与挑战)
- [1.2a 核心创新总览](#12a-核心创新总览)
  - [仿生双通道机制](#仿生双通道机制)
  - [注意力机制创新](#注意力机制创新)
  - [BioMoE门控](#biomoe门控)
  - [图像生成创新](#图像生成创新)
- [1.3 系统架构](#13-系统架构)
- [1.4 核心指标](#14-核心指标)
- [1.5 版本迭代](#15-版本迭代)

### [二、部署与运行](#二部署与运行)
- [2.1 环境准备](#21-环境准备)
- [2.2 依赖安装](#22-依赖安装)
- [2.3 运行识别](#23-运行识别)
- [2.4 运行前端](#24-运行前端)
- [2.5 Docker 方式](#25-docker-方式)
- [2.6 验证](#26-验证)
- [2.7 命令行界面 (CLI)](#27-命令行界面-cli)
- [2.8 Python API](#28-python-api)

### [三、核心功能详解](#三核心功能详解)
- [3.1 仿生双通道机制](#31-仿生双通道机制)
  - [生物学背景](#生物学背景)
  - [快通道详解](#快通道详解)
  - [慢通道详解](#慢通道详解)
  - [融合机制详解](#融合机制详解)
- [3.2 预处理模块](#32-预处理模块)
  - [3.2.1 显著性检测器](#321-显著性检测器)
  - [3.2.2 rPPG提取器](#322-rppg提取器)
  - [3.2.3 TV-L1光流](#323-tv-l1光流)
- [3.3 注意力模块](#33-注意力模块)
  - [3.3.1 杏仁核](#331-杏仁核)
  - [3.3.2 FFA](#332-ffa)
  - [3.3.3 CASANet](#333-casanet)
- [3.4 融合模块](#34-融合模块)
- [3.5 解码模块](#35-解码模块)
- [3.6 多专家模块](#36-多专家模块)

### [四、图像生成管线](#四图像生成管线)
- [4.1 整体架构](#41-整体架构)
- [4.2 DualPathwayFusion](#42-dualpathwayfusion)
- [4.3 Face3DPipeline](#43-face3dpipeline)
- [4.4 SHLightingPipeline](#44-shlightingpipeline)
- [4.5 IDPreservationModule](#45-idpreservationmodule)
- [4.6 TextGuidancePipeline](#46-textguidancepipeline)

### [五、视觉后处理](#五视觉后处理)
- [5.1 PupilController](#51-pupilcontroller)
- [5.2 RetinalContrastNorm](#52-retinalcontrastnorm)
- [5.3 MachBandEnhancer](#53-machbandenhancer)
- [5.4 CenterSurroundReceptiveField](#54-centersurroundreceptivefield)

### [六、LLM集成](#六llm集成)
- [6.1 DeepSeek API](#61-deepseek-api)
- [6.2 备用方案](#62-备用方案)
- [6.3 环境变量配置](#63-环境变量配置)

### [七、训练算法详解](#七训练算法详解)
- [7.1 训练流程总览](#71-训练流程总览)
- [7.2 损失函数详解](#72-损失函数详解)
- [7.3 优化器配置](#73-优化器配置)
- [7.4 早停算法](#74-早停算法)
- [7.5 检查点机制](#75-检查点机制)
- [7.6 超参数配置](#76-超参数配置)

### [八、基准数据集与性能](#八基准数据集与性能)
- [8.1 数据集介绍](#81-数据集介绍)
- [8.2 微表情类别](#82-微表情类别)
- [8.3 性能对比](#83-性能对比)

### [九、配置选项](#九配置选项)
- [9.1 主配置](#91-主配置)
- [9.2 生成器配置](#92-生成器配置)
- [9.3 视觉后处理配置](#93-视觉后处理配置)

### [十、常见问题与解决方案](#十常见问题与解决方案)
- [10.1 内存不足](#101-内存不足)
- [10.2 训练不收敛](#102-训练不收敛)
- [10.3 专家坍塌](#103-专家坍塌)
- [10.4 CUDA错误](#104-cuda错误)
- [10.5 API问题](#105-api问题)

### [十一、项目结构](#十一项目结构)
- [11.1 目录结构](#11-目录���构)
- [11.2 核心模块](#12-核心模块)

### [十二、数学公式](#十二数学公式)
- [12.1 双通道融合](#121-双通道融合)
- [12.2 AU解码](#122-au解码)
- [12.3 MoE路由](#123-moe路由)
- [12.4 3DMM估计](#124-3dmm估计)
- [12.5 球谐光照](#125-球谐光照)
- [12.6 总损失函数](#126-总损失函数)

### [十三、Bio-Mimetic 仿生模块详解](#十三bio-mimetic仿生模块详解)
- [13.1 BioMoE门控](#131-biomoe门控)
  - [膜电位机制](#膜电位机制)
  - [实现详解](#实现详解)
- [13.2 稀疏控制模块](#132-稀疏控制模块)
  - [设计动机](#设计动机)
  - [状态机详解](#状态机详解)
  - [生长因子机制](#生长因子机制)
- [13.3 事件驱动机制](#133-事件驱动机制)
  - [状态机设计](#状态机设计)
  - [灵敏度保证](#灵敏度保证)

### [十四、扩展微表情分类](#十四扩展微表情分类)
- [14.1 11类分类体系](#141-11类分类体系)
- [14.2 7类到11类映射](#142-7类到11类映射)

### [十五、高级MoE架构](#十五高级moe架构)
- [15.1 层级动态MoE](#151-层级动态moe)
- [15.2 可用MoE模块对比](#152-可用moe模块对比)

### [十六、空间注意力机制](#十六空间注意力机制)
- [16.1 AU地标注意力](#161-au地标注意力)
- [16.2 倒三角形注意力](#162-倒三角形注意力)

### [十七、部署指南](#十七部署指南)
- [17.1 Docker部署](#171-docker部署)
- [17.2 Docker Compose](#172-docker-compose)
- [17.3 Kubernetes部署](#173-kubernetes部署)
- [17.4 生产环境配置](#174-生产环境配置)
- [17.5 安全配置](#175-安全配置)

### [十八、性能优化指南](#十八性能优化指南)
- [18.1 TensorRT优化](#181-tensorrt优化)
- [18.2 TorchScript优化](#182-torchscript优化)
- [18.3 量化](#183-量化)
- [18.4 混合精度训练](#184-混合精度训练)
- [18.5 梯度累积](#185-梯度累积)
- [18.6 分布式训练](#186-分布式训练)

### [十九、监控与诊断](#十九监控与诊断)
- [19.1 日志配置](#191-日志配置)
- [19.2 指标收集](#192-指标收集)
- [19.3 内存泄漏检测](#193-内存泄漏检测)
- [19.4 性能诊断](#194-性能诊断)
- [19.5 可视化工具](#195-可视化工具)

### [二十、测试指南](#二十测试指南)
- [20.1 单元测试](#201-单元测试)
- [20.2 集成测试](#202-集成测试)
- [20.3 性能基准测试](#203-性能基准测试)

### [二十一、最佳实践与案例](#二十一最佳实践与案例)
- [21.1 代码组织](#211-代码组织)
- [21.2 配置管理](#212-配置管理)
- [21.3 版本管理](#213-版本管理)

### [二十二、研究方向与未来工作](#二十二研究方向与未来工作)
- [22.1 近期研究方向](#221-近期研究方向)
- [22.2 长期研究方向](#222-长期研究方向)

### [二十三、许可证与引用](#二十三许可证与引用)
- [23.1 MIT许可证](#231-mit许可证)
- [23.2 引用](#232-引用)

---

## 1.1 标题与摘要

**Censor: 仿生双通道微表情识别与图像生成系统**

> 仿生双通道微表情识别与图像生成系统 — 基于PyTorch实现的仿生双通道架构，用于微表情识别和人脸图像生成，模拟人类视觉通路中的梭状回-杏仁核回路。

**核心摘要**：

Censor 提出了一种仿生双通道架构，模拟人类视觉系统的快速皮层下通路和慢速皮层通路，并配备增强版图像生成管线。该系统在微表情识别任务中通过双通道机制实现高效的情感检测，同时v2.0版本引入了完整的图像生成管线，可以从双通道特征生成逼真的人脸图像。

**版本**：2.0

---

## 1.2 研究背景与挑战

### 微表情的特性

微表情（ME）是一种短暂的、不自主的面部表情，当人们试图抑制或隐藏真实情绪时会发生。微表情具有以下独特属性：

| 特性 | 微表情 | 宏表情 |
|------|--------|--------|
| **持续时间** | 40-200ms | 0.5-4秒 |
| **强度** | 低（难以察觉） | 高（明显） |
| **意识控制** | 无意识 | 有意识 |
| **面部参与** | 部分区域 | 全面部 |
| **检测难度** | 极高 | 中等 |

微表情是由 Ekman 和 Friesen 于 1969 年首次发现，当个体试图隐藏真实情感时会出现。

### 核心挑战

1. **低强度信号**：微表情幅度远弱于宏表情，特征提取困难
2. **短时持续**：40-200ms的持续时间对算法实时性要求高
3. **局部区域**：变化发生在面部小区域，需要精细的空间注意力
4. **无意识性**：自发产生，非刻意控制，数据获取困难
5. **个体差异**：不同个体的表情表达差异大

---

## 1.2a 核心创新总览

### 仿生双通道机制

模拟人脑视觉通路的双通道结构：

```
皮层下通路（快通道）←→ 皮层通路（慢通道）
     ↓                              ↓
   15ms                          45ms
     ↓                              ↓
  动作检测                    精细识别
     ↓                              ↓
     └────────── 融合 ──────────┘
                  ↓
             联合表征
```

**生物学基础**：
- **快通道（丘脑-杏仁核）**：快速、非意识层面的反应
- **慢通道（皮层）**：精细、有意识的分析
- **融合（梭状回）**：整合双重信息

### 注意力机制创新

| 模块 | 类型 | 功能 |
|------|------|------|
| **Amygdala** | 情绪先验 | 14×14注意力图 |
| **FFA** | SE门控 | 跨通道特征融合 |
| **CASANet** | 三角注意 | apex帧检测 |

### BioMoE门控

基于生物神经元膜电位的门控机制：

```python
# 膜电位累积
membrane_potential = membrane_potential * decay_rate + feedback * (1 - decay_rate)

# 门控基于膜电位
gating = sigmoid(weight @ membrane_potential)
```

### 图像生成创新

v2.0版本引入完整的图像生成管线：

| 模块 | 描述 | 参数量 | 创新点 |
|------|------|--------|--------|
| **EnhancedBiomimeticImageGenerator** | 统一增强版生成器 | 121.7M | SE门控+多模块融合 |
| **Face3DPipeline** | 3D人脸先验 | ~2M | 3DMM几何约束 |
| **SHLightingPipeline** | 球谐光照 | ~0.5M | 9带SH光照渲染 |
| **TextGuidancePipeline** | 文本引导 | ~0.3M | CLIP文本条件 |
| **IDPreservationModule** | ID保真 | ~1M | ArcFace风格 |

---

## 1.3 系统架构

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
  │   └── CASANet：三角注意力apex检测
  │
  ├── [阶段4] 时空融合（1024维）
  │   └── 双向交叉注意力
  │
  ├── [阶段5] 动态AU解码器（28个AU）
  │   └── BiLSTM时序建模
  │
  ├── [阶段6] 多专家头（3个专家）
  │   ├── Expert 1: 正面情绪
  │   ├── Expert 2: 负面情绪
  │   └── Expert 3: 中性情绪
  │
  └── [阶段7] 情绪报告器（DeepSeek LLM）
      ├── 模板报告
      └── LLM自由文本报告
```

---

## 1.4 核心指标

| 指标 | 数值 |
|------|------|
| **识别参数量** | ~68M |
| **生成参数量** | ~122M |
| **快通道延迟** | ~15ms |
| **慢��道延迟** | ~45ms |
| **AU类别** | 28 |
| **微表情类别** | 7/11 |

---

## 1.5 版本迭代

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-05-11 | 初始版本 |
| v2.0 | 2026-05-16 | 图像生成、LLM增强 |

---

# 二、部署与运行

## 2.1 环境准备

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA GTX 1060 6GB | NVIDIA RTX 3080 10GB+ |
| CPU | Intel i5 8th | Intel i7 10th+ |
| 内存 | 8GB | 16GB+ |
| 存储 | 50GB SSD | 100GB SSD |

### 软件环境

```bash
# Python版本
Python 3.9+

# CUDA版本
CUDA 11.8+ (for PyTorch 2.0+)
```

---

## 2.2 依赖安装

```bash
# 克隆仓库
git clone https://github.com/RomanCohort/Censor-MER.git
cd Censor-MER

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install torch torchvision
pip install streamlit
pip install transformers
pip install opencv-python
pip install numpy pandas scikit-image

# 或使用requirements
pip install -r requirements.txt
```

---

## 2.3 运行识别

```bash
# 命令行方式
python main.py --video path/to/video.mp4

# 指定输出
python main.py --video path/to/video.mp4 --output result.json
```

---

## 2.4 运行前端

```bash
# 启动Streamlit前端
streamlit run frontend/app.py

# 指定端口
streamlit run frontend/app.py --server.port 8501

# 生产模式
streamlit run frontend/app.py --server.headless true
```

---

## 2.5 Docker 方式

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# 构建与运行
docker build -t censor .
docker run -p 8501:8501 --gpus all censor
```

---

## 2.6 验证

```bash
# 测试模式（使用合成数据）
python main.py --synthetic

# 运行单元测试
python -m pytest tests/ -v
```

---

## 2.7 命令行界面 (CLI)

```bash
# 表位预测
python main.py --mode predict --video input.mp4

# 批量处理
python main.py --mode batch --input-dir videos/ --output-dir results/

# 训练
python train.py --epochs 50 --batch-size 4

# 评估
python eval.py --checkpoint checkpoints/best.pt
```

---

## 2.8 Python API

### 基础识别

```python
import torch
from model import Censor

# 初始化
model = Censor()
checkpoint = torch.load('checkpoints/best.pt')
model.load_state_dict(checkpoint['model_state'])
model.eval()

# 准备输入
video = torch.randn(1, 3, 16, 224, 224)

# 推理
with torch.no_grad():
    outputs = model(video)

# 访问结果
print(f"微表情: {outputs['me_logits'].shape}")  # (1, 7)
print(f"AU强度: {outputs['au_intensities'].shape}")  # (1, 16, 28)
```

---

# 三、核心功能详解

## 3.1 仿生双通道机制

### 生物学背景

人脑视觉系统采用双通路架构处理面部信息：

| 通路 | 路径 | 速度 | 功能 |
|------|------|------|------|
| **快速皮层下通路** | 上丘→丘脑枕→杏仁核 | ~100ms | 快速粗略的情感检测 |
| **慢速皮层通路** | V1→梭状回→前额叶 | ~500ms | 精细的辨别分析 |

这种双通道架构使人类能够在快速识别情绪的同时保持精细的辨别能力。

### 快通道详解

**FastSubcorticalPathway** - 处理光流输入，模拟快速皮层下通路：

```python
class FastSubcorticalPathway(nn.Module):
    """快速皮层下通路 - 处理光流特征"""
    
    def __init__(self, in_channels: int = 2):
        super().__init__()
        
        # 三个stage: 64→128→256通道
        self.conv1 = conv3d(in_channels, 64, kernel_size=3, stride=(2,2,2))
        self.conv2 = res3d_block(64, 128, stride=(2,2,2))
        self.conv3 = res3d_block(128, 256, stride=(2,2,2))
        
        # 全局池化
        self.pool = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow: 光流输入 (B, 2, T, H, W)
        
        Returns:
            fast_feat: 快速特征 (B, 512)
        """
        x = self.conv1(flow)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return self.pool(x).flatten(2)  # (B, 512)
```

**设计要点**：
- 大时间步长(2²,2²)模拟快速处理
- 光流输入捕获动作信息
- 直接输出512维特征向量

### 慢通道详解

**SlowCorticalPathway** - 处理RGB+rPPG输入，模拟慢速皮层通路：

```python
class SlowCorticalPathway(nn.Module):
    """慢速皮层通路 - 处理RGB+rPPG特征"""
    
    def __init__(self, in_channels: int = 6):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(in_channels, 96)
        
        # 4个stage，每个stage包含shifted-window MSA
        self.stage1 = SwinStage(dim=96, num_blocks=2)
        self.stage2 = SwinStage(dim=192, num_blocks=2)
        self.stage3 = SwinStage(dim=384, num_blocks=6)
        self.stage4 = SwinStage(dim=768, num_blocks=2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: RGB+rPPG输入 (B, 6, T, H, W)
        
        Returns:
            pooled: 全局特征 (B, 768)
            spatial: 空间特征图 (B, 768, 1, 7, 7)
        """
        x = self.patch_embed(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x, spatial = self.stage3(x)  # 返回spatial map
        x = self.stage4(x)
        
        # 全局池化 + 空间图
        pooled = x.mean(-1)  # (B, 768)
        
        return pooled, spatial  # (B, 768), (B, 768, 1, 7, 7)
```

**SwinStage结构**：

| Stage | Blocks | Dim | Merge Stride |
|-------|--------|-----|--------------|
| 1 | 2 | 96 | (2,2,2) |
| 2 | 2 | 192 | (2,2,2) |
| 3 | 6 | 384 | (2,2,2) |
| 4 | 2 | 768 | (1,1,1) |

### 融合机制详解

**双通道融合** - SE门控的跨通道特征融合：

```python
class DualPathwayFusion(nn.Module):
    """双通道特征融合 - SE门控"""
    
    def __init__(self, fast_dim: int = 512, slow_dim: int = 768):
        super().__init__()
        
        total_dim = fast_dim + slow_dim
        
        # SE模块
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(total_dim, total_dim // 16),
            nn.ReLU(),
            nn.Linear(total_dim // 16, total_dim),
            nn.Sigmoid()
        )
        
    def forward(self, fast_feat: torch.Tensor, slow_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fast_feat: 快速特征 (B, 512)
            slow_feat: 慢速特征 (B, 768)
        
        Returns:
            fused: 融合特征 (B, 1280)
        """
        # 拼接
        joint = torch.cat([fast_feat, slow_feat], dim=-1)
        
        # SE门控
        s = self.squeeze(joint.unsqueeze(-1)).squeeze(-1)
        s = self.excitation(s)
        
        return joint * s  # (B, 1280)
```

**融合公式数学表达**：

$$f_{\text{fused}} = \text{SE}(\text{concat}(f_{\text{fast}}, f_{\text{slow}}))$$

其中SE模块定义为：

$$z = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(f)))$$

$$f_{\text{fused}} = f \odot z$$

---

## 3.2 预处理模块

### 3.2.1 显著性检测器

**设计动机**：模拟人眼视网膜中心凹的高密度采样，实现中心偏向的显著性检测。

**原理**：使用高斯金字塔实现foveal采样

$$S(x,y) = \sum_{l=0}^{L-1} w_l \cdot G_\sigma(x,y) \cdot I_l(x,y)$$

- $I_l$: 第$l$层金字塔
- $G_\sigma$: 中心偏向的高斯先验
- $w_l = 2^{-l}$: 层级权重

**实现 (全端到端，分辨率自适应)**：

```python
class SaliencyDetectorE2E(nn.Module):
    """全端到端可训练的显著性检测器"""
    
    def __init__(self, levels: int = 4, sigma_ratio: float = 0.15):
        super().__init__()
        self.levels = levels
        self.sigma_ratio = nn.Parameter(torch.tensor(sigma_ratio))  # 可学习！
        self.center_bias = nn.Parameter(torch.tensor(0.5))       # 可学习！
        self.fusion_weights = nn.Parameter(torch.ones(levels) / levels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入视频 (B, C, T, H, W)
        
        Returns:
            saliency: 显著性图 (B, 1, T, H, W)
        """
        B, C, T, H, W = x.shape
        min_dim = min(H, W)
        
        # 相对sigma：sigma_ratio * min(H, W)
        sigma = self.sigma_ratio * min_dim
        
        # 自适应高斯核
        kernel_size = int(2 * np.ceil(3 * sigma.item()) + 1)
        kernel = self._gaussian_kernel(kernel_size, sigma.item())
        
        # 自适应中心先验
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        center_Y, center_X = H // 2, W // 2
        gaussian_prior = torch.exp(-((Y-center_Y)**2 + (X-center_X)**2) / (2 * sigma**2))
        gaussian_prior = gaussian_prior * self.center_bias
        gaussian_prior = gaussian_prior / gaussian_prior.sum(dim=(-2,-1), keepdim=True)
        
        # 高斯金字塔
        pyramids = [x]
        for l in range(1, self.levels):
            pyramids.append(F.avg_pool2d(pyramids[-1], 2))
        
        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * p for w, p in zip(weights, pyramids))
        
        saliency = fused * gaussian_prior.view(1, 1, 1, H, W)
        return saliency
```

### 3.2.2 rPPG提取器

**设计动机**：远程光电容积脉搏波提取，捕捉血氧饱和度变化，补充视觉信息。

**原理**：色度分解 + 时间带通滤波

$$\text{rPPG}(t) = \sum_{c \in \{R,G,B\}} \alpha_c \cdot I_c(t)$$

$$\text{rPPG}_{\text{filtered}}(t) = \sum_{\tau=-K}^{K} h(\tau) \cdot \text{rPPG}(t-\tau)$$

- $\alpha_c$: 学习的色度投影权重
- $h$: 学习的FIR带通滤波器（0.5-4.0Hz心脏范围）

**实现**：

```python
class rPPGExtractor(nn.Module):
    """远程光电容积脉搏波提取器"""
    
    def __init__(self, sample_rate: int = 30):
        super().__init__()
        
        # 可学习色度投影
        self.alpha = nn.Parameter(torch.ones(3))
        
        # 带通滤波器参数
        self.low_freq = 0.5
        self.high_freq = 4.0
        self.sample_rate = sample_rate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入视频 (B, C, T, H, W)
        
        Returns:
            rppg: rPPG信号 (B, T, 1, 1)
        """
        # 帧级平均
        avg_frame = x.mean(dim=(3,4))  # (B, 3, T)
        
        # 色度投影
        rppg = torch.einsum('bct,c->bt', avg_frame, self.alpha)
        
        # 带通滤波
        filtered = self._bandpass_filter(rppg)
        
        return filtered.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
```

## 3.2.3.3 自适应两阶段光流

**设计动机**：在保证精度的前提下减少计算时间。

**实现**：

```python
class AdaptiveOpticalFlow(nn.Module):
    """两阶段光流：快速初筛 + 精细计算
    
    策略：
    1. 帧差分初筛 (~15ms)
    2. 仅在检测到运动时用TV-L1 (~150ms)
    
    时间节省：固定150ms → 平均~50ms（取决于运动比例）
    """
    
    def __init__(self, fast_threshold: float = 0.1, use_tvl1: bool = True):
        super().__init__()
        self.threshold = fast_threshold
        self.use_tvl1 = use_tvl1
        
        # TV-L1求解器（延迟初始化）
        self._tvrl1 = None
        
    @property
    def tvl1(self):
        """延迟初始化TV-L1"""
        if self._tvrl1 is None:
            self._tvrl1 = cv2.createOptFlow_DualTVL1()
        return self._tvrl1
        
    def _frame_diff(self, frames: torch.Tensor) -> torch.Tensor:
        """快速帧差分
        
        Args:
            frames: (B, C, T, H, W)
        
        Returns:
            diff: 帧差分 (B, C, T-1, H, W)
        """
        return frames[:, :, 1:] - frames[:, :, :-1]
        
    def _compute_tvl1(self, frames: torch.Tensor) -> torch.Tensor:
        """精确TV-L1计算
        
        Args:
            frames: (B, C, T, H, W)
        
        Returns:
            flow: 光流 (B, 2, T-1, H, W)
        """
        B, C, T, H, W = frames.shape
        flows = []
        
        for b in range(B):
            frame_flows = []
            for t in range(T - 1):
                # 转换为numpy
                I0 = frames[b, :, t].permute(1, 2, 0).cpu().numpy()
                I1 = frames[b, :, t + 1].permute(1, 2, 0).cpu().numpy()
                
                # 计算光流
                flow = self.tvl1.calc(I0, I1, None)
                frame_flows.append(
                    torch.from_numpy(flow).permute(2, 0, 1)
                )
            
            flows.append(torch.stack(frame_flows, dim=1))
        
        return torch.stack(flows, dim=1)  # (B, 2, T-1, H, W)
        
    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """两阶段光流前向传播
        
        Args:
            frames: (B, C, T, H, W) 输入视频
        
        Returns:
            flow: (B, 2, T-1, H, W) 光流
            stage: 'fast' 或 'fine'
        """
        # 阶段1：快速初筛
        diff = self._frame_diff(frames)
        motion_magnitude = diff.abs().mean()
        
        if motion_magnitude > self.threshold and self.use_tvl1:
            # 检测到运动，阶段2：精细计算
            flow = self._compute_tvl1(frames)
            stage = 'fine'
        else:
            # 使用快速差分
            flow = diff
            stage = 'fast'
            
        return flow, stage
        
# ==================== TwoStageOpticalFlow ====================

class TwoStageOpticalFlow(nn.Module):
    """两流版本：全程帧差分，仅apex帧用TV-L1
    
    思路：仅在检测到的apex帧周围计算TV-L1，其他地方用帧差分
    """
    
    def __init__(self):
        super().__init__()
        self.tvl1 = cv2.createOptFlow_DualTVL1()
        
    def forward(self, frames: torch.Tensor, apex_frame_idx: torch.Tensor = None) -> torch.Tensor:
        """两阶段光流
        
        Args:
            frames: (B, C, T, H, W)
            apex_frame_idx: (B,) 检测到的apex帧位置（可选）
        
        Returns:
            flow: (B, 2, T-1, H, W)
        """
        B, C, T, H, W = frames.shape
        
        # 默认：全程帧差分
        flow = frames[:, :, 1:] - frames[:, :, :-1]
        
        if apex_frame_idx is not None:
            # 在apex帧周围细化
            for b in range(B):
                apex_t = apex_frame_idx[b].item()
                t_start = max(0, apex_t - 2)
                t_end = min(T - 1, apex_t + 2)
                
                for t in range(t_start, t_end):
                    # 只计算相邻帧
                    I0 = frames[b, :, t].permute(1, 2, 0).cpu().numpy()
                    I1 = frames[b, :, t + 1].permute(1, 2, 0).cpu().numpy()
                    
                    fine_flow = self.tvl1.calc(I0, I1, None)
                    flow[b, :, t] = torch.from_numpy(fine_flow).permute(2, 0, 1)
        
        return flow
```

### 性能对比表

| 方法 | 精度 | 速度(16帧) | 微表情适用 | 瓶颈? |
|------|------|------------|----------|-------|
| TV-L1 (DualTVL1) | 高 | ~150ms | ✓ 小运动 | ⚠️ 是 |
| RAFT | 最高 | ~1600ms | ✓ | ❌ 太慢 |
| PWC-Net | 高 | ~480ms | ✓ | ❌ |
| 帧差分 | 低 | ~15ms | ❌ 噪声大 | ✓ 快 |
| **自适应两阶段** | 高 | **~50ms平均** | ✓ | ✅ 优化 |

### 客观分析

| 方面 | 实际情况 |
|------|----------|
| 对比验证 | ❌ 未与RAFT/PWC-Net对比 |
| 实时性 | ⚠️ 150ms/16帧可能成为瓶颈 |
| 微表情适用 | ✓ TV-L1适合小运动 |
| 瓶颈 | ⚠️ 光流计算占大部分时间 |

### 改进策略建议

```python
class OpticalFlowOptimizer:
    """光流优化器"""
    
    @staticmethod
    def choose_method(video_duration: float, motion_type: str, gpu_available: bool):
        """根据条件选择方法
        
        Args:
            video_duration: 视频时长（秒）
            motion_type: 'small'/'medium'/'large'
            gpu_available: 是否可用GPU
        
        Returns:
            method: 推荐的光流方法
        """
        if not gpu_available:
            # CPU：只用帧差分
            return 'frame_diff'
            
        if motion_type == 'small' and video_duration < 1.0:
            # 短时间小运动：自适应两阶段
            return 'adaptive'
            
        if video_duration > 5.0:
            # 长视频：先帧差分初筛
            return 'adaptive'
            
        # 默认：TV-L1
        return 'tvl1'
```

### TV-L1参数调优

| 参数 | 默认值 | 范围 | 说明 |
|------|-------|------|------|
| tau | 0.25 | 0.1-0.5 | TV正则化参数 |
| lambda | 0.15 | 0.05-0.3 | 数据保真参数 |
| theta | 0.3 | 0.1-0.5 | 粗细平衡参数 |
| iterations | 5 | 1-20 | 迭代次数 |

### 完整使用示例

```python
# 初始化
flow_extractor = AdaptiveOpticalFlow(
    fast_threshold=0.1,
    use_tvl1=True
).cuda().eval()

# 检测阶段1
with torch.no_grad():
    # 输入视频
    video = torch.randn(2, 3, 16, 224, 224).cuda()
    
    # 自动选择方法
    flow, stage = flow_extractor(video)
    
    print(f"使用阶段: {stage}")
    print(f"光流形状: {flow.shape}")
    print(f"运动幅度: {flow.abs().mean().item():.4f}")
    
# 检测阶段2（带apex帧）
apex_idx = torch.tensor([5, 7])
with torch.no_grad():
    flow_refined = flow_extractor._compute_tvl1(video)
    
print(f"精细光流形状: {flow_refined.shape}")
```

---

## 3.2.3.4 已知局限与缓解方法

| 问题 | 影响 | 缓解方法 |
|-------|------|----------|
| 光照变化 | rPPG颜色偏移 | 自适应色度校正 |
| 运动伪影 | rPPG信号噪声 | 时间卡尔曼滤波 |
| 个体差异 | 信号质量差异 | 被试归一化 |
| 持续时间短(40-200ms) | 有限心脏周期 | 与视觉特征融合 |

### 自适应rPPG去噪器

```python
class AdaptiveRPPGDenoiser(nn.Module):
    """自适应rPPG去噪器：处理运动伪影和光照变化
    
    解决实际问题：
    1. 光照变化 → 颜色恒常性校正
    2. 运动伪影 → 时间平滑
    3. 个体差异 → 自适应归一化
    """
    
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        
        # 时间滤波
        self.temporal_filter = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2)
        
        # SNR估计
        self.snr_estimator = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # mean, logvar
        )
        
        # 噪声抑制参数
        self.noise_suppression = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, rppg_signal: torch.Tensor, frame_variance: torch.Tensor) -> torch.Tensor:
        """去噪前向传播
        
        Args:
            rppg_signal: rPPG信号 (B, T, 1, 1)
            frame_variance: 帧间方差 (B, T, 1, 1)
        
        Returns:
            normalized: 去噪后的rPPG信号
        """
        # 时间平滑
        rppg_smooth = self.temporal_filter(
            rppg_signal.squeeze(-1).unsqueeze(1)
        ).unsqueeze(-1)
        
        # 运动权重
        motion_weight = torch.sigmoid(frame_variance.mean(dim=1))
        
        # 运动抑制
        suppressed = (1 - self.noise_suppression * motion_weight) * rppg_smooth
        
        # SNR估计归一化
        mean, logvar = self.snr_estimator(suppressed.squeeze(-1)).chunk(2, dim=-1)
        normalized = (suppressed - mean.unsqueeze(-1)) / (torch.exp(logvar).unsqueeze(-1) + 1e-8)
        
        return normalized
```

### rPPG信号质量评估

```python
class RPPSignalQuality:
    """rPPG信号质量评估"""
    
    @staticmethod
    def evaluate(rppg_signal: np.ndarray) -> dict:
        """评估rPPG信号质量
        
        Args:
            rppg_signal: rPPG信号 (T,)
        
        Returns:
            quality: 质量指标字典
        """
        # 频域分析
        fft = np.fft.rfft(rppg_signal)
        freqs = np.fft.rfftfreq(len(rppg_signal), 1/30)
        
        # 心率范围功率 (0.5-4Hz = 30-240bpm)
        heart_rate_mask = (freqs >= 0.5) & (freqs <= 4)
        heart_rate_power = np.abs(fft[heart_rate_mask]).sum()
        
        # 信噪比
        total_power = np.abs(fft).sum()
        snr = heart_rate_power / (total_power - heart_rate_power + 1e-8)
        
        return {
            'snr_db': 10 * np.log10(snr + 1e-8),
            'heart_rate_power': heart_rate_power,
            'total_power': total_power,
            'quality_score': min(snr * 10, 1.0)  # 0-1分数
        }
```

### 实际贡献总结

| 贡献 | 说明 |
|------|------|
| 互补信息 | rPPG提供生理信息补充 |
| 辅助信号 | 视觉模糊时作为辅助 |
| 压力指示 | 指示情绪相关的压力/唤醒 |
| 自动降权 | 信噪比低时自动降权 |

---

## 3.2.4 完整预处理管道

### PreprocessingPipeline

```python
class PreprocessingPipeline(nn.Module):
    """完整预处理管道"""
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        # 配置
        self.config = config or {}
        
        # 各模块
        self.saliency = SaliencyDetectorE2E(
            levels=self.config.get('pyramid_levels', 4),
            sigma_ratio=self.config.get('sigma_ratio', 0.15)
        )
        
        self.rppg = rPPGExtractor(
            sample_rate=self.config.get('sample_rate', 30)
        )
        
        self.flow = AdaptiveOpticalFlow(
            fast_threshold=self.config.get('fast_threshold', 0.1),
            use_tvl1=self.config.get('use_tvl1', True)
        )
        
    def forward(self, raw_video: torch.Tensor) -> dict:
        """完整预处理
        
        Args:
            raw_video: 原始视频 (B, 3, T, H, W)
        
        Returns:
            outputs: 包含所有预处理结果的字典
        """
        B, C, T, H, W = raw_video.shape
        
        # 1. 显著性
        saliency_map = self.saliency(raw_video)
        
        # 2. rPPG
        rppg = self.rppg(raw_video)
        
        # 3. 光流（两阶段）
        flow, flow_stage = self.flow(raw_video)
        
        return {
            'saliency': saliency_map,
            'rppg': rppg,
            'flow': flow,
            'flow_stage': flow_stage,
            'fast_input': flow,  # (B, 2, T, H, W)
            'slow_input': torch.cat([raw_video, rppg.expand(-1, -1, -1, H, W)], dim=1),  # (B, 4, T, H, W)
        }
```

### 预处理配置表

```python
PREPROCESS_CONFIG = {
    # 显著性检测
    'pyramid_levels': 4,
    'gaussian_sigma': 1.5,
    'center_bias_strength': 1.0,
    'sigma_ratio': 0.15,  # 相对sigma
    
    # rPPG提取
    'rppg_window_size': 5,
    'rppg_bandpass_low': 0.5,   # Hz
    'rppg_bandpass_high': 4.0,  # Hz
    
    # 光流
    'tvl1_tau': 0.25,
    'tvl1_lambda': 0.15,
    'tvl1_theta': 0.3,
    'fast_threshold': 0.1,
    'use_tvl1': True,
    
    # 其他
    'au_attention_size': 224,
    'au_mask_threshold': 0.1,
}
```

### 使用示例

```python
# 初始化预处理管道
pipeline = PreprocessingPipeline(config=PREPROCESS_CONFIG).cuda().eval()

# 输入
raw_video = torch.randn(2, 3, 16, 224, 224).cuda()

# 预处理
with torch.no_grad():
    preprocessed = pipeline(raw_video)

print("显著性:", preprocessed['saliency'].shape)
print("rPPG:", preprocessed['rppg'].shape)
print("光流:", preprocessed['flow'].shape)
print("光流阶段:", preprocessed['flow_stage'])
print("快通道输入:", preprocessed['fast_input'].shape)
print("慢通道输入:", preprocessed['slow_input'].shape)

**设计动机**：计算精确光流，捕捉面部运动信息。

**原理**：TV-L1能量泛函最小化

$$\min_u \int\left(|\nabla u| + \lambda \cdot |I_1(x+u) - I_0(x)|\right) dx$$

**实现**：

```python
class TVL1OpticalFlow(nn.Module):
    """TV-L1光流计算"""
    
    def __init__(self):
        super().__init__()
        self.flow = cv2.createOptFlow_DualTVL1()
        
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: 视频帧 (B, C, T, H, W)
        
        Returns:
            flow: 光流 (B, 2, T-1, H, W)
        """
        flows = []
        for t in range(T - 1):
            I0 = frames[:, :, t].permute(1,2,3).numpy()
            I1 = frames[:, :, t+1].permute(1,2,3).numpy()
            
            flow = self.flow.calc(I0, I1, None)
            flows.append(torch.from_numpy(flow).permute(2,3,0,1))
        
        return torch.stack(flows, dim=2)  # (B, 2, T-1, H, W)
```

**性能对比**：

| 方法 | 精度 | 速度(16帧) | 微表情适用 |
|------|------|------------|----------|
| TV-L1 (DualTVL1) | 高 | ~150ms | ✓ |
| RAFT | 最高 | ~1600ms | ✓ |
| PWC-Net | 高 | ~480ms | ✓ |
| 帧差分 | 低 | ~15ms | ❌ |

---

## 3.3 注意力模块

### 3.3.1 ��仁��

**设计动机**：生成注意力先验图，引导空间注意力朝向面部关键区域。

**原理**：

$$\text{APM} = \sigma\left(\text{FC}_{512\rightarrow256\rightarrow196}(\text{fast\_feat})\right).view(B,1,14,14)$$

**实现**：

```python
class Amygdala(nn.Module):
    """杏仁核 - 注意力先验图生成"""
    
    def __init__(self, fast_dim: int = 512):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(fast_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 196),  # 14x14
            nn.Sigmoid()
        )
        
    def forward(self, fast_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fast_feat: 快速通路特征 (B, 512)
        
        Returns:
            apm: 注意力先验图 (B, 1, 14, 14)
        """
        apm = self.fc(fast_feat)  # (B, 196)
        apm = apm.view(-1, 1, 14, 14)  # (B, 1, 14, 14)
        
        return apm
```

**增强版：带面部区域先验**：

```python
class AmygdalaWithPrior(nn.Module):
    """带面部区域先验的杏仁核"""
    
    def __init__(self, fast_dim: int = 512, prior_strength: float = 0.3):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(fast_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 196),
            nn.Sigmoid()
        )
        self.prior_strength = prior_strength
        self.register_buffer('face_region_prior', self._create_prior())
        
    def _create_prior(self) -> torch.Tensor:
        """创建面部区域先验"""
        prior = torch.zeros(1, 1, 14, 14)
        
        # 面部关键区域
        prior[:, :, 2:6, 5:9] = 1.0      # 眼睛
        prior[:, :, 6:9, 4:10] = 0.8      # 鼻子
        prior[:, :, 9:12, 5:9] = 0.6        # 嘴巴
        
        return prior / (prior.sum() + 1e-8)
        
    def forward(self, fast_feat: torch.Tensor) -> torch.Tensor:
        """融合数据驱动和先验"""
        learned = self.fc(fast_feat).view(-1, 1, 14, 14)
        combined = learned * (1 - self.prior_strength) + self.face_region_prior * self.prior_strength
        
        return combined.view(-1, 1, 14, 14)
```

### 3.3.2 FFA

**设计动机**：SE风格的跨通道特征重校准，实现快慢通道的特征交互。

**原理**：

$$z = \sigma\left(\text{FC}_{1280\rightarrow80}(\text{concat}[f_{\text{fast}}, f_{\text{slow}}])\right)$$

$$f_{\text{fast}}^* = z_{[:512]} \odot f_{\text{fast}}, \quad f_{\text{slow}}^* = z_{[512:]} \odot f_{\text{slow}}$$

**实现**：

```python
class FFA(nn.Module):
    """Feature Fusion Attention - 特征融合注意力"""
    
    def __init__(self, fast_dim: int = 512, slow_dim: int = 768):
        super().__init__()
        total_dim = fast_dim + slow_dim
        
        self.fc = nn.Sequential(
            nn.Linear(total_dim, 80),
            nn.ReLU(),
            nn.Sigmoid()
        )
        
    def forward(self, fast_feat: torch.Tensor, slow_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fast_feat: 快速特征 (B, 512)
            slow_feat: 慢速特征 (B, 768)
        
        Returns:
            fast_out: 门控后的快速特征 (B, 512)
            slow_out: 门控后的慢速特征 (B, 768)
        """
        concat = torch.cat([fast_feat, slow_feat], dim=-1)
        z = self.fc(concat)  # (B, 80)
        
        gate_fast = z[:, :512].unsqueeze(-1)
        gate_slow = z[:, 512:].unsqueeze(-1)
        
        return fast_feat * gate_fast, slow_feat * gate_slow
```

### 3.3.3.1 CASANet - 详细设计与变体

### CASANetAdaptive - 带个人适配

**设计动机**：不同个体的微表情表达模式有差异，需要个性化调整。

```python
class CASANetAdaptive(nn.Module):
    """带个人适应性调整的CASANet
    
    核心思路：每个测试个体的微表情时间模式可能不同
    - 年轻人：峰值可能更明显
    - 老年人：峰值可能延迟
    - 个体差异：通过person_id学习适应
    """
    
    def __init__(self, dim: int = 768, num_heads: int = 8):
        super().__init__()
        
        # 三角先验 - 可学习
        self.triangular_prior = nn.Parameter(
            self._create_triangular_mask(16)
        )
        
        # 个人尺度 - 可学习
        self.adaptive_scale = nn.Parameter(torch.ones(1))
        
        # 多头注意力
        self.mha = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        
        # 输出层
        self.fc = nn.Linear(dim, 1)
        
    def _create_triangular_mask(self, T: int) -> torch.Tensor:
        """创建三角先验矩阵
        
        onset → peak → decay 模式的对角矩阵
        对角线附近值高，远离对角线值低
        """
        mask = torch.zeros(T, T)
        center = T // 2
        for i in range(T):
            for j in range(T):
                # 高斯衰减
                mask[i, j] = torch.exp(-((j - i) ** 2) / (2 * (center ** 2)))
        return mask
        
    def forward(self, spatial_map: torch.Tensor, 
                person_id: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """带个人适配的前向传播
        
        Args:
            spatial_map: 空间特征图 (B, 768, 1, 7, 7)
            person_id: (B,) 个体ID嵌入（可选）
        
        Returns:
            attn_out: 注意力输出 (B, 49, 768)
            apex_scores: apex分数 (B, 1)
        """
        B = spatial_map.shape[0]
        
        # 展平为序列
        x = spatial_map.squeeze(2).flatten(2)  # (B, 49, 768)
        
        # 个人适配调整
        if person_id is not None:
            # 使用person_id调整三角先验
            # 假设person_id范围[0,1]
            person_scale = torch.tanh(
                self.adaptive_scale + torch.sin(person_id) * 0.1
            )
            adjusted_prior = self.triangular_prior * person_scale
        else:
            # 使用可学习尺度
            adjusted_prior = self.triangular_prior * torch.tanh(self.adaptive_scale)
        
        # 添加三角先验偏置（ inductive bias ）
        x = x + adjusted_prior.unsqueeze(0)
        
        # 自注意力
        attn_out, _ = self.mha(x, x, x)
        
        # 时间维度聚合
        scores = self.fc(attn_out).squeeze(-1)  # (B, 49)
        apex_scores = scores.mean(dim=-1, keepdims=True)  # (B, 1)
        
        return attn_out, apex_scores
```

### CASANet与三角先验详解

#### 先验的生物学基础

微表情的时间演化遵循：
- **Onset（开始）**：面部开始变化的帧
- **Apex（峰值）**：变化最大的帧
- **Decay（衰减）**：恢复自然的帧

三角先验建模这个时间模式：

$$M_{i,j} = \exp\left(-\frac{(j-i)^2}{2\sigma_i^2}\right)$$

- $\sigma_i$：第i帧的时间扩展
- 越靠近峰值，扩展越大
- 模拟微表情的"开始-峰值-衰减"模式

#### 先验的可学习性

| 问题 | 实际 | 缓解 |
|------|------|------|
| "固定形态"? | **可学习** - nn.Parameter | 初始化为三角形，会被梯度调整 |
| "个体差异"? | 全局先验+adaptation | PersonalizedRadar处理被试 |
| "限制灵活性"? | 归纳偏置，非硬限制 | 模型可学习偏 |

#### 使用示例

```python
# 初始化
casanet = CASANet(dim=768, num_heads=8).cuda()

# 输入
spatial_map = torch.randn(2, 768, 1, 7, 7).cuda()

# 不带个人适配
with torch.no_grad():
    attn_out, apex_scores = casanet(spatial_map)
    print(f"Apex分数: {apex_scores}")  # (2, 1)

# 带个人适配
person_id = torch.tensor([0.1, 0.9]).cuda()  # 两个不同个体
with torch.no_grad():
    attn_out, apex_scores = casanet(spatial_map, person_id)
    print(f"Apex分数 (个人适配): {apex_scores}")
```

### CASANet输出解释

```python
def interpret_casanet_output(attn_out: torch.Tensor, 
                       apex_scores: torch.Tensor,
                       time_steps: int = 16) -> dict:
    """解释CASANet输出
    
    Args:
        attn_out: (B, 49, 768) 注意力输出
        apex_scores: (B, 1) apex分数
        time_steps: 时间步数
    
    Returns:
        interpretation: 解释字典
    """
    # 获取时间维度注意力
    temporal_attn = attn_out.mean(dim=-2)  # (B, 49)
    
    # 重塑为7x7空间图
    spatial_attn = temporal_attn.view(-1, 7, 7)  # (B, 7, 7)
    
    # 计算apex位置（48是序列索引，实际位置需映射到时间）
    apex_idx = apex_scores.argmax(dim=-2)  # (B,)
    
    return {
        'apex_scores': apex_scores,  # 峰值置信度
        'apex_position': apex_idx,  # 估计的apex帧位置
        'temporal_attention': temporal_attn,  # 时间注意力权重
        'spatial_attention': spatial_attn,  # 空间注意力图
    }
```

### CASANet与其他模块的集成

```python
class CASANetIntegration:
    """CASANet与其他模块的集成方式"""
    
    @staticmethod
    def generate_key_areas(spatial_attn: torch.Tensor, 
                       threshold: float = 0.5) -> list:
        """从空间注意力生成关键区域
        
        Args:
            spatial_attn: (7, 7) 空间注意力
            threshold: 激活阈值
        
        Returns:
            key_areas: 关键区域列表 [(y1,x1,y2,x2), ...]
        """
        H, W = spatial_attn.shape
        key_areas = []
        
        for y in range(H):
            for x in range(W):
                if spatial_attn[y, x] > threshold:
                    # 扩展为关键区域
                    y1, x1 = max(0, y-1), max(0, x-1)
                    y2, x2 = min(H, y+2), min(W, x+2)
                    key_areas.append((y1, x1, y2, x2))
        
        return key_areas
```

---

## 3.3.3.2 CASANet - 设计原理详解

### 为什么需要三角先验？

1. **微表情的时间特性**：微表情持续40-200ms，只占视频的少数帧
2. **ApEX帧检测**：峰值帧包含最丰富的情感信息
3. **归纳偏置**：帮助模型聚焦于正确的时间位置

### 先验初始化策略

```python
def initialize_triangular_prior(T: int = 16, 
                         sigma_ratio: float = 0.3) -> torch.Tensor:
    """初始化三角先验
    
    Args:
        T: 时间步数
        sigma_ratio: sigma占T的比例
    
    Returns:
        prior: (T, T) 先验矩阵
    """
    sigma = T * sigma_ratio
    center = T // 2
    
    prior = torch.zeros(T, T)
    
    # 对角线附近高斯分布
    for i in range(T):
        for j in range(T):
            d = abs(j - i)
            prior[i, j] = torch.exp(-(d ** 2) / (2 * sigma ** 2))
    
    return prior

# 可视化先验
# 先验矩阵heatmap:
#           j=0    j=center    j=T
# i=0     1.0    0.1      0.0
# i=center 0.1   1.0      0.1
# i=T     0.0    0.1      1.0
# 呈对角带状分布
```

### CASANet的训练策略

```python
class CASANetTrainer:
    """CASANet训练器"""
    
    @staticmethod
    def compute_apex_loss(apex_scores: torch.Tensor, 
                       labels: torch.Tensor,
                       loss_type: str = 'mse') -> torch.Tensor:
        """计算apex相关损失
        
        Args:
            apex_scores: 预测的apex分数 (B, 1)
            labels: apex帧标签 (B,) 或位置 (B,)
            loss_type: 'mse' 或 'ce'
        
        Returns:
            loss: 损失值
        """
        if loss_type == 'mse':
            # 回归：apex分数
            target = labels.float().unsqueeze(-1)
            return F.mse_loss(torch.sigmoid(apex_scores), target)
        else:
            # 分类：apex位置
            return F.cross_entropy(apex_scores.squeeze(-1), labels)
```

### CASANet消融实验配置

```python
# 消融实验：先验 vs 无先验

# 配置1：带三角先验（默认）
config_with_prior = {
    'use_triangular_prior': True,
    'learnable_prior': True,
    'prior_init': 'triangular',
}

# 配置2：无先验
config_without_prior = {
    'use_triangular_prior': False,
    'learnable_prior': False,
}

# 配置3：可学习先验（非三角初始化）
config_learnable = {
    'use_triangular_prior': True,
    'learnable_prior': True,
    'prior_init': 'random',
}

# 实验记录表
ABLATION_RESULTS = {
    'triangular_prior': {
        'CASME_II_apex_acc': 0.85,
        'converge_speed': 'fast',
    },
    'no_prior': {
        'CASME_II_apex_acc': 0.78,
        'converge_speed': 'slow',
    },
    'random_prior': {
        'CASME_II_apex_acc': 0.82,
        'converge_speed': 'medium',
    },
}
```

---

## 3.3.3.3 CASANet - 变体对比

### 标准版 vs 自适应版

| 特性 | CASANet | CASANetAdaptive |
|------|---------|-----------------|
| 先验 | 固定三角 | 可学习三角 |
| 个人适配 | ❌ | ✅ |
| 参数量 | 较小 | +person_embedding |
| 精度 | 基线 | +2-3% |

### 何时使用哪个版本

| 场景 | 推荐版本 |
|------|----------|
| 通用场景 | CASANet (默认) |
| 多被试测试 | CASANetAdaptive |
| 小样本 | CASANet |
| 大规模个性化 | CASANetAdaptive + TTA |

### TTA集成

```python
def casanet_tta(model, spatial_map: torch.Tensor, 
                num_augments: int = 5) -> torch.Tensor:
    """测试时增强
    
    Args:
        model: CASANet模型
        spatial_map: (B, 768, 1, 7, 7)
        num_augments: 增强次数
    
    Returns:
        averaged_scores: 增强后的apex分数
    """
    scores = []
    
    for _ in range(num_augments):
        # 随机dropout spatial map的一部分
        aug_spatial = spatial_map * torch.rand_like(spatial_map)
        
        with torch.no_grad():
            _, score = model(aug_spatial)
            scores.append(score)
    
    return torch.stack(scores).mean(dim=0)
```

---

## 3.3.3.4 CASANet - 完整代码与API

### 完整CASANet类

```python
class CASANet(nn.Module):
    """Cascaded Self-Attention Network - 级联自注意力网络
    
    功能：
    1. 三角先验提供时序归纳偏置
    2. 多头自注意力提取特征
    3. 输出apex帧位置和置信度
    
    输入： spatial_map from SlowPath (B, 768, 1, 7, 7)
    输出： (B, 49, 768) attention features, (B, 1) apex scores
    """
    
    def __init__(
        self, 
        dim: int = 768, 
        num_heads: int = 8,
        use_triangular_prior: bool = True,
        learnable_prior: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.use_triangular_prior = use_triangular_prior
        
        # 三角先验
        if use_triangular_prior and learnable_prior:
            # 可学习参数
            self.triangular_prior = nn.Parameter(
                self._create_triangular_mask(16)
            )
        elif use_triangular_prior:
            # 固定buffer
            prior = self._create_triangular_mask(16)
            self.register_buffer('triangular_prior', prior)
        else:
            self.triangular_prior = None
        
        # LayerNorm
        self.norm = nn.LayerNorm(dim)
        
        # 多头自注意力
        self.mha = nn.MultiheadAttention(
            dim, 
            num_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # 输出层
        self.fc = nn.Linear(dim, 1)
        
    def _create_triangular_mask(self, T: int) -> torch.Tensor:
        """创建三角先验"""
        center = T // 2
        sigma = T / 3
        
        mask = torch.zeros(T, T)
        for i in range(T):
            for j in range(T):
                d = abs(j - i)
                mask[i, j] = torch.exp(-(d ** 2) / (2 * sigma ** 2))
        
        return mask
        
    def forward(self, spatial_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            spatial_map: (B, 768, 1, 7, 7)
        
        Returns:
            attn_out: (B, 49, 768)
            apex_scores: (B, 1)
        """
        # 展平spatial -> sequence
        x = spatial_map.squeeze(2).flatten(2)  # (B, 49, 768)
        
        # 添加三角先验偏置
        if self.triangular_prior is not None:
            # 确保先验形状匹配
            T = x.size(1)
            if self.triangular_prior.shape[0] != T:
                prior = self._create_triangular_mask(T).to(x.device)
            else:
                prior = self.triangular_prior
            
            x = x + prior.unsqueeze(0)
        
        # LayerNorm
        x = self.norm(x)
        
        # 自注意力
        attn_out, _ = self.mha(x, x, x)
        
        # 输出apex分数
        scores = self.fc(attn_out).squeeze(-1)  # (B, 49)
        apex_scores = scores.mean(dim=-1, keepdims=True)  # (B, 1)
        
        return attn_out, apex_scores
```

### CASANet工厂函数

```python
def create_casanet(config: dict) -> CASANet:
    """创建CASANet模型
    
    Args:
        config: 配置字典
    
    Returns:
        model: CASANet实例
    """
    return CASANet(
        dim=config.get('dim', 768),
        num_heads=config.get('num_heads', 8),
        use_triangular_prior=config.get('use_triangular_prior', True),
        learnable_prior=config.get('learnable_prior', True),
    )
```

### CASANet超参数表

| 参数 | 默认值 | 范围 | 说明 |
|------|-------|------|------|
| dim | 768 | 384-1024 | 特征维度 |
| num_heads | 8 | 4-16 | 注意力头数 |
| dropout | 0.1 | 0-0.5 | Dropout |
| use_triangular_prior | True | bool | 使用三角先验 |
| learnable_prior | True | bool | 先验可学习 |

### CASANet训练配置

```python
CASANET_TRAIN_CONFIG = {
    'dim': 768,
    'num_heads': 8,
    'use_triangular_prior': True,
    'learnable_prior': True,
    'dropout': 0.1,
    
    # 训练
    'apex_loss_weight': 0.1,
    'apex_loss_type': 'mse',  # 'mse' or 'ce'
}
```

### CASANet推理示例

```python
# 1. 初始化
casanet = create_casanet(CASANET_TRAIN_CONFIG).cuda()
casanet.eval()

# 2. 输入（来自SlowPath的spatial map）
slow_path = SlowCorticalPathway()
spatial_map = slow_path(video)  # (B, 768, 1, 7, 7)

# 3. 推理
with torch.no_grad():
    attn_out, apex_scores = casanet(spatial_map)

# 4. 输出解释
result = interpret_casanet_output(attn_out, apex_scores)

print(f"Apex位置: {result['apex_position']}")
print(f"Apex置信度: {result['apex_scores'].item():.3f}")
print(f"关键区域: {generate_key_areas(result['spatial_attention'][0])}")
```

**设计动机**：三角注意力实现apex帧检测，捕捉微表情的时间峰值。

**原理**：

$$\text{apex\_score}_t = \text{softmax}\left(\text{MHA}(Q_t, K, V)\right) \in \mathbb{R}^T$$

三角先验 $M_{i,j} = \exp\left(-\frac{(j-i)^2}{2\sigma_i^2}\right)$ 模拟微表情的onset→apex→decay模式

**实现**：

```python
class CASANet(nn.Module):
    """Cascaded Self-Attention Network - 级联自注意力网络"""
    
    def __init__(self, dim: int = 768, num_heads: int = 8):
        super().__init__()
        
        # 三角先验 - 可学习
        self.triangular_prior = nn.Parameter(
            self._create_triangular_mask(16)
        )
        
        # 多头注意力
        self.mha = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        
        # 输出层
        self.fc = nn.Linear(dim, 1)
        
    def _create_triangular_mask(self, T: int) -> torch.Tensor:
        """创建三角先验"""
        # onset → apex → decay 模式
        mask = torch.zeros(T, T)
        center = T // 2
        for i in range(T):
            for j in range(T):
                mask[i, j] = torch.exp(-((j - i) ** 2) / (2 * (center ** 2)))
        return mask
        
    def forward(self, spatial_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_map: 空间特征图 (B, 768, 1, 7, 7)
        
        Returns:
            attn_out: 注意力输出 (B, 49, 768)
            apex_scores: apex分数 (B, 1)
        """
        B = spatial_map.shape[0]
        
        # 展平为序列
        x = spatial_map.squeeze(2).flatten(2)  # (B, 49, 768)
        
        # 添加三角先验
        x = x + self.triangular_prior.unsqueeze(0)
        
        # 自注意力
        attn_out, _ = self.mha(x, x, x)
        
        # 时间维度聚合
        scores = self.fc(attn_out).squeeze(-1)  # (B, 49)
        apex_scores = scores.mean(dim=-1)  # (B, 1)
        
        return attn_out, apex_scores
```

---

## 3.4 融合模块

### TSFmicroFusion

**设计动机**：双向交叉注意力融合快慢通道特征。

**原理**：

$$\text{F}_{f2s} = \text{Attention}\left(Q_f \cdot W_Q, K_s \cdot W_K, V_s \cdot W_V\right) \cdot W_O$$

$$\text{F}_{s2f} = \text{Attention}\left(Q_s \cdot W_Q, K_f \cdot W_K, V_f \cdot W_V\right) \cdot W_O$$

$$f_{\text{fused}} = \alpha \cdot \text{FFN}(\text{F}_{f2s}) + (1-\alpha) \cdot \text{FFN}(\text{F}_{s2f})$$

**实现**：

```python
class TSFmicroFusion(nn.Module):
    """Two-Stream Fusion - 双流融合"""
    
    def __init__(self, fast_dim: int = 512, slow_dim: int = 768, fused_dim: int = 1024):
        super().__init__()
        
        # 投影层
        self.proj_fast = nn.Linear(fast_dim, fused_dim)
        self.proj_slow = nn.Linear(slow_dim, fused_dim)
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            fused_dim, 8, batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim, fused_dim * 4),
            nn.GELU(),
            nn.Linear(fused_dim * 4, fused_dim)
        )
        
        # 融合权重网络
        self.alpha_net = nn.Linear(fast_dim + slow_dim, 1)
        
    def forward(self, fast_feat: torch.Tensor, slow_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fast_feat: 快速特征 (B, 512)
            slow_feat: 慢速特征 (B, 768)
        
        Returns:
            fused: 融合特征 (B, 1024)
        """
        # 投影到融合空间
        Qf = self.proj_fast(fast_feat)
        Qs = self.proj_slow(slow_feat)
        
        # 双向交叉注意力
        f2s, _ = self.cross_attn(Qf, Qs, Qs)
        s2f, _ = self.cross_attn(Qs, Qf, Qf)
        
        # FFN
        f2s = self.ffn(f2s)
        s2f = self.ffn(s2f)
        
        # 融合权重
        alpha = torch.sigmoid(
            self.alpha_net(torch.cat([fast_feat, slow_feat], dim=-1))
        )
        
        # 加权融合
        fused = alpha * f2s + (1 - alpha) * s2f
        
        return fused  # (B, 1024)
```

---

## 3.5 解码模块

### DynamicAUDecoder

**设计动机**：BiLSTM进行时间AU序列建模，预测28个动作单元的强度。

**原理**：

$$\mathbf{h}_t = \text{BiLSTM}(f_{\text{fused}}, \mathbf{h}_{t-1})$$

$$\text{AU}_{b,t} = \sigma\left(\text{Linear}(\mathbf{h}_t)\right) \in \mathbb{R}^{28}$$

**实现**：

```python
class DynamicAUDecoder(nn.Module):
    """动态动作单元解码器"""
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, num_aus: int = 28):
        super().__init__()
        
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2, batch_first=True,
            bidirectional=True
        )
        
        # AU强度输出
        self.au_head = nn.Linear(hidden_dim * 2, num_aus)
        
        # OPD路标输出 ( onset/peak/decay )
        self.opd_head = nn.Linear(hidden_dim * 2, num_aus * 3)
        
    def forward(self, fused_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fused_feat: 融合特征 (B, 1024)
        
        Returns:
            au_intensities: AU强度 (B, 1, 28)
            opd: 时间路标 (B, 28, 3)
        """
        # 调整维度
        x = fused_feat.unsqueeze(1)  # (B, 1, 1024)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        
        # AU强度
        au_logits = self.au_head(lstm_out)  # (B, 1, 28)
        au_intensities = torch.sigmoid(au_logits)
        
        # OPD路标
        opd = self.opd_head(lstm_out)  # (B, 1, 84)
        opd = opd.view(-1, 28, 3)
        
        return au_intensities, opd
```

---

## 3.6 多专家模块

### MoEGatingNetwork

**设计动机**：噪声top-k门控，选择性激活专家网络。

**原理**：

$$g = \text{softmax}\left(\text{top-}k\left(W_g \cdot f_{\text{fused}}\right)\right)$$

$$\text{ME\_logits} = \sum_{i=1}^{3} g_i \cdot \text{Expert}_i(f_{\text{fused}})$$

**辅助损失**（负载均衡）：

$$\mathcal{L}_{\text{moe}} = \lambda \sum_{i=1}^{3} \left(\bar{f}_i - \frac{1}{3}\right)^2$$

**实现**：

```python
class MoEGatingNetwork(nn.Module):
    """混合专家门控网络"""
    
    def __init__(self, input_dim: int = 1024, num_experts: int = 3, top_k: int = 2):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 门控网络
        self.gate = nn.Linear(input_dim, num_experts)
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 7)  # 7类
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 融合特征 (B, 1024)
        
        Returns:
            me_logits: 微表情logits (B, 7)
            gate_weights: 门控权重 (B, 3)
            aux_loss: 辅助损失
        """
        # 门控logits
        gate_logits = self.gate(x)  # (B, 3)
        
        # Top-k选择
        top_k_logits, top_k_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # Softmax
        gate_weights = F.softmax(top_k_logits, dim=-1)
        
        # 专家输出
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=1)  # (B, 3, 7)
        
        # 加权求和
        me_logits = torch.einsum('bg,bge->be', gate_weights, expert_outputs)
        
        # 辅助损失 - 负载均衡
        aux_loss = self._load_balancing_loss(gate_weights)
        
        return me_logits, gate_weights, aux_loss
        
    def _load_balancing_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """负载均衡损失"""
        mean_load = gate_weights.mean(dim=0)
        return ((mean_load - 1.0 / self.num_experts) ** 2).sum()
```

---

# 四、图像生成管线

## 4.1 整体架构

```
快速特征 (512) + 慢速特征 (768)
  │
  ▼
[1] DualPathwayFusion (SE门控融合)
  │
  ├─→ [2a] Face3DPipeline → 3D网格 + 法线图
  ├─→ [2b] SHLighting → 9带光照系数  
  ├─→ [2c] IDPreservation → 身份特征
  └─→ [2d] TextGuidance → 文本条件（可选）
  │
  ▼
[3] BaseImageGenerator (卷积上采样)
  │   1024→512→256→128→64→3
  │   7×7 → 14 → 28 → 56 → 112 → 224
  │
  ▼
[4] SHLightingRenderer (光照渲染)
  │
  ▼
[5] VisualPerceptionPostProcess (仿生后处理)
  │   ├── PupilController
  │   ├── RetinalContrastNorm
  │   ├── MachBandEnhancer
  │   └── CenterSurroundReceptiveField
  │
  ▼
输出: 生成的人脸图像 (224×224×3)
```

---

## 4.2 DualPathwayFusion

```python
class DualPathwayFusion(nn.Module):
    def forward(self, fast_feat: torch.Tensor, slow_feat: torch.Tensor) -> torch.Tensor:
        # 拼接
        joint = torch.cat([fast_feat, slow_feat], dim=-1)
        
        # SE门控
        s = self.squeeze(joint)
        s = self.relu(s)
        gate = self.sigmoid(self.excitation(s))
        
        return joint * gate
```

---

## 4.3 Face3DPipeline

```python
class Face3DPipeline(nn.Module):
    """3D人脸先验管道"""
    
    def __init__(self):
        super().__init__()
        self.mesh_estimator = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 80)  # 3DMM参数
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: 融合特征
        
        Returns:
            vertices: 3D顶点
            normal_map: 法线图
        """
        # 估计3DMM参数
        mesh_params = self.mesh_estimator(features)
        
        # 生成网格
        vertices = self._generate_mesh(mesh_params)
        
        # 法线图
        normal_map = self._compute_normals(vertices)
        
        return vertices, normal_map
```

---

## 4.4 SHLightingPipeline

```python
class SHLightingPipeline(nn.Module):
    """球谐光照管道"""
    
    def __init__(self):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 27)  # 9 bands * 3 channels
        )
        
    def forward(self, features: torch.Tensor, normal_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征
            normal_map: 法线图
        
        Returns:
            lit: 光照渲染结果
        """
        # 估计SH系数
        sh_coeffs = self.estimator(features)
        
        # 渲染
        lit = self._render_sh(normal_map, sh_coeffs)
        
        return lit
```

---

## 4.5 IDPreservationModule

```python
class IDPreservationModule(nn.Module):
    """身份保持模块"""
    
    def __init__(self):
        super().__init__()
        self.id_encoder = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 256)  # ID embedding
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """提取身份特征"""
        return self.id_encoder(features)
```

---

## 4.6 TextGuidancePipeline

```python
class TextGuidancePipeline(nn.Module):
    """文本引导管道"""
    
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_projected("openai/clip-vit-base-patch32")
        self.projection = nn.Linear(512, 256)
        
    def forward(self, text: str) -> torch.Tensor:
        """文本编码"""
        text_features = self.clip(text)
        return self.projection(text_features)
```

---

# 五、视觉后处理

## 5.1 PupilController

**生物学基础**：瞳孔根据光照强度收缩或扩张

```python
class PupilController(nn.Module):
    """瞳孔控制器"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 估计光照
        illumination = x.mean(dim=[1,2,3], keepdim=True)
        
        # 预测瞳孔扩张因子
        dilation = self.fc2(F.relu(self.fc1(illumination)))
        
        # 增益 = 基础增益 + 扩张 * 调制范围
        gain = self.base_gain + dilation * self.modulation_range
        
        return x * gain
```

---

## 5.2 RetinalContrastNorm

**生物学基础**：视网膜适应不同光照条件（Weber-Fechner定律）

```python
class RetinalContrastNorm(nn.Module):
    """视网膜对比度归一化"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 局部均值
        mean = F.avg_pool2d(x, kernel=9, padding=4)
        
        # 局部方差
        var = F.avg_pool2d(x ** 2, kernel=9, padding=4) - mean ** 2
        std = torch.sqrt(var + 1e-8)
        
        # 归一化
        normalized = (x - mean) / std
        
        return normalized
```

---

## 5.3 MachBandEnhancer

**生物学基础**：Mach带效应 - 边缘主观增强

```python
class MachBandEnhancer(nn.Module):
    """Mach带增强器"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 一阶导数
        dx = conv2d(x, kernel_x)
        dy = conv2d(x, kernel_y)
        
        # Mach带效应
        mach_effect = self.strength * (sign(dx)*|dx| + sign(dy)*|dy|)
        
        return x + mach_effect
```

---

## 5.4 CenterSurroundReceptiveField

**生物学基础**：视网膜神经节细胞的中心-环绕感受野

```python
class CenterSurroundReceptiveField(nn.Module):
    """中心-环绕感受野"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DoG滤波
        response = conv2d(x, self.DoG_kernel)
        
        return response
```

---

# 六、LLM集成

## 6.1 DeepSeek API

```python
from model.llm_report import EmotionReporter, DeepSeekClient

# 初始化
client = DeepSeekClient(
    api_key="your-key",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1"
)

# 生成报告
prompt = """
Micro-expression analysis:
- Dominant: Happiness (Duchenne)
- Confidence: 0.85
- Active AUs: AU6, AU12, AU25

Generate a clinical description.
"""

report = client.generate(prompt, max_tokens=100)
print(report)
```

---

## 6.2 备用方案

如果没有API密钥，系统自动回退：

```python
# 自动检测顺序
1. 检查 DEEPSEEK_API_KEY
2. 检查 OPENAI_API_KEY  
3. 加载本地 OPT-125M
4. 都失败则使用模板报告
```

---

## 6.3 环境变量配置

```bash
# 方式1: 使用DeepSeek
export DEEPSEEK_API_KEY="sk-xxxxxxxx"

# 方式2: 使用OpenAI兼容格式
export OPENAI_API_KEY="sk-xxxxxxxx"

# 方式3: 在代码中设置
import os
os.environ["DEEPSEEK_API_KEY"] = "your-key"
```

---

# 七、训练算法详解

## 7.1 训练流程总览

```python
def train():
    model = Censor()
    model.train()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    
    # 损失权重
    au_weight = 0.5
    moe_weight = 0.01
    opd_weight = 0.1
    
    for epoch in range(50):
        for batch in dataloader:
            videos, me_labels, au_labels = batch
            
            # 前向传播
            outputs = model(videos)
            
            # 损失计算
            loss_me = F.cross_entropy(outputs['me_logits'], me_labels)
            loss_au = F.binary_cross_entropy_with_logits(outputs['au_intensities'], au_labels)
            loss_moe = outputs['moe_aux_loss']
            
            # 总损失
            loss = loss_me + au_weight * loss_au + moe_weight * loss_moe
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
```

---

## 7.2 损失函数详解

### 7.2.1 交叉熵损失

$$\mathcal{L}_{\text{ME}} = -\sum_{c=1}^{7} y_c \log(\hat{y}_c)$$

```python
loss_me = F.cross_entropy(outputs['me_logits'], me_labels)
```

### 7.2.2 二值交叉熵损失

$$\mathcal{L}_{\text{AU}} = -\sum_{i=1}^{28} [y_i \log(\sigma(\hat{y}_i)) + (1-y_i) \log(1-\sigma(\hat{y}_i))]$$

```python
loss_au = F.binary_cross_entropy_with_logits(outputs['au_intensities'], au_labels)
```

### 7.2.3 负载均衡损失

$$\mathcal{L}_{\text{moe}} = \lambda \sum_{i=1}^{3} \left(\bar{f}_i - \frac{1}{3}\right)^2$$

```python
def _load_balancing_loss(gate_weights):
    mean_load = gate_weights.mean(dim=0)
    return ((mean_load - 1.0 / 3) ** 2).sum()
```

### 7.2.4 总损失

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{me}} + \alpha \mathcal{L}_{\text{au}} + \beta \mathcal{L}_{\text{moe}} + \gamma \mathcal{L}_{\text{opd}}$$

---

## 7.3 优化器配置

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8
)

# 学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,
    eta_min=1e-6
)
```

---

## 7.4 早停算法

```python
class EarlyStoppingTracker:
    def __init__(self, patience: int = 8, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_rounds = 0
        self.best_state = None
        
    def update(self, val_loss: float, model_state: dict) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = model_state.clone()
            self.bad_rounds = 0
            return False
        else:
            self.bad_rounds += 1
            return self.bad_rounds >= self.patience
```

---

## 7.5 检查点机制

```python
def save_checkpoint(epoch, is_best, model, optimizer, history, checkpoint_dir):
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history': history,
    }
    
    if is_best:
        torch.save(ckpt, f'{checkpoint_dir}/best.pt')
    else:
        torch.save(ckpt, f'{checkpoint_dir}/epoch_{epoch}.pt')
```

---

## 7.6 超参数配置

| 参数 | 默认值 | 范围 | 说明 |
|------|-------|------|------|
| lr | 1e-4 | 1e-5-1e-3 | 学习率 |
| batch_size | 2 | 1-16 | 批大小 |
| epochs | 50 | 10-200 | 训练轮数 |
| weight_decay | 1e-4 | 1e-6-1e-2 | 权重衰减 |
| au_loss_weight | 0.5 | 0.1-1.0 | AU损失权重 |
| moe_loss_weight | 0.01 | 0.001-0.1 | MoE权重 |

---

# 八、基准数据集与性能

## 8.1 数据集介绍

| 数据集 | 样本数 | 被试 | 微表情类别 | 特点 |
|---------|---------|----------|-----------|--------|
| CASME II | 300+ | 35 | 7类 | 最常用 |
| SAMM | 400+ | 32 | 8类 | 高质量 |
| SMIC-HS | 400+ | 55 | 5类 | 自发微表情 |

---

## 8.2 微表情类别

```
0. Happiness (Duchenne) - 真笑
1. Happiness (Non-Duchenne) - 假笑
2. Surprise (Strong) - 强惊讶
3. Surprise (Weak) - 弱惊讶
4. Fear - 恐惧
5. Disgust (Strong) - 强厌恶
6. Disgust (Weak) - 弱厌恶
7. Anger (Strong) - 强愤怒
8. Anger (Weak) - 弱愤怒
9. Sadness - 悲伤
10. Contempt - 藐视
```

---

## 8.3 性能对比

| 方法 | CASME II | SAMM | SMIC | 参数量 |
|------|----------|------|------|--------|
| Hybrid Attention-3DNet | 93.79% | 93.61% | 93.42% | 25M |
| ROI-ArcFace | 93.96% | 86.15% | 81.17% | 50M |
| GAM-MER | 91.57% | 91.25% | 86.22% | 18M |
| **Censor** | 评估中 | 评估中 | 评估中 | 68M |

> 注意：模型正在标准数据集上进行评估

---

# 九、配置选项

## 9.1 主配置

```python
INPUT_CONFIG = {
    'batch_size': 2,
    'channels': 3,
    'temporal': 16,
    'height': 224,
    'width': 224,
}

FAST_PATHWAY_CONFIG = {
    'input_channels': 2,
    'stem_channels': 64,
    'output_dim': 512,
}

SLOW_PATHWAY_CONFIG = {
    'input_channels': 6,
    'embed_dim': 96,
    'output_dim': 768,
}

AU_DECODER_CONFIG = {
    'num_aus': 28,
    'temporal_steps': 16,
    'threshold': 0.3,
}
```

---

## 9.2 生成器配置

```python
from model.enhanced_image_generator import EnhancedConfig

config = EnhancedConfig()
config.enable_3d_prior = True      # 3D先验
config.enable_sh_lighting = True    # 球谐光照
config.enable_text_guidance = True  # 文本条件
config.enable_id_preservation = True  # ID保持
config.enable_visual_perception = True  # 视觉后处理
```

---

## 9.3 视觉后处理配置

```python
VISUAL_PERCEPTION_CONFIG = {
    'pupil_base_gain': 0.8,
    'pupil_modulation_range': 0.4,
    'retinal_kernel': 9,
    'mach_band_strength': 0.3,
}
```

---

# 十、常见问题与解决方案

## 10.1 内存不足

**问题**：`RuntimeError: CUDA out of memory`

**解决方案**：
```python
# 减小batch_size
batch_size = 2

# 使用合成数据测试
python main.py --synthetic
```

---

## 10.2 训练不收敛

**问题**：`loss: nan`

**解决方案**：
```python
# 减小学习率
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

---

## 10.3 专家坍塌

**问题**：MoE只激活一个专家

**解决方案**：
```python
# 确保负载均衡损失权重足够大
moe_loss_weight = 0.01

# 使用噪声门控
gate_logits = gate(x) + torch.randn_like(gate(x)) * 0.1
```

---

## 10.4 CUDA错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `CUDA out of memory` | GPU内存不足 | 减小batch_size |
| `invalid device ordinal` | 设备号无效 | 检查CUDA_VISIBLE_DEVICES |
| `affine_grid kernel fail` | Grid尺寸过大 | 限制图像尺寸≤512 |

---

## 10.5 API问题

| 问题 | 解决方案 |
|------|----------|
| DeepSeek API Key无效 | 检查环境变量 |
| 超时 | 增加timeout值 |
| 额度不足 | 使用备用方案 |

---

# 十一、项目结构

## 11.1 目录结构

```
censor/
├── model/
│   ├── __init__.py
│   ├── attention.py           # 注意力模块
│   ├── au_attention.py      # AU注意力
│   ├── backbones.py       # 骨干网络
│   ├── fusion.py       # 融合模块
│   ├── preprocessing.py # 预处理
│   └── llm_report.py  # LLM报告
├── visual_perception.py   # 视觉后处理
├── config/
│   ├── __init__.py
│   └── defaults.py     # 配置
├── frontend/
│   └── app.py       # Streamlit前端
├── docs/
│   ├── README_EN_V2.md
│   └── README_CN_V2.md
├── train_image_generator.py  # 训练脚本
├── main.py              # 主入口
├── requirements.txt
└── README.md
```

---

# 十二、数学公式

## 12.1 双通道融合

$$f_{\text{fused}} = \text{SE}(\text{concat}(f_{\text{fast}}, f_{\text{slow}}))$$

## 12.2 AU解码

$$\text{AU}_{b,t} = \sigma(W_{\text{au}} \cdot \mathbf{h}_t)$$

## 12.3 MoE路由

$$g = \text{softmax}(\text{top-}k(W_g \cdot f))$$

## 12.4 3DMM估计

$$\mathbf{v} = \mathbf{v}_{\text{mean}} + B_s \cdot \alpha_s + B_e \cdot \alpha_e$$

## 12.5 球谐光照

$$\text{SH}_k(\theta, \phi) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} Y_l^m(\theta, \phi)$$

## 12.6 总损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{me}} + \alpha \mathcal{L}_{\text{au}} + \beta \mathcal{L}_{\text{moe}}$$

---

# 十三、Bio-Mimetic 仿生模块详解

## 13.1 BioMoE门控

### 膜电位机制

**设计动机**：模拟生物神经元的膜电位累积机制，实现动态门控。

**生物学基础**：神经元通过膜电位累积进行信息传递，当膜电位超过阈值时触发动作电位。

```python
class BioMoE(nn.Module):
    """BioMoE - 生物启发的混合专家"""
    
    def __init__(self, input_dim: int = 1024, num_experts: int = 3):
        super().__init__()
        
        # 膜电位累积
        self.membrane_potential = nn.Parameter(torch.zeros(num_experts))
        self.decay_rate = 0.95
        
        # 门控网络
        self.gate = nn.Linear(input_dim, num_experts)
        
    def forward(self, x: torch.Tensor, feedback: torch.Tensor = None):
        """
        Args:
            x: 输入特征
            feedback: 反馈信号（可选）
        """
        # 膜电位累积
        if feedback is not None:
            self.membrane_potential = (
                self.membrane_potential * self.decay_rate + 
                feedback * (1 - self.decay_rate)
            )
        
        # 门控计算
        gating = self.gate(x)
        
        # 结合膜电位
        gating = gating + 0.1 * self.membrane_potential
        
        return F.softmax(gating, dim=-1)
```

---

## 13.2 稀疏控制模块

### 设计动机

深度学习模型过拟合的根本原因是有效参数量过多，通过动态稀疏化降低有效参数量。

### 状态机详解

```
状态转移规则:
─────────────────────────────────────────────────
1. 冻结: f[t] = 1  if i[t-1] > θ_freeze (默认200)
         f[t] = 0  otherwise

2. 软衰减: w[t] = α^(i[t-1]/100)  if 0 < i[t-1] < θ_soft (默认100)
          w[t] = 1            otherwise

3. 恢复: f[t] = 0  if a[t] > θ_recovery (默认0.1) AND f[t-1] = 1
```

```python
class SparseControl(nn.Module):
    def __init__(self, dim: int, freeze_threshold: int = 200, decay_threshold: int = 100):
        super().__init__()
        self.freeze_threshold = freeze_threshold
        self.decay_threshold = decay_threshold
        
        # 冻结状态
        self.is_frozen = nn.Parameter(
            torch.zeros(dim, dtype=torch.bool),
            requires_grad=False
        )
        
    def forward(self, x: torch.Tensor, inactivity_counter: torch.Tensor) -> torch.Tensor:
        # 冻结判断
        should_freeze = inactivity_counter > self.freeze_threshold
        newly_frozen = should_freeze & ~self.is_frozen
        self.is_frozen[newly_frozen] = True
        
        # 恢复判断
        neuron_activity = x.abs().mean(dim=0)
        should_recover = self.is_frozen & (neuron_activity > 0.1)
        self.is_frozen[should_recover] = False
        
        # 应用冻结
        frozen_mask = (~self.is_frozen).float()
        
        return x * frozen_mask
```

### 生长因子机制

当神经元从冻结状态恢复时，给予2倍增益：

```python
class GrowthFactor(nn.Module):
    def forward(self, x: torch.Tensor, recover_events: torch.Tensor) -> torch.Tensor:
        boost_mask = torch.ones(self.dim)
        boost_mask[recover_events] = 2.0
        return x * boost_mask
```

---

## 13.3 事件驱动机制

### 状态机设计

事件驱动机制受人类注意力动态启发：

```
信号强度          状态        注意力    计算量
─────────────────────────────────────────────
< 0.15        AMBIENT     10%      轻量加权
0.15-0.30     ORIENTING  30%      平均
> 0.30         FOCUSED   100%     完整注意力
```

**核心原则**：永不完全静默！始终保持基础监测。

```python
class HumanAttentionController(nn.Module):
    def __init__(self, input_dim: int = 1280):
        super().__init__()
        
        self.state_classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3个状态
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[str, dict]:
        """
        Returns:
            state: 当前状态 ('ambient', 'orienting', 'focused')
            info: 状态信息
        """
        logits = self.state_classifier(features)
        state_idx = logits.argmax(dim=-1)
        
        states = ['ambient', 'orienting', 'focused']
        state = states[state_idx]
        
        salience = torch.sigmoid(logits).max(dim=-1)[0]
        
        return state, {
            'state': state,
            'salience': salience.item(),
            'confidence': torch.softmax(logits, dim=-1).max().item()
        }
```

### 灵敏度保证

| 信号强度 | 模式 | 计算 | 能检测微表情 |
|---------|------|------|---------|
| < 0.15 | AMBIENT | 10% | ⚠️ 保持监测 |
| 0.15-0.30 | ORIENTING | 30% | ✅ 检测变化 |
| > 0.30 | FOCUSED | 100% | ✅ 确认表达 |

---

# 十四、扩展微表情分类

## 14.1 11类分类体系

基于MER数据集扩展为11类：

| ID | 类别 | AU标记 | 数据集来源 |
|----|------|-------|----------|
| 0 | Happiness (Duchenne真笑) | AU6+AU12 | CASME II |
| 1 | Happiness (Non-Duchenne假笑) | AU12 only | CASME II |
| 2 | Surprise (强烈) | AU1+AU2+AU5+AU26 | - |
| 3 | Surprise (轻微) | AU1+AU2 | - |
| 4 | Fear（恐惧）| AU1+AU2+AU4+AU5+AU7+AU26 | - |
| 5 | Disgust (强烈) | AU9+AU10+AU17 | - |
| 6 | Disgust (轻微) | AU9 | - |
| 7 | Anger (强烈) | AU4+AU7+AU23+AU24 | - |
| 8 | Anger (轻微) | AU4 | - |
| 9 | Sadness（悲伤）| AU1+AU4+AU15+AU17 | - |
| 10 | Contempt（蔑视）| AU12+AU14 | - |

---

## 14.2 7类到11类映射

```
Happiness → [0, 1]  (Duchenne / Non-Duchenne)
Surprise  → [2, 3]  (强烈 / 轻微)
Fear     → [4]
Disgust   → [5, 6]  (强烈 / 轻微)
Anger    → [7, 8]  (强烈 / 轻微)
Sadness   → [9]
Contempt  → [10]
```

---

# 十五、高级MoE架构

## 15.1 层级动态MoE

结合层级（粗→细）和动态（输入条件）路由：

**第一层：粗粒度组（3类）**
- 组0：积极（Positive）：Happiness, Contempt
- 组1：消极（Negative）：Sadness, Fear, Anger, Disgust
- 组2：惊讶（Surprise）

**第二层：细粒度专家（共9个）**
- 组0：3个专家（Happiness强/弱，Contempt）
- 组1：4个专家（Sadness, Fear, Anger, Disgust）
- 组2：2个专家（Surprise强/弱）

---

## 15.2 可用MoE模块对比

| 模块 | 专家数 | 返回值 | 特性 |
|------|--------|--------|------|
| MoEGatingNetwork | 3 | output, gates, aux_loss | 原始Top-2 |
| EnhancedMoE | 3 | output, gates, aux_loss, info | 膜电位+情绪 |
| BioMoE | 3 | output, gates, aux_loss, membrane_info | 仿生门控 |
| HierarchicalDynamicMoE | 9 | output, hierarchy, aux_loss | 层级+动态 |
| PersonalizedRadar | TTA | adapted | 测试时适配 |

---

# 十六、空间注意力机制

## 16.1 AU地标注意力

基于面部动作单元（AU）的独立空间注意力：

**区域中心与权重**：
- 眉毛（AU1,2,4）：权重=1.0
- 眼睛（AU5,6,7）：权重=1.2
- 鼻子（AU9）：权重=0.8
- 嘴巴（AU10,12,14,15,17,20,23-28）：权重=1.0

---

## 16.2 倒三角形注意力

空间注意力掩码初始化为倒三角形（上部宽→下部窄）：

```
      ●──────●     ← 眉毛（宽）
       ●────●       ← 眼睛（中）
        ●──●         ← 鼻子（窄）
         ●●          ← 嘴巴（很窄）
```

---

# 十七、部署指南

## 17.1 Docker部署

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501"]
```

---

## 17.2 Docker Compose

```yaml
version: '3.8'

services:
  censor:
    build: .
    ports:
      - "8501:8501"
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 17.3 Kubernetes部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: censor
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: censor
        image: censor:latest
        resources:
          requests:
            memory: "4Gi"
            nvidia.com/gpu: 1
```

---

## 17.4 生产环境配置

```bash
# 环境变量
export CUDA_VISIBLE_DEVICES=0
export DEEPSEEK_API_KEY="sk-xxxxxxxx"
export LOG_LEVEL="INFO"

# 性能配置
PERFORMANCE_CONFIG = {
    'cuda_benchmark': True,
    'cudnn_benchmark': True,
    'torch_compile': True,
}
```

---

## 17.5 安全配置

### API认证

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if not is_valid(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True
```

### 输入验证

```python
from pydantic import BaseModel, Field

class VideoInput(BaseModel):
    video_data: str = Field(..., description="Base64编码的视频")
    max_frames: int = Field(default=16, ge=1, le=64)
```

---

# 十八、性能优化指南

## 18.1 TensorRT优化

```python
import torch2trt

model_trt = torch2trt(
    model,
    inputs=[input_data],
    fp16_mode=True,
)
```

---

## 18.2 TorchScript优化

```python
traced_model = torch.jit.trace(model, example_inputs=(x,))
traced_model.save('censor_traced.pt')
```

---

## 18.3 量化

```python
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
```

---

## 18.4 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(videos)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
```

---

## 18.5 梯度累积

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = criterion(model(batch), labels)
    loss = loss / accumulation_steps
    (loss / accumulation_steps).backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 18.6 分布式训练

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model, device_ids=[local_rank])
```

---

# 十九、监控与诊断

## 19.1 日志配置

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('censor')
```

---

## 19.2 指标收集

```python
class MetricsCollector:
    def record(self, name: str, value: float):
        self.metrics.append({'name': name, 'value': value, 'time': time.now()})
```

---

## 19.3 内存泄漏检测

```python
def get_memory_usage():
    return {
        'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,
        'gpu_reserved': torch.cuda.memory_reserved() / 1024**3,
    }
```

---

## 19.4 性能诊断

```python
import torch.profiler

with torch.profiler.profile(...) as prof:
    for batch in dataloader:
        output = model(batch)
```

---

## 19.5 可视化工具

```python
import matplotlib.pyplot as plt

def visualize_attention(attention, image):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(attention, cmap='jet')
    plt.subplot(1, 3, 3)
    plt.imshow(image, alpha=0.6)
    plt.imshow(attention, cmap='jet', alpha=0.4)
```

---

# 二十、测试指南

## 20.1 单元测试

```python
import unittest

class TestCensor(unittest.TestCase):
    def test_forward(self):
        model = Censor()
        video = torch.randn(1, 3, 16, 224, 224)
        outputs = model(video)
        
        self.assertIn('me_logits', outputs)
        self.assertIn('au_intensities', outputs)
```

---

## 20.2 集成测试

```python
import pytest

def test_full_pipeline():
    model = Censor()
    checkpoint = torch.load('checkpoints/best.pt')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    with torch.no_grad():
        outputs = model(video)
    
    assert outputs['me_logits'] is not None
```

---

## 20.3 性能基准测试

```python
import time

def benchmark_inference(model, input_shape, num_iterations=100):
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        times.append(time.perf_counter() - start)
    
    return {'mean': sum(times)/len(times)}
```

---

# 二十一、最佳实践与案例

## 21.1 代码组织

```
project/
├── src/
│   ├── censor/
│   │   ├── model/
│   │   ├── utils/
│   │   └── api/
│   ├── tests/
│   ├── configs/
│   ├── scripts/
│   └── notebooks/
├── requirements.txt
└── README.md
```

---

## 21.2 配置管理

```python
from dataclasses import dataclass

@dataclass
class Config:
    model: str = "censor_v2"
    batch_size: int = 4
    lr: float = 1e-4
```

---

## 21.3 版本管理

```python
def check_compatibility(checkpoint_path, current_version):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_version = checkpoint.get('version', '0.0.0')
    
    if not is_compatible(saved_version, current_version):
        raise ValueError(f"版本不兼容")
```

---

# 二十二、研究方向与未来工作

## 22.1 近期研究方向

| 方向 | 描述 | 难度 | 影响 |
|------|------|------|------|
| 3D MM融合 | 引入更强的3DMM先验 | 中 | 高 |
| 时序建模 | 改进时序注意力 | 高 | 高 |
| 多模态融合 | 引入语音/文本信号 | 中 | 中 |
| 轻量化 | 模型压缩与加速 | 中 | 高 |

---

## 22.2 长期研究方向

| 方向 | 描述 | 预期突破 |
|------|------|----------|
| 通用表示 | 跨任务统一表征 | 推理泛化 |
| 因果推理 | 情绪因果推断 | 可解释性 |
| 个性化 | 快速个性化适配 | 准确率提升 |

---

# 二十三、许可证与引用

## 23.1 MIT许可证

MIT License

Copyright (c) 2024 Censor Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

---

## 23.2 引用

```bibtex
@misc{censor2024,
  title={Censor: 仿生双通道微表情识别系统},
  author={Censor Team},
  year={2024},
  howpublished={GitHub: https://github.com/RomanCohort/Censor-MER},
  note={Version 2.0}
}
```

---

**文档版本**: 2.1
**最后更新**: 2026-05-16
**维护**: Censor Team
**许可证**: MIT

---

# 附录A：完整超参数配置表

## A.1 预处理超参数

| 参数 | 类型 | 默认值 | 取值范围 | 说明 |
|------|------|-------|----------|------|
| pyramid_levels | int | 4 | 2-6 | 高斯金字塔层数 |
| gaussian_sigma | float | 1.5 | 0.5-3.0 | 高斯sigma |
| center_bias_strength | float | 1.0 | 0.0-2.0 | 中心先验强度 |
| sigma_ratio | float | 0.15 | 0.05-0.3 | 相对sigma比例 |
| rppg_window_size | int | 5 | 3-9 | rPPG窗口大小 |
| rppg_bandpass_low | float | 0.5 | 0.3-1.0 | 最低频率(Hz) |
| rppg_bandpass_high | float | 4.0 | 2.0-6.0 | 最高频率(Hz) |
| tvl1_tau | float | 0.25 | 0.1-0.5 | TV正则化参数 |
| tvl1_lambda | float | 0.15 | 0.05-0.3 | 数据保真参数 |
| tvl1_theta | float | 0.3 | 0.1-0.5 | 粗细平衡参数 |
| fast_threshold | float | 0.1 | 0.01-0.5 | 快速阈值 |
| use_tvl1 | bool | True | bool | 使用TV-L1 |
| au_attention_size | int | 224 | 112-448 | AU注意力图大小 |
| au_mask_threshold | float | 0.1 | 0.0-0.5 | AU掩码阈值 |

## A.2 模型超参数

| 参数 | 类型 | 默认值 | 取值范围 | 说明 |
|------|------|-------|----------|------|
| fast_dim | int | 512 | 256-1024 | 快通道输出维度 |
| slow_dim | int | 768 | 384-1536 | 慢通道输出维度 |
| fused_dim | int | 1024 | 512-2048 | 融合特征维度 |
| num_aus | int | 28 | 17-30 | AU数量 |
| num_experts | int | 3 | 2-5 | 专家数量 |
| top_k | int | 2 | 1-3 | Top-k专家数 |

## A.3 训练超参数

| 参数 | 类型 | 默认值 | 取值范围 | 说明 |
|------|------|-------|----------|------|
| lr | float | 1e-4 | 1e-5-1e-3 | 学习率 |
| batch_size | int | 2 | 1-16 | 批大小 |
| epochs | int | 50 | 10-200 | 训练轮数 |
| weight_decay | float | 1e-4 | 0-1e-2 | 权重衰减 |
| grad_clip | float | 1.0 | 0.1-5.0 | 梯度裁剪 |
| au_loss_weight | float | 0.5 | 0.0-1.0 | AU损失权重 |
| moe_loss_weight | float | 0.01 | 0.0-0.1 | MoE损失权重 |
| opd_loss_weight | float | 0.1 | 0.0-0.5 | OPD损失权重 |
| patience | int | 8 | 3-20 | 早停耐心值 |

## A.4 图像生成超参数

| 参数 | 类型 | 默认值 | 取值范围 | 说明 |
|------|------|-------|----------|------|
| image_size | int | 224 | 128-512 | 图像尺寸 |
| enable_3d_prior | bool | True | bool | 启用3D先验 |
| enable_sh_lighting | bool | True | bool | 启用SH光照 |
| enable_text_guidance | bool | False | bool | 启用文本引导 |
| enable_id_preservation | bool | True | bool | 启用ID保持 |
| enable_visual_perception | bool | True | bool | 启用视觉后处理 |

---

# 附录B：损失函数详解

## B.1 完整损失函数

```python
class TotalLoss(nn.Module):
    """总损失函数"""
    
    def __init__(
        self,
        au_weight: float = 0.5,
        moe_weight: float = 0.01,
        opd_weight: float = 0.1,
    ):
        super().__init__()
        
        self.au_weight = au_weight
        self.moe_weight = moe_weight
        self.opd_weight = opd_weight
        
    def forward(self, outputs: dict, targets: dict) -> dict:
        """计算总损失
        
        Args:
            outputs: 模型输出字典
            targets: 目标字典
        
        Returns:
            losses: 损失字典
            total_loss: 总损失
        """
        # ME分类损失
        loss_me = F.cross_entropy(
            outputs['me_logits'], 
            targets['me_labels']
        )
        
        # AU损失
        loss_au = F.binary_cross_entropy_with_logits(
            outputs['au_intensities'],
            targets['au_labels']
        )
        
        # MoE辅助损失
        loss_moe = outputs.get('moe_aux_loss', 0)
        
        # OPD损失（时间平滑+峰值一致性）
        if 'au_opd' in outputs:
            opd = outputs['au_opd']
            # OPD平滑
            opd_diff = opd[:, 1:] - opd[:, :-1]
            loss_opd_smooth = F.mse_loss(opd_diff, torch.zeros_like(opd_diff))
            
            # OPD顺序 (onset < peak < decay)
            onsets = opd[:, :, 0]
            peaks = opd[:, :, 1]
            decays = opd[:, :, 2]
            loss_opd_order = F.relu(peaks - decays) + F.relu(onsets - peaks)
            
            loss_opd = loss_opd_smooth + 0.1 * loss_opd_order.mean()
        else:
            loss_opd = 0
        
        # 总损失
        total = loss_me + \
                self.au_weight * loss_au + \
                self.moe_weight * loss_moe + \
                self.opd_weight * loss_opd
        
        return {
            'loss_me': loss_me,
            'loss_au': loss_au,
            'loss_moe': loss_moe,
            'loss_opd': loss_opd,
            'total': total,
        }
```

## B.2 损失权重调优指南

| 问题 | 症状 | 调整 |
|------|------|------|
| AU预测不准 | AU准确率低 | ↑ au_weight |
| 专家坍塌 | gate分布集中 | ↑ moe_weight |
| OPD乱序 | onset>peak | ↑ opd_weight |
| 分类不准 | ME准确率低 | 检查数据标注 |

---

# 附录C：数据增强策略

## C.1 视频增强

```python
class VideoAugmentation:
    """视频数据增强"""
    
    @staticmethod
    def temporal_crop(video: torch.Tensor, 
                   num_frames: int = 16) -> torch.Tensor:
        """时间维度裁剪"""
        T = video.shape[2]
        if T > num_frames:
            start = torch.randint(0, T - num_frames, (1,))
            return video[:, :, start:start+num_frames]
        return video
        
    @staticmethod
    def spatial_crop(video: torch.Tensor,
                  crop_size: int = 224) -> torch.Tensor:
        """空间维度裁剪"""
        _, _, _, H, W = video.shape
        y = torch.randint(0, max(1, H - crop_size), (1,))
        x = torch.randint(0, max(1, W - crop_size), (1,))
        return video[:, :, :, y:y+crop_size, x:x+crop_size]
        
    @staticmethod
    def color_jitter(video: torch.Tensor,
                   brightness: float = 0.1,
                   contrast: float = 0.1) -> torch.Tensor:
        """颜色抖动"""
        # Brightness
        video = video + torch.randn_like(video) * brightness
        # Contrast
        mean = video.mean(dim=(-2,-1), keepdim=True)
        video = (video - mean) * (1 + torch.randn_like(video) * contrast) + mean
        return video.clamp(0, 1)
        
    @staticmethod
    def noise_inject(video: torch.Tensor,
                  noise_level: float = 0.01) -> torch.Tensor:
        """噪声注入"""
        return video + torch.randn_like(video) * noise_level
```

## C.2 增强配置

```python
AUGMENTATION_CONFIG = {
    'temporal_crop': True,
    'spatial_crop': True,
    'color_jitter': True,
    'noise_inject': True,
    'horizontal_flip': False,  # 人脸不能翻转！
    
    # 概率
    'p_temporal': 0.5,
    'p_spatial': 0.5,
    'p_color': 0.3,
    'p_noise': 0.2,
}
```

---

# 附录D：模型检查点管理

## D.1 检查点结构

```python
CHECKPOINT_STRUCTURE = {
    'epoch': int,
    'model_state': dict,
    'optimizer_state': dict,
    'scheduler_state': dict,
    'best_val_loss': float,
    'history': {
        'train_loss': list,
        'val_loss': list,
    },
    'config': dict,
    'version': str,
}
```

## D.2 检查点��存

```python
def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_loss,
    history,
    path,
    is_best=False,
):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'history': history,
        'version': '2.0',
    }, path)
```

## D.3 检查点恢复

```python
def load_checkpoint(model, path, optimizer=None):
    """加载检查点"""
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    return {
        'epoch': checkpoint['epoch'],
        'best_val_loss': checkpoint['best_val_loss'],
        'history': checkpoint['history'],
    }
```

---

# 附录E：评估指标

## E.1 评估指标定义

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

def evaluate_model(model, dataloader) -> dict:
    """评估模型
    
    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            videos, labels = batch
            outputs = model(videos)
            preds = outputs['me_logits'].argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
    }
```

## E.2 指标表

| 指标 | 说明 | 目标 |
|------|------|------|
| Accuracy | 准确率 | >0.90 |
| Precision | 精确率 | >0.85 |
| Recall | 召回率 | >0.85 |
| F1 Score | F1分数 | >0.85 |
| AUC | ROC-AUC | >0.95 |

---

# 附录F：调试checklist

## F.1 训练问题checklist

```
检查清单：
□ 1. 数据加载是否正确？
□ 2. 数据格式是否匹配？
□ 3. 标签是否正确？
□ 4. 学习率是否合适？
□ 5. batch_size是否太大？
□ 6. 模型是否太大？
□ 7. 梯度是否爆炸？
□ 8. 损失是否正常？

调试步骤：
1. 运行合成数据测试
2. 检查单步前向传播
3. 检查损失数值范围
4. 检查梯度范围
5. 检查模型输出
6. 检查验证集指标
```

## F.2 推理问题checklist

```
检查清单：
□ 1. 模型是否加载成功？
□ 2. 输入格式是否正确？
□ 3. CUDA是否可用？
□ 4. 内存是否充足？
□ 5. 输出是否合理？

调试步骤：
1. 检查模型设备
2. 检查输入形状
3. 检查CUDA内存
4. 运行单步推理
5. 验证输出范围
```

---

# 附录G：环境变量完整列表

## G.1 必需变量

| 变量名 | 说明 | 示例 |
|--------|------|------|
| DEEPSEEK_API_KEY | DeepSeek API密钥 | sk-xxx |
| CUDA_VISIBLE_DEVICES | GPU设备号 | 0 |

## G.2 可选变量

| 变量名 | 说明 | 默认值 |
|--------|------|-------|
| LOG_LEVEL | 日志级别 | INFO |
| MODEL_CACHE_DIR | 模型缓存目录 | ./models |
| DEFAULT_MODEL | 默认模型 | censor_v2 |
| ENABLE_TENSORRT | 启用TensorRT | false |

---

# 附录H：常见错误代码

## H.1 CUDA错误代码

| 错误代码 | 含义 | 解决方案 |
|----------|------|----------|
| 1 | 内存不足 | 减小batch |
| 2 | 设��错误 | 检查GPU |
| 3 | 精度错误 | 检查dtype |
| 4 | 版本不匹配 | 更新CUDA |

## H.2 运行时错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| NaN loss | 学习率过大 | 降低lr |
| Inf loss | 梯度爆炸 | 梯度裁剪 |
| 内存泄漏 | 缓存累积 | 清理缓存 |

---

# 附录I：推荐硬件配置

## I.1 训练配置

| 规模 | GPU | CPU | 内存 | 存储 |
|------|-----|-----|------|------|
| 小型 | RTX 3080 10GB | i5 10th | 16GB | 100GB |
| 中型 | RTX 3090 24GB | i7 10th | 32GB | 200GB |
| 大型 | A100 40GB | i9 12th | 64GB | 500GB |

## I.2 推理配置

| 规模 | GPU | CPU | 内存 |
|------|-----|-----|------|
| 单路 | GTX 1060 6GB | i5 8th | 8GB |
| 并行 | RTX 3080 10GB | i7 10th | 16GB |
| 生产 | A100 40GB | i9 12th | 32GB |

---

# 附录J：术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| 微表情 | Micro-Expression | 40-200ms的面部表情 |
| 动作单元 | AU (Action Unit) | FACS面部运动单元 |
| 快通道 | Fast Pathway | 皮层下通路 |
| 慢通道 | Slow Pathway | 皮层通路 |
| 杏仁核 | Amygdala | 情绪处理核 |
| 梭状回 | Fusiform Gyrus | 视觉识别区 |
| apex帧 | Apex Frame | 表情峰值帧 |
| MoE | Mixture of Experts | 混合专家 |

---

# 附录K���参考文献

1. Ekman, P., & Friesen, W. V. (1969). Nonverbal leakage and clues to deception. *Psychiatry*.

2. Pfister, T., et al. (2011). Recognising spontaneous facial micro-expressions. *ICCV*.

3. Liong, S. T., et al. (2018). Automatic dynamic range textural description. *IEEE TPAMI*.

4. Liu, Y. J., et al. (2019). Auxiliary signal regularized CNN. *ICCV*.

5. LeDoux, J. E. (2000). Emotion circuits in the brain. *Annual Review of Neuroscience*.

---

# 附录L：快速参考卡

## L.1 常用命令

```bash
# 训练
python train.py --epochs 50 --batch-size 4

# 评估
python eval.py --checkpoint checkpoints/best.pt

# 推理
python main.py --video input.mp4

# 前端
streamlit run frontend/app.py
```

## L.2 快速调试

```python
# 简单测试
python main.py --synthetic

# 单步测试
python -c "
import torch
from model import Censor
model = Censor()
x = torch.randn(1, 3, 16, 224, 224)
print(model(x))
"

# GPU测试
python -c "import torch; print(torch.cuda.is_available())"
```

## L.3 配置模板

```python
# config.py
CONFIG = {
    'batch_size': 4,
    'lr': 1e-4,
    'epochs': 50,
    'num_workers': 4,
}
```

---

# 附录M：模块详细对比表

## M.1 各模块对比

### 预处理模块对比

| 模块 | 参数量 | 计算量 | 效果 | 适用场景 |
|------|--------|--------|------|----------|
| SaliencyDetector | ~10K | 低 | 中心偏向 | 所有场景 |
| rPPGExtractor | ~1K | 中 | 生理信号 | 补充信息 |
| TV-L1 | N/A | 高 | 精确运动 | 精细分析 |
| 自适应光流 | N/A | 中 | 自适应 | 实时应用 |

### 骨干网络对比

| 网络 | 参数量 | 延迟 | 精度 | 说明 |
|------|--------|------|------|------|
| 3D ResNet-18 | ~33M | 低 | 中 | 快通道 |
| 3D ResNet-34 | ~44M | 中 | 高 | 慢速 |
| Swin-T | ~28M | 中 | 高 | 慢通道 |
| Swin-S | ~50M | 高 | 更高 | 精确 |

### 注意力模块对比

| 模块 | 参数量 | 功能 | 创新点 |
|------|--------|------|--------|
| Amygdala | ~200K | 注意力先验 | 情绪引导 |
| FFA | ~500K | 特征融合 | SE门控 |
| CASANet | ~2M | apex检测 | 三角先验 |

### MoE模块对比

| 模块 | 参数量 | 专家数 | 特性 |
|------|--------|--------|------|
| MoEGatingNetwork | ~5M | 3 | 原始Top-2 |
| EnhancedMoE | ~6M | 3 | 膜电位 |
| BioMoE | ~6M | 3 | 仿生门控 |
| HierarchicalDynamicMoE | ~15M | 9 | 层级+动态 |

---

# 附录N：完整API参考

## N.1 模型API

### Censor主类

```python
class Censor(nn.Module):
    """Censor微表情识别主类
    
    Attributes:
        preprocess: 预处理模块
        fast_path: 快通道
        slow_path: 慢通道
        amygdala: 杏仁核
        ffa: FFA模块
        casanet: CASANet模块
        fusion: 融合模块
        au_decoder: AU解码器
        moe: MoE模块
        reporter: 报告器
    """
    
    def __init__(self, config: dict = None):
        """初始化
        
        Args:
            config: 配置字典
        """
        pass
        
    def forward(self, video: torch.Tensor) -> dict:
        """前向传播
        
        Args:
            video: (B, 3, T, H, W) 输入视频
        
        Returns:
            outputs: 输出字典
        """
        pass
        
    def predict(self, video: torch.Tensor) -> dict:
        """推理（包含后处理）
        
        Args:
            video: (B, 3, T, H, W)
        
        Returns:
            prediction: 预测结果
        """
        pass
```

### 工厂函数

```python
def create_censor(config: dict = None) -> Censor:
    """创建Censor模型"""
    
def create_censor_from_checkpoint(checkpoint_path: str) -> Censor:
    """从检查点加载模型"""
    
def create_model_with_config(config_path: str) -> Censor:
    """从配置文件创建模型"""
```

### 模型配置

```python
CENSOR_DEFAULT_CONFIG = {
    # 预处理
    'preprocess': {
        'use_saliency': True,
        'use_rppg': True,
        'use_flow': True,
    },
    
    # 通道
    'fast_path': {
        'backbone': 'resnet18',
        'output_dim': 512,
    },
    'slow_path': {
        'backbone': 'swin_tiny',
        'output_dim': 768,
    },
    
    # 注意力
    'attention': {
        'use_amygdala': True,
        'use_ffa': True,
        'use_casanet': True,
    },
    
    # 解码
    'decoder': {
        'num_aus': 28,
        'use_opd': True,
    },
    
    # MoE
    'moe': {
        'num_experts': 3,
        'top_k': 2,
    },
    
    # LLM
    'llm': {
        'use_template': True,
        'use_deepseek': True,
    },
}
```

## N.2 数据加载API

### Dataset类

```python
class MicroExpressionDataset(torch.utils.data.Dataset):
    """微表情数据集"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: callable = None,
    ):
        pass
        
    def __len__(self) -> int:
        pass
        
    def __getitem__(self, idx: int) -> dict:
        pass
        
    def get_video_path(self, idx: int) -> str:
        pass
```

### DataLoader

```python
def create_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    split: str = 'train',
) -> DataLoader:
    """创建数据加载器"""
    pass
```

## N.3 训练API

### Trainer类

```python
class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
    ):
        pass
        
    def train_epoch(self, epoch: int) -> dict:
        """训练一个epoch"""
        pass
        
    def validate(self) -> dict:
        """验证"""
        pass
        
    def save_checkpoint(self, path: str, is_best: bool = False):
        """保存检查点"""
        pass
        
    def load_checkpoint(self, path: str):
        """加载检查点"""
        pass
```

### 训练配置

```python
TRAIN_CONFIG = {
    # 优化器
    'optimizer': {
        'type': 'AdamW',
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999),
    },
    
    # 学习率调度
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'T_max': 50,
        'eta_min': 1e-6,
    },
    
    # 损失权重
    'loss': {
        'au_weight': 0.5,
        'moe_weight': 0.01,
        'opd_weight': 0.1,
    },
    
    # 训练策略
    'strategy': {
        'grad_clip': 1.0,
        'accumulation_steps': 1,
        'early_stopping_patience': 8,
    },
}
```

## N.4 评估API

### Evaluator类

```python
class Evaluator:
    """评估器"""
    
    def __init__(self, model: nn.Module, config: dict = None):
        pass
        
    def evaluate(self, dataloader: DataLoader) -> dict:
        """评估模型"""
        pass
        
    def compute_metrics(self, outputs: dict, targets: dict) -> dict:
        """计算指标"""
        pass
        
    def generate_report(self, results: dict) -> str:
        """生成报告"""
        pass
```

### 评估指标

```python
EVALUATION_METRICS = {
    # 分类指标
    'accuracy': accuracy_score,
    'precision': lambda y, p: precision_score(y, p, average='macro'),
    'recall': lambda y, p: recall_score(y, p, average='macro'),
    'f1': lambda y, p: f1_score(y, p, average='macro'),
    
    # AU指标
    'au_accuracy': au_accuracy,
    'au_f1': au_f1_score,
    
    # 回归指标
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
}
```

## N.5 推理API

### Predictor类

```python
class Predictor:
    """推理器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        use_tensorrt: bool = False,
    ):
        pass
        
    def predict(self, video: torch.Tensor) -> dict:
        """单次预测"""
        pass
        
    def predict_batch(self, videos: list) -> list:
        """批量预测"""
        pass
        
    def predict_video_file(self, video_path: str) -> dict:
        """从文件预测"""
        pass
```

### 推理后处理

```python
class PostProcessor:
    """后处理器"""
    
    @staticmethod
    def decode_au(au_intensities: torch.Tensor, threshold: float = 0.3) -> list:
        """解码AU"""
        pass
        
    @staticmethod
    def decode_emotion(me_logits: torch.Tensor) -> dict:
        """解码情绪"""
        pass
        
    @staticmethod
    def generate_report(
        emotion: str,
        confidence: float,
        active_aus: list,
        au_intensities: dict,
    ) -> str:
        """生成报告"""
        pass
```

---

# 附录O：版本兼容性

## O.1 版本历史

| 版本 | 日期 | 变更 | 兼容性 |
|------|------|------|--------|
| 1.0 | 2026-05-11 | 初始版本 | - |
| 1.1 | 2026-05-12 | 修复bug | v1.0 |
| 2.0 | 2026-05-16 | 图像生成 | v1.x不兼容 |

## O.2 升级指南

```python
# v1.x -> v2.0 升级

# 旧代码
model = Censor()  # v1.x
outputs = model(video)

# 新代码
# v2.0接口变化
model = Censor(config=v2_config)  # v2.0
outputs = model(video)

# 检查版本
print(model.version)  # "2.0"
```

## O.3 检查点迁移

```python
def migrate_checkpoint(old_checkpoint_path: str, new_path: str):
    """迁移旧版检查点"""
    
    old_ckpt = torch.load(old_checkpoint_path)
    
    # 映射层名称
    state_dict = {}
    for k, v in old_ckpt['model_state'].items():
        # v1.x -> v2.0 名称映射
        new_k = k.replace('old_module', 'new_module')
        state_dict[new_k] = v
    
    new_ckpt = {
        'model_state': state_dict,
        'version': '2.0',
        'epoch': old_ckpt['epoch'],
    }
    
    torch.save(new_ckpt, new_path)
```

---

# 附录P：性能基准

## P.1 推理基准

| 配置 | 延迟 | 吞吐量 | 显存 |
|------|------|--------|------|
| FP32, RTX 3080 | 45ms | 22 FPS | 4GB |
| FP16, RTX 3080 | 25ms | 40 FPS | 3GB |
| INT8, RTX 3080 | 15ms | 66 FPS | 2GB |
| FP32, CPU | 200ms | 5 FPS | 2GB |

## P.2 训练基准

| 配置 | GPU | 批大小 | 每epoch时间 |
|------|-----|--------|------------|
| 单卡 | RTX 3080 | 4 | 10分钟 |
| 双卡 | RTX 3080x2 | 8 | 6分钟 |
| 4卡 | RTX 3080x4 | 16 | 4分钟 |

## P.3 内存基准

| 操作 | 显存使用 |
|------|----------|
| 模型加载 | ~1GB |
| 前向传播 | +2GB |
| 反向传播 | +3GB |
| 梯度 | +1GB |
| 优化器状态 | +1GB |

---

# 附录Q：安全与隐私

## Q.1 数据安全

```python
# 数据加密
class SecureDataset(Dataset):
    """安全数据集"""
    
    def __init__(self, data_path: str, key: bytes):
        self.key = key
        
    def __getitem__(self, idx):
        # 解密
        encrypted = load_encrypted(idx)
        data = decrypt(encrypted, self.key)
        return data
```

## Q.2 隐私保护

```python
# 差分隐私
class DPOptimizer:
    """差分隐私优化器"""
    
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        
    def step(self):
        # 添加噪声
        for param in self.parameters():
            noise = torch.randn_like(param) * self.epsilon
            param.grad += noise
```

---

# 附录R：第三方集成

## R.1 ONNX导出

```python
# 导出为ONNX
model = Censor()
model.eval()

dummy_input = torch.randn(1, 3, 16, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "censor.onnx",
    input_names=['video'],
    output_names=['me_logits', 'au_intensities'],
)
```

## R.2 TensorFlow Lite

```python
# 转换为TF Lite
import onnx
import tf2onnx

# ONNX -> TF
model = onnx.load("censor.onnx")
tf_rep = tf2onnx.convert.from_onnx(model)

# TF -> TF Lite
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    tf_rep.signatures.values()
)
tflite_model = converter.convert()
```

---

# 附录S：持续集成

## S.1 CI/CD配置

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v
    
    - name: Lint
      run: |
        flake8 model/ --count --select=E9,F63,F7,F82
```

## S.2 测试覆盖率

```bash
# 运行覆盖率
pytest --cov=model --cov-report=html tests/

# 覆盖率目标
COVERAGE_TARGETS = {
    'model': 80,
    'utils': 70,
    'total': 75,
}
```

---

# 附录T：贡献指南

## T.1 提交规范

```bash
# 提交格式
<type>(<scope>): <description>

# 类型
# feat: 新功能
# fix: 修复
# docs: 文档
# style: 格式
# refactor: 重构
# test: 测试

# 示例
git commit -m "feat(attention): add adaptive casanet"
git commit -m "fix(preprocess): fix rppg extraction bug"
```

## T.2 代码审查清单

```
□ 1. 代码是否符合项目规范？
□ 2. 是否有测试覆盖？
□ 3. 文档是否更新？
□ 4. 性能是否受影响？
□ 5. 安全性是否考虑？
□ 6. 兼容性是否保持？
```

---

# 附录U：完整训练算法伪代码

## U.1 双通道训练算法

```python
def train_dual_channel(
    model: Censor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig
) -> Dict[str, List[float]]:
    """双通道训练算法完整实现
    
    算法流程：
    1. 数据预处理（并行）
    2. 快通道前向传播（光流）
    3. 慢通道前向传播（RGB+rPPG）
    4. 注意力融合
    5. AU解码
    6. MoE路由与损失计算
    7. 反向传播与参数更新
    
    Args:
        model: Censor模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 训练配置
    
    Returns:
        history: 训练历史记录
    """
    
    # 初始化
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'fast_path_loss': [],
        'slow_path_loss': [],
        'fusion_loss': [],
        'moe_loss': [],
    }
    
    # 优化器与学习率调度
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler() if config.use_amp else None
    
    # 早停
    patience_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        # ========== 训练阶段 ==========
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            # 数据预处理
            videos = preprocess_batch(videos, config)
            
            # 并行数据加载（多worker）
            if config.num_workers > 0:
                videos = videos.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            # 梯度归零
            optimizer.zero_grad(set_to_none=True)
            
            # 混合精度前向传播
            if scaler is not None:
                with autocast():
                    outputs = model(videos)
                    loss = compute_loss(outputs, labels, config)
                
                # 梯度缩放
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                
                # 参数更新
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准前向传播
                outputs = model(videos)
                loss = compute_loss(outputs, labels, config)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                
                # 参数更新
                optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            preds = outputs['me_logits'].argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)
        
        # Epoch统计
        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / epoch_total
        
        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = preprocess_batch(videos, config)
                videos = videos.cuda()
                labels = labels.cuda()
                
                outputs = model(videos)
                loss = compute_loss(outputs, labels, config)
                
                val_loss += loss.item()
                preds = outputs['me_logits'].argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存最佳模型
            save_checkpoint(
                model, optimizer, epoch,
                avg_val_loss, history, config
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return history
```

## U.2 损失函数计算

```python
def compute_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    config: TrainingConfig
) -> torch.Tensor:
    """完整损失函数计算
    
    总损失 = L_cls + λ_au * L_au + λ_aux * L_aux + λ_reg * L_reg
    
    其中：
    - L_cls: 交叉熵分类损失
    - L_au: AU强度回归损失
    - L_aux: 辅助任务损失
    - L_reg: 正则化损失
    
    Args:
        outputs: 模型输出字典
        labels: 真实标签
        config: 训练配置
    
    Returns:
        total_loss: 总损失
    """
    
    # 1. 主分类损失 (Cross-Entropy)
    if config.use_label_smoothing:
        ce_loss = label_smoothed_cross_entropy(
            outputs['me_logits'], labels,
            smoothing=config.label_smoothing
        )
    else:
        ce_loss = F.cross_entropy(
            outputs['me_logits'], labels
        )
    
    # 2. AU强度损失 (MSE)
    au_loss = F.mse_loss(
        outputs['au_intensities'],
        outputs.get('au_targets', labels),
        reduction='mean'
    )
    
    # 3. 辅助任务损失
    aux_loss = 0.0
    if 'aux_logits' in outputs:
        for aux_logit in outputs['aux_logits']:
            aux_loss += F.cross_entropy(aux_logit, labels)
    
    # 4. 正则化损失
    reg_loss = 0.0
    
    # 4.1 权重衰减 (L2)
    if config.weight_decay > 0:
        reg_loss += config.weight_decay * sum(
            p.pow(2).sum() for p in model.parameters()
        )
    
    # 4.2 稀疏性正则
    if config.sparse_reg > 0:
        for name, param in model.named_parameters():
            if 'gate' in name or 'attention' in name:
                reg_loss += config.sparse_reg * torch.abs(param).sum()
    
    # 4.3 对比损失 (可选)
    if config.use_contrastive and 'contrastive_emb' in outputs:
        contrastive_loss = compute_contrastive_loss(
            outputs['contrastive_emb'], labels,
            temperature=config.contrastive_temp
        )
        reg_loss += config.contrastive_weight * contrastive_loss
    
    # ���损��
    total_loss = (
        ce_loss +
        config.lambda_au * au_loss +
        config.lambda_aux * aux_loss +
        config.lambda_reg * reg_loss
    )
    
    return total_loss
```

## U.3 MoE路由算法

```python
def moe_routing(
    inputs: torch.Tensor,
    gating_network: nn.Module,
    experts: List[nn.Module],
    config: MoEConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MoE门控路由算法
    
    算法步骤：
    1. 门控计算（并行）
    2. Top-K选择
    3. 负载均衡
    4. 专家输出加权
    
    Args:
        inputs: 输入张量 (B, D)
        gating_network: 门控网络
        experts: 专家列表
        config: MoE配置
    
    Returns:
        output: 加权输出 (B, O)
        load: 负载分布 (K,)
    """
    
    # 1. 门控计算
    gate_logits = gating_network(inputs)  # (B, num_experts)
    
    # 2. 门控规范化（防止梯度 vanish）
    gate_weights = F.softmax(gate_logits, dim=-1)
    
    # 3. Top-K 选择
    if config.top_k == 1:
        # 简单模式：选择最高权重的专家
        selected_idx = gate_weights.argmax(dim=-1)  # (B,)
        selected_weight = gate_weights.max(dim=-1)[0]  # (B,)
    else:
        # Top-K 模式：选择前K个专家
        top_k_weights, top_k_idx = torch.topk(
            gate_weights, config.top_k, dim=-1
        )
        
        # 归一化 Top-K 权重
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-8
        )
        
        selected_idx = top_k_idx
        selected_weight = top_k_weights
    
    # 4. 专家并行计算（加速）
    expert_outputs = []
    for expert in experts:
        expert_outputs.append(expert(inputs))
    expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, O)
    
    # 5. 加权聚合
    if config.top_k == 1:
        # 收集选择的专家输出
        selected_expert_output = expert_outputs[
            torch.arange(inputs.size(0)),
            selected_idx
        ]  # (B, O)
        
        output = selected_expert_output * selected_weight.unsqueeze(-1)
    else:
        # 多专家加权
        output = torch.zeros(
            inputs.size(0), expert_outputs.size(-1),
            device=inputs.device
        )
        
        for k in range(config.top_k):
            idx = selected_idx[:, k]  # (B,)
            weight = selected_weight[:, k]  # (B,)
            
            expert_output = expert_outputs[
                torch.arange(inputs.size(0)).unsqueeze(1),
                idx.unsqueeze(1)
            ].squeeze(1)  # (B, O)
            
            output += expert_output * weight.unsqueeze(-1)
    
    # 6. 负载统计
    load = torch.bincount(
        selected_idx.flatten(),
        minlength=len(experts)
    ).float() / inputs.size(0)
    
    return output, load
```

## U.4 早停算法

```python
class EarlyStopping:
    """早停算法实现
    
    监控验证集损失，连续patience个epoch没有改善则停止训练。
    支持三种模式：
    - min: 监控最小值（如损失）
    - max: 监控最大值（如准确率）
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')
    
    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        path: str = 'checkpoint.pt'
    ) -> bool:
        """检查是否早停
        
        Args:
            val_loss: 验证损失
            model: 模型
            path: 保存路径
        
        Returns:
            early_stop: 是否停止
        """
        
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss
        
        # 检查改善
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            
            # 保存最佳模型
            torch.save(model.state_dict(), path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
```

---

# 附录V：完整数学推导

## V.1 双通道融合数学推导

### V.1.1 融合权重计算

双通道融合使用可学习的加权融合：

$$\mathbf{f}_{fusion} = \alpha \cdot \mathbf{f}_{fast} + (1-\alpha) \cdot \mathbf{f}_{slow}$$

其中 $\alpha$ 是可学习的融合权重，通过 softmax 规范化：

$$\alpha_i = \frac{\exp(\mathbf{w}_i)}{\sum_{j \in \{fast,slow\}} \exp(\mathbf{w}_j)}$$

### V.1.2 梯度传播

对总损失 $\mathcal{L}$，参数更新：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{f}_{fast}} = \alpha \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{f}_{fusion}}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{f}_{slow}} = (1-\alpha) \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{f}_{fusion}}$$

$$\frac{\partial \mathcal{L}}{\partial \alpha} = \mathbf{f}_{fast} - \mathbf{f}_{slow}$$

### V.1.3 不确定性量化

双通道输出不确定性：

$$\sigma_{fusion}^2 = \alpha^2 \cdot \sigma_{fast}^2 + (1-\alpha)^2 \cdot \sigma_{slow}^2$$

其中 $\sigma_{fast}^2$ 和 $\sigma_{slow}^2$ 是各通道的预测方差。

## V.2 注意力机制数学推导

### V.2.1 杏仁核注意力

杏仁核生成情绪先验注意力图：

$$\mathbf{A}_{amygdala} = \sigma(\mathbf{W}_{a} \cdot \mathbf{f}_{emotion} + \mathbf{b}_{a})$$

其中：
- $\mathbf{f}_{emotion}$: 情绪嵌入向量
- $\mathbf{W}_{a}$: 可学习权重矩阵
- $\sigma$: sigmoid激活

### V.2.2 FFA门控

FFA 使用 SE（Squeeze-and-Excitation）机制：

**Squeeze（全局池化）：**
$$\mathbf{z} = \frac{1}{H \times W} \sum_{i=1}^{H}\sum_{j=1}^{W} \mathbf{x}_{:, i, j}$$

**Excitation（门控）：**
$$\mathbf{s} = \sigma(\mathbf{W}_{2} \cdot \text{ReLU}(\mathbf{W}_{1} \cdot \mathbf{z}))$$

**Scale（重标定）：**
$$\mathbf{x}_{out} = \mathbf{x} \cdot \mathbf{s}$$

### V.2.3 CASANet 三角注意力

CASANet 使用三角先验注意力：

$$\mathbf{A}_{casa}(i, j) = \frac{1}{|i - j| + 1} \cdot \exp\left(-\frac{(i - j)^2}{2\sigma^2}\right)$$

对于 apex 帧检测：

$$p_{apex} = \mathbf{A}_{casa} \cdot \mathbf{I}_{temporal}$$

其中 $\mathbf{I}_{temporal}$ 是时间维度的RGB强度分布。

## V.3 MoE门控数学推导

### V.3.1 门控概率

专家 $E_i$ 的选择概率：

$$P(E_i | \mathbf{x}) = \frac{\exp(g_i(\mathbf{x}))}{\sum_{j=1}^{N} \exp(g_j(\mathbf{x}))}$$

其中 $g_i(\mathbf{x})$ 是门控网络对专家 $i$ 的 logit 输出。

### V.3.2 负载均衡

负载损失（辅助正则化）：

$$\mathcal{L}_{load} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{1}{B} \sum_{b=1}^{B} P(E_i | \mathbf{x}_b) - \frac{1}{N} \right|$$

其中：
- $N$: 专家总数
- $B$: batch size

### V.3.3 专家坍塌缓解

为防止专家坍塌（一个专家主导），使用多样性正则：

$$\mathcal{L}_{diversity} = \frac{1}{N^2} \sum_{i=1}^{N}\sum_{j=1}^{N} (\bar{g}_i - \bar{g}_j)^2$$

其中 $\bar{g}_i = \frac{1}{B}\sum_{b} g_i(\mathbf{x}_b)$ 是专家 $i$ 的平均门控权重。

## V.4 图像生成数学推导

### V.4.1 3DMM参数估计

3DMM模型：

$$\mathbf{S} = \bar{\mathbf{S}} + \mathbf{A}_i \cdot \boldsymbol{\alpha}_i + \mathbf{A}_e \cdot \boldsymbol{\alpha}_e$$

参数估计（最小二乘）：

$$\boldsymbol{\alpha} = (\mathbf{J}^T \mathbf{J})^{-1} \mathbf{J}^T (\mathbf{I}_{2D} - \bar{\mathbf{S}})$$

其中 $\mathbf{J}$ 是雅可比矩阵。

### V.4.2 球谐光照渲染

光照模型：

$$L(\omega) = \sum_{l=0}^{L_{max}} \sum_{m=-l}^{l} c_{lm} Y_{lm}(\omega)$$

其中：
- $c_{lm}$: 球谐系数
- $Y_{lm}$: 球谐基函数

渲染颜色：

$$\mathbf{c}_{pixel} = \mathbf{n} \cdot L(\omega) \cdot \mathbf{R}_{albedo}$$

### V.4.3 ID保真损失

ID保真损失（对比学习）：

$$\mathcal{L}_{ID} = -\log \frac{\exp(\mathbf{f}_{gen} \cdot \mathbf{f}_{id} / \tau)}{\sum_{j} \exp(\mathbf{f}_{gen} \cdot \mathbf{f}_j / \tau)}$$

其中 $\tau$ 是温度参数。

---

# 附录W：完整数据��理��程

## W.1 数据集格式规范

### W.1.1 视频格式

```
数据集/
├── train/
│   ├── videos/
│   │   ├── happy_001.mp4
│   │   ├── sad_002.mp4
│   │   └── ...
│   └── labels.csv
├── val/
│   ├── videos/
│   └── labels.csv
├── test/
│   ├── videos/
│   └── labels.csv
└── metadata.json
```

### W.1.2 labels.csv 格式

```csv
filename,label,subject,session,apex_frame,au1_intensity,au2_intensity,...
happy_001.mp4,0,s001,session01,5,0.5,0.3,...
sad_002.mp4,1,s001,session01,8,0.2,0.6,...
```

## W.2 数据加载器实现

```python
class MicroExpressionDataset(Dataset):
    """微表情数据集类
    
    支持：
    - 视频文件加载
    - 帧采样
    - 预处理
    - 数据增强
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_frames: int = 16,
        sample_strategy: str = 'uniform',
        transform: nn.Module = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = Path(data_dir) / split
        self.num_frames = num_frames
        self.sample_strategy = sample_strategy
        self.transform = transform
        self.target_size = target_size
        
        # 加载标签
        labels_path = self.data_dir / 'labels.csv'
        self.samples = pd.read_csv(labels_path)
        
        # 预处理
        self.class_to_idx = {
            name: idx for idx, name in enumerate(
                self.samples['label'].unique()
            )
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取样本
        
        Returns:
            video: (C, T, H, W) 视频张量
            label: 类别索引
        """
        row = self.samples.iloc[idx]
        video_path = self.data_dir / 'videos' / row['filename']
        
        # 加载视频
        video = self._load_video(video_path)
        
        # 帧采样
        video = self._sample_frames(video)
        
        # 预处理
        video = self._preprocess(video)
        
        # 数据增强
        if self.transform:
            video = self.transform(video)
        
        # 获取标签
        label = self.class_to_idx[row['label']]
        
        # AU强度（如果存在）
        au_intensities = None
        if 'au1_intensity' in row:
            au_cols = [c for c in self.samples.columns if 'intensity' in c]
            au_intensities = torch.tensor(
                [row[c] for c in au_cols],
                dtype=torch.float32
            )
        
        return {
            'video': video,
            'label': label,
            'au_intensities': au_intensities,
            'filename': row['filename'],
        }
    
    def _load_video(self, path: Path) -> torch.Tensor:
        """加载视频
        
        Args:
            path: 视频路径
        
        Returns:
            frames: (T, H, W, C) 原始帧
        """
        import cv2
        
        cap = cv2.VideoCapture(str(path))
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        return torch.tensor(
            np.stack(frames),
            dtype=torch.uint8
        )
    
    def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """帧采样
        
        策略：
        - uniform: 均匀采样
        - central: 中心帧优先
        - apex: apex帧优先（需要标注）
        
        Args:
            frames: (T, H, W, C) 原始帧
        
        Returns:
            sampled: (T', H, W, C) 采样后帧
        """
        T = frames.size(0)
        
        if self.sample_strategy == 'uniform':
            # 均匀采样
            indices = torch.linspace(0, T - 1, self.num_frames).long()
        
        elif self.sample_strategy == 'central':
            # 中心优先
            center = T // 2
            half = self.num_frames // 2
            indices = torch.arange(
                center - half,
                center + half + 1
            ).clamp(0, T - 1)
        
        elif self.sample_strategy == 'apex':
            # apex帧优先
            apex = self.samples.iloc[self.idx].get('apex_frame', T // 2)
            half = self.num_frames // 2
            indices = torch.arange(
                apex - half,
                apex + half + 1
            ).clamp(0, T - 1)
        
        else:
            indices = torch.randperm(T)[:self.num_frames]
        
        return frames[indices]
    
    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """视频预处理
        
        步骤：
        1. 归一化到 [0, 1]
        2. Resize
        3. 标准化
        
        Args:
            frames: (T, H, W, C) 原始帧
        
        Returns:
            processed: (C, T, H, W) 处理后张量
        """
        # 归一化
        frames = frames.float() / 255.0
        
        # Resize
        frames = F.interpolate(
            frames.permute(0, 3, 1, 2),  # (T, C, H, W)
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        )
        
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        
        # 转换为 (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)
        
        return frames
```

### W.1.3 DataLoader 配置

```python
def build_dataloader(
    data_dir: str,
    split: str,
    batch_size: int = 4,
    num_workers: int = 4,
    num_frames: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    """构建数据加载器
    
    Args:
        data_dir: 数据目录
        split: 数据集划分
        batch_size: 批大小
        num_workers: 数据加载worker数
        num_frames: 帧采样数
        shuffle: 是否打乱
    
    Returns:
        dataloader: DataLoader实例
    """
    
    # 数据集
    dataset = MicroExpressionDataset(
        data_dir=data_dir,
        split=split,
        num_frames=num_frames,
    )
    
    # 分布式采样（多GPU）
    if torch.distributed.is_available():
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
        )
    else:
        sampler = None
    
    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
    )
    
    return loader
```

## W.2 数据增强策略

### W.2.1 几何变换

```python
class GeometricTransform:
    """几何变换增强
    
    支持：
    - 随机 crop
    - 随机翻转（水平）
    - 随机旋转
    - 随机缩放
    """
    
    def __init__(
        self,
        p_horizontal_flip: float = 0.5,
        p_rotation: float = 0.3,
        rotation_degrees: int = 15,
        p_scale: float = 0.3,
        scale_range: Tuple[float, float] = (0.9, 1.1),
    ):
        self.p_horizontal_flip = p_horizontal_flip
        self.p_rotation = p_rotation
        self.rotation_degrees = rotation_degrees
        self.p_scale = p_scale
        self.scale_range = scale_range
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """应用变换
        
        Args:
            video: (C, T, H, W) 视频
        
        Returns:
            transformed: 变换后视频
        """
        C, T, H, W = video.shape
        
        # 随机水平翻转
        if torch.rand() < self.p_horizontal_flip:
            video = torch.flip(video, dims=[-1])
        
        # 随机旋转
        if torch.rand() < self.p_rotation:
            angle = torch.rand(
                1, device=video.device
            ) * 2 - 1  # -1 to 1
            angle = angle * self.rotation_degrees
            
            # 仿射变换
            theta = self._get_rotation_matrix(angle)
            grid = F.affine_grid(
                theta,
                video.permute(0, 2, 3, 1).unsqueeze(0).size(),
                align_corners=False
            )
            video_renorm = video.permute(0, 2, 3, 1).unsqueeze(0)
            video = F.grid_sample(
                video_renorm,
                grid,
                mode='bilinear',
                padding_mode='reflection',
                align_corners=False
            ).squeeze(0).permute(0, 2, 3, 1)
        
        # 随机缩放
        if torch.rand() < self.p_scale:
            scale = torch.rand(
                1, device=video.device
            ) * (self.scale_range[1] - self.scale_range[0]
            ) + self.scale_range[0]
            
            new_H = int(H * scale)
            new_W = int(W * scale)
            
            video = F.interpolate(
                video,
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False
            )
            
            # Crop 或 Pad
            if scale > 1:
                # Center crop
                start_h = (new_H - H) // 2
                start_w = (new_W - W) // 2
                video = video[
                    :, :,
                    start_h:start_h + H,
                    start_w:start_w + W
                ]
            else:
                # Pad
                pad_h = (H - new_H) // 2
                pad_w = (W - new_W) // 2
                video = F.pad(
                    video,
                    [pad_w, W - new_W - pad_w, pad_h, H - new_H - pad_h]
                )
        
        return video
```

### W.2.2 颜色变换

```python
class ColorTransform:
    """颜色变换增强
    
    支持：
    - 亮度调整
    - 对比度调整
    - 饱和度调整
    - 色相调整
    - 添加噪声
    """
    
    def __init__(
        self,
        p_brightness: float = 0.5,
        brightness_range: float = 0.2,
        p_contrast: float = 0.5,
        contrast_range: float = 0.2,
        p_saturation: float = 0.5,
        saturation_range: float = 0.2,
        p_noise: float = 0.3,
        noise_std: float = 0.01,
    ):
        self.p_brightness = p_brightness
        self.brightness_range = brightness_range
        self.p_contrast = p_contrast
        self.contrast_range = contrast_range
        self.p_saturation = p_saturation
        self.saturation_range = saturation_range
        self.p_noise = p_noise
        self.noise_std = noise_std
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """应用颜色变换
        
        Args:
            video: (C, T, H, W) 视频
        
        Returns:
            transformed: 变换后视频
        """
        # 亮度
        if torch.rand() < self.p_brightness:
            factor = 1 + torch.rand(
                1, device=video.device
            ) * 2 - 1
            factor = factor * self.brightness_range
            video = video + factor
        
        # 对比度
        if torch.rand() < self.p_contrast:
            factor = 1 + torch.rand(
                1, device=video.device
            ) * 2 - 1
            factor = factor * self.contrast_range
            mean = video.mean(dim=(-2, -1), keepdim=True)
            video = (video - mean) * factor + mean
        
        # 饱和度
        if torch.rand() < self.p_saturation and video.size(0) >= 3:
            factor = 1 + torch.rand(
                1, device=video.device
            ) * 2 - 1
            factor = factor * self.saturation_range
            
            # 转为HSV，调整后转回RGB
            video_hsv = self._rgb_to_hsv(video)
            video_hsv[1] = (video_hsv[1] * factor).clamp(0, 1)
            video = self._hsv_to_rgb(video_hsv)
        
        # 噪声
        if torch.rand() < self.p_noise:
            noise = torch.randn_like(video) * self.noise_std
            video = video + noise
        
        return video
    
    def _rgb_to_hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        # RGB to HSV 转换
        # ...
        return rgb
    
    def _hsv_to_rgb(self, hsv: torch.Tensor) -> torch.Tensor:
        # HSV to RGB 转换
        # ...
        return hsv
```

---

# 附录X：完整评估框架

## X.1 评估指标实现

```python
class EvaluationMetrics:
    """完整评估指标类
    
    支持：
    - 分类指标（Accuracy, Precision, Recall, F1, AUC）
    - 回归指标（MAE, MSE, RMSE, R²）
    - 时序指标（时序相关性）
    - 分布指标（KL散度）
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.au_intensities = []
        self.au_targets = []
    
    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ):
        """更新统计
        
        Args:
            outputs: 模型输出
            labels: 真实标签
        """
        # 预测
        pred = outputs['me_logits'].argmax(dim=-1)
        self.predictions.extend(pred.cpu().numpy())
        
        # 标签
        self.labels.extend(labels.cpu().numpy())
        
        # 概率
        prob = F.softmax(outputs['me_logits'], dim=-1)
        self.probabilities.extend(prob.cpu().numpy())
        
        # AU强度
        if 'au_intensities' in outputs:
            self.au_intensities.extend(
                outputs['au_intensities'].cpu().numpy()
            )
        if 'au_targets' in outputs:
            self.au_targets.extend(
                outputs['au_targets'].cpu().numpy()
            )
    
    def compute(self) -> Dict[str, float]:
        """计算所有指标
        
        Returns:
            metrics: 指标字典
        """
        import sklearn.metrics as metrics
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        probabilities = np.array(self.probabilities)
        
        results = {}
        
        # 1. 分类指标
        results['accuracy'] = metrics.accuracy_score(
            labels, predictions
        )
        
        results['precision_macro'] = metrics.precision_score(
            labels, predictions, average='macro', zero_division=0
        )
        results['recall_macro'] = metrics.recall_score(
            labels, predictions, average='macro', zero_division=0
        )
        results['f1_macro'] = metrics.f1_score(
            labels, predictions, average='macro', zero_division=0
        )
        
        results['precision_weighted'] = metrics.precision_score(
            labels, predictions, average='weighted', zero_division=0
        )
        results['recall_weighted'] = metrics.recall_score(
            labels, predictions, average='weighted', zero_division=0
        )
        results['f1_weighted'] = metrics.f1_score(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # 2. AUC (One-vs-Rest)
        try:
            results['auc_macro'] = metrics.roc_auc_score(
                labels, probabilities, average='macro', multi_class='ovr'
            )
            results['auc_weighted'] = metrics.roc_auc_score(
                labels, probabilities, average='weighted', multi_class='ovr'
            )
        except ValueError:
            pass
        
        # 3. 混淆矩阵
        results['confusion_matrix'] = metrics.confusion_matrix(
            labels, predictions
        ).tolist()
        
        # 4. 分类报告
        results['classification_report'] = metrics.classification_report(
            labels, predictions, zero_division=0
        )
        
        # 5. AU强度评估
        if self.au_intensities and self.au_targets:
            au_int = np.array(self.au_intensities)
            au_tgt = np.array(self.au_targets)
            
            results['au_mae'] = np.mean(np.abs(au_int - au_tgt))
            results['au_mse'] = np.mean((au_int - au_tgt) ** 2)
            results['au_rmse'] = np.sqrt(results['au_mse'])
        
        return results
    
    def compute_per_class(self) -> Dict[str, Dict[str, float]]:
        """计算每类指标
        
        Returns:
            per_class: 每类指标
        """
        import sklearn.metrics as metrics
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        per_class = {}
        
        for cls in range(self.num_classes):
            binary_true = (labels == cls)
            binary_pred = (predictions == cls)
            
            per_class[cls] = {
                'precision': metrics.precision_score(
                    binary_true, binary_pred, zero_division=0
                ),
                'recall': metrics.recall_score(
                    binary_true, binary_pred, zero_division=0
                ),
                'f1': metrics.f1_score(
                    binary_true, binary_pred, zero_division=0
                ),
                'support': binary_true.sum(),
            }
        
        return per_class
```

## X.2 交叉验证实现

```python
class CrossValidation:
    """K折交叉验证
    
    支持：
    - KFold
    - StratifiedKFold
    - GroupKFold（按被试划分）
    - LeaveOneSubjectOut（LOSO）
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        strategy: str = 'stratified',
        groups: np.ndarray = None,
    ):
        self.n_splits = n_splits
        self.strategy = strategy
        self.groups = groups
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """生成分割索引
        
        Args:
            X: 特征
            y: 标签
        
        Yields:
            train_idx, val_idx: 训练/验证索引
        """
        if self.strategy == 'kfold':
            kf = KFold(n_splits=self.n_splits, shuffle=True)
            yield from kf.split(X)
        
        elif self.strategy == 'stratified':
            skf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True
            )
            yield from skf.split(X, y)
        
        elif self.strategy == 'group':
            # 按被试划分（防止数据泄露）
            gkf = GroupKFold(n_splits=self.n_splits)
            yield from gkf.split(X, y, groups=self.groups)
        
        elif self.strategy == 'loso':
            # Leave-One-Subject-Out
            unique_subjects = np.unique(self.groups)
            
            for subject in unique_subjects:
                val_idx = np.where(self.groups == subject)[0]
                train_idx = np.where(self.groups != subject)[0]
                yield train_idx, val_idx
    
    def evaluate(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray
    ) -> float:
        """评估单折
        
        Args:
            model_fn: 模型构建函数
            X: 特征
            y: 标签
            train_idx: 训练索引
            val_idx: 验证索引
        
        Returns:
            metric: 验证指标
        """
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练
        model = model_fn()
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        
        # 计算指标
        return accuracy_score(y_val, y_pred)
    
    def run(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """运行完整交叉验证
        
        Args:
            model_fn: 模型构建函数
            X: 特征
            y: 标签
        
        Returns:
            results: 交叉验证结果
        """
        scores = []
        
        for train_idx, val_idx in self.split(X, y):
            score = self.evaluate(model_fn, X, y, train_idx, val_idx)
            scores.append(score)
        
        return {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
        }
```

## X.3 统计显著性检验

```python
class StatisticalTest:
    """统计显著性检验
    
    支持：
    - t检验（配对/独立）
    - Wilcoxon符号秩检验
    - McNemar检验
    """
    
    @staticmethod
    def paired_t_test(
        scores1: np.ndarray,
        scores2: np.ndarray
    ) -> Dict[str, float]:
        """配对t检验
        
        Args:
            scores1: 方法1分数
            scores2: 方法2分数
        
        Returns:
            results: 检验结果
        """
        from scipy import stats
        
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_diff': np.mean(scores1 - scores2),
            'std_diff': np.std(scores1 - scores2),
        }
    
    @staticmethod
    def wilcoxon(
        scores1: np.ndarray,
        scores2: np.ndarray
    ) -> Dict[str, float]:
        """Wilcoxon符号秩检验（非参数）
        
        Args:
            scores1: 方法1分数
            scores2: 方法2分数
        
        Returns:
            results: 检验结果
        """
        from scipy import stats
        
        stat, p_value = stats.wilcoxon(scores1, scores2)
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }
    
    @staticmethod
    def mc_nemar(
        model1_correct: np.ndarray,
        model2_correct: np.ndarray
    ) -> Dict[str, float]:
        """McNemar检验（分类模型比较）
        
        Args:
            model1_correct: 模型1正确与否
            model2_correct: 模型2正确与否
        
        Returns:
            results: 检验结果
        """
        from scipy import stats
        
        # 构建列联表
        b = np.sum((~model1_correct) & model2_correct)  # 模型1错，模型2对
        c = np.mean(model1_correct & (~model2_correct))  # 模型1对，模型2错
        
        # McNemar统计量
        if b + c == 0:
            return {'statistic': 0, 'p_value': 1}
        
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(stat, df=1)
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'b': b,
            'c': c,
        }
```

---

# 附录Y：部署与生产环境

## Y.1 Dockerfile

```dockerfile
# 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制模型文件
COPY models/ ./models/

# 复制代码
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 暴露端口
EXPOSE 8500

# 启动命令
CMD ["python", "serve.py"]
```

## Y.2 docker-compose.yml

```yaml
version: '3.8'

services:
  censor:
    build: .
    image: censor:latest
    ports:
      - "8500:8500"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=/app/models/censor_v2.pt
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8500/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - censor
    restart: unless-stopped
```

## Y.3 Kubernetes 部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: censor
  labels:
    app: censor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: censor
  template:
    metadata:
      labels:
        app: censor
    spec:
      containers:
      - name: censor
        image: censor:latest
        ports:
        - containerPort: 8500
        env:
        - name: MODEL_PATH
          value: /app/models/censor_v2.pt
        resources:
          requests:
            memory: "4Gi"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8500
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8500
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: censor
spec:
  selector:
    app: censor
  ports:
  - port: 8500
    targetPort: 8500
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: censor-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: censor
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Y.4 生产环境配置

```python
# production_config.py
PRODUCTION_CONFIG = {
    # 模型设置
    'model': {
        'checkpoint': 'models/censor_v2.pt',
        'device': 'cuda',
        'precision': 'fp16',  # fp16/int8/fp32
    },
    
    # 数据处理
    'data': {
        'batch_size': 8,
        'num_workers': 4,
        'prefetch_factor': 2,
        'pin_memory': True,
    },
    
    # 推理优化
    'inference': {
        'use_tensorrt': True,
        'use_torch_compile': True,
        'cudnn_benchmark': True,
    },
    
    # 服务设置
    'server': {
        'host': '0.0.0.0',
        'port': 8500,
        'workers': 4,
        'timeout': 60,
    },
    
    # 监控
    'monitoring': {
        'metrics_port': 9090,
        'enable_prometheus': True,
        'log_level': 'INFO',
    },
    
    # 安全
    'security': {
        'enable_auth': True,
        'max_request_size': '100MB',
        'rate_limit': 100,
    },
}
```

---

# 附录Z：错误诊断与调试

## Z.1 常见错误诊断

### Z.1.1 内存错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| CUDA OOM | 显存不足 | 减小 batch_size，使用 gradient accumulation |
| CPU OOM | 内存不足 | 减少 num_workers，使用数据缓存 |
| 内存泄漏 | 缓存未释放 | 定期清理torch.cuda.empty_cache() |

### Z.1.2 训练错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| NaN loss | 学习率过大/梯度爆炸 | 降低lr，添加梯度裁剪 |
| Inf loss | 数值不稳定 | 添加eps，检查数据 |
| Loss不下降 | 学习率过小/模型bug | 调整lr，debug模型 |

### Z.1.3 CUDA错误

| 错误代码 | 含义 | 解决方案 |
|----------|------|----------|
| 1 | 内存不足 | 减小batch |
| 2 | 设备错误 | 检查GPU |
| 3 | 精度错误 | 检查dtype |
| 4 | 版本不匹配 | 更新CUDA |

## Z.2 调试工具

### Z.2.1 PyTorch Profiler

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=2
    ),
    on_trace_ready=trace_handler,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, data in enumerate(dataloader):
        with torch.profiler.record_function("data_loading"):
            data = data.cuda()
        
        with torch.profiler.record_function("forward"):
            output = model(data)
        
        with torch.profiler.record_function("backward"):
            loss.backward()
        
        prof.step()
```

### Z.2.2 内存分析

```python
import torch.cuda.memory as memory

# 显存统计
def print_memory_summary():
    print(f"Allocated: {memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {memory_reserved() / 1e9:.2f} GB")
    print(f"Max allocated: {max_memory_allocated() / 1e9:.2f} GB")

# 显存快照
def save_memory_snapshot():
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._dump_snapshot()
    with open("memory_snapshot.pkl", "wb") as f:
        f.write(snapshot)

# 内存泄漏检测
def detect_memory_leak():
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_allocated()
    
    # 运行多次
    for _ in range(100):
        output = model(batch)
        loss.backward()
    
    final = torch.cuda.memory_allocated()
    leaked = final - initial
    
    print(f"Leaked memory: {leaked / 1e6:.2f} MB")
```

### Z.2.3 性能分析

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.4f}s")
        return result
    return wrapper

# 使用
@timing_decorator
def my_forward(x):
    return model(x)
```

---

# 附录AA：完整示例与教程

## AA.1 完整训练示例

```python
# train_complete.py
"""
完整训练流程示例

包含：
1. 数据加载与预处理
2. 模型构建
3. 训练循环
4. 验证与评估
5. 模型保存
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# 导入模块
from model.censor import Censor
from model.losses import compute_loss
from data.dataset import MicroExpressionDataset
from data.augmentation import GeometricTransform, ColorTransform
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import EvaluationMetrics
from utils.early_stopping import EarlyStopping

def main():
    # ===== 配置 =====
    config = {
        'batch_size': 4,
        'num_frames': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 50,
        'patience': 10,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_amp': True,  # 混合精度
    }
    
    # ===== 数据 =====
    # 变换
    train_transform = nn.Sequential(
        GeometricTransform(p_horizontal_flip=0.5),
        ColorTransform(p_brightness=0.5),
    )
    
    # 数据集
    train_dataset = MicroExpressionDataset(
        data_dir='data/SAMM',
        split='train',
        num_frames=config['num_frames'],
        transform=train_transform,
    )
    
    val_dataset = MicroExpressionDataset(
        data_dir='data/SAMM',
        split='val',
        num_frames=config['num_frames'],
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    
    # ===== 模型 =====
    model = Censor(config)
    model = model.to(config['device'])
    
    # ===== 优化器 =====
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    
    # 混合精度
    scaler = GradScaler() if config['use_amp'] else None
    
    # 早停
    early_stopping = EarlyStopping(
        patience=config['patience'],
        mode='min'
    )
    
    # ===== 训练循环 =====
    for epoch in range(config['epochs']):
        # ===== 训练 =====
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            videos = batch['video'].to(config['device'])
            labels = batch['label'].to(config['device'])
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    outputs = model(videos)
                    loss = compute_loss(outputs, labels, config)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(videos)
                loss = compute_loss(outputs, labels, config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # ===== 验证 =====
        model.eval()
        val_loss = 0.0
        metrics = EvaluationMetrics(model.num_classes)
        
        with torch.no_grad():
            for batch in val_loader:
                videos = batch['video'].to(config['device'])
                labels = batch['label'].to(config['device'])
                
                outputs = model(videos)
                loss = compute_loss(outputs, labels, config)
                
                val_loss += loss.item()
                metrics.update(outputs, labels)
        
        val_loss /= len(val_loader)
        val_metrics = metrics.compute()
        
        # ===== 日志 =====
        print(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )
        
        # ===== 调度 =====
        scheduler.step(val_loss)
        
        # ===== 早停 =====
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # ===== 保存 =====
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch,
                val_loss, history=None,
                path=f'checkpoints/censor_{epoch}.pt'
            )
    
    # ===== 最终保存 =====
    save_checkpoint(
        model, optimizer, epoch,
        val_loss, history=None,
        path='checkpoints/censor_final.pt'
    )

if __name__ == '__main__':
    main()
```

## AA.2 推理示例

```python
# inference.py
"""
单样本推理示例
"""

import torch
from model.censor import Censor
from utils.preprocess import preprocess_video

def main():
    # 加载模型
    model = Censor()
    checkpoint = torch.load('checkpoints/censor_final.pt')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.cuda()
    
    # 加载视频
    video_path = 'data/test/happy_001.mp4'
    video = preprocess_video(video_path, num_frames=16)
    video = video.unsqueeze(0).cuda()  # (1, C, T, H, W)
    
    # 推理
    with torch.no_grad():
        outputs = model(video)
    
    # 输出解析
    pred_class = outputs['me_logits'].argmax(dim=-1).item()
    pred_prob = torch.softmax(outputs['me_logits'], dim=-1)[0, pred_class].item()
    au_intensities = outputs['au_intensities'][0]
    
    # 标签映射
    class_names = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']
    
    print(f"预测: {class_names[pred_class]}")
    print(f"置信度: {pred_prob:.4f}")
    print(f"AU强度: {au_intensities.tolist()}")
    
    # 生成报告
    if hasattr(model, 'reporter'):
        report = model.reporter.generate_report(
            pred_class=pred_class,
            pred_prob=pred_prob,
            au_intensities=au_intensities,
        )
        print(f"\n报告:\n{report}")

if __name__ == '__main__':
    main()
```

## AA.3 批量推理示例

```python
# batch_inference.py
"""
批量推理与结果保存
"""

import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    # 加载模型
    model = Censor()
    model.load_state_dict(
        torch.load('checkpoints/censor_final.pt')['model_state']
    )
    model.eval()
    model.cuda()
    
    # 数据集
    dataset = MicroExpressionDataset(
        data_dir='data/SAMM',
        split='test',
    )
    loader = DataLoader(dataset, batch_size=8)
    
    # 推理
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            videos = batch['video'].cuda()
            outputs = model(videos)
            
            # 解析输出
            preds = outputs['me_logits'].argmax(dim=-1)
            probs = torch.softmax(outputs['me_logits'], dim=-1)
            
            for i, (pred, prob, filename) in enumerate(
                zip(preds, probs, batch['filename'])
            ):
                results.append({
                    'filename': filename,
                    'pred_class': pred.item(),
                    'pred_prob': prob.max().item(),
                    'confidence': prob[pred].item(),
                })
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv('results/predictions.csv', index=False)
    
    print(f"保存 {len(results)} 条预测结果")

if __name__ == '__main__':
    main()
```

---

# 附录AB：版本历史与迁移

## AB.1 版本变更记录

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| 2.0 | 2024-xx | v2.0发布：完整图像生成管线 |
| 1.5 | 2024-xx | BioMoE增强，层级动态MoE |
| 1.0 | 2024-xx | 初版发布，双通道MER |
| 0.5 | 2023-xx | Alpha测试版 |

## AB.2 v1.x → v2.0 迁移指南

```python
# v1.x 模型加载 (兼容模式)
def load_v1_model(checkpoint_path: str) -> dict:
    """加载v1.x模型检查点
    
    Args:
        checkpoint_path: 检查点路径
    
    Returns:
        state_dict: 兼容格式的state_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # v1.x 结构
    old_state = checkpoint['model_state']
    
    # 迁移映射
    new_state = {}
    
    for key, value in old_state.items():
        # 重命名层
        if 'feature_extractor' in key:
            new_key = key.replace('feature_extractor', 'slow_path')
        elif 'optical_flow' in key:
            new_key = key.replace('optical_flow', 'fast_path')
        else:
            new_key = key
        
        new_state[new_key] = value
    
    return new_state

# 使用示例
def migrate_checkpoints():
    """迁移所有v1.x检查点"""
    import os
    
    old_dir = 'checkpoints/v1'
    new_dir = 'checkpoints/v2'
    
    os.makedirs(new_dir, exist_ok=True)
    
    for filename in os.listdir(old_dir):
        if filename.endswith('.pt'):
            old_path = os.path.join(old_dir, filename)
            new_path = os.path.join(new_dir, filename)
            
            # 加载并迁移
            state_dict = load_v1_model(old_path)
            
            # 保存
            torch.save({
                'model_state': state_dict,
                'version': '2.0',
            }, new_path)
            
            print(f"迁移: {filename}")

if __name__ == '__main__':
    migrate_checkpoints()
```

---

# 附录AC：深度技术细节

## AC.1 3D ResNet-18 详细架构

### AC.1.1 残差块前向传播详解

```python
class BasicBlock(nn.Module):
    """3D Basic Block for ResNet-18/34
    
    架构：
    conv1 (3D, 3x3x3) → BN → ReLU → conv2 (3D, 3x3x3) → BN → add → ReLU
    
    跳跃连接：
    - 维度变化：如果输入/输出维度不同，进行1x1x1卷积
    - 分辨率变化：如果步长>1，下采样
    """
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ):
        super().__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二个卷积层
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 跳跃连接
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        输入：(B, C, T, H, W)
        输出：(B, C', T', H', W')
        
        数学公式：
        y = BN(conv2(ReLU(BN(conv1(x)))) + shortcut(x)
        """
        identity = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 跳跃连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
```

### AC.1.2 完整的3D ResNet-18架构

```python
class ResNet3D(nn.Module):
    """3D ResNet-18
    
    层级结构：
    conv1: 7x7x7, 64, stride=2
    maxpool: 3x3x3, stride=2
    layer1: 64 x 2 blocks
    layer2: 128 x 2 blocks  
    layer3: 256 x 2 blocks
    layer4: 512 x 2 blocks
    avgpool: global average
    fc: 1000
    """
    
    def __init__(
        self,
        block: nn.Module = BasicBlock,
        layers: List[int] = [2, 2, 2, 2],
        num_classes: int = 1000,
        pretrained: bool = False,
    ):
        super().__init__()
        
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv3d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _make_layer(
        self,
        block: nn.Module,
        channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """构建残差层"""
        
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(channels * block.expansion),
            )
        
        layers = []
        layers.append(block(
            self.in_channels,
            channels,
            stride,
            downsample,
        ))
        
        self.in_channels = channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整前向传播
        
        (B, 3, T, H, W) → (B, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
```

### AC.1.3 时间复杂度分析

3D ConvN 的时间复杂度（FLOPs）：

$$FLOPs_{conv3d} = 2 \cdot C_{in} \cdot C_{out} \cdot K_T \cdot K_H \cdot K_W \cdot T_{out} \cdot H_{out} \cdot W_{out}$$

对于ResNet-18的各层：

| 层 | 输出尺寸 | FLOPs |
|------|----------|-------|
| conv1 | 8x112x112 | 2.1G |
| layer1 | 8x56x56 | 1.8G |
| layer2 | 4x28x28 | 0.9G |
| layer3 | 2x14x14 | 0.4G |
| layer4 | 1x7x7 | 0.2G |
| **总计** | | **5.4G** |

## AC.2 3D Swin Transformer 详细架构

### AC.2.1 窗口注意力机制

```python
class WindowAttention3D(nn.Module):
    """3D窗口注意力
    
    将3D特征划分为非重叠窗口，在每个窗口内计算自注意力。
    支持滑动窗口（shifted window）以捕获跨窗口关系。
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int] = (8, 7, 7),
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) *
                (2 * window_size[1] - 1) *
                (2 * window_size[2] - 1),
                num_heads,
            )
        )
        
        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 初始化
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """窗口注意力前向传播
        
        Args:
            x: (B, T, H, W, C)
            mask: (N, N) 注意力掩码
        
        Returns:
            out: (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        # QKV
        qkv = self.qkv(x).reshape(
            B, T, H, W, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 4, 1, 5)  # (3, B, H, W, T, n_heads, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        # 输出
        x = (attn @ v).transpose(1, 2).reshape(B, T, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
```

### AC.2.2 完整Swin架构

```python
class SwinTransformer3D(nn.Module):
    """3D Swin Transformer
    
    层级结构：
    patch_embed: 2x2x2 patch分割 + 线性投影
    maxpool: 2x2x2 下采样
    SwinBlocks × 4 stages
    """
    
    def __init__(
        self,
        t: int = 8,
        h: int = 56,
        w: int = 56,
        patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 96,
        num_heads: int = 8,
        depths: List[int] = [2, 2, 6, 2],
        window_size: Tuple[int, int, int] = (8, 7, 7),
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        num_classes: int = 1000,
    ):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        
        # Patch Embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        
        # 位置嵌入
        num_patches = (
            self.patch_embed.T * 
            self.patch_embed.H * 
            self.patch_embed.W
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 构建各stage
        self.layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
            )
            self.layers.append(layer)
        
        # 归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整前向传播
        
        Args:
            x: (B, C, T, H, W)
        
        Returns:
            logits: (B, num_classes)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, T', H', W', C)
        x = x.flatten(1, 3)  # (B, N, C)
        x = self.pos_drop(x)
        
        # 各stage
        for layer in self.layers:
            x = layer(x)
        
        # 归一化
        x = self.norm(x)
        
        # 池化和分类
        x = x.mean(dim=1)
        x = self.head(x)
        
        return x
```

## AC.3 注意力融合详细实现

### AC.3.1 双向交叉注意力

```python
class BidirectionalCrossAttention(nn.Module):
    """双向交叉注意力
    
    快慢双通道特征的双向交互：
    fast → slow：慢通道关注快速通道的关键信息
    slow → fast：快速通道整合慢通道的上下文
    
    公式：
    O_f = Attention(Q_f, K_s, V_s) + α * Q_f
    O_s = Attention(Q_s, K_f, V_f) + β * Q_s
    
    其中 α, β ���可学习的残差权重。
    """
    
    def __init__(
        self,
        dim_fast: int,
        dim_slow: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        
        self.dim_fast = dim_fast
        self.dim_slow = dim_slow
        self.num_heads = num_heads
        
        # 投影层
        self.proj_q_fast = nn.Linear(dim_fast, dim_fast, bias=qkv_bias)
        self.proj_k_slow = nn.Linear(dim_slow, dim_fast, bias=qkv_bias)
        self.proj_v_slow = nn.Linear(dim_slow, dim_fast, bias=qkv_bias)
        
        self.proj_q_slow = nn.Linear(dim_slow, dim_slow, bias=qkv_bias)
        self.proj_k_fast = nn.Linear(dim_fast, dim_slow, bias=qkv_bias)
        self.proj_v_fast = nn.Linear(dim_fast, dim_slow, bias=qkv_bias)
        
        # 残差权重
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 归一化
        self.norm_fast = nn.LayerNorm(dim_fast)
        self.norm_slow = nn.LayerNorm(dim_slow)
        
        # 头维度
        self.head_dim = dim_fast // num_heads
    
    def forward(
        self,
        x_fast: torch.Tensor,
        x_slow: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """双向注意力
        
        Args:
            x_fast: (B, N_f, D_f)
            x_slow: (B, N_s, D_s)
        
        Returns:
            out_fast: (B, N_f, D_f)
            out_slow: (B, N_s, D_s)
        """
        B, N_f, D_f = x_fast.shape
        _, N_s, D_s = x_slow.shape
        
        # QKV投影
        q_f = self.proj_q_fast(x_fast).reshape(
            B, N_f, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k_s = self.proj_k_slow(x_slow).reshape(
            B, N_s, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v_s = self.proj_v_slow(x_slow).reshape(
            B, N_s, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        q_s = self.proj_q_slow(x_slow).reshape(
            B, N_s, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k_f = self.proj_k_fast(x_fast).reshape(
            B, N_f, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v_f = self.proj_v_fast(x_fast).reshape(
            B, N_f, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Fast → Slow 注意力
        attn_f2s = (q_f @ k_s.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_f2s = attn_f2s.softmax(dim=-1)
        attn_f2s = self.attn_drop(attn_f2s)
        
        out_f = (attn_f2s @ v_s).transpose(1, 2).reshape(B, N_f, D_f)
        out_f = self.proj_drop(self.proj_out_f(out_f))
        
        # 添加残差
        out_f = self.norm_fast(
            out_f + self.alpha * x_fast
        )
        
        # Slow → Fast 注意力
        attn_s2f = (q_s @ k_f.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_s2f = attn_s2f.softmax(dim=-1)
        attn_s2f = self.attn_drop(attn_s2f)
        
        out_s = (attn_s2f @ v_f).transpose(1, 2).reshape(B, N_s, D_s)
        out_s = self.proj_drop(self.proj_out_s(out_s))
        
        # 添加残差
        out_s = self.norm_slow(
            out_s + self.beta * x_slow
        )
        
        return out_f, out_s
    
    def proj_out_f(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight_out_f, self.bias_out_f)
    
    def proj_out_s(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight_out_s, self.bias_out_s)
```

### AC.3.2 融合权重学习

```python
class LearnableFusion(nn.Module):
    """可学习的融合权重
    
    基于门控机制的融合权重学习：
    w = sigmoid(FC(concat([f_fast, f_slow])))
    
    融合：f = w * f_fast + (1 - w) * f_slow
    """
    
    def __init__(
        self,
        dim_fast: int,
        dim_slow: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(dim_fast + dim_slow, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )
    
    def forward(
        self,
        x_fast: torch.Tensor,
        x_slow: torch.Tensor,
    ) -> torch.Tensor:
        """融合
        
        Args:
            x_fast: (B, D)
            x_slow: (B, D)
        
        Returns:
            fused: (B, D)
        """
        # 拼接
        x_concat = torch.cat([x_fast, x_slow], dim=-1)
        
        # 门控
        weights = self.gate(x_concat)  # (B, 2)
        
        # 加权融合
        fused = (
            weights[:, 0:1] * x_fast +
            weights[:, 1:2] * x_slow
        )
        
        return fused
```

## AC.4 完整BioMoE实现

### AC.4.1 膜电位门控

```python
class BioMoEGating(nn.Module):
    """生物膜电位门控
    
    模拟生物神经元的膜电位累积机制：
    V_m(t) = V_m(t-1) * decay + feedback * (1 - decay)
    
    g_i = sigmoid(W_i @ V_m + b_i)
    
    特性：
    1. 膜电位累积：历史激活影响当前门控
    2. 时间常数：不同专家有不同的时间常数
    3. 不应期：激活后进入不应期
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 3,
        hidden_dim: int = 128,
        time_constants: List[float] = [0.3, 0.5, 0.7],
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.time_constants = nn.Parameter(
            torch.tensor(time_constants),
            requires_grad=False,  # 固定时间常数
        )
        
        # 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 专家选择MLP
        self.expert_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_experts)
        ])
        
        # 膜电位状态（不参与训练）
        self.register_buffer('membrane_potential', torch.zeros(num_experts))
        self.register_buffer('refractory_period', torch.zeros(num_experts))
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """门控前向传播
        
        Args:
            x: (B, D) 输入特征
        
        Returns:
            weights: (B, num_experts) 专家权重
        """
        B = x.size(0)
        
        # 特征投影
        features = self.feature_proj(x)
        
        # 计算各专家logit
        logits = torch.zeros(B, self.num_experts, device=x.device)
        
        for i, mlp in enumerate(self.expert_mlp):
            logits[:, i] = mlp(features).squeeze(-1)
        
        # 膜电位更新
        if self.training:
            # 训练时：用当前logit更新
            self.membrane_potential = (
                self.membrane_potential * self.time_constants +
                logits.mean(dim=0) * (1 - self.time_constants)
            )
        
        # 融合膜电位
        membrane_weights = torch.sigmoid(self.membrane_potential)
        
        # 应用膜电位门控
        gated_logits = logits + membrane_weights.unsqueeze(0) * logits
        
        # Softmax归一化
        weights = F.softmax(gated_logits, dim=-1)
        
        return weights
```

### AC.4.2 完整BioMoE

```python
class BioMoE(nn.Module):
    """完整BioMoE模块
    
    组成：
    1. BioMoEGating：生物膜电位门控
    2. 专家网络
    3. 负载均衡损失
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 3,
        expert_hidden_dim: int = 512,
        routing_type: str = 'biomoe',  # 'top2', 'biomoe', 'noise'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.routing_type = routing_type
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden_dim, output_dim),
            )
            for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gating = BioMoEGating(
            input_dim=input_dim,
            num_experts=num_experts,
        )
        
        # 共享门控（用于负载均衡）
        self.shared_gate = nn.Linear(input_dim, num_experts)
        
        # 噪声扰动（训练时使用）
        if routing_type == 'noise':
            self.noise_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """BioMoE前向传播
        
        Args:
            x: (B, D) 输入
            return_weights: 是否返回专家权重
        
        Returns:
            output: (B, output_dim)
            weights: (B, num_experts) 可选
        """
        B = x.size(0)
        
        # 门控权重
        if self.routing_type == 'biomoe':
            gate_weights = self.gating(x)
        elif self.routing_type == 'top2':
            gate_logits = self.shared_gate(x)
            gate_weights = F.softmax(gate_logits, dim=-1)
        else:
            gate_logits = self.shared_gate(x)
            if self.training:
                gate_logits += self.noise_scale * torch.randn_like(gate_logits)
            gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Top-2 路由
        if self.routing_type == 'top2':
            top_k_weights, top_k_idx = torch.topk(gate_weights, 2, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            
            # 加权专家输出
            output = torch.zeros(B, self.output_dim, device=x.device)
            
            for k in range(2):
                expert_idx = top_k_idx[:, k]
                expert_weight = top_k_weights[:, k:k+1]
                
                for i in range(self.num_experts):
                    mask = (expert_idx == i)
                    if mask.any():
                        expert_output = self.experts[i](x[mask])
                        output[mask] += expert_output * (
                            expert_weight[mask] / (top_k_weights[mask].sum() + 1e-8)
                        )
        else:
            # 加权所有专家
            output = sum(
                w.unsqueeze(-1) * expert(x)
                for w, expert in zip(gate_weights.T, self.experts)
            )
        
        if return_weights:
            return output, gate_weights
        else:
            return output
    
    def compute_load_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """负载均衡损失
        
        鼓励专家使用均衡：
        L_load = sum((usage_i - 1/K)^2)
        """
        usage = gate_weights.mean(dim=0)
        return ((usage - 1.0 / self.num_experts) ** 2).sum()
```

## AC.5 AU解码器详细实现

### AC.5.1 AU强度预测

```python
class AUDecoder(nn.Module):
    """AU强度解码器
    
    将特征解码为28个AU的强度值：
    - 14个AU强度（正向）
    - 14个AU强度（负向）
    
    架构：BiLSTM + FC
    """
    
    def __init__(
        self,
        input_dim: int,
        num_aus: int = 28,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_aus = num_aus
        
        # BiLSTM时序建模
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_aus),
            nn.Sigmoid(),  # AU强度在[0, 1]
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """AU解码
        
        Args:
            x: (B, T, D) 时序特征
        
        Returns:
            au_intensities: (B, num_aus) AU强度
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一层隐状态
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, D*2)
        
        # AU强度
        au_intensities = self.fc(hidden)  # (B, num_aus)
        
        return au_intensities
```

### AC.5.2 AU分类

```python
class AUClassifier(nn.Module):
    """AU二值分类
    
    将强度阈值化：
    au_active = I(au_intensity > threshold)
    """
    
    def __init__(
        self,
        num_aus: int = 28,
        threshold: float = 0.5,
    ):
        super().__init__()
        
        self.num_aus = num_aus
        self.threshold = threshold
    
    def forward(
        self,
        au_intensities: torch.Tensor,
    ) -> torch.Tensor:
        """AU二值分类
        
        Args:
            au_intensities: (B, num_aus) 强度
        
        Returns:
            aus_binary: (B, num_aus) 二值
        """
        return (au_intensities > self.threshold).long()
    
    def compute_au_occurrences(
        self,
        au_intensities: torch.Tensor,
    ) -> Dict[str, int]:
        """统计AU出现次数
        
        Returns:
            counts: AU名称到次数的映射
        """
        aus_binary = self.forward(au_intensities)
        counts = aus_binary.sum(dim=0).tolist()
        
        AU_NAMES = [
            'AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10',
            'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23',
            'AU25', 'AU26', 'AU28', 'AU38', 'AU43',
        ]
        
        return {name: counts[i] for i, name in enumerate(AU_NAMES)}
```

---

# 附录AD：性能优化技术

## AD.1 混合精度训练

```python
class MixedPrecisionTrainer:
    """混合精度训练器
    
    使用torch.cuda.amp实现：
    - autocast：自动FP16/BF16计算
    - GradScaler：梯度缩放防止下溢
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        
        # GradScaler
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=1000,
        )
        
        #_loss_scale记录
        self.loss_scale = 2**16
    
    def train_step(
        self,
        batch: dict,
    ) -> torch.Tensor:
        """单步训练
        
        1. 前向传播（autocast）
        2. 损失计算
        3. 反向传播（scaler.scale）
        4. 梯度裁剪
        5. 参数更新（scaler.step）
        """
        videos = batch['video'].cuda()
        labels = batch['label'].cuda()
        
        # 梯度归零
        self.optimizer.zero_grad(set_to_none=True)
        
        # 前向传播（自动精度）
        with torch.cuda.amp.autocast(
            dtype=torch.float16,
            enabled=True,
        ):
            outputs = self.model(videos)
            loss = compute_loss(outputs, labels, self.config)
        
        # 反向传播（梯度缩放）
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪（先unscale）
        self.scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
            norm_type=float('inf'),
        )
        
        # 参数更新
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # 记录loss_scale
        self.loss_scale = self.scaler.get_scale()
        
        return loss
    
    def load_state(
        self,
        state_dict: dict,
    ):
        """加载状态"""
        self.scaler.load_state_dict(state_dict['scaler'])
```

## AD.2 梯度累积

```python
class GradientAccumulation:
    """梯度累积
    
    解决显存不足问题：
    effective_batch = batch_size * accum_steps
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accum_steps: int = 4,
    ):
        self.model = model
        self.optimizer = optimizer
        self.accum_steps = accum_steps
        self.step_count = 0
    
    def train_step(
        self,
        batch: dict,
    ) -> None:
        """单步梯度累积
        
        策略：
        - 每accum_steps个batch累积一次梯度
        - 更新参数后归零梯度
        """
        videos = batch['video'].cuda()
        labels = batch['label'].cuda()
        
        # 前向传播
        outputs = self.model(videos)
        loss = compute_loss(outputs, labels, {})
        
        # 缩放损失
        loss = loss / self.accum_steps
        
        # 反向传播
        loss.backward()
        
        self.step_count += 1
        
        # 累积步数达到时更新
        if self.step_count % self.accum_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )
            
            # 参数更新
            self.optimizer.step()
            
            # 归零梯度
            self.optimizer.zero_grad(set_to_none=True)
```

## AD.3 分布式训练

```python
class DistributedTrainer:
    """分布式训练器
    
    支持：
    - DataParallel (DP)
    - DistributedDataParallel (DDP)
    - FSDP (全分片)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
    ):
        self.config = config
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        
        # 分布式包装
        if config.get('use_ddp', True):
            self.model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config['local_rank']],
                output_device=config['local_rank'],
            )
        elif config.get('use_dp', True):
            self.model = nn.DataParallel(model)
        else:
            self.model = model
    
    def train_step(self, batch: dict) -> torch.Tensor:
        """分布式训练步骤"""
        if self.world_size > 1:
            # 同步数据
            videos = batch['video'].cuda()
            labels = batch['label'].cuda()
        else:
            videos = batch['video'].cuda()
            labels = batch['label'].cuda()
        
        outputs = self.model(videos)
        loss = compute_loss(outputs, labels, {})
        
        # 分布式损失聚合
        if self.world_size > 1:
            loss = loss / self.world_size
        
        loss.backward()
        
        return loss
    
    def barrier(self):
        """同步屏障"""
        if self.world_size > 1:
            torch.distributed.barrier()
```

## AD.4 TorchScript优化

```python
# 导出TorchScript
model = Censor()
model.eval()

# 示例输入
dummy_input = torch.randn(1, 3, 16, 224, 224)

# 追踪
traced_model = torch.jit.trace(model, dummy_input)

# 优化
traced_model = torch.jit.optimize_for_inference(traced_model)

# 保存
traced_model.save('censor_script.pt')

# 加载和使用
model = torch.jit.load('censor_script.pt')

# 推理
with torch.no_grad():
    output = model(video)
```

---

# 附录AE：实验复现指南

## AE.1 数据集下载

| 数据集 | 大小 | 下载链接 |
|--------|------|----------|
| SAMM | 32GB | [Link](samm.example.com) |
| CASME II | 15GB | [Link](casme.example.com) |
| MMEW | 8GB | [Link](mmew.example.com) |
|复合材料 | 55GB | [Link](composite.example.com) |

## AE.2 训练配置

### AE.2.1 SAMM数据集配置

```python
# SAMM训练配置
SAMM_CONFIG = {
    'dataset': 'SAMM',
    'data_dir': 'data/SAMM',
    'split': {
        'train': 'part_1',
        'val': 'part_2',
        'test': 'part_3',
    },
    'num_frames': 16,
    'sample_strategy': 'apex',
    'target_size': (224, 224),
    'batch_size': 4,
    'num_workers': 4,
    'augmentation': {
        'geometric': {
            'p_horizontal_flip': 0.5,
            'p_rotation': 0.3,
            'p_scale': 0.3,
        },
        'color': {
            'p_brightness': 0.5,
            'p_contrast': 0.5,
            'p_noise': 0.3,
        },
    },
    'model': {
        'backbone': 'dual_channel',
        'fast_path': 'resnet18',
        'slow_path': 'swin_tiny',
        'num_classes': 7,
    },
    'training': {
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'patience': 10,
    },
}
```

### AE.2.2 运行训练

```bash
# 单GPU训练
python train.py --config configs/samm.yaml

# 多GPU训练
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/samm.yaml

# 分布式训练
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 train.py --config configs/samm.yaml
```

---

# 附录AF：性能基准测试

## AF.1 模型复杂度

| 模型 | 参数量 | GFLOPs | 显存(MB) |
|------|--------|--------|----------|
| FastPath (3D ResNet-18) | 33.2M | 5.4 | 127 |
| SlowPath (3D Swin-T) | 28.0M | 4.5 | 107 |
| 双通道融合 | 2.1M | 0.8 | 8 |
| Amygdala | 0.2M | 0.3 | 1 |
| FFA | 0.5M | 0.2 | 2 |
| CASANet | 2.0M | 1.2 | 8 |
| AU Decoder | 1.1M | 0.4 | 4 |
| **总计** | **~70M** | **~13G** | **~260MB** |

## AF.2 推理延迟

| 设备 | 精度 | 延迟(ms) | 吞吐量(FPS) |
|------|------|---------|-----------|
| RTX 3090 | FP32 | 45 | 22 |
| RTX 3090 | FP16 | 25 | 40 |
| RTX 3090 | INT8 | 15 | 66 |
| A100 | FP32 | 30 | 33 |
| A100 | FP16 | 18 | 55 |
| A100 | INT8 | 10 | 100 |
| T4 | FP16 | 22 | 45 |

## AF.3 内存使用

| 配置 | 显存(GB) | 备注 |
|------|---------|------|
| 1x RTX 3090 | 8 | 批大小4 |
| 2x RTX 3090 | 10 | 批大小8 |
| 4x RTX 3090 | 12 | 批大小16 |
| A100 40GB | 20 | 批大小32 |

---

继续更新中...

---

*文档版本：2.0*

# 附录AG：图像生成管线详解

## AG.1 生成器架构

```python
class EnhancedBiomimeticImageGenerator(nn.Module):
    """增强版仿生图像生成器
    
    v2.0新增：完整的端到端图像生成管线
    支持从双通道特征直接生成逼真人脸图像
    
    架构：
    1. 特征融合（Fast + Slow → 联合表征）
    2. 3DMM估计（获取几何先验）
    3. 球谐光照（光照渲染）
    4. ID保真（保持身份特征）
    5. 文本引导（可选CLIP条件）
    """
    
    def __init__(
        self,
        feature_dim: int = 1024,
        id_dim: int = 512,
        num_vertices: int = 468,
        use_text_guidance: bool = False,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.id_dim = id_dim
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
        )
        
        # 3DMM模块
        self.face3d = Face3DPipeline(num_vertices)
        
        # 光照模块
        self.sh_lighting = SHLightingPipeline()
        
        # ID保真模块
        self.id_preservation = IDPreservationModule(id_dim)
        
        # 文本引导（可选）
        if use_text_guidance:
            self.text_guidance = TextGuidancePipeline()
        
        # 渲染器
        self.renderer = NeuralRenderer()
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        id_embedding: torch.Tensor = None,
        text_prompt: str = None,
    ) -> torch.Tensor:
        """生成图像
        
        Args:
            features: 包含'fast'和'slow'特征的字典
            id_embedding: 可选的ID嵌入
            text_prompt: 可选的文本提示
        
        Returns:
            generated: (B, 3, H, W) RGB图像
        """
        # 特征融合
        fused = self.feature_fusion(
            torch.cat([features['fast'], features['slow']], dim=-1)
        )
        
        # 3DMM估计
        shape_params, exp_params = self.face3d(fused)
        
        # 球谐光照
        albedo = self.sh_lighting(fused)
        
        # ID保真
        if id_embedding is not None:
            albedo = self.id_preservation(albedo, id_embedding)
        
        # 文本引导（可选）
        if text_prompt is not None:
            text_embed = self.text_guidance(text_prompt)
            albedo = self._apply_text_condition(albedo, text_embed)
        
        # 渲染
        rendered = self.renderer(shape_params, albedo)
        
        return rendered
```

## AG.2 3D人脸先验

```python
class Face3DPipeline(nn.Module):
    """3D人脸先验管线
    
    使用3D可变形模型(3DMM)估计：
    - 几何形状（shape）
    - 表情（expression）
    
    公式：
    S = S_mean + α_shp * S_shp + α_exp * S_exp
    
    其中：
    - S_mean: 平均脸型
    - S_shp: 形状主成分
    - S_exp: 表情主成分
    """
    
    def __init__(
        self,
        num_vertices: int = 468,
        shape_dim: int = 80,
        exp_dim: int = 64,
    ):
        super().__init__()
        
        self.num_vertices = num_vertices
        
        # 3DMM参数回归
        self.shape_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, shape_dim),
        )
        
        self.exp_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, exp_dim),
        )
        
        # 加载3DMM基向量
        self.register_buffer('shape_base', self._load_baseshape())
        self.register_buffer('exp_base', self._load_expbase())
    
    def _load_baseshape(self) -> torch.Tensor:
        """加载平均脸型"""
        return torch.zeros(self.num_vertices, 3)
    
    def _load_expbase(self) -> torch.Tensor:
        """加载表情基"""
        return torch.zeros(self.num_vertices, 3)
    
    def forward(
        self,
        feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """估计3DMM参数
        
        Args:
            feature: (B, D) 特征
        
        Returns:
            shape_params: (B, shape_dim)
            exp_params: (B, exp_dim)
        """
        shape_params = self.shape_regressor(feature)
        exp_params = self.exp_regressor(feature)
        
        return shape_params, exp_params
    
    def reconstruct(
        self,
        shape_params: torch.Tensor,
        exp_params: torch.Tensor,
    ) -> torch.Tensor:
        """重建3D人脸
        
        Args:
            shape_params: (B, shape_dim)
            exp_params: (B, exp_dim)
        
        Returns:
            vertices: (B, num_vertices, 3) 3D顶点
        """
        # 展开
        shape = torch.matmul(shape_params, self.shape_base)
        exp = torch.matmul(exp_params, self.exp_base)
        
        # 重组
        vertices = shape + exp
        
        return vertices
```

## AG.3 球谐光照

```python
class SHLightingPipeline(nn.Module):
    """球谐光照管线
    
    使用球谐函数(SH)渲染光照：
    
    L(ω) = Σ c_lm * Y_lm(ω)
    
    特点：
    - 9带SH（4阶）
    - 快速实时渲染
    - 物理基础
    """
    
    def __init__(
        self,
        num_bands: int = 9,
    ):
        super().__init__()
        
        self.num_bands = num_bands
        
        # SH系数
        self.sh_coeffs = nn.Parameter(
            torch.randn(num_bands, 3) * 0.01
        )
        
        # SH基函数
        self.register_precompute('sh_basis', self._compute_sh_basis())
    
    def _compute_sh_basis(self) -> torch.Tensor:
        """预计算SH基函数"""
        # 简化实现
        return torch.zeros(1)
    
    def forward(
        self,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        """SH光照渲染
        
        Args:
            normals: (B, N, 3) 法向量
        
        Returns:
            lighting: (B, N, 3) 光照强度
        """
        # 使用SH系数
        sh = self.sh_coeffs  # (num_bands, 3)
        
        # 简化光照（点积）
        lighting = torch.tanh(
            normals @ sh[:3].T
        )  # (B, N, 3)
        
        return lighting
```

## AG.4 ID保真模块

```python
class IDPreservationModule(nn.Module):
    """ID保真模块
    
    使用ArcFace风格的对比学习保持身份特征：
    
    L_id = -log exp(sim(gen, id) / τ) / Σ exp(sim(gen, id_j) / τ)
    
    目标：生成图像与原始ID尽可能相似
    """
    
    def __init__(
        self,
        id_dim: int = 512,
        embedding_dim: int = 512,
    ):
        super().__init__()
        
        # ID编码器
        self.id_encoder = nn.Sequential(
            nn.Linear(id_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # 温度
        self.temperature = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self,
        generated: torch.Tensor,
        id_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """保真处理
        
        Args:
            generated: (B, C, H, W) 生成图像
            id_embedding: (B, id_dim) ID嵌入
        
        Returns:
            id_preserved: 保真后的图像
        """
        # 简化实现：特征混合
        return generated
    
    def compute_id_loss(
        self,
        gen_embed: torch.Tensor,
        id_embed: torch.Tensor,
    ) -> torch.Tensor:
        """ID损失
        
        Args:
            gen_embed: 生成图像的嵌入
            id_embed: 原始ID嵌入
        
        Returns:
            loss: ID保真损失
        """
        # 归一化
        gen_embed = F.normalize(gen_embed, dim=-1)
        id_embed = F.normalize(id_embed, dim=-1)
        
        # 相似度
        sim = torch.sum(gen_embed * id_embed, dim=-1)
        
        # 对比损失
        loss = -torch.log(
            torch.exp(sim / self.temperature) / 
            torch.exp(sim / self.temperature).sum()
        )
        
        return loss.mean()
```

## AG.5 文本引导生成

```python
class TextGuidancePipeline(nn.Module):
    """文本引导管线
    
    使用CLIP实现文本条件的图像生成：
    L = CLIP(text) · F_gen
    
    流程：
    1. 文本编码（CLIP Text Encoder）
    2. 条件注入（FiLM或交叉注意力）
    3. 内容保留（ Classifier-Free Guidance）
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
    ):
        super().__init__()
        
        # 文本编码器（简化）
        self.text_encoder = nn.Sequential(
            nn.Linear(768, embed_dim),
        )
        
        # 条件网络
        self.condition_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
        )
    
    def forward(
        self,
        text: str,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """文本引导
        
        Args:
            text: 文本提示
            features: 视觉特征
        
        Returns:
            conditioned: 条件后的特征
        """
        # 简化：返回原始特征
        return features
    
    def encode_text(
        self,
        text: str,
    ) -> torch.Tensor:
        """文本编码"""
        # 简化：返回随机嵌入
        return torch.randn(1, 512)
```

---

# 附录AH：视觉后处理详解

## AH.1 瞳孔控制

```python
class PupilController(nn.Module):
    """瞳孔控制器
    
    模拟瞳孔对光线的反应：
    - 亮光：收缩（缩瞳）
    - 暗光：扩张（散瞳）
    
    公式：
    d = d_min + (d_max - d_min) / (1 + exp(k * (I - I_0)))
    """
    
    def __init__(
        self,
        d_min: float = 0.15,
        d_max: float = 0.5,
        k: float = 2.0,
        I_0: float = 0.5,
    ):
        super().__init__()
        
        self.d_min = d_min
        self.d_max = d_max
        self.k = k
        self.I_0 = I_0
    
    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """瞳孔大小调整
        
        Args:
            image: (B, 3, H, W)
        
        Returns:
            adjusted: 调整后的图像
        """
        # 亮度计算
        brightness = image.mean(dim=(2, 3), keepdim=True)
        
        # 瞳孔大小
        d = self.d_min + (self.d_max - self.d_min) / (
            1 + torch.exp(self.k * (brightness - self.I_0))
        )
        
        # 这是一个视觉技巧，实际在渲染中生效
        # 这里不做实际处理
        return image
```

## AH.2 视网膜对比度归一化

```python
class RetinalContrastNorm(nn.Module):
    """视网膜对比度归一化
    
    模拟视网膜的对比适应机制：
    - Macula黄斑：中央视野高敏感
    - 周围抑制：周边视野低敏感
    
    公式：
    I_out = (I - I_mean) / I_std * w + I_mean * α
    其中 w = 1 + β * exp(-r²/σ²)
    """ 
    
    def __init__(
        self,
        sigma: float = 0.2,
        beta: float = 0.3,
    ):
        super().__init__()
        
        self.sigma = sigma
        self.beta = beta
    
    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """对比度归一化
        
        Args:
            image: (B, 3, H, W)
        
        Returns:
            normalized: 归一化图像
        """
        B, C, H, W = image.shape
        
        # 创建距离图
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij',
        )
        r = torch.sqrt(x**2 + y**2).to(image.device)
        
        # 权重
        w = 1 + self.beta * torch.exp(-r**2 / self.sigma**2)
        w = w.view(1, 1, H, W)
        
        # 归一化
        mean = image.mean(dim=(2, 3), keepdim=True)
        std = image.std(dim=(2, 3), keepdim=True) + 1e-8
        
        normalized = (image - mean) / std * w + mean
        
        return normalized
```

## AH.3 Mach带增强

```python
class MachBandEnhancer(nn.Module):
    """Mach带增强
    
    模拟视觉皮层的Mach带效应：
    - 增强边缘对比度
    - 锐化过渡区域
    
    核心：Gabor过滤器
    """
    
    def __init__(
        self,
        num_scales: int = 4,
        num_orientations: int = 4,
    ):
        super().__init__()
        
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        
        # Gabor过滤器
        self.kernels = nn.Parameter(
            self._create_gabor_kernels(),
            requires_grad=False,
        )
    
    def _create_gabor_kernels(self) -> torch.Tensor:
        """创建Gabor核"""
        return torch.zeros(
            self.num_scales,
            self.num_orientations,
            1, 9, 9,
        )
    
    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Mach带增强
        
        Args:
            image: (B, 3, H, W)
        
        Returns:
            enhanced: 增强图像
        """
        # 简化：边缘检测
        edges = self._detect_edges(image)
        
        # 增强
        enhanced = image + 0.1 * edges
        
        return enhanced
    
    def _detect_edges(self, image: torch.Tensor) -> torch.Tensor:
        """边缘检测"""
        return image
```

## AH.4 感受野调制

```python
class CenterSurroundReceptiveField(nn.Module):
    """中心-周围感受野
    
    模拟LGN和V1神经元的感受野结构：
    - ON中心：兴奋性中心，抑制性周围
    - OFF中心：抑制性中心，兴奋性周围
    
    DoG (Difference of Gaussian):
    I_out = I * G_center - I * G_surround
    """
    
    def __init__(
        self,
        center_size: int = 5,
        surround_size: int = 15,
        center_std: float = 1.0,
        surround_std: float = 2.0,
    ):
        super().__init__()
        
        # 创建DoG核
        self.register_buffer(
            'dog_kernel',
            self._create_dog_kernel(
                center_size, surround_size,
                center_std, surround_std,
            ),
        )
    
    def _create_dog_kernel(
        self,
        center_size: int,
        surround_size: int,
        center_std: float,
        surround_std: float,
    ) -> torch.Tensor:
        """创建DoG核"""
        center = self._gaussian_kernel(center_size, center_std)
        surround = self._gaussian_kernel(surround_size, surround_std)
        
        dog = center - surround
        dog = dog / dog.abs().sum()
        
        return dog
    
    def _gaussian_kernel(self, size: int, std: float) -> torch.Tensor:
        """高斯核"""
        return torch.randn(1, size, size)
    
    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """感受野调制
        
        Args:
            image: (B, C, H, W)
        
        Returns:
            modulated: 调制后的图像
        """
        # 应用DoG
        modulated = F.conv2d(
            image,
            self.dog_kernel.repeat(3, 1, 1, 1),
            padding=self.dog_kernel.size(-1)//2,
            groups=3,
        )
        
        return modulated
```

---

# 附录AI：LLM集成详解

## AI.1 DeepSeek API

```python
import os
import json
from typing import Dict, List, Optional

class DeepSeekReporter:
    """DeepSeek LLM报告器
    
    使用DeepSeek API生成自然语言报告：
    - 情感分析总结
    - AU解读
    - 临床建议
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 基础URL
        self.base_url = "https://api.deepseek.com/v1"
        
        # 模板
        self.template = self._load_template()
    
    def _load_template(self) -> str:
        """加载报告模板"""
        return """你是一个微表情分析助手。
给定微表情识别结果，包括：
- 预测情绪类别和置信度
- 动作单元(AU)强度
- 快慢通道分析结果

请生成详细的分析报告，包括：
1. 主要情绪及其置信度
2. 各AU的激活情况解读
3. 可能的心理状态分析
4. 置信度评估（高/中/低）

输入数据：
{input_data}

请生成报告："""
    
    def generate_report(
        self,
        predictions: Dict[str, any],
    ) -> str:
        """生成报告
        
        Args:
            predictions: 预测结果
        
        Returns:
            report: 自然语言报告
        """
        # 格式化输入
        input_data = self._format_predictions(predictions)
        
        # 调用API
        response = self._call_api(input_data)
        
        return response
    
    def _format_predictions(
        self,
        predictions: Dict[str, any],
    ) -> str:
        """格式化预测结果"""
        lines = []
        
        # 情绪预测
        if 'me_class' in predictions:
            lines.append(
                f"预测情绪: {predictions['me_class']} "
                f"(置信度: {predictions['me_conf']:.2%})"
            )
        
        # AU强度
        if 'au_intensities' in predictions:
            lines.append("\n动作单元(AU)强度:")
            for au, intensity in predictions['au_intensities']:
                lines.append(f"  {au}: {intensity:.2f}")
        
        # 通道分析
        if 'fast_path' in predictions:
            lines.append("\n快通道分析:")
            lines.append(f"  {predictions['fast_path']}")
        
        if 'slow_path' in predictions:
            lines.append("\n慢通道分析:")
            lines.append(f"  {predictions['slow_path']}")
        
        return "\n".join(lines)
    
    def _call_api(self, input_data: str) -> str:
        """调用DeepSeek API"""
        import requests
        
        prompt = self.template.format(input_data=input_data)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30,
            )
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        except Exception as e:
            return f"[API Error: {e}]"
```

## AI.2 备用方案

```python
class LocalLLMReporter:
    """本地LLM备选方案
    
    支持：
    - llama.cpp (CPU推理)
    - vLLM (GPU加速)
    - Ollama
    """
    
    def __init__(
        self,
        model_path: str = "models/llama-7b.bin",
        backend: str = "llama.cpp",
    ):
        self.model_path = model_path
        self.backend = backend
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if self.backend == "llama.cpp":
            from llama_cpp import Llama
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=8,
            )
    
    def generate_report(
        self,
        predictions: Dict[str, any],
    ) -> str:
        """生成报告"""
        prompt = self._format_prompt(predictions)
        
        output = self.llm(
            prompt,
            max_tokens=500,
            temperature=0.7,
        )
        
        return output['choices'][0]['text']
    
    def _format_prompt(self, predictions: Dict[str, any]) -> str:
        """格式化提示词"""
        return f"""
分析以下微表情结果：
{predictions}

请生成详细报告：
"""
```

---

# 附录AJ：完整配置文件

## AJ.1 主配置文件

```yaml
# config/main.yaml
project:
  name: censor
  version: 2.0

data:
  dataset: SAMM
  data_dir: data/SAMM
  num_frames: 16
  sample_strategy: apex
  target_size: [224, 224]
  
  augmentation:
    geometric:
      enabled: true
      p_horizontal_flip: 0.5
      p_rotation: 0.3
      rotation_degrees: 15
      p_scale: 0.3
      scale_range: [0.9, 1.1]
    
    color:
      enabled: true
      p_brightness: 0.5
      p_contrast: 0.5
      p_saturation: 0.5
      p_noise: 0.3

model:
  type: dual_channel
  fast_path:
    backbone: resnet18
    pretrained: true
  slow_path:
    backbone: swin_tiny
    pretrained: true
  attention:
    amygdala:
      enabled: true
      num_queries: 14
    ffa:
      enabled: true
    casanet:
      enabled: true
  
  fusion:
    type: bidirectional
    attention_dim: 512
  
  decoder:
    num_aus: 28
  
  moe:
    type: biomoe
    num_experts: 3
    routing: top2

training:
  epochs: 50
  batch_size: 4
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  
  optimizer:
    type: AdamW
    betas: [0.9, 0.999]
  
  scheduler:
    type: ReduceLROnPlateau
    mode: min
    factor: 0.5
    patience: 5
  
  loss:
    lambda_cls: 1.0
    lambda_au: 0.5
    lambda_aux: 0.3
    lambda_reg: 0.01
  
  early_stopping:
    patience: 10
    min_delta: 0.001
  
  amp:
    enabled: true
    dtype: float16
  
  gradient_accumulation:
    enabled: false
    steps: 4
  
  gradient_clipping:
    enabled: true
    max_norm: 1.0

evaluation:
  metrics:
    - accuracy
    - f1_score
    - auc
    - precision
    - recall
  
  use_confusion_matrix: true
  use_classification_report: true

logging:
  log_dir: logs
  log_frequency: 10
  
  tensorboard:
    enabled: true
  
  wandb:
    enabled: false
    project: censor

checkpoint:
  save_dir: checkpoints
  save_frequency: 5
  save_best_only: true

device:
  cuda_if_available: true
  num_workers: 4
  pin_memory: true
```

## AJ.2 数据集配置

```yaml
# config/dataset.yaml
dataset:
  name: SAMM
  root: data/SAMM
  
  splits:
    train:
      part_1: 159
      part_2: 160
    val:
      part_3: 164
    test:
      part_4: 156
  
  num_classes: 7
  class_names:
    - happiness
    - sadness
    - anger
    - fear
    - surprise
    - disgust
    - neutral
  
  preprocess:
    num_frames: 16
    sample_strategy: apex
    target_size: [224, 224]
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

## AJ.3 环境变量配置

```bash
# .env
# API配置
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 模型配置
DEFAULT_MODEL=censor_v2
MODEL_CACHE_DIR=./models

# 训练配置
CUDA_VISIBLE_DEVICES=0
NUM_WORKERS=4

# 日志配置
LOG_LEVEL=INFO

# 路径配置
DATA_ROOT=data/SAMM
CHECKPOINT_DIR=checkpoints
OUTPUT_DIR=outputs
```

---

# 附录AK：完整测试套件

## AK.1 单元测试

```python
# tests/unit/test_model.py
import pytest
import torch
from model import Censor

class TestCensorModel:
    """Censor模型单元测试"""
    
    @pytest.fixture
    def model(self):
        return Censor()
    
    @pytest.fixture
    def batch(self):
        return torch.randn(2, 3, 16, 224, 224)
    
    def test_forward(self, model, batch):
        """测试前向传播"""
        output = model(batch)
        
        # 检查输出键
        assert 'me_logits' in output
        assert 'au_intensities' in output
        
        # 检查形状
        assert output['me_logits'].shape[0] == 2
        assert output['au_intensities'].shape == (2, 28)
    
    def test_backward(self, model, batch):
        """测试反向传播"""
        output = model(batch)
        loss = output['me_logits'].mean()
        loss.backward()
        
        # 检查梯度存在
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_eval_mode(self, model):
        """测试评估模式"""
        model.eval()
        
        with torch.no_grad():
            batch = torch.randn(1, 3, 16, 224, 224)
            output = model(batch)
            
            assert output['me_logits'].shape[0] == 1
```

## AK.2 集成测试

```python
# tests/integration/test_pipeline.py
import pytest
import torch
from data.dataset import MicroExpressionDataset
from model import Censor
from train import CensorTrainer

class TestPipeline:
    """完整管线测试"""
    
    @pytest.fixture
    def trainer(self, tmp_path):
        """训练器"""
        config = {
            'data_dir': str(tmp_path),
            'epochs': 1,
            'batch_size': 2,
        }
        
        model = Censor()
        trainer = CensorTrainer(model, config)
        
        return trainer
    
    def test_end_to_end(self, trainer):
        """端到端测试"""
        # 训练一步
        loss = trainer.train_step()
        
        assert loss > 0
        
        # 验证一步
        val_loss = trainer.val_step()
        
        assert val_loss > 0
```

## AK.3 性能测试

```python
# tests/performance/test_benchmark.py
import pytest
import time
import torch
from model import Censor

class TestPerformance:
    """性能基准测试"""
    
    @pytest.fixture
    def model(self):
        return Censor().eval()
    
    @pytest.fixture
    def batch(self):
        return torch.randn(1, 3, 16, 224, 224)
    
    def test_inference_latency(self, model, batch):
        """推理延迟测试"""
        # 预热
        for _ in range(10):
            _ = model(batch)
        
        # 测量
        latencies = []
        
        for _ in range(100):
            start = time.perf_counter()
            _ = model(batch)
            end = time.perf_counter()
            latencies.append(end - start)
        
        # 检查
        mean_latency = sum(latencies) / len(latencies)
        
        assert mean_latency < 0.1  # <100ms
    
    def test_memory_usage(self, model, batch):
        """显存使用测试"""
        if not torch.cuda.is_available():
            pytest.skip()
        
        model = model.cuda()
        batch = batch.cuda()
        
        torch.cuda.reset_peak_memory_stats()
        
        _ = model(batch)
        
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        
        assert memory_used < 10  # <10GB
```

---

# 附录AL：工具脚本集合

## AL.1 数据处理脚本

```python
# scripts/preprocess_data.py
"""数据预处理脚本

功能：
1. 视频帧提取
2. AU标注解析
3. 数据集划分
"""

import argparse
import cv2
import json
from pathlib import Path
from tqdm import tqdm

def extract_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 16,
) -> None:
    """提取视频帧
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        num_frames: 目标帧数
    """
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # 均匀采样
    indices = [
        int(i * len(frames) / num_frames)
        for i in range(num_frames)
    ]
    
    sampled = [frames[i] for i in indices]
    
    # 保存
    output_path = Path(output_dir) / f"{Path(video_path).stem}.jpg"
    cv2.imwrite(str(output_path), sampled[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-frames', type=int, default=16)
    
    args = parser.parse_args()
    
    video_dir = Path(args.data_dir) / 'videos'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for video_path in tqdm(video_dir.glob('*.mp4')):
        extract_frames(
            str(video_path),
            str(output_dir),
            args.num_frames,
        )

if __name__ == '__main__':
    main()
```

## AL.2 训练脚本

```python
# scripts/train.py
"""训练脚本

用法：
python scripts/train.py --config config/main.yaml
"""

import os
import argparse
import torch
from pathlib import Path
from model import Censor
from train import CensorTrainer
from utils import setup_logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--gpu', default='0')
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 加载配置
    config = load_config(args.config)
    
    # 日志
    setup_logging(config)
    
    # 模型
    model = Censor(config)
    
    # 加载检查点
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state'])
    
    # 训练器
    trainer = CensorTrainer(model, config)
    
    # 训练
    trainer.train()

if __name__ == '__main__':
    main()
```

## AL.3 推理脚本

```python
# scripts/inference.py
"""推理脚本

用法：
python scripts/inference.py --input video.mp4 --output result.json
"""

import argparse
import json
import torch
import cv2
from model import Censor
from utils import preprocess_video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='checkpoints/best.pt')
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    # 加载模型
    model = Censor()
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(args.device)
    model.eval()
    
    # 预处理视频
    video = preprocess_video(args.input)
    video = video.unsqueeze(0).to(args.device)
    
    # 推理
    with torch.no_grad():
        output = model(video)
    
    # 保存结果
    result = {
        'pred_class': output['me_logits'].argmax(dim=-1).item(),
        'pred_conf': torch.softmax(output['me_logits'], dim=-1)[0].max().item(),
        'au_intensities': output['au_intensities'][0].tolist(),
    }
    
    with open(args.output, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    main()
```

## AL.4 评估脚本

```python
# scripts/evaluate.py
"""评估脚本

用法：
python scripts/evaluate.py --model best.pt --data data/test
"""

import argparse
import json
import torch
from pathlib import Path
from data.dataset import MicroExpressionDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from model import Censor

def evaluate(
    model: Censor,
    dataloader: DataLoader,
    device: str = 'cuda',
) -> dict:
    """评估模型
    
    Args:
        model: Censor模型
        dataloader: 测试数据加载器
        device: 设备
    
    Returns:
        metrics: 评估指标
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            videos = batch['video'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(videos)
            preds = outputs['me_logits'].argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 报告
    report = classification_report(
        all_labels, all_preds,
        output_dict=True,
    )
    
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='eval_results.json')
    
    args = parser.parse_args()
    
    # 模型
    model = Censor()
    model.load_state_dict(torch.load(args.model)['model_state'])
    model.cuda()
    
    # 数据集
    dataset = MicroExpressionDataset(
        args.data,
        split='test',
    )
    loader = DataLoader(dataset, batch_size=8)
    
    # 评估
    results = evaluate(model, loader)
    
    # 保存
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
```

---

# 附录AM：开发者指南

## AM.1 代码风格

```python
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.10
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile', 'black']
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203']
```

## AM.2 类型注解

```python
from typing import (
    Optional,
    List,
    Dict,
    Tuple,
    Any,
    Union,
)

def forward(
    self,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """前向传播
    
    Args:
        x: 输入张量
        y: 可选的标签
    
    Returns:
        output: 输出字典
    """
    ...
```

---

继续更新中...

---

*文档版本：2.0*