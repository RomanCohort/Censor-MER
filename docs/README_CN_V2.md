# Censor: 仿生双通道微表情识别与图像生成系统

> **仿生双通道微表情识别与图像生成系统** — 基于PyTorch实现的仿生双通道架构，用于微表情识别和人脸图像生成，模拟人类视觉通路中的梭状回-杏仁核回路。

---

## 目录

- [概述](#概述)
- [新功能 (v2.0)](#新功能-v20)
- [核心创新](#核心创新)
- [系统架构](#系统架构)
- [数学公式](#数学公式)
- [图像生成管线](#图像生成管线)
- [视觉后处理](#视觉后处理)
- [LLM集成](#llm集成)
- [基准数据集](#基准数据集)
- [性能对比](#性能对比)
- [配置选项](#配置选项)
- [快速开始](#快速开始)
- [Python API](#python-api)
- [项目结构](#项目结构)
- [训练](#训练)
- [常见问题](#常见问题)
- [引用](#引用)

---

## 概述

微表情（ME）是一种短暂的、不自主的面部表情，当人们试图抑制或隐藏真实情绪时会发生。微表情持续时间仅为**40-200毫秒**，其特征是通过面部动作编码系统（FACS）中的动作单元（AU）衡量的细微肌肉活动。

**Censor** 提出了一种仿生双通道架构，模拟人类视觉系统的快速皮层下通路和慢速皮层通路，并配备增强版图像生成管线：

| 指标 | 数值 |
|------|------|
| 识别参数量 | ~68M |
| 生成参数量 | ~122M |
| 快通道延迟 | ~15ms |
| 慢通道延迟 | ~45ms |

---

## 新功能 (v2.0)

### 🔥 图像生成（全新！）

v2.0版本引入了完整的图像生成管线，可以从双通道特征生成逼真的人脸图像：

| 模块 | 描述 | 参数量 | 创新点 |
|------|------|--------|--------|
| **EnhancedBiomimeticImageGenerator** | 统一增强版生成器 | 121.7M | SE门控+多模块融合 |
| **Face3DPipeline** | 3D人脸先验 | ~2M | 3DMM几何约束 |
| **SHLightingPipeline** | 球谐光照 | ~0.5M | 9带SH光照渲染 |
| **TextGuidancePipeline** | 文本引导 | ~0.3M | CLIP文本条件 |
| **IDPreservationModule** | ID保真 | ~1M | ArcFace风格 |
| **VisualPerceptionPostProcess** | 视觉后处理 | ~0.1M | 仿生生理机制 |

### 🤖 LLM增强（更新）

- **主模型**: DeepSeek API（云端）
- **备用模型**: OPT-125M（本地）
- 支持自动回退
- 带临床描述的情绪报告生成

### 🎨 综合前端

基于Streamlit的集成平台，包含：
- 🎨 图像生成
- 😐 情绪识别
- 📝 LLM报告
- 📊 训练监测
- ⚙️ 模型管理
- 🔧 参数设置

---

## 核心创新

### 1. 仿生双通道机制

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

### 2. 注意力机制创新

| 模块 | 类型 | 功能 |
|------|------|------|
| **Amygdala** | 情绪先验 | 14×14注意力图 |
| **FFA** | SE门控 | 跨通道特征融合 |
| **CASANet** | 三角注意 | apex帧检测 |

### 3. BioMoE门控

基于生物神经元膜电位的门控机制：

```python
# 膜电位累积
membrane_potential = membrane_potential * decay_rate + feedback * (1 - decay_rate)

# 门控基于膜电位
gating = sigmoid(weight @ membrane_potential)
```

### 4. 图像生成创新

- **3D先验**：基于3DMM的几何约束
- **球谐光照**：9带球谐函数光照估计
- **ID保持**：ArcFace风格身份保持
- **文本条件**：CLIP引导的生成

---

## 系统架构

```
输入视频 (B×3×16×224×224)
  │
  ├── [阶段1] 仿生预处理
  │   ├── SaliencyDetector（高斯金字塔显著性检测）
  │   │   └── 模拟视网膜中心凹采样
  │   ├── rPPGExtractor（远程光电容积脉搏波）
  │   │   └── 心率估计 + 血流量分析
  │   └── TVL1OpticalFlow（光流）
  │       └── 动作检测
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

## 数学公式

### 双通道融合

```python
# 融合公式
fused = SE_block(concat(fast, slow))

# SE门控
channel_weights = sigmoid(W2 @ ReLU(W1 @ pooled))
fused = fused * channel_weights
```

### AU解码

```python
# AU强度预测
au_logits = W_au @ LSTM(hidden_states)
au_intensities = sigmoid(au_logits)  # (B, T, 28)
```

### MoE路由

```python
# 门控计算
gating = softmax(W_g @ fused)

# Top-k专家选择
top_k_indices = topk(gating, k=2)
selected_experts = experts[top_k_indices]

# 加权输出
output = sum(gating[k] * expert_k(output) for k in top_k_indices)
```

### 3DMM估计

```python
# 人脸顶点
vertices = mean_face + ShapeBasis @ shape_coeffs + ExprBasis @ expr_coeffs
```

### 球谐光照

```python
# SH基函数评估
sh_basis = evaluate_sh(normals, degree=2)  # 9 bands

# 光照渲染
lit = SH_basis @ lighting_coeffs
```

---

## 图像生成管线

### 增强版生成流程

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
  │   │   └── 模拟瞳孔光照适应
  │   ├── RetinalContrastNorm
  │   │   └── 模拟视网膜对比度适应
  │   ├── MachBandEnhancer
  │   │   └── 模拟Mach带边缘锐化
  │   └── CenterSurroundReceptiveField
  │       └── 模拟感受野边缘检测
  │
  ▼
输出: 生成的人脸图像 (224×224×3)
```

### 模块详解

#### DualPathwayFusion

```python
class DualPathwayFusion(nn.Module):
    def forward(self, fast_feat, slow_feat):
        # 拼接
        joint = torch.cat([fast_feat, slow_feat], dim=-1)
        
        # SE门控
        s = self.squeeze(joint)
        s = self.relu(s)
        gate = self.sigmoid(self.excitation(s))
        
        return joint * gate
```

#### Face3DPipeline

```python
class Face3DPipeline(nn.Module):
    def forward(self, features):
        # 估计3DMM参数
        mesh_params = self.mesh_estimator(features)
        
        # 生成网格
        vertices = self.mesh_generator(
            mesh_params['shape_coeffs'],
            mesh_params['expr_coeffs']
        )
        
        # 法线图
        normal_map = self.normal_mapper(vertices)
        
        return vertices, normal_map
```

#### SHLightingPipeline

```python
class SHLightingPipeline(nn.Module):
    def forward(self, features, normal_map):
        # 估计9个SH带系数
        sh_coeffs = self.estimator(features)
        
        # 渲染
        lit = self.renderer(normal_map, sh_coeffs)
        
        return lit
```

---

## 视觉后处理

### 仿生机制详解

#### 1. PupilController（瞳孔控制器）

**生物学基础**：瞳孔根据光照强度收缩或扩张

```python
class PupilController(nn.Module):
    def forward(self, x):
        # 估计光照
        illumination = x.mean(dim=[1,2,3], keepdim=True)
        
        # 预测瞳孔扩张因子
        dilation = self.fc2(F.relu(self.fc1(illumination)))
        
        # 增益 = 基础增益 + 扩张 * 调制范围
        gain = self.base_gain + dilation * self.modulation_range
        
        return x * gain
```

#### 2. RetinalContrastNorm（视网膜对比度归一化）

**生物学基础**：视网膜适应不同光照条件（Weber-Fechner定律）

```python
class RetinalContrastNorm(nn.Module):
    def forward(self, x):
        # 局部均值
        mean = F.avg_pool2d(x, kernel,...)
        
        # 局部标准差
        std = sqrt(E[x²] - E[x]²)
        
        # 归一化
        normalized = alpha * (x - mean) / (std + eps) + beta
        
        return normalized
```

#### 3. MachBandEnhancer（增强器）

**生物学基础**：Mach带效应 - 边缘主观增强

```python
class MachBandEnhancer(nn.Module):
    def forward(self, x):
        # 一阶导数
        dx = conv2d(x, kernel_x)
        dy = conv2d(x, kernel_y)
        
        # Mach带效应
        mach_effect = strength * (sign(dx)*|dx| + sign(dy)*|dy|)
        
        return x + mach_effect
```

#### 4. CenterSurroundReceptiveField（感受野）

**生物学基础**：视网膜神经节细胞的中心-环绕感受野

```python
class CenterSurroundReceptiveField(nn.Module):
    def forward(self, x):
        # DoG滤波
        response = conv2d(x, DoG_kernel)
        
        return response
```

---

## LLM集成

### DeepSeek API

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

### 环境变量配置

```bash
# 方式1: 使用DeepSeek
export DEEPSEEK_API_KEY="sk-xxxxxxxx"

# 方式2: 使用OpenAI兼容格式
export OPENAI_API_KEY="sk-xxxxxxxx"

# 方式3: 在代码中设置
import os
os.environ["DEEPSEEK_API_KEY"] = "your-key"
```

### 备用方案

如果API密钥不可用，系统自动回退到本地OPT-125M：

```python
# 自动检测顺序
1. 检查 DEEPSEEK_API_KEY
2. 检查 OPENAI_API_KEY  
3. 加载本地 OPT-125M
4. 都失败则使用模板报告
```

---

## 基准数据集

| 数据集 | 样本数 | 被试 | 微表情类别 | 特点 |
|---------|---------|----------|-----------|--------|
| CASME II | 300+ | 35 | 7类 | 最常用 |
| SAMM | 400+ | 32 | 8类 | 高质量 |
| SMIC-HS | 400+ | 55 | 5类 | 自发微表情 |

### 微表情类别

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

## 性能对比

| 方法 | CASME II | SAMM | SMIC | 参数量 |
|------|----------|------|------|--------|
| Hybrid Attention-3DNet | 93.79% | 93.61% | 93.42% | 25M |
| ROI-ArcFace | 93.96% | 86.15% | 81.17% | 50M |
| GAM-MER | 91.57% | 91.25% | 86.22% | 18M |
| **Censor** | - | - | - | 68M |

> 注意：模型正在标准数据集上进行评估

---

## 配置选项

### 主配置 (config/defaults.py)

```python
# 输入配置
INPUT_CONFIG = {
    'batch_size': 2,
    'channels': 3,
    'temporal': 16,
    'height': 224,
    'width': 224,
}

# 快通道配置
FAST_PATHWAY_CONFIG = {
    'input_channels': 2,
    'stem_channels': 64,
    'output_dim': 512,
}

# 慢通道配置
SLOW_PATHWAY_CONFIG = {
    'input_channels': 6,
    'embed_dim': 96,
    'output_dim': 768,
}

# AU解码器配置
AU_DECODER_CONFIG = {
    'num_aus': 28,
    'temporal_steps': 16,
    'threshold': 0.3,
}

# 视觉后处理配置
VISUAL_PERCEPTION_CONFIG = {
    'pupil_base_gain': 0.8,
    'pupil_modulation_range': 0.4,
    'retinal_kernel': 9,
    'mach_band_strength': 0.3,
}
```

### 生成器配置

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

## 快速开始

### 安装依赖

```bash
pip install torch torchvision
pip install streamlit
pip install transformers
pip install opencv-python
pip install numpy pandas scikit-image
```

### 运行识别

```bash
# 命令行
python main.py --video path/to/video.mp4

# 或运行前端
streamlit run frontend/app.py
```

### 运行图像生成

```bash
# 测试模式
python train_image_generator.py --test

# 完整训练
python train_image_generator.py
```

---

## Python API

### 基础生成器

```python
import torch
from model.biomimetic_image_generator import BiomimeticImageGenerator

# 创建
generator = BiomimeticImageGenerator({
    'fast_dim': 512,
    'slow_dim': 768,
    'fused_dim': 1024,
})

# 生成
fast_feat = torch.randn(2, 512)
slow_feat = torch.randn(2, 768)
au_intensities = torch.rand(2, 16, 28)

with torch.no_grad():
    image = generator(fast_feat, slow_feat, au_intensities)

# 输出: (2, 3, 224, 224)
print(image.shape)
```

### 增强版生成器

```python
import torch
from model.enhanced_image_generator import EnhancedBiomimeticImageGenerator, EnhancedConfig

# 配置
config = EnhancedConfig()
config.enable_3d_prior = True
config.enable_sh_lighting = True
config.enable_id_preservation = True

# 创建
generator = EnhancedBiomimeticImageGenerator(config)

# 生成
fast_feat = torch.randn(2, 512)
slow_feat = torch.randn(2, 768)

with torch.no_grad():
    image, details = generator(
        fast_feat=fast_feat,
        slow_feat=slow_feat,
        return_details=True
    )

print(f"生成图像: {image.shape}")
print(f"中间结果: {list(details.keys())}")
```

### LLM报告

```python
from model.llm_report import EmotionReporter

# 创建
reporter = EmotionReporter()

# 生成报告
fused_feat = torch.randn(1, 1024)
au_intensities = torch.rand(1, 16, 28)
me_logits = torch.randn(1, 7)

template_reports, llm_reports = reporter(fused_feat, au_intensities, me_logits)

print("模板报告:", template_reports)
print("LLM报告:", llm_reports)
```

### 视觉后处理

```python
import torch
from visual_perception import VisualPerceptionPostProcess
from config.defaults import VISUAL_PERCEPTION_CONFIG

# 创建
vpp = VisualPerceptionPostProcess(VISUAL_PERCEPTION_CONFIG)

# 处理
image = torch.randn(2, 3, 224, 224)
processed = vpp(image)

print(f"处理后: {processed.shape}")
```

---

## 项目结构

```
censor/
├── model/
│   ├── __init__.py
│   ├── attention.py           # 注意力模块
│   ├── au_attention.py      # AU注意力
│   ├── backbones.py       # 骨干网络
│   ├── biomimetic_enhance.py  # 仿生增强
│   ├── biomimetic_image_generator.py  # 基础生成器
│   ├── biomoe.py           # BioMoE
│   ├── brain_event.py      # 事件驱动
│   ├── enhanced_image_generator.py  # 增强生成器
│   ├── enhanced_moe.py    # 增强MoE
│   ├── event_driven_wrappers.py
│   ├── face_3d_prior.py  # 3D先验
│   ├���─ fusion.py          # 融合模块
│   ├── hierarchical_dynamic_moe.py
│   ├── human_attention.py
│   ├── identity_preservation.py  # ID保持
│   ├── llm_report.py    # LLM报告
│   ├── moe_head.py     # MoE头
│   ├── preprocessing.py # 预处理
│   ├── sh_lighting.py # SH光照
│   └── text_guided_generation.py  # 文本引导
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

## 训练

### 图像生成训练

```bash
# 基础训练
python train_image_generator.py

# 指定配置
python train_image_generator.py --config path/to/config.json

# 断点继续
python train_image_generator.py --resume path/to/checkpoint.pt
```

### 训练超参数

```python
TrainingConfig = {
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'batch_size': 4,
    'epochs': 50,
    'warmup_epochs': 5,
    
    # 损失权重
    'lambda_l2': 1.0,
    'lambda_perceptual': 0.1,
    'lambda_smooth': 0.01,
    'lambda_sparse': 0.001,
    'lambda_contrastive': 0.05,
}
```

### 损失函数

| 损失 | 公式 | 用途 |
|------|------|------|
| L2重建 | `||G - T||²` | 像素级重建 |
| Perceptual | `||VGG(G) - VGG(T)||²` | 感知相似 |
| 光照平滑 | `||L_t - L_{t-1}||²` | 时间一致性 |
| 稀疏正则 | `||W||₁` | 防过拟合 |
| 对比 | `-log(sim(I, I'))` | ID保持 |

---

## 常见问题

### Q: 如何设置DeepSeek API Key？

A: 使用环境变量 `export DEEPSEEK_API_KEY="your-key"`，或在前端设置。

### Q: 没有API Key怎么办？

A: 系统会自动回退到本地OPT-125M（需要下载约250MB）。

### Q: 图像生成需要GPU吗？

A: 推荐使用GPU，CPU也可以但速度较慢。

### Q: 可以生成多人的图像吗？

A: 当前版本支持单人图像生成，多人需要修改代码。

### Q: 如何训练自己的数据？

A: 修改 `train_image_generator.py` 中的 `ImageGenerationDataset` 类。

---

## 引用

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

**最后更新**: 2024-05-16
**版本**: 2.0
**维护**: Censor Team