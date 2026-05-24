# Censor: 仿生双通道微表情识别系统

> 基于人类视觉通路双通道架构（杏仁核-梭状回回路）的PyTorch实现
> 灵感来源：生物神经系统的快速皮层下通路（40ms）+ 慢速皮层通路（200ms+）

---

## 目录

- [概述与动机](#概述与动机)
- [架构详解](#架构详解)
  - [阶段1: 仿生预处理](#阶段1-仿生预处理)
  - [阶段2: 双通道主干网络](#阶段2-双通道主干网络)
  - [阶段3: 梭状回-杏仁核注意力](#阶段3-梭状回-杏仁核注意力)
  - [阶段4: 时空融合](#阶段4-时空融合)
  - [阶段5: 动态AU解码器](#阶段5-动态au解码器)
  - [阶段6: 混合专家头](#阶段6-混合专家头)
  - [阶段7: 情绪报告器](#阶段7-情绪报告器)
- [仿生增强机制详解](#仿生增强机制详解)
  - [1. 动态拓扑网络DTN](#1-动态拓扑网络dtn)
  - [2. 元学习记忆Meta-Plasticity](#2-元学习记忆meta-plasticity)
  - [3. 生物门控MoE BioMoE](#3-生物门控moe-biomoe)
  - [4. EnhancedMoE封装](#4-enhancedmoe封装)
- [数学形式化完整推导](#数学形式化完整推导)
- [数据集与基准](#数据集与基准)
- [训练流程与参数](#训练流程与参数)
- [快速开始与推理](#快速开始与推理)
- [API参考](#api参考)
- [测试与验证](#测试与验证)
- [常见问题](#常见问题)
- [技术细节扩展](#技术细节扩展)
  - [3D Swin-Transformer 详解](#3d-swin-transformer-详解)
  - [BiLSTM时序建模详解](#bilstm时序建模详解)
  - [OPD界标检测](#opd-onset-peak-decay-界标检测)
  - [混合专家MoE详解](#混合专家moe详解)
  - [训练细节扩展](#训练细节扩展)
  - [推理优化](#推理优化)
- [新增功能预告](#新增功能预告)
- [引用与参考](#引用与参考)

---

## 概述与动机

### 问题背景

微表情（Micro-expressions）是人类三大情绪泄露途径之一（语言、姿态、微表情），具有以下特征：

| 特征 | 数值 | 含义 |
|------|------|------|
| **持续时间** | 40-200ms | 极短暂，易被忽略 |
| **强度** | < 5% 面部动作 | 细微，难以检测 |
| **出现场景** | 被压抑的真实情绪 | 测谎可用 |
| **分布** | 眉毛、嘴角、眼周 | 特定AU组合 |

### 生物学动机

人类视觉系统天然的双通道架构：

```
视觉输入 
    ↓
    ├── 快速通路 (Tectum → Pulvinar → Amygdala)    ~40ms
    │  粗糙、低分辨率、情绪预警
    │
    └── 慢速通路 (Retina → LGN → V1 → FFA)    ~200ms+
        精细、高分辨率、具体识别
```

Censor正是模拟这一生物学机制：
- FastPath = 3D ResNet-18 (光流输入)
- SlowPath = 3D Swin-Transformer (RGB输入)
- 两者融合 = 梭状回-杏仁核回路

---

## 架构详解

### 整体流水线

```
输入: (B, 3, T=16, H=224, W=224) RGB视频
    ↓
阶段1: 仿生预处理 → (B, 1+3+2, T, H, W) = 显著性+rPPG+光流
    ↓
阶段2: 双通道 → Fast (B,512) + Slow (B,768)
    ↓
阶段3: 注意力 → (B,512) + (B,768) + 顶帧分数
    ↓
阶段4: 融合 → (B,1024)
    ↓
阶段5: AU解码 → (B,T,28) AU + (B,28,3) OPD
    ↓
阶段6: MoE → (B,7) ME logits + (B,3) gates
    ↓
阶段7: 报告 → ["模板报告", "LLM报告"]
```

---

### 阶段1: 仿生预处理

#### 1.1 SaliencyDetector — 视网膜中央凹采样

**生物学原理**：
- 人眼视网膜中央凹（fovea）锥细胞密度最高，1-2度视角范围分辨率最高
- 周边视野使用神经节细胞，粗糙但敏感

**数学模型**：
$$S_{saliency}(x,y) = \sum_{l=0}^{L-1} w_l \cdot G_\sigma(x,y) \cdot I_l(x,y)$$

其中：
- $I_l$: 第l层高斯金字塔图像
- $G_\sigma(x,y) = \exp(-\frac{x^2+y^2}{2\sigma^2})$: 中心优先的高斯核
- $w_l = 2^{-l}$: 层级权重

**实现**：
```python
class SaliencyDetector(nn.Module):
    def __init__(self, levels=3, sigma=1.5):
        self.levels = levels
        self.sigma = sigma
        
    def forward(self, x):
        # 构建高斯金字塔
        pyramid = [x]
        for l in range(self.levels - 1):
            pyramid.append(F.max_pool2d(pyramid[-1], 2))
        
        # 加权融合 + 中心高斯先验
        h, w = x.shape[-2:]
        Y, X = torch.meshgrid(torch.arange(h), torch.arange(w))
        center_Y, center_X = h // 2, w // 2
        gaussian_prior = torch.exp(-((Y-center_Y)**2 + (X-center_X)**2 / (2*self.sigma**2))
        
        return weighted_sum(pyramid, gaussian_prior)
```

**输出**: (B, 1, T, H, W) 显著性图

#### 1.2 rPPGExtractor — 远程光电容积图

**生物学原理**：
- 心脏搏动导致面部血流变化
- 皮下血管透光性随血压波动
- 频率范围：0.5-4.0 Hz (30-240 BPM)

**数学模型**：
$$\text{rPPG}(t) = \sum_{c \in \{R,G,B\}} \alpha_c \cdot I_c(t)$$

$$\text{rPPG}_{filtered}(t) = \sum_{\tau=-K}^{K} h(\tau) \cdot \text{rPPG}(t-\tau)$$

其中 $\alpha_c$ 是学习到的 chrominance 投影权重，$h(\tau)$ 是带通FIR滤波器。

**实现**：
```python
class rPPGExtractor(nn.Module):
    def __init__(self):
        # 颜色空间分解
        self.chromacity_proj = nn.Linear(3, 3)
        
        # 带通滤波器 (0.5-4.0 Hz)
        self.bandpass = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # 时间维度平均
        temporal = x.mean(dim=[-2, -1]  # (B, C, T)
        
        # 颜色空间投影
        chrom = self.chromacity_proj(temporal)  # (B, 3, T)
        
        # 带通滤波
        filtered = self.bandpass(chrom)
        
        return filtered  # (B, 3, T, H, W)
```

**输出**: (B, 3, T, H, W) 血流热力图

#### 1.3 TVL1OpticalFlow — 光流

**生物学原理**：
- 运动信息是微表情检测的关键
- 光流场直接反映面部肌肉运动

**算法**：OpenCV DualTVL1
$$\min_u \int\left(|\nabla u| + \lambda \cdot |I_1(x+u) - I_0(x)|\right) dx$$

**实现**：
```python
class TVL1OpticalFlow(nn.Module):
    def __init__(self):
        self.flow = cv2.optflow.createDualTVL1OpticalFlow()
        
    def forward(self, frame0, frame1):
        # frame0, frame1: (B, C, H, W)
        flow = []
        for i in range(B):
            f0 = frame0[i].permute(1, 2, 0).numpy()
            f1 = frame1[i].permute(1, 2, 0).numpy()
            flow.append(self.flow.calc(f0, f1, None))
        return torch.tensor(flow).permute(0, 3, 1, 2)
```

**输出**: (B, 2, T, H, W) 光流图 (u, v)

---

### 阶段2: 双通道主干网络

#### 2.1 FastPath — 3D ResNet-18

**设计原理**：
- 快速皮层下通路不需要细节，只需运动信息
- 光流作为输入，捕捉时序变化

**架构**：
```python
FastSubcorticalPathway:
  # 修改版3D ResNet-18 (3 stages)
  Stage1: 3D Conv (2→64) + 3D ResBlock ×2
  Stage2: 3D Conv (64→128, stride=2²) + 3D ResBlock ×2
  Stage3: 3D Conv (128→256, stride=2²) + 3D ResBlock ×2
  GlobalAvgPool → 512维
```

**特点**：
- 大时间步长 (2²) 模拟快速通路
- 无需过多参数量

#### 2.2 SlowPath — 3D Swin-Transformer

**设计原理**：
- 皮层通路需要空间细节和语义
- 使用Transformer捕获长程依赖

**架构**：

| Stage | Blocks | 维度 | 合并步长 | 分辨率 |
|-------|--------|------|----------|--------|
| 1 | 2 | 96 | (2,2,2) | T/2, H/2, W/2 |
| 2 | 2 | 192 | (2,2,2) | T/4, H/4, W/4 |
| 3 | 6 | 384 | (2,2,2) | T/8, H/8, W/8 |
| 4 | 2 | 768 | (1,1,1) | T/16, H/32, W/32 |

**输出**：
- 全局池化: (B, 768)
- 空间图: (B, 768, T/16, H/32, W/32)

---

### 阶段3: 梭状回-杏仁核注意力

#### 3.1 Amygdala — 杏仁核

**生物学**：杏仁核接收快速通路信号，产生情绪预警

**数学**：
$$\text{APM} = \sigma\left(\text{FC}_{512\rightarrow256\rightarrow196}(f_{fast})\right).view(B,1,14,14)$$

#### 3.2 FFA — 梭状回

**生物学**：梭状回整合多通道信息，产生特征重校准

**数学**：
```python
# SE风格门控
s = FC([f_fast; f_slow])  # (B, 1280) → (B, 80)
gate = σ(FC(s))         # (B, 80) → (B, 1280)
f_fast_gated = f_fast × gate[:512]
f_slow_gated = f_slow × gate[512:]
```

#### 3.3 CASANet — 顶帧检测

**生物学**：微表情有明确的 onset→apex→decay 过程

**数学**：
```python
# 逆三角形先验
apex_score_t = Softmax(MHA(Q_t, K, V))  # 时序注意力
M_{i,j} = exp(-(j-i)² / 2σ_i²)  # 三角形
```

---

### 阶段4: 时空融合

**双向交叉注意力**：
```python
F_f2s = Attention(Q_f·W_Q, K_s·W_K, V_s·W_V)·W_O
F_s2f = Attention(Q_s·W_Q, K_f·W_K, V_f·W_V)·W_O

α = σ(W_α[f_fast; f_slow])
f_fused = α·FFN(F_f2s) + (1-α)·FFN(F_s2f)
```

---

### 阶段5: 动态AU解码器

**BiLSTM时序建模**：
```python
h_t = BiLSTM(f_fused, h_{t-1})  # (B, 512)
AU_t = σ(Linear(h_t)) ∈ (0,1)^28  # AU强度
```

**OPD界标检测**：
```python
# onset = 首次超过阈值且上升
# peak = 最高点
# decay = 最后超过阈值且下降
OPD_u = [t_onset, t_peak, t_decay] ∈ ℝ³
```

---

### 阶段6: 混合专家头

#### 标准MoE架构

```python
# 3个专家
Expert_i: MLP(1024 → 512 → 7)

# 门控
gate_logits = W_g · x
gate = Softmax(Top2(gate_logits))

# 输出
y = Σ g_i · Expert_i(x)
```

#### 增强版BioMoE（见下文）

---

### 阶段7: 情绪报告器

```python
class EmotionReporter:
    # 模板报告
    report = template.format(
        AU_list=threshold(AU > 0.5),
        ME=argmax(logits),
        OPD_landmarks=opd
    )
    
    # LLM报告 (可选)
    llm_report = OPT125M(prompt)
```

---

## 仿生增强机制详解

### 1. 动态拓扑网络DTN

#### 生物学原型

细胞骨架（微管、微丝）撑起细胞内部：

| 组分 | 直径 | 功能 |
|------|------|------|
| 微管 | ~25nm | 物质运输 |
| 微丝 | ~7nm | 形态维持 |

当细胞受压形变时，骨架产生**张力** → 打开**机械力敏感通道** → 离子流入 → 电信号

这是"**形态决定功能**"的典型例子。

#### AI映射：非欧几里得特征图

**痛点**：ViT/CNN把图像当作静态网格处理，忽视了"形变"的物理含义。

**突破**：
- 特征图 → 弹性薄膜
- 边权重 → 由输入形变实时调制
- 门控阈值 → 可学习参数

#### 数学形式化

```python
# 1. 构建拓扑图
G = (V, E)  # 节点=特征, 边=连接

# 2. 张力计算 (特征梯度)
tension = ||∇feature||  = √(∂x² + ∂y²)

# 3. 机械门控
gate = sigmoid(gain × tension - threshold)

# 4. 消息传递
h_i^{(l+1)} = UPDATE(h_i^{(l)}, Σ MSG(h_j) · gate_ij)
```

#### 实现代码

```python
class DynamicTopologyLayer(nn.Module):
    def __init__(self, in_dim, k=8):
        self.tension_proj = nn.Linear(in_dim, 1)
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.gain = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        # x: (B, N, D) 特征序列
        
        # 计算张力 (梯度)
        x_centered = x - x.mean(dim=1, keepdims=True)
        tension = torch.norm(x_centered, dim=-1)  # (B, N)
        
        # 门控
        gate = torch.sigmoid(self.gain * tension - self.threshold)
        
        # 门控输出
        out = x * gate.unsqueeze(-1)
        
        return out
```

#### 效果

- 不再把"侧翻的卡车"认成"行驶的卡车"
- 底层理解重力与形变
- 物理先验编码

---

### 2. 元学习记忆Meta-Plasticity

#### 生物学原型

DNA甲基化：化学标记不改变序列，但影响基因表达

| ��级 | 可塑性 | 遗传性 | 时间 |
|------|--------|--------|------|
| DNA序列 | 低 | 高 | 百万年 |
| 甲基化 | 中 | 中 | 年 |
| 蛋白表达 | 高 | 无 | 天 |
| 神经活动 | 极高 | 无 | 秒 |

经典案例：
- 荷兰饥荒(1944) → 子代代谢异常
- 创伤后应激 → FKBP5基因去甲基化

**核心**：事件**强度**决定是否触发甲基化。

#### AI映射：双轨记忆系统

```python
输入对话
    ↓
    ├── 短期: KV Cache (会话级) ← 类似神经活动
    │   - GPU内存
    │   - 每次会话重置
    │
    └── 长期: LoRA固化 (持久) ← 类似DNA甲基化
        - 特定权重文件
        - 跨会话保留
        - 时间戳标记
```

#### 触发机制

```python
class EmotionStimulusDetector:
    def __init__(self):
        self.classifier = nn.Sequential(...)
        
    def forward(self, context):
        score = self.classifier(context)  # (0, 1)
        
        if score > strong_threshold:
            trigger_methylation_update(score)  # 固化权重
        elif score > weak_threshold:
            enhance_kv_cache(score)  # 强化短期记忆
```

#### 实现代码

```python
class MetaPlasticityMemory(nn.Module):
    def __init__(self, num_slots=4, rank=8):
        self.emotion_detector = EmotionStimulusDetector()
        self.slots = nn.ModuleList([
            MethylationSlot(rank=rank) for _ in range(num_slots)
        ])
        
    def forward(self, x, context):
        score = self.emotion_detector(context)
        
        if score > 0.8:
            # 强刺激 → 触发甲基化
            self.trigger_consolidation(score)
        
        # 应用累积记忆
        for slot in self.slots:
            x = x + slot.get_delta() * slot.intensity
        
        return x
```

#### 效果

- 真正的"成长轨迹"
- 分级记忆（重要 vs 日常）
- 可追溯、可撤销

---

### 3. 生物门控MoE BioMoE

#### 生物学原型

神经元膜电位：

```
输入信号 → 树突整合 → 膜电位变化 → 动作电位 → 轴突输出
```

关键特性：
- **累积性**：膜电位由历史输入塑造
- **阈值性**：超过阈值才发放动作电位
- **不应期**：发放后电位回落

#### 情绪影响认知

- 好心情 → 更自信、更有创造力
- 差心情 → 更保守、更谨慎

#### AI映射： membran + emotion门控

```python
# 标准MoE
gate_logits = W_g · x

# BioMoE
gate_logits = W_g · x                    # 基础门控
           + membrane_bias             # 历史累积
           + emotion_gain × mood       # 情绪调制
```

#### 膜电位更新

```python
class MembranePotential(nn.Module):
    def __init__(self):
        self.potential = nn.Parameter(torch.zeros(1))
        self.decay = 0.95
        
    def forward(self, feedback):
        # feedback: 1.0 (正确), 0.0 (错误), -1.0 (纠正)
        
        if feedback is not None:
            # 直接用反馈更新
            positive_delta = relu(feedback) * (1 - self.decay)
            negative_delta = relu(-feedback) * (1 - self.decay) * 0.5
            
            self.potential += positive_delta - negative_delta
            self.potential = self.potential.clamp(-1, 1)
            
            self.positive_count += max(0, feedback)
            self.negative_count += max(0, -feedback)
        else:
            # 无反馈 → 衰减
            self.potential *= self.decay
            
        return self.potential
```

#### 情绪状态

```python
class EmotionalState:
    def forward(self, feedback_stats):
        # 统计正确/错误比例
        pos = feedback_stats.positive_count
        neg = feedback_stats.negative_count
        ratio = pos / (pos + neg + 1e-8)
        
        # mood ∈ (-1, 1): 正=自信, 负=保守
        mood = (ratio - 0.5) * 2
        return mood
```

#### 完整BioMoE

```python
class BioMoE(nn.Module):
    def __init__(self, num_experts=3):
        self.experts = nn.ModuleList([MLP() for _ in range(num_experts)])
        self.membrane = MembranePotential()
        self.emotion = EmotionalState()
        self.gate = nn.Linear(...)
        
    def forward(self, x):
        # 1. 膜电位
        potential, activation = self.membrane(x)
        
        # 2. 情绪
        stats = self.membrane.get_state()
        mood = self.emotion(stats)
        
        # 3. 门控 (input + membrane + emotion)
        gate = self.gate(x)
        gate = gate + self.gate.membrane_bias * 0.1
        gate = gate + self.gate.emotion_gain * mood * 0.1
        
        # 4. 专家输出
        outputs = [expert(x) for expert in self.experts]
        
        return weighted_sum(outputs, gate)
```

#### 反馈接口

```python
# 训练时自动应用
moe.apply_feedback(1.0)   # 预测正确
moe.apply_feedback(0.0)   # 预测错误
moe.apply_feedback(-1.0)  # 用户纠正

# 获取状态
state = moe.get_state()
# {
#   'positive_count': 10,
#   'negative_count': 3,
#   'accuracy': 0.77,
#   'membrane_potential': 0.2
# }
```

---

### 4. EnhancedMoE封装

#### 三种模式

| 模式 | 门控机制 | 推荐场景 |
|------|----------|---------|
| `standard` | f(input) 原始 | 基线对比 |
| `bio` | f(input)+membrane+emotion | 实验研究 |
| `hybrid` | f(input)+membrane+emo 推荐 | **实际训练** |

#### 使用示例

```python
from model.enhanced_moe import EnhancedMoE

# 推荐：混合模式
moe = EnhancedMoE(
    mode="hybrid",
    enable_membrane=True,
    enable_emotion=True,
    decay_rate=0.95
)

# 前向传播
output, gates, aux_loss, info = moe(x)

# 输出结构
# output: (B, 7) logits
# gates: (B, 3) routing weights  
# aux_loss: scalar load balancing
# info: {
#     'mode': 'hybrid',
#     'membrane_activation': 0.83,
#     'emotional_state': 0.15,
#     'expert_usage': [0.4, 0.3, 0.3]
# }

# 外部反馈
moe.apply_feedback(1.0)

# 状态查询
state = moe.get_state()

# 重置
moe.reset_state()
```

#### 训练集成

在`train.py`中已自动集成：

```python
# train.py 约 line 343
if hasattr(self.model, 'moe') and hasattr(self.model.moe, 'apply_feedback'):
    preds = outputs['me_logits'].argmax(dim=1)
    correct = (preds == me_labels).float()
    fb = correct.mean()
    
    if self.model.moe.mode == 'hybrid':
        self.model.moe.apply_feedback(fb)
```

---

## 数学形式化完整推导

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{ME} + \alpha\mathcal{L}_{AU} + \beta\mathcal{L}_{MoE} + \gamma\mathcal{L}_{OPD}$$

| 损失项 | 公式 | 说明 |
|--------|------|------|
| $\mathcal{L}_{ME}$ | $\text{CE}(\hat{y}, y)$ | 7类交叉熵 |
| $\mathcal{L}_{AU}$ | $\text{BCE}(\hat{a}, a)$ | 28AU二分类交叉熵 |
| $\mathcal{L}_{MoE}$ | $\sum_i(\bar{f}_i - \frac{1}{N})^2$ | 负载均衡 |
| $\mathcal{L}_{OPD}$ | $\lambda_1\|\partial_t a\|_2 + \lambda_2\|t_{peak} - \text{argmax}(a)\|$ | 时序平滑+峰值一致性 |

### 默认权重

$$\alpha=0.5, \beta=0.01, \gamma=0.1, \lambda_1=0.1, \lambda_2=0.1$$

### 维度变化

```
Input:          (B, 3, T=16, H=224, W=224)
  ↓ Saliency:    (B, 1, T, H, W)
  ↓ rPPG:      (B, 3, T, H, W)
  ↓ Flow:       (B, 2, T, H, W)

Stage2:
  ↓ Fast:      (B, 512)
  ↓ Slow:      (B, 768) + (B, 768, 1, 7, 7)

Stage4:
  ↓ Fused:    (B, 1024)

Stage5:
  ↓ AU:       (B, T, 28)
  ↓ OPD:      (B, 28, 3)

Stage6:
  ↓ ME:       (B, 7)
  ↓ Gates:    (B, 3)
```

---

## 数据集与基准

### 主要数据集

| 数据集 | 样本 | 被试 | 帧率 | 分辨率 | 类别 | 获取 |
|--------|------|------|------|--------|------|------|
| **CASME II** | 247 | 26 | 200 | 640×480 | 5-7 | 申请 |
| **SAMM** | 159 | 32 | 200 | 2040×1088 | 7-8 | 申请 |
| **SMIC-HS** | 164 | 16 | 100 | 640×480 | 3 | 申请 |
| **MMEW** | 300 | 36 | 90 | 1920×1080 | 7 | GitHub |

### 基准测试

| 年份 | 比赛 | 冠军方案 |
|------|------|---------|
| 2022 | MEGC ACM MM | USTC-IAT-United |
| 2023 | MEGC ACM MM | CAS-IA + BUST |
| 2024 | MEGC ACM MM | USTC + HIT |

### SOTA对比

| 方法 | Backbone | CASME II | SAMM | SMIC |
|------|----------|---------|------|------|
| Hybrid Attention-3DNet | 3D CNN+SE | 93.79% | 93.61% | 93.42% |
| ROI-ArcFace | CNN+ROI | 93.96% | 86.15% | 81.17% |
| GAM-MER | Graph Attn | 91.57% | 91.25% | 86.22% |
| **Censor** | Dual+BioMoE | 待测 | 待测 | 待测 |

---

## 训练流程与参数

### 超参数

| 参数 | 默认值 | 范围 | 说明 |
|------|-------|------|------|
| `epochs` | 50 | 1-200 | 训练轮数 |
| `batch_size` | 2 | 1-16 | 批次大小 |
| `lr` | 1e-4 | 1e-5-1e-3 | 学习率 |
| `weight_decay` | 1e-4 | 1e-6-1e-2 | 权重衰减 |
| `max_grad_norm` | 1.0 | 0.1-10 | 梯度裁剪 |
| `au_loss_weight` (α) | 0.5 | 0-1 | AU损失权重 |
| `moe_loss_weight` (β) | 0.01 | 0-1 | 负载均衡权重 |
| `landmark_loss_weight` (γ) | 0.1 | 0-1 | OPD损失权重 |

### 训练命令

```bash
# 合成数据调试
python train.py --synthetic_data --epochs 2 --batch_size 2

# CASME II 数据集
python train.py --dataset casme2 --data_root ./data/CASME_II --epochs 50

# SAMM 数据集
python train.py --dataset samm --data_root ./data/SAMM --epochs 50

# 多GPU训练
python train.py --data_root ./data/CASME_II --epochs 100 --batch_size 8 --num_workers 4

# 恢复训练
python train.py --resume ./checkpoints/model_epoch_20.pt
```

### 训练技巧

1. **梯度累积**：batch_size=1时，使用 `--accum_steps 4`
2. **混合精度**：自动启用AMP加速
3. **早停**：添加 `--early_stop_patience 10`
4. **学习率调度**：CosineAnnealingLR

---

## 快速开始与推理

### 前向测试

```bash
python main.py

# 预期输出:
[1] Censor: Biomimetic Dual-Pathway MER System
[1] Total parameters:     68,353,230
[1] Input video: torch.Size([2, 3, 16, 224, 224])
[1] ME Logits:       torch.Size([2, 7])
[1] AU Intensities:  torch.Size([2, 16, 28])
```

### 单样本推理

```python
import torch
from main import Censor

# 加载模型
model = Censor()
checkpoint = torch.load('./checkpoints/best.pt')
model.load_state_dict(checkpoint['model'])
model.eval()

# 推理
video = torch.randn(1, 3, 16, 224, 224)
with torch.no_grad():
    output = model(video)
    
print(f"预测类别: {output['me_logits'].argmax()}")
print(f"Top-3: {output['me_logits'].topk(3)}")
```

### 视频预处理

```python
from dataset import VideoPreprocessor

preprocessor = VideoPreprocessor(
    num_frames=16,
    resize=(224, 224),
    normalize=True
)

video = preprocessor.load_video('path/to/video.mp4')
```

---

## API参考

### 主模型

```python
from main import Censor

model = Censor()
output = model(video)
# output keys:
#   'me_logits': (B, 7)
#   'au_intensities': (B, T, 28)
#   'au_opd': (B, 28, 3)
#   'apex_scores': (B, 1)
#   'expert_gates': (B, 3)
#   'template_report': list[str]
```

### 组件

```python
# 预处理
from model.preprocessing import SaliencyDetector, rPPGExtractor, TVL1OpticalFlow

# 主干网络
from model.backbones import FastSubcorticalPathway, SlowCorticalPathway

# 注意力
from model.attention import Amygdala, FFA, CASANet

# 融合
from model.fusion import TSFmicroFusion

# 解码器
from model.decoders import DynamicAUDecoder

# MoE
from model.moe_head import MoEGatingNetwork, PersonalizedRadar

# 报告
from model.llm_report import EmotionReporter

# 仿生增强
from model.biomimetic_enhance import DTNEnhancedFFA, MetaPlasticityMemory
from model.biomoe import BioMoE
from model.enhanced_moe import EnhancedMoE
```

### 配置

```python
from config.defaults import (
    INPUT_CONFIG,
    FAST_PATHWAY_CONFIG,
    SLOW_PATHWAY_CONFIG,
    AMYGDALA_CONFIG,
    FFA_CONFIG,
    CASA_CONFIG,
    FUSION_CONFIG,
    AU_DECODER_CONFIG,
    MOE_CONFIG,
)
```

---

## 测试与验证

### 测试文件

```bash
# 概念测试
python test_concepts.py

# 增强模块测试
python test_enhance.py

# BioMoE测试
python test_biomoe.py

# 反馈测试
python test_feedback.py

# 完整流程测试
python main.py
```

### 单元测试覆盖

| 模块 | 测试文件 | 覆盖 |
|------|----------|------|
| Preprocessing | test_concepts.py | Saliency, rPPG, Flow |
| Backbones | main.py | FastPath, SlowPath |
| Attention | main.py | Amygdala, FFA, CASA |
| Fusion | main.py | TSFmicroFusion |
| Decoders | main.py | AU, OPD |
| MoE | test_biomoe.py, test_feedback.py | 全部门控 |
| BioMoE | test_biomoe.py | 全部新机制 |

---

## 常见问题

### Q1: 显存不足？

```bash
# 减小batch_size
python train.py --batch_size 1

# 开启梯度累积
python train.py --batch_size 1 --accum_steps 4
```

### Q2: 需要预训练模型？

当前版本从随机初始化开始训练。后续可添加预训练backbone。

### Q3: 支持其他数据集？

需要按格式准备：
- 视频文件 or 帧序列
- 类别标签CSV
- AU标注JSON

### Q4: 如何使用自定义专家？

```python
from model.moe_head import MoEGatingNetwork

# 定义专家
custom_experts = nn.ModuleList([
    nn.Sequential(Linear(1024, 512), ReLU(), Linear(512, 7)),
    nn.Sequential(Linear(1024, 512), GELU(), Linear(512, 7)),
    nn.Sequential(Linear(1024, 512), SiLU(), Linear(512, 7)),
])

# 修改配置
config = MOE_CONFIG.copy()
config['custom_experts'] = custom_experts

# 使用
moe = MoEGatingNetwork(config)
```

### Q5: 如何添加BioMoE到现有训练？

```python
# 方式1: 直接替换
from model.enhanced_moe import EnhancedMoE
model.moe = EnhancedMoE(mode="hybrid")

# 方式2: 修改main.py中的初始化
# 将 self.moe = MoEGatingNetwork(MOE_CONFIG)
# 改为 self.moe = EnhancedMoE(mode="hybrid")
```

---

## 引用与参考

```bibtex
@article{censor2025,
  title={Censor: 仿生双通道微表情识别系统 with Fusiform-Amygdala Circuit and Mixture-of-Experts},
  author={},
  journal={},
  year={2025}
}
```

### 参考资料

- [CASME II Database](http://casme.psych.ac.cn/casme/c2)
- [SAMM Micro-Expression Database](https://www.mmu.ac.uk)
- [SMIC Database](https://www.oulu.fi)
- [MMEW Dataset](https://github.com/benxianyeteam/MMEW-Dataset)
- [iMER Benchmark](https://github.com/ZhengQinLai/IMER-benchmark)
- [Video-Based Facial Micro-Expression Analysis: A Survey](https://ar5iv.labs.arxiv.org/html/2201.12728)
- [Hua, R. (2025) 动态拓扑网络的生物学基础]
- [DualPrompt Learning for iMER](https://github.com/facebookresearch/convnext)
- [SwinTransformer V2](https://github.com/microsoft/Swin-TransformerV2)

---

## 技术细节扩展

### 3D Swin-Transformer 详解

#### 窗口注意力机制 (Window Attention)

3D Swin-Transformer使用** shifted window multi-head self-attention (W-MSA)** 来降低计算复杂度：

$$\Omega(\text{W-MSA}) = \frac{4N^d \cdot M^2}{d} + 2M \cdot d$$

其中：
- $N^d$: 空间 patch 数量
- $M$: 窗口内的 patch 数量 ($7\times7 = 49$)
- $d$: 特征维度

#### 移位窗口 (Shifted Windows)

在连续层之间交替使用**常规窗口**和**移位窗口**：

```
Layer l:     Layer l+1:
┌───┬───┐   ┌───┬───┐
│ A │ B │   │███│   │
├───┼───┤   ├───┼───┤
│ C │ D │   │   │███│
└───┴───┘   └───┴───┘
  Regular       Shifted (+⌊M/2⌋)
```

实现代码：
```python
def shifted_window_attn(x, shift_size):
    # 移位操作
    x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
    # 计算注意力
    attn = self.attn(x)
    # 移位回来
    x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))
    return attn
```

#### 3D位置偏置

Censor使用3D网格相对位置偏置：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + B_{3D}\right) V$$

其中 $B_{3D} \in \mathbb{R}^{(2M-1)\times(2M-1)\times(2M-1)}$ 是可学习的3D位置偏置。

### BiLSTM时序建模详解

#### 双向LSTM结构

$$\mathbf{f}_t = \sigma(W_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_f)$$
$$\mathbf{i}_t = \sigma(W_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_i)$$
$$\tilde{\mathbf{C}}_t = \tanh(W_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_C)$$
$$\mathbf{o}_t = \sigma(W_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_o)$$

$$\mathbf{C}_t = \mathbf{f}_t \cdot \mathbf{C}_{t-1} + \mathbf{i}_t \cdot \tilde{\mathbf{C}}_t$$
$$\mathbf{h}_t = \mathbf{o}_t \cdot \tanh(\mathbf{C}_t)$$

#### 前向与后向拼接

$$\mathbf{h}_t = [\mathbf{h}_t^f; \mathbf{h}_t^b]$$

其中：
- $\mathbf{h}_t^f$: 前向隐藏状态 (从 $t=0$ 到 $t=T$)
- $\mathbf{h}_t^b$: 后向隐藏状态 (从 $t=T$ 到 $t=0$)

### OPD (Onset-Peak-Decay) 界标检测

#### 算法流程

```python
def compute_opd(au_sequence, threshold=0.5):
    """
    au_sequence: (T,) AU强度序列

    Returns:
        opd: (3,) [t_onset, t_peak, t_decay]
    """
    T = len(au_sequence)

    # 1. 找到峰值时刻
    t_peak = torch.argmax(au_sequence)
    peak_value = au_sequence[t_peak]

    # 2. 回溯寻找onset (首次超过阈值且上升)
    onset_candidates = torch.where(
        (au_sequence[:t_peak] > threshold) &
        (torch.diff(au_sequence)[:t_peak-1] > 0)
    )[0]
    t_onset = onset_candidates[0] if len(onset_candidates) > 0 else 0

    # 3. 前溯寻找decay (最后超过阈值且下降)
    decay_candidates = torch.where(
        (au_sequence[t_peak:] > threshold) &
        (torch.diff(au_sequence)[t_peak:] < 0)
    )[0]
    t_decay = t_peak + decay_candidates[-1] if len(decay_candidates) > 0 else T - 1

    return torch.tensor([t_onset, t_peak, t_decay])
```

#### OPD损失函数

$$\mathcal{L}_{\text{OPD}} = \lambda_1 \|\partial_t \mathbf{AU}\|_2 + \lambda_2 \|t_{peak} - \text{argmax}(\mathbf{AU})\|$$

### 混合专家(MoE)详解

#### Noisy Top-K Gating

为每个token添加噪声以增加路由多样性：

$$g_i = \text{Softmax}(\text{top-}k(\text{logits} + \epsilon))$$

其中 $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 是可学习的噪声。

#### 负载均衡损失

$$\mathcal{L}_{\text{load}} = \lambda \sum_{i=1}^{N} \left(\bar{f}_i - \frac{1}{N}\right)^2$$

其中 $\bar{f}_i = \frac{1}{B}\sum_b g_i^{(b)}$ 是专家 $i$ 的平均使用频率。

#### PersonalizedRadar TTA

```python
def personalized_radar(feat, support_frames, num_steps=5, lr=0.01):
    """
    个性化测试时自适应

    Args:
        feat: (B, D) 查询特征
        support_frames: (K, D) 支持帧特征
        num_steps: 内部优化步数

    Returns:
        adapted_feat: (B, D) 适应后特征
    """
    # 初始化残差适配器 (恒等映射)
    delta = torch.zeros_like(feat)

    for step in range(num_steps):
        # 前向传播
        adapted = feat + delta

        # 计算对比损失
        loss = contrastive_loss(adapted, support_frames)

        # SGD更新
        delta.grad = torch.autograd.grad(loss, delta)
        delta = delta - lr * delta.grad

    return feat + delta
```

### 训练细节扩展

#### 梯度累积

```python
# 当显存不足时使用梯度累积
accum_steps = 4
effective_batch_size = batch_size * accum_steps

# 累积梯度
optimizer.zero_grad()
for step in range(accum_steps):
    loss = model(batch[step])
    loss = loss / accum_steps  # 归一化损失
    loss.backward()

# 更新参数
optimizer.step()
```

#### 混合精度训练

```python
scaler = GradScaler()

for epoch in range(epochs):
    for batch in dataloader:
        with autocast():  # 自动 FP16
            outputs = model(batch)
            loss = compute_loss(outputs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

#### 早停机制

```python
early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001,
    mode='max'  # 监控准确率最大化
)

for epoch in range(epochs):
    # 训练
    val_acc = validate()
    early_stopping(val_acc)

    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

### 推理优化

#### ONNX导出

```python
model.eval()

# 导出为ONNX
torch.onnx.export(
    model,
    dummy_input,
    "censor.onnx",
    input_names=['video'],
    output_names=['me_logits', 'au_intensities'],
    dynamic_axes={
        'video': {0: 'batch_size'},
        'me_logits': {0: 'batch_size'}
    }
)
```

#### TensorRT加速

```python
import tensorrt as trt

# 构建TensorRT引擎
builder = trt.Builder()
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# 解析ONNX模型
with open("censor.onnx", "rb") as f:
    parser.parse(f.read())

# 构建引擎
engine = builder.build_serialized_cuda_engine(network)
```

---

## 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2025-01 | v1.0 | 初始版本，完整流水线 |
| 2025-05 | v1.1 | 新增DTN、Meta-Plasticity、BioMoE增强机制 |
| 2025-10 | v1.2 | 扩展技术细节：3D Swin、BiLSTM、MoE数学推导 |
| 2026-05 | v1.3 | 添加推理优化、TensorRT支持、OPD损失详解 |

---

## 新增功能预告

### [规划中] V2架构演进

- [ ] Vision Mamba (SSM) 骨干网络
- [ ] 多模态大模型集成
- [ ] 实时推理优化 (FP16/INT8)
- [ ] 移动端部署 (ONNX + TF-Lite)

### [规划中] 数据增强

- [ ] 时序插值增强
- [ ] AU噪声注入
- [ ] 对抗训练 (AT)

### [规划中] 应用场景

- [ ] 在线实时推理
- [ ] 批量视频处理
- [ ] API服务封装

---

## 许可证

MIT License