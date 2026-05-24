# Censor 技术文档

> 仿生双通道微表情识别系统 - 详细技术规格说明书 v1.0

---

## 一、项目总览与研究背景

### 1.1 研究动机

Censor是一个基于PyTorch实现的**仿生双通道微表情识别（MER）**架构，模拟人类视觉通路中的梭状回-杏仁核神经回路。本项目的核心研究问题是：**如何借鉴人脑的视觉-情感处理机制，设计更精确、更可解释的微表情识别系统？**

#### 1.1.1 微表情的特性

| 特性 | 微表情 | 宏表情 |
|------|--------|--------|
| **持续时间** | 40-200ms | 0.5-4秒 |
| **强度** | 低（难以察觉） | 高（明显） |
| **意识控制** | 无意识 | 有意识 |
| **面部参与** | 部分区域 | 全面部 |
| **检测难度** | 极高 | 中等 |

微表情是由Ekman和Friesen于1969年首次发现，当个体试图隐藏真实情感时会出现。

#### 1.1.2 双通道通路的神经科学基础

人脑视觉系统采用双通路架构处理面部信息：

| 通路 | 路径 | 速度 | 功能 |
|------|------|------|------|
| **快速皮层下通路** | 上丘→丘脑枕→杏仁核 | ~100ms | 快速粗略的情感检测 |
| **慢速皮层通路** | V1→梭状回→前额叶 | ~500ms | 精细的辨别分析 |

---

## 二、系统架构

### 2.1 整体架构图

```mermaid
flowchart TD
    subgraph Stage1["阶段1: 预处理"]
        S1[显著性检测<br/>SaliencyDetector] --> S2[rPPG提取<br/>rPPGExtractor]
        S2 --> S3[TV-L1光流<br/>TVL1OpticalFlow]
    end
    
    subgraph Stage2["阶段2: 双通道骨干"]
        S3 --> F[快通道<br/>3D ResNet-18<br/>光流→512D]
        S2 --> SL[慢通道<br/>3D Swin-T<br/>RGB+rPPG→768D]
    end
    
    subgraph Stage3["阶段3: 注意力回路"]
        F --> A[杏仁核<br/>Amygdala<br/>注意力先验]
        SL --> FFA[FFA<br/>跨通道门控]
        A --> CA[CASANet<br/>Apex检测]
    end
    
    subgraph Stage4["阶段4: 融合"]
        FFA -.-> FU[TSFmicroFusion<br/>双向交叉注意力<br/>1024D]
    end
    
    subgraph Stage5["阶段5: 解码"]
        FU --> AU[动态AU解码器<br/>BiLSTM→28 AU]
    end
    
    subgraph Stage6["阶段6: 多专家"]
        AU --> MOE[MoE头<br/>3专家+TTA]
    end
    
    MOE --> OUT[输出<br/>情绪报告]
    
    style Stage1 fill:#f9f,stroke:#333
    style Stage2 fill:#ff9,stroke:#333
    style Stage3 fill:#9ff,stroke:#333
    style Stage4 fill:#f9f,stroke:#333
    style Stage5 fill:#ff9,stroke:#333
    style Stage6 fill:#9ff,stroke:#333
```

### 2.2 核心指标

| 指标 | 数值 |
|------|-------|
| **总参数量** | 68,353,230 |
| **架构** | 双通道: 3D ResNet-18 + 3D Swin-Transformer |
| **预处理** | 高斯显著性 + rPPG + OpenCV TV-L1 |
| **注意力** | 杏仁核(FC) + FFA(SE) + CASANet |
| **融合** | 双向交叉注意力，1024维 |
| **AU解码** | BiLSTM → 28 sigmoid输出 |
| **多专家** | 3专家，top-2门控 |

---

## 三、核心模块详解

### 3.1 预处理模块

#### 3.1.1 显著性检测器（SaliencyDetector）

**功能**：模拟人眼视网膜中心凹的高密度采样，实现中心偏向的显著性检测

**原理**：使用高斯金字塔实现foveal采样

$$S(x,y) = \sum_{l=0}^{L-1} w_l \cdot G_\sigma(x,y) \cdot I_l(x,y)$$

- $I_l$: 第$l$层金字塔
- $G_\sigma$: 中心偏向的高斯先验
- $w_l = 2^{-l}$: 层级权重

**关键问题解答**:

1. **是端到端训练的吗？**
   - **当前版本**：部分端到端 - 只有`fusion_weights`可学习，高斯核和中心先验是固定buffer
   - **改进（全端到端）**：
   ```python
   class SaliencyDetectorE2E(nn.Module):
       """全端到端可训练的显著性检测器"""
       def __init__(self, levels=4, sigma_ratio=0.15):
           super().__init__()
           self.levels = levels
           self.sigma_ratio = nn.Parameter(torch.tensor(sigma_ratio))  # 可学习！
           self.center_bias = nn.Parameter(torch.tensor(0.5))   # 可学习！
           self.fusion_weights = nn.Parameter(torch.ones(levels) / levels)
   ```

2. **固定sigma适用于变分辨率吗？**
   - **问题**：sigma=0.15（绝对像素值）在不同分辨率下失效：
     - 224×224 → 有效σ = 33.6px（宽度的15%）
     - 112×112 → 有效σ = 16.8px（宽度的15%）
     - 448×448 → 有效σ = 67.2px（过大！）
   - **解决方案**：使用相对sigma = `sigma_ratio * min(H,W)` → 始终保持15%

**实现**（全端到端，分辨率自适应）：
```python
class SaliencyDetectorE2E(nn.Module):
    def __init__(self, levels=4, sigma_ratio=0.15):
        super().__init__()
        self.levels = levels
        self.sigma_ratio = nn.Parameter(torch.tensor(sigma_ratio))  # 可学习！
        self.center_bias = nn.Parameter(torch.tensor(0.5))       # 可学习！
        self.fusion_weights = nn.Parameter(torch.ones(levels) / levels)
        
    def forward(self, x):
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

#### 3.1.2 rPPG提取器（rPPGExtractor）

**功能**：远程光电容积脉搏波提取，捕捉血氧饱和度变化

**原理**：色度分解 + 时间带通滤波

$$\text{rPPG}(t) = \sum_{c \in \{R,G,B\}} \alpha_c \cdot I_c(t)$$

$$\text{rPPG}_{\text{filtered}}(t) = \sum_{\tau=-K}^{K} h(\tau) \cdot \text{rPPG}(t-\tau)$$

- $\alpha_c$: 学习的色度投影权重
- $h$: 学习的FIR带通滤波器（0.5-4.0Hz心脏范围）

**实现**：
```python
class rPPGExtractor(nn.Module):
    def __init__(self, sample_rate=30):
        super().__init__()
        #Learnable chrominance projection
        self.alpha = nn.Parameter(torch.ones(3))
        
        # Bandpass filter parameters
        self.low_freq = 0.5
        self.high_freq = 4.0
        self.sample_rate = sample_rate
        
    def forward(self, x):
        # 帧级平均
        avg_frame = x.mean(dim=(3,4))  # (B, 3, T)
        
        # 色度投影
        rppg = torch.einsum('bct,c->bt', avg_frame, self.alpha)
        
        # 带通滤波
        filtered = self._bandpass_filter(rppg)
        
        return filtered.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
```

**已知局限与缓解方法**：

| 问题 | 影响 | 缓解方法 |
|-------|------|----------|
| 光照变化 | rPPG颜色偏移 | 自适应色度校正 |
| 运动伪影 | rPPG信号噪声 | 时间卡尔曼滤波 |
| 个体差异 | 信号质量差异 | 被试归一化 |
| 持续时间短(40-200ms) | 有限心脏周期 | 与视觉特征融合 |

**实际贡献**：
- rPPG提供互补的生理信息
- 当视觉特征模糊时可作为辅助信号
- 可指示与某些情绪相关的压力/唤醒水平
- 信噪比低时自动降权（学习抑制）


class AdaptiveRPPGDenoiser(nn.Module):
    """自适应rPPG去噪器：处理运动伪影和光照变化
    
    解决实际问题：
    1. 光照变化 → 颜色恒常性校正
    2. 运动伪影 → 时间平滑
    3. 个体差异 → 自适应归一化
    """
```python
    def __init__(self, kernel_size=5):
        super().__init__()
        self.temporal_filter = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2)
        self.snr_estimator = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.noise_suppression = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, rppg_signal, frame_variance):
        rppg_smooth = self.temporal_filter(rppg_signal.squeeze(-1)).unsqueeze(-1)
        motion_weight = torch.sigmoid(frame_variance.mean(dim=1))
        suppressed = (1 - self.noise_suppression * motion_weight) * rppg_smooth
        mean, logvar = self.snr_estimator(suppressed.squeeze(-1)).chunk(2, dim=-1)
        normalized = (suppressed - mean) / (torch.exp(logvar) + 1e-8)
        return normalized
```

#### 3.1.3 TV-L1光流（TVL1OpticalFlow）

**功能**：使用OpenCV的DualTVL1算法计算精确光流

**原理**：TV-L1能量泛函最小化

$$\min_u \int\left(|\nabla u| + \lambda \cdot |I_1(x+u) - I_0(x)|\right) dx$$

**实现**：
```python
class TVL1OpticalFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow = cv2.createOptFlow_DualTVL1()
        
    def forward(self, frames):
        # frames: (B, C, T, H, W)
        flows = []
        for t in range(T - 1):
            I0 = frames[:, :, t].permute(1,2,3).numpy()
            I1 = frames[:, :, t+1].permute(1,2,3).numpy()
            
            flow = self.flow.calc(I0, I1, None)
            flows.append(torch.from_numpy(flow).permute(2,3,0,1))
        
        return torch.stack(flows, dim=2)  # (B, 2, T-1, H, W)
```

**性能对比**：

| 方法 | 精度 | 速度(16帧) | 微表情适用 | 瓶颈? |
|------|------|------------|----------|-------|
| TV-L1 (DualTVL1) | 高 | ~150ms | ✓ 小运动 | ⚠️ 是 |
| RAFT | 最高 | ~1600ms | ✓ | ❌ 太慢 |
| PWC-Net | 高 | ~480ms | ✓ | ❌ |
| 帧差分 | 低 | ~15ms | ❌ 噪声大 | ✓ 快 |

**诚实分析**：

| 方面 | 实际情况 |
|------|----------|
| 对比验证 | ❌ 未与RAFT/PWC-Net对比 |
| 实时性 | ⚠️ 150ms/16帧可能成为瓶颈 |
| 微表情适用 | ✓ TV-L1适合小运动 |
| 瓶颈 | ⚠️ 光流计算占大部分时间 |

**客观建议**：
- 快速初筛用帧差分
- 精细分析用TV-L1
- 或者离线预计算光流

**改进版**：
```python
class AdaptiveOpticalFlow(nn.Module):
    """两阶段光流：快速初筛 + 精细计算
    
    策略：
    1. 帧差分初筛 (~15ms)
    2. 仅在检测到运动时用TV-L1 (~150ms)
    
    时间节省：固定150ms → 平均~50ms（取决于运动比例）
    """
    def __init__(self, fast_threshold=0.1, use_tvl1=True):
        super().__init__()
        self.threshold = fast_threshold
        self.use_tvl1 = use_tvl1
        
        # TV-L1求解器
        self._tvrl1 = None
        
    @property
    def tvl1(self):
        if self._tvrl1 is None:
            self._tvrl1 = cv2.createOptFlow_DualTVL1()
        return self._tvrl1
        
    def _frame_diff(self, frames):
        """快速帧差分"""
        return frames[:, :, 1:] - frames[:, :, :-1]
        
    def _compute_tvl1(self, frames):
        """精确TV-L1计算"""
        B, C, T, H, W = frames.shape
        flows = []
        
        for b in range(B):
            frame_flows = []
            for t in range(T - 1):
                I0 = frames[b, :, t].permute(1, 2, 0).numpy()
                I1 = frames[b, :, t + 1].permute(1, 2, 0).numpy()
                
                flow = self.tvl1.calc(I0, I1, None)
                frame_flows.append(torch.from_numpy(flow).permute(2, 0, 1))
            
            flows.append(torch.stack(frame_flows, dim=1))
        
        return torch.stack(flows, dim=1)
        
    def forward(self, frames):
        """两阶段光流
        
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
            # 阶段2：检测到运动时精细计算
            flow = self._compute_tvl1(frames)
            stage = 'fine'
        else:
            # 使用快速差分
            flow = diff
            stage = 'fast'
            
        return flow, stage


class TwoStageOpticalFlow(nn.Module):
    """两流版本：全程帧差分，仅apex帧用TV-L1
    
    思路：仅在检测到的apex帧周围计算TV-L1，其他地方用帧差分
    """
    def __init__(self):
        super().__init__()
        self.tvl1 = cv2.createOptFlow_DualTVL1()
        
    def forward(self, frames, apex_frame_idx=None):
        """两阶段光流
        
        Args:
            frames: (B, C, T, H, W)
            apex_frame_idx: (B,) 检测到的apex帧位置
            
        Returns:
            flow: (B, 2, T-1, H, W)
        """
        B, C, T, H, W = frames.shape
        
        # 默认用帧差分
        flow = frames[:, :, 1:] - frames[:, :, :-1]
        
        if apex_frame_idx is not None:
            # 在apex帧周围细化
            for b in range(B):
                apex_t = apex_frame_idx[b].item()
                t_start = max(0, apex_t - 2)
                t_end = min(T - 1, apex_t + 2)
                
                for t in range(t_start, t_end):
                    I0 = frames[b, :, t].permute(1, 2, 0).numpy()
                    I1 = frames[b, :, t + 1].permute(1, 2, 0).numpy()
                    fine_flow = self.tvl1.calc(I0, I1, None)
                    flow[b, :, t] = torch.from_numpy(fine_flow).permute(2, 0, 1)
        
        return flow
```

### 3.2 双通道骨干网络

#### 3.2.1 快通道 - 3D ResNet-18

**功能**：处理光流输入，模拟快速皮层下通路

**结构**：
- 3个stage: 64→128→256通道
- 大时间步长(2²,2²)模拟快速处理

**实现**：
```python
class FastSubcorticalPathway(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        
        self.conv1 = conv3d(in_channels, 64, kernel_size=3, stride=(2,2,2))
        self.conv2 = res3d_block(64, 128, stride=(2,2,2))
        self.conv3 = res3d_block(128, 256, stride=(2,2,2))
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, flow):
        x = self.conv1(flow)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return self.pool(x).flatten(2)  # (B, 512)
```

#### 3.2.2 慢通道 - 3D Swin-Transformer

**功能**：处理RGB+rPPG输入，模拟慢速皮层通路

**结构**：

| Stage | Blocks | Dim | Merge Stride |
|-------|--------|-----|--------------|
| 1 | 2 | 96 | (2,2,2) |
| 2 | 2 | 192 | (2,2,2) |
| 3 | 6 | 384 | (2,2,2) |
| 4 | 2 | 768 | (1,1,1) |

**实现**：
```python
class SlowCorticalPathway(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(in_channels, 96)
        
        # 4 stages with shifted-window MSA
        self.stage1 = SwinStage(dim=96, num_blocks=2)
        self.stage2 = SwinStage(dim=192, num_blocks=2)
        self.stage3 = SwinStage(dim=384, num_blocks=6)
        self.stage4 = SwinStage(dim=768, num_blocks=2)
        
    def forward(self, x):
        # x: (B, 6, T, H, W)
        x = self.patch_embed(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x, spatial = self.stage3(x)  # 返回spatial map
        x = self.stage4(x)
        
        # Global pool + spatial map
        pooled = x.mean(-1)  # (B, 768)
        
        return pooled, spatial  # (B, 768), (B, 768, 1, 7, 7)
```

### 3.3 注意力模块

#### 3.3.1 杏仁核（Amygdala）

**功能**：生成注意力先验图，引导空间注意力朝向面部关键区域

**原理**：
$$\text{APM} = \sigma\left(\text{FC}_{512\rightarrow256\rightarrow196}(\text{fast\_feat})\right).view(B,1,14,14)$$

**实现**：
```python
class Amygdala(nn.Module):
    """Attention Prior Map from fast pathway features"""
    def __init__(self, fast_dim=512):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(fast_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 196),  # 14x14
            nn.Sigmoid()
        )
        
    def forward(self, fast_feat):
        apm = self.fc(fast_feat)  # (B, 196)
        apm = apm.view(-1, 1, 14, 14)  # (B, 1, 14, 14)
        
        return apm
```

**已知局限**：

| 问题 | 影响 | 缓解方法 |
|-------|------|----------|
| 纯数据驱动 | 可能学到错误区域 | 加入面部区域先验 |
| 训练数据少(~3K) | 过拟合风险 | 关键点弱监督 |
| 无关键点监督 | 不可解释 | 辅助损失 |

**增强版：带面部区域先验**：
```python
class AmygdalaWithPrior(nn.Module):
    """带面部区域先验的杏仁核"""
    def __init__(self, fast_dim=512, prior_strength=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(fast_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 196),
            nn.Sigmoid()
        )
        self.prior_strength = prior_strength
        self.register_buffer('face_region_prior', self._create_prior())
        
    def _create_prior(self):
        prior = torch.zeros(1, 1, 14, 14)
        prior[:, :, 2:6, 5:9] = 1.0
        prior[:, :, 6:9, 4:10] = 0.8
        prior[:, :, 9:12, 5:9] = 0.6
        prior = prior / (prior.sum() + 1e-8)
        return prior
        
    def forward(self, fast_feat):
        learned = self.fc(fast_feat).view(-1, 1, 14, 14)
        combined = learned * (1 - self.prior_strength) + self.face_region_prior * self.prior_strength
        return combined.view(-1, 1, 14, 14)
```

#### 3.3.2 FFA（Feature Fusion Attention）

**功能**：SE风格的跨通道特征重校准

**原理**：
$$z = \sigma\left(\text{FC}_{1280\rightarrow80}(\text{concat}[f_{\text{fast}}, f_{\text{slow}}])\right)$$

$$f_{\text{fast}}^* = z_{[:512]} \odot f_{\text{fast}}, \quad f_{\text{slow}}^* = z_{[512:]} \odot f_{\text{slow}}$$

**实现**：
```python
class FFA(nn.Module):
    """Feature Fusion Attention"""
    def __init__(self, fast_dim=512, slow_dim=768):
        super().__init__()
        total_dim = fast_dim + slow_dim
        
        self.fc = nn.Sequential(
            nn.Linear(total_dim, 80),
            nn.ReLU(),
            nn.Sigmoid()
        )
        
    def forward(self, fast_feat, slow_feat):
        concat = torch.cat([fast_feat, slow_feat], dim=-1)
        z = self.fc(concat)  # (B, 80)
        
        gate_fast = z[:, :512].unsqueeze(-1)
        gate_slow = z[:, 512:].unsqueeze(-1)
        
        return fast_feat * gate_fast, slow_feat * gate_slow
```

#### 3.3.3 CASANet

**功能**：三角注意力实现apex帧检测

**原理**：
$$\text{apex\_score}_t = \text{softmax}\left(\text{MHA}(Q_t, K, V)\right) \in \mathbb{R}^T$$

三角先验 $M_{i,j} = \exp\left(-\frac{(j-i)^2}{2\sigma_i^2}\right)$ 模拟微表情的onset→apex→decay模式

**实现**：
```python
class CASANet(nn.Module):
    """Cascaded Self-Attention Network for Apex Detection"""
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        
        # 三角先验
        self.triangular_prior = nn.Parameter(
            self._create_triangular_mask(16)
        )
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        
        # 输出层
        self.fc = nn.Linear(dim, 1)
        
    def forward(self, spatial_map):
        # spatial_map: (B, 768, 1, 7, 7)
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

**设计说明**：

| 问题 | 实际 | 缓解 |
|------|------|------|
| "固定形态"? | **可学习** - nn.Parameter | 初始化为三角形，会被梯度调整 |
| "个体差异"? | 全局先验+adaptation | PersonalizedRadar处理被试 |
| "限制灵活性"? | 归纳偏置，非硬限制 | 模型可学偏 |

**增强版：CASANetAdaptive**
```python
class CASANetAdaptive(nn.Module):
    """带个人适应性调整的CASANet"""
    def __init__(self, dim=768, num_heads=8):
        self.triangular_prior = nn.Parameter(self._create_triangular_mask(16))
        self.adaptive_scale = nn.Parameter(torch.ones(1))
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.fc = nn.Linear(dim, 1)
        
    def forward(self, spatial_map, person_id=None):
        x = spatial_map.squeeze(2).flatten(2)
        
        if person_id is not None:
            person_scale = torch.tanh(self.adaptive_scale + torch.sin(person_id) * 0.1)
            adjusted = self.triangular_prior * person_scale
        else:
            adjusted = self.triangular_prior
            
        x = x + adjusted.unsqueeze(0)
        attn_out, _ = self.mha(x, x, x)
        scores = self.fc(attn_out).squeeze(-1)
        return attn_out, scores.mean(dim=-1, keepdims=True)
```

### 3.4 融合模块

#### 3.4.1 TSFmicroFusion

**功能**：双向交叉注意力融合

**原理**：
$$\text{F}_{f2s} = \text{Attention}\left(Q_f \cdot W_Q, K_s \cdot W_K, V_s \cdot W_V\right) \cdot W_O$$

$$\text{F}_{s2f} = \text{Attention}\left(Q_s \cdot W_Q, K_f \cdot W_K, V_f \cdot W_V\right) \cdot W_O$$

$$f_{\text{fused}} = \alpha \cdot \text{FFN}(\text{F}_{f2s}) + (1-\alpha) \cdot \text{FFN}(\text{F}_{s2f})$$

**实现**：
```python
class TSFmicroFusion(nn.Module):
    """Two-Stream Fusion Micro-Expression Fusion"""
    def __init__(self, fast_dim=512, slow_dim=768, fused_dim=1024):
        super().__init__()
        
        # 投影
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
        
        # 融合权重
        self.alpha_net = nn.Linear(fast_dim + slow_dim, 1)
        
    def forward(self, fast_feat, slow_feat):
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

### 3.5 解码模块

#### 3.5.1 DynamicAUDecoder

**功能**：BiLSTM进行时间AU序列建模

**原理**：
$$\mathbf{h}_t = \text{BiLSTM}(f_{\text{fused}}, \mathbf{h}_{t-1})$$

$$\text{AU}_{b,t} = \sigma\left(\text{Linear}(\mathbf{h}_t)\right) \in \mathbb{R}^{28}$$

$$\text{OPD}_{b,u} = [t_{\text{onset}}, t_{\text{peak}}, t_{\text{decay}}] \in \mathbb{R}^3$$

**实现**：
```python
class DynamicAUDecoder(nn.Module):
    """Dynamic Action Unit Decoder"""
    def __init__(self, input_dim=1024, hidden_dim=512, num_aus=28):
        super().__init__()
        
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2, batch_first=True,
            bidirectional=True
        )
        
        # AU强度输出
        self.au_head = nn.Linear(hidden_dim * 2, num_aus)
        
        # OPD路标输出
        self.opd_head = nn.Linear(hidden_dim * 2, num_aus * 3)
        
    def forward(self, fused_feat):
        # fused_feat: (B, 1024) -> (B, 1, 1024)
        x = fused_feat.unsqueeze(1)
        
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

### 3.6 多专家模块

#### 3.6.1 MoEGatingNetwork

**功能**：噪声top-k门控，3专家MLP

**原理**：
$$g = \text{softmax}\left(\text{top-}k\left(W_g \cdot f_{\text{fused}}\right)\right)$$

$$\text{ME\_logits} = \sum_{i=1}^{3} g_i \cdot \text{Expert}_i(f_{\text{fused}})$$

**辅助损失**：
$$\mathcal{L}_{\text{moe}} = \lambda \sum_{i=1}^{3} \left(\bar{f}_i - \frac{1}{3}\right)^2$$

**实现**：
```python
class MoEGatingNetwork(nn.Module):
    """Mixture of Experts Gating"""
    def __init__(self, input_dim=1024, num_experts=3, top_k=2):
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
                nn.Linear(256, 7)  # 7-class
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, x):
        # 门控 logits
        gate_logits = self.gate(x)  # (B, 3)
        
        # Top-k选择
        top_k_logits, top_k_idx = torch.topk(
            gate_logits, self.top_k, dim=-1
        )
        
        # Softmax
        gate_weights = F.softmax(top_k_logits, dim=-1)
        
        # 专家输出
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=1)  # (B, 3, 7)
        
        # 加权求和
        me_logits = torch.einsum('bg,bge->be', gate_weights, expert_outputs)
        
        # 辅助损失
        aux_loss = self._load_balancing_loss(gate_weights)
        
        return me_logits, gate_weights, aux_loss
```

**设计意图 vs 实际**：

| 方面 | 设计意图 | 需验证 |
|------|----------|--------|
| Expert 0 | 正面情绪(joy, surprise) | 需可视化 |
| Expert 1 | 负面情绪(fear, anger) | 需可视化 |
| Expert 2 | 中性(disgust, contempt) | 需可视化 |

**验证代码**：
```python
def visualize_gating(model, test_loader):
    """可视化门控激活，验证专家分化"""
    gating_activations = []
    labels = []
    
    for batch, label in test_loader:
        _, gate_weights, _ = model(batch)
        gating_activations.append(gate_weights)
        labels.append(label)
    
    # 每类情绪的专家分布
    emotion_names = ['joy', 'sadness', 'trust', 'disgust', 'fear', 'anger', 'surprise']
    for i, name in enumerate(emotion_names):
        mask = labels == i
        dist = gating_activations[mask].mean(dim=0)
        print(f"{name}: {dist.numpy()}")
    
    # 预期结果
    """
    有分化：
      joy:       [0.8, 0.1, 0.1]
      sadness:   [0.1, 0.8, 0.1]
    
    无分化（坍塌）：
      All:       [0.34, 0.33, 0.33]
    """
```

**客观性分析**：

| 方面 | 设计声称 | 现实 | 客观性 |
|------|--------|------|------|
| 膜电位 | 通过反馈累积 | 无自动反馈 | ⚠️ 需要人工反馈 |
| 情绪状态 | 影响路由 | 从正确率推导 | ⚠️ 简单映射 |
| 生物启发 | 神经科学 | 类比 | ⚠️ 类比非精确模型 |

**诚实评估**：BioMoE是"生物启发的架构模式"，而非精确生物模型。"膜电位"本质是内存缓冲区，"情绪状态"只是滚动正确率。

**更客观版本**（用模型自身的置信度）：
```python
class MoEGatingNetworkObjective(nn.Module):
    """客观MoE：用模型置信度作为隐式反馈"""
    def forward(self, x):
        gate_logits = self.gate(x)
        
        # 用最大置信度作为隐式"膜电位"
        max_conf = gate_logits.max(dim=-1, keepdims=True)[0]
        
        # 略微提升置信度高的专家
        confidence_bonus = torch.sigmoid(max_conf - 0.5) * 0.1
        gate_logits = gate_logits + confidence_bonus
        
        # 标准top-k选择
        return me_logits, gate_weights, aux_loss
```

**建议**：客观评估时使用标准MoEGatingNetwork。BioMoE可作为消融实验。

**强制专业化版**（如果无分化）：
```python
class SpecializedMoE(nn.Module):
    """带强制专业化的MoE"""
    def __init__(self, input_dim=1024, num_experts=3, use_forced=False):
        super().__init__()
        self.use_forced = use_forced
        
        # 专家分工
        self.expert_specialty = {
            0: [0, 6],   # joy, surprise
            1: [3, 4, 5], # fear, anger, sadness
            2: [1, 2],    # trust, disgust
        }
        
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([...])
        
    def forward(self, x, emotion_label=None):
        gate_logits = self.gate(x)
        
        if self.use_forced and emotion_label is not None:
            # 强制使用对应专家
            specialty_mask = torch.zeros(3)
            for exp_id, emotions in self.expert_specialty.items():
                if emotion_label.item() in emotions:
                    specialty_mask[exp_id] = 1.0
            gate_logits = gate_logits * specialty_mask
        
        return me_logits, gate_weights, None
```python
        
        # 负载均衡辅助损失
        aux_loss = self._load_balancing_loss(gate_weights)
        
        return me_logits, gate_weights, aux_loss
    
    def _load_balancing_loss(self, gate_weights):
        mean_load = gate_weights.mean(dim=0)
        return ((mean_load - 1.0 / self.num_experts) ** 2).sum()
```

#### 3.6.2 PersonalizedRadar

**功能**：测试时个性化适配

```python
class PersonalizedRadar(nn.Module):
    """Test-Time Personalization"""
    def __init__(self, input_dim=1024, steps=5, lr=0.01):
        super().__init__()
        
        self.steps = steps
        self.lr = lr
        
        # 残差适配器
        self.adapter = nn.Linear(input_dim, input_dim)
        
    def adapt(self, model, support_x, support_labels):
        """内循环SGD适配"""
        model.eval()
        
        # 克隆参数
        adapted_model = copy.deepcopy(model)
        
        for step in range(self.steps):
            outputs = adapted_model(support_x)
            loss = F.cross_entropy(
                outputs['me_logits'], support_labels
            )
            
            # SGD更新
            loss.backward()
            with torch.no_grad():
                for param in adapted_model.parameters():
                    param -= self.lr * param.grad
        
        return adapted_model
```

**设计分析**：

| 方面 | 当前值 | 局限 | 缓解 |
|-------|--------|------|------|
| 步数 | 5 | 可能不足 | 10-15步+渐进LR |
| 学习率 | 固定0.01 | 可能过调 | 预热+衰减 |
| 适配器 | 线性 | 容量有限 | MLP残差连接 |
| Support样本 | 少 | 过拟合 | 更多样本+正则化 |

**为什么5步？**
- Support集通常很小（5-20样本）
- 更多步数会过拟合
- 权衡：适配 vs 泛化

**误差来源分析**（主要）：

| 误差来源 | 影响 | 占比 |
|----------|------|------|
| **个体差异** | 微笑vs轻蔑幅度相似 | 30-40% |
| 数据不平衡 | 某些情绪样本少 | 20-30% |
| 光照/姿态 | 特征质量下降 | 15-20% |
| 标注噪声 | AU标注主观性 | 10-15% |
| 其他 | 设备/算法 | 5-10% |

**增强版**：
```python
class PersonalizedRadarEnhanced(nn.Module):
    """增强版TTA：预热学习率+残差适配器"""
    def __init__(self, input_dim=1024, steps=10, lr=0.01):
        super().__init__()
        self.steps = steps
        self.lr = lr
        
        # 残差适配器
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim)
        )
        
    def adapt(self, model, support_x, support_labels):
        adapted_model = copy.deepcopy(model)
        
        for step in range(self.steps):
            # 预热调度：早期大LR，后期小LR
            progress = step / max(self.steps - 1, 1)
            current_lr = self.lr * (1 - progress) * 0.7 + self.lr * 0.3
            
            outputs = adapted_model(support_x)
            loss = F.cross_entropy(outputs['me_logits'], support_labels)
            loss.backward()
            
            with torch.no_grad():
                for param in adapted_model.parameters():
                    if param.grad is not None:
                        param -= current_lr * param.grad
                        
        return adapted_model


class SubjectNormalizedRadar(nn.Module):
    """被试归一化：解决个体差异
    
    核心思路：被试内相对比较，而非绝对值
    """
    def __init__(self):
        self.subject_stats = {}
        
    def normalize(self, features, subject_id):
        if subject_id not in self.subject_stats:
            self.subject_stats[subject_id] = {
                'mean': features.mean(dim=0, keepdims=True),
                'std': features.std(dim=0, keepdims=True) + 1e-8
            }
        return (features - self.subject_stats[subject_id]['mean']) / self.subject_stats[subject_id]['std']
```

### 3.8 微表情增强模块

#### 3.8.1 MicroExpressionBoost — 低强度信号增强

**设计动机**: 微表情有三个独特属性需要专门处理：
- **无意识性**：自发产生，非刻意控制
- **低强度**：幅度远弱于宏表情
- **局部区域**：变化发生在面部小区域

**原理**:
$$\text{feat}_{\text{enhanced}} = \text{feat} \cdot \sigma(W_{\text{spatial}} \cdot \text{feat})$$

**实现**:
```python
class MicroExpressionBoost(nn.Module):
    """增强低强度微表情信号"""
    def __init__(self, channels=512, num_regions=8):
        super().__init__()
        self.region_coords = {
            'brow': [(32, 20), (48, 28)],
            'eye': [(28, 35), (52, 40)],
            'mouth': [(25, 55), (55, 62)],
        }
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.pyramid = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=i)
            for i in [1, 2, 3]
        ])
        
    def forward(self, x):
        B, C, H, W = x.shape
        attn = self.spatial_attention(x)
        x_enhanced = x * attn
        pyramid_feats = [F.relu(p(x_enhanced)) for p in self.pyramid]
        weights = torch.softmax(torch.randn(B, 3, H, W), dim=1)
        x_fused = sum(w * f for w, f in zip(weights.unbind(1), pyramid_feats))
        return x_fused + x_enhanced


class UnconsciousContrastiveLoss(nn.Module):
    """对比损失：区分微表情vs宏表情"""
    def __init__(self, margin=0.5, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, micro_emb, macro_emb, labels):
        micro_emb = F.normalize(micro_emb, dim=-1)
        macro_emb = F.normalize(macro_emb, dim=-1)
        sim_cross = torch.mm(micro_emb, macro_emb.t()) / self.temperature
        loss = F.relu(self.margin - sim_cross.mean())
        return loss
```

### 3.9 报告模块

#### 3.7.1 EmotionReporter

**功能**：基于模板的临床报告生成

```python
class EmotionReporter(nn.Module):
    """Template-based emotion report"""
    def __init__(self):
        super().__init__()
        
        # AU模板
        self.au_templates = {
            1: "眉毛内侧抬高",
            4: "眉毛降低",
            6: "颧骨抬高",
            12: "嘴角抬高",
            # ...
        }
        
        # 表情模板
        self.me_templates = {
            0: "高兴",
            1: "悲伤",
            2: "惊讶",
            3: "恐惧",
            4: "愤怒",
            5: "厌恶",
            6: "藐视"
        }
        
    def generate_report(self, au_intensities, me_pred, apex_frame):
        reports = []
        
        for b in range(au_intensities.shape[0]):
            # 活跃AU
            active_aus = []
            for au_idx in range(28):
                if au_intensities[b, au_idx] > 0.3:
                    active_aus.append(
                        self.au_templates.get(au_idx, f"AU{au_idx}")
                    )
            
            # 表情
            emotion = self.me_templates[me_pred[b]]
            
            # 生成报告
            report = f"检测到{emotion}情绪，主要动作单元：{', '.join(active_aus) if active_aus else '无明显AU'}，apex帧位置：{apex_frame[b].item()}"
            reports.append(report)
        
        return reports
```

---

## 四、数学公式

### 4.1 总损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{me}} + \alpha \mathcal{L}_{\text{au}} + \beta \mathcal{L}_{\text{moe}} + \gamma \mathcal{L}_{\text{opd}}$$

| 损失 | 类型 | 描述 |
|------|------|------|
| $\mathcal{L}_{\text{me}}$ | 交叉熵 | 7类微表情分类 |
| $\mathcal{L}_{\text{au}}$ | 二值交叉熵 | 28类AU多标签识别 |
| $\mathcal{L}_{\text{moe}}$ | 负载均衡辅助 | 防止专家坍塌 |
| $\mathcal{L}_{\text{opd}}$ | L2平滑 + 峰值一致性 | Onset-peak-decay时间模式 |

**损失权重选择**：

| 权重 | 默认值 |  rationale | 调整优先级 |
|------|--------|------------|
| α (AU) | 0.5 | AU辅助监督，中等权重 | 2级 |
| β (MoE) | 0.01 | 负载均衡正则化，小权重 | 3级 |
| γ (OPD) | 0.1 | 时间路标，中等权重 | 1级 |

**当前是经验值**：基于验证集选择，非理论最优。未来工作：梯度自动调优或不确定性加权。

**调整策略**：
- OPD不准：先改进时序建模，再↑γ
- AU不准：先检查标注可靠性，再↑α  
- MoE坍塌：↑β

### 4.2 架构维度

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
ME logits:       (B, 7)       ← 7类交叉熵
专家门控:        (B, 3)       ← top-2 softmax
```

---

## 五、训练流程

### 5.1 训练脚本

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
            loss_me = F.cross_entropy(
                outputs['me_logits'], me_labels
            )
            loss_au = F.binary_cross_entropy_with_logits(
                outputs['au_intensities'], au_labels
            )
            loss_moe = outputs['moe_aux_loss']
            loss_opd = F.mse_loss(
                outputs['au_opd'][:, 1:] - outputs['au_opd'][:, :-1]
            )
            
            # 总损失
            loss = loss_me + au_weight * loss_au + \
                  moe_weight * loss_moe + opd_weight * loss_opd
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
```

### 5.2 训练参数

| 参数 | 默认值 | 范围 | 描述 |
|------|-------|------|------|
| lr | 1e-4 | 1e-5-1e-3 | 学习率 |
| batch_size | 2 | 1-16 | 批大小 |
| epochs | 50 | 10-200 | 训练轮数 |
| weight_decay | 1e-4 | 1e-6-1e-2 | 权重衰减 |
| au_loss_weight | 0.5 | 0.1-1.0 | AU损失权重 |
| moe_loss_weight | 0.01 | 0.001-0.1 | MoE权重 |
| opd_loss_weight | 0.1 | 0.01-0.5 | OPD权重 |

---

## 六、使用示例

### 6.1 基础用法

```python
import torch
from model import Censor

# 初始化模型
model = Censor()

# 准备输入
video = torch.randn(1, 3, 16, 224, 224)

# 前向传播
with torch.no_grad():
    outputs = model(video)

# 访问结果
me_logits = outputs['me_logits']           # (B, 7)
au_intensities = outputs['au_intensities']  # (B, T, 28)
apex_scores = outputs['apex_scores']        # (B, 1)
reports = outputs['template_report']         # List[str]
```

### 6.2 训练用法

```python
# 使用合成数据训练
python train.py --epochs 5 --batch_size 2 --synthetic_data

# 使用真实数据训练
python train.py --epochs 50 --batch_size 4 --data_root ./data/CASME_II
```

### 6.3 测试时适配

```python
from model import Censor, PersonalizedRadar

# 加载基础模型
model = Censor()
checkpoint = torch.load('./checkpoints/best.pt')
model.load_state_dict(checkpoint['model_state'])

# 个性化适配
radar = PersonalizedRadar(steps=5)
adapted_model = radar.adapt(model, support_videos, support_labels)
```

---

## 七、项目结构

```
censor/
├── main.py                 # Censor编排器 + 前向传播测试
├── train.py                # 训练流程（多任务损失、AMP、检查点）
├── requirements.txt
├── config/
│   └── defaults.py         # 中央超参数字典
└── model/
    ├── __init__.py         # 重新导出所有类
    ├── preprocessing.py   # 显著性检测器、rPPG提取器、TVL1光流
    ├��─ backbones.py       # 快皮层下通路、慢皮层通路
    ├── attention.py      # 杏仁核、FFA、CASANet
    ├── fusion.py        # TSFmicroFusion
    ├── decoders.py     # 动态AU解码器
    ├── moe_head.py    # MoE门控网络、个性化Radar
    ├── llm_report.py   # 情绪报告器
    ├── biomimetic_enhance.py # 动态拓扑网络 + 元学习可塑性
    └── biomoe.py       # 生物门控（BioMoE）
```

---

## 八、常见问题与解决方案

### 8.1 内存不足

**问题**：`RuntimeError: CUDA out of memory`

**解决方案**：
```python
# 减小batch_size
batch_size = 2

# 使用合成数据测试
python main.py --synthetic
```

### 8.2 训练不收敛

**问题**：`loss: nan`

**解决方案**：
```python
# 减小学习率
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### 8.3 专家坍塌

**问题**：MoE只激活一个专家

**解决方案**：
```python
# 确保负载均衡损失权重足够大
moe_loss_weight = 0.01

# 使用噪声门控
gate_logits = gate(x) + torch.randn_like(gate(x)) * 0.1
```

---

## 九、API参考

### 9.1 模型加载

```python
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
```

### 9.2 完整示例

```python
import torch
from model import Censor

model = Censor()
model.eval()

video = torch.randn(2, 3, 16, 224, 224)

with torch.no_grad():
    outputs = model(video)

print(f"ME Logits: {outputs['me_logits'].shape}")
print(f"AU: {outputs['au_intensities'].shape}")
print(f"Apex: {outputs['apex_scores'].shape}")
```

---

## 十、参考文献

1. Ekman, P., & Friesen, W. V. (1969). Nonverbal leakage and clues to deception. *Psychiatry*.

2. Pfister, T., et al. (2011). Recognising spontaneous facial micro-expressions. *ICCV*.

3. Liong, S. T., et al. (2018). Automatic dynamic range textural description. *IEEE TPAMI*.

4. Liu, Y. J., et al. (2019). Auxiliary signal regularized CNN. *ICCV*.

5. Verburg, M., & Menegato, G. (2024). Graph attention networks for MER. *Heliyon*.

6. Xia, R., & Deng, J. (2025). Hybrid Attention-3DNet. *JJCIT*.

7. Cao, J., et al. (2024). Video Swin Transformer. *CVPR*.

8. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.

9. LeDoux, J. E. (2000). Emotion circuits in the brain. *Annual Review of Neuroscience*.

10. Hebb, D. O. (1949). *The Organization of Behavior*. Wiley.

---

## 十一、版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-05-11 | 初始版本 |

---

## 附录A：扩展微表情分类

### A.1 11类分类体系

基于MER数据集（CASME II, SAMM, SMIC, MMEW），将原始7类扩展为11类：

| ID | 类别 | AU标记 | 数据集来源 |
|----|------|-------|----------|
| 0 | Happiness (Duchenne真笑) | AU6+AU12（眼轮匝肌）| CASME II |
| 1 | Happiness (Non-Duchenne假笑) | AU12 only | CASME II |
| 2 | Surprise (强烈) | AU1+AU2+AU5+AU26 | - |
| 3 | Surprise (轻微) | AU1+AU2 | - |
| 4 | Fear（恐惧）| AU1+AU2+AU4+AU5+AU7+AU26 | - |
| 5 | Disgust (强烈) | AU9+AU10+AU17 | - |
| 6 | Disgust (轻微) | AU9 | - |
| 7 | Anger (强烈) | AU4+AU7+AU23+AU24 | - |
| 8 | Anger (轻微) | AU4 | - |
| 9 | Sadness（悲伤）| AU1+AU4+AU15+AU17 | - |
| 10 | Contempt（蔑视）| AU12+AU14（单侧）| - |

关键区分：**Duchenne vs Non-Duchenne** 微笑（CASME II特色标注）

### A.2 7类到11类映射

用于向后兼容7类数据集：

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

## 附录B：高级MoE架构

### B.1 层级动态MoE（HieDyMoE）

结合**层级**（粗→细）和**动态**（输入条件）路由：

**第一层：粗粒度组（3类）**
- 组0：积极（Positive）：Happiness, Contempt
- 组1：消极（Negative）：Sadness, Fear, Anger, Disgust
- 组2：惊讶（Surprise）

**第二层：细粒度专家（共9个）**
- 组0：3个专家（Happiness强/弱，Contempt）
- 组1：4个专家（Sadness, Fear, Anger, Disgust）
- 组2：2个专家（Surprise强/弱）

**动态路由：**
```
输入特征 → 条件编码器 → （光照、遮挡、运动强度）
                              ↓
                           组内Top-k选择
```

### B.2 可用MoE模块

| 模块 | 专家数 | 返回值 | 特性 |
|------|--------|--------|------|
| MoEGatingNetwork | 3 | output, gates, aux_loss | 原始Top-2 |
| EnhancedMoE | 3 | output, gates, aux_loss, info | 膜电位+情绪 |
| BioMoE | 3 | output, gates, aux_loss, membrane_info | 仿生门控 |
| HierarchicalDynamicMoE | 9 | output, hierarchy, aux_loss | 层级+动态 |
| HierarchicalDynamicMoELite | 3 | output, gates, aux_loss | 轻量级 |
| PersonalizedRadar | TTA | adapted | 测试时适配 |

---

## 附录C：空间注意力机制

### C.1 AU地标注意力

基于面部动作单元（AU）的独立空间注意力，关注面部关键区域：

**区域中心与权重：**
- 眉毛（AU1,2,4）：权重=1.0
- 眼睛（AU5,6,7）：权重=1.2
- 鼻子（AU9）：权重=0.8
- 嘴巴（AU10,12,14,15,17,20,23-28）：权重=1.0

使用方式：
```python
from model.au_attention import create_au_attention_map, AUMaskedAttention

# 独立函数
attn = create_au_attention_map(224)  # (1, 1, 224, 224)

# 带上掩码
masker = AUMaskedAttention(size=224)
masked, attn = masker(features, apply_mask=True)
```

### C.2 倒三角形注意力

空间注意力掩码初始化为倒三角形（上部宽→下部窄）：

```
      ●──────●     ← 眉毛（宽）
       ●────●       ← 眼睛（中）
        ●──●         ← 鼻子（窄）
         ●●          ← 嘴巴（很窄）
```

在**CASANet**中实现（`model/attention.py`）：
- 可学习空间掩码（7×7）
- 结合时间多头注意力进行**顶帧检测**

---

## 附录D：配置参数汇总

### PREPROCESS_CONFIG

```python
{
    'pyramid_levels': 4,
    'gaussian_sigma': 1.5,
    'center_bias_strength': 1.0,
    'sigma_ratio': 0.15,
    'rppg_window_size': 5,
    'rppg_bandpass_low': 0.5,   # Hz
    'rppg_bandpass_high': 4.0,  # Hz
    'tvl1_tau': 0.25,
    'tvl1_lambda': 0.15,
    'tvl1_theta': 0.3,
    'fast_threshold': 0.1,
    'use_tvl1': True,
    'au_attention_size': 224,
    'au_mask_threshold': 0.1,
}
```

### MOE_CONFIG

```python
{
    'input_dim': 1024,
    'hidden_dim': 512,
    'num_experts': 3,
    'gating_hidden_dim': 128,
    'num_classes': 11,  # 扩展后
    'top_k': 2,
    'load_balancing_lambda': 0.01,
    'use_dynamic_routing': True,
    'condition_hidden_dim': 64,
}
```

## 附录E：事件驱动机制与人类注意力模型

### E.1 设计理念

事件驱动机制受人类注意力动态启发，实现"走神→定向反应→专心致志"的注意力切换：

- **默认模式 (AMBIENT)**: "走神"状态，保持基础监测（10%注意力）
- **定向反应 (ORIENTING)**: 检测到变化时加强注意（30%注意力）
- **专心致志 (FOCUSED)**: 表情涌现时全力分析（100%注意力）

核心原则：**永不完全静默**！始终保持基础监测，在变化出现时快速集中。

### E.2 状态机

```
信号强度          状态        注意力    计算量
─────────────────────────────────────────────
< 0.15        AMBIENT     10%      轻量加权
0.15-0.30     ORIENTING  30%      平均
> 0.30         FOCUSED   100%     完整注意力
```

状态转换：
```
AMBIENT → (salience > 0.15) → ORIENTING → (confidence > 0.4) → FOCUSED
  ↑                                              ↓
  ←←←←←←←←←← 衰减回归 ←←←←←←←←←←←←←←←←
```

### E.3 核心模块

#### E.3.1 HumanAttentionController

```python
from model.human_attention import HumanAttentionController

controller = HumanAttentionController(input_dim=1280)
state, info = controller(combined_features)
# info: {'state': 'orienting', 'salience': 0.25, 'confidence': 0.6}
```

#### E.3.2 EventDrivenFusionHuman

```python
from model.human_attention import create_human_attention_fusion

fusion = create_human_attention_fusion({})
fused, info = fusion(fast_feat, slow_feat)
# info: {'state': 'FOCUSED', 'method': 'focused_full', 'salience': 0.45}
# fused: (B, 1024)
```

#### E.3.3 EventDrivenAUDecoderHuman

```python
from model.human_attention import create_human_attention_au_decoder

au_decoder = create_human_attention_au_decoder({})
au_intensities, opd, info = au_decoder(fused_feat)
# info: {'state': 'FOCUSED', 'magnitude': 0.8}
```

#### E.3.4 EventDrivenMoEHuman

```python
from model.human_attention import create_human_attention_moe

moe = create_human_attention_moe({})
output, info = moe(x, expression_type='happiness')
# info: {'state': 'FOCUSED', 'active_expert': 0}
```

### E.4 模块选择

| 模块 | 文件 | 适用场景 |
|------|------|---------|
| `NeuralPlasticityCycle` | brain_event.py | 复杂状态机，需突触可塑性 |
| `EventDrivenFusion` | event_driven_wrappers.py | 需要跳过计算时 |
| `HumanAttentionController` | human_attention.py | 人类注意力模式（推荐） |

### E.5 灵敏度保证

| 信号强度 | 模式 | 计算 | 能检测微表情 |
|---------|------|------|---------|
| < 0.15 | AMBIENT | 10% | ⚠️ 保持监测 |
| 0.15-0.30 | ORIENTING | 30% | ✅ 检测变化 |
| > 0.30 | FOCUSED | 100% | ✅ 确认表达 |

### E.6 性能优化

使用事件驱动机制可实现：
- **计算节省**: 弱信号时仅需30%计算
- **快速响应**: 状态切换延迟 < 1帧
- **灵敏度保持**: 永不漏检真实表情

---

## 附录F：长期记忆动态稀疏化控制

### F.1 设计动机与生物学背景

#### F.1.1 生物学启发

本模块的设计灵感来源于真实神经系统的**突触可塑性(Synaptic Plasticity)**机制：

**突触消退(Synaptic Silencing)**: 在生物大脑中，如果一个神经元通路长期不被使用，突触连接会逐渐减弱甚至消失。这是神经系统的"用进废退"机制，防止能量浪费在不常用的神经通路上。

**神经发生(Neurogenesis)**: 当大脑重新激活一个长期休眠的神经元时，会释放**脑源性神经营养因子(BDNF)**来加强新形成的神经连接。这允许大脑在需要时重新启用重要的神经通路。

**本模块的实现**正是模拟这两个机制：
1. 通过追踪神经元使用频率，对长期不活跃的神经元进行"冻结"(减少其参与计算)
2. 当检测到高活动信号时，自动"解冻"并给予增益(2x boost)

#### F.1.2 对抗过拟合

深度学习模型过拟合的根本原因是**有效参数量过多**，导致模型有能力"记住"训练数据而非学习泛化特征。本模块通过以下方式降低有效参数量：

| 机制 | 效果 |
|------|------|
| 硬冻结 | 将长期不活跃神经元输出置零，梯度截断 |
| 软衰减 | 对即将冻结的神经元进行权重衰减 |
| 随机Dropout | 额外引入15%随机屏蔽，模拟数据增强 |
| L2正则化 | 限制权重幅度，防止个别权重过大 |

#### F.1.3 与传统方法的对比

| 方法 | 本模块 | 传统Dropout | 传统L2 |
|------|-------|-----------|--------|
| 时序特性 | ✅ 基于累积使用频率 | ❌ 固定概率 | ❌ 固定惩罚 |
| 可恢复性 | ✅ 支持解冻+增益 | ❌ 一次性 | ❌ 固定惩罚 |
| 多层次 | ✅ 5个应用位置 | ❌ 单一层 | ❌ 单一层 |
| 自适应 | ✅ 动态调整 | ❌ 固定 | ❌ 固定 |

---

### F.2 系统架构

#### F.2.1 整体流程

```
输入特征 x ∈ ℝ^(B×D)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. NeuronUsageTracker                                    │
│    - 记录每个神经元的使用计数 usage_count[d]              │
│    - 记录最后活跃步 last_active_step[d]                  │
│    - 记录连续不活跃步数 inactivity_steps[d]         │
│    - 计算活动掩码 activity_mask[d]                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. HardFreezePath (硬冻结路径)                          │
│    - 判断: inactivity > hard_freeze_threshold → 冻结        │
│    - 检测高活动输入 → 自动解冻                          │
│    - 输出: masked_features = features × frozen_mask       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. SoftDecayPath (软衰减路径)                           │
│    - 判断: 0 < inactivity < soft_decay_threshold       │
│    - 衰减: weight = decay_factor^(inactivity/100)       │
│    - 输出: features × decay_weight                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 随机Dropout混合 (仅训练时)                         │
│    - 随机生成: random_mask ∼ Bernoulli(1-p)          │
│    - 只对未冻结神经元应用                              │
│    - 输出: features × (active_mask × random_mask)      │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. GrowthFactorSignal (生长因子)                        │
│    - 检测恢复事件: previously_frozen → now_active      │
│    - 增益: boost_factor (默认2.0)                  │
│    - 渐进衰减: 30步内从boost恢复到1.0                │
└─────────────────────────────────────────────────────┘
    │
    ▼
    输出特征 x' ∈ ℝ^(B×D)
```

#### F.2.2 关键组件详解

##### F.2.2.1 NeuronUsageTracker

负责追踪每个神经元的活跃状态：

```python
class NeuronUsageTracker(nn.Module):
    def __init__(self, dim):
        self.register_buffer('usage_count', torch.zeros(dim))           # 总使用次数
        self.register_buffer('last_active_step', torch.zeros(dim))     # 最后活跃步
        self.register_buffer('inactivity_steps', torch.zeros(dim))       # 连续不活跃步数
        self.register_buffer('cumulative_activity', torch.zeros(dim))     # 累积活跃度
    
    def update(self, activity_mask, current_step):
        # 所有神经元不活跃步数+1
        self.inactivity_steps += 1
        
        # 重置活跃神经元的计数
        active = activity_mask.bool()
        self.inactivity_steps[active] = 0
        self.usage_count[active] += 1
        self.last_active_step[active] = current_step
        self.cumulative_activity += activity_mask
```

**关键指标**:
- `usage_count[d]`: 神经元d被使用的总次数
- `inactivity_steps[d]`: 神经元d连续不活跃的步数
- `cumulative_activity[d]`: 神经元d的累积活跃强度

##### F.2.2.2 HardFreezePath

硬冻结路径通过将输出置零来实现梯度截断：

```python
class HardFreezePath(nn.Module):
    def __init__(self, dim, freeze_threshold=200, recovery_threshold=0.1):
        self.register_buffer('is_frozen', torch.zeros(dim, dtype=torch.bool))
        
    def forward(self, features, inactivity_counter):
        # 1. 判断是否应该冻结
        should_freeze = inactivity_counter > self.freeze_threshold
        newly_frozen = should_freeze & ~self.is_frozen
        self.is_frozen[newly_frozen] = True
        
        # 2. 检测高活动输入，实现自动恢复
        neuron_activity = features.abs().mean(dim=0)
        should_recover = self.is_frozen & (neuron_activity > self.recovery_threshold)
        self.is_frozen[should_recover] = False
        
        # 3. 应用冻结掩码
        frozen_mask = (~self.is_frozen).float().view(1, -1)
        return features * frozen_mask, should_recover.float()
```

**梯度截断机制**: 由于输出被置零，反向传播时冻结神经元的梯度也为零，该神经元不再参与训练。

##### F.2.2.3 SoftDecayPath

软衰减路径提供缓冲区域，对即将冻结的神经元进行渐进式衰减：

```python
class SoftDecayPath(nn.Module):
    def forward(self, features, inactivity_counter):
        # 软衰减区间: 0 < inactivity < 200
        soft_decay_zone = (inactivity_counter > 0) & (inactivity_counter < 200)
        
        if soft_decay_zone.any():
            # 渐进衰减: decay_factor^(inactivity/100)
            decay_weights = torch.where(
                soft_decay_zone,
                self.decay_mask ** (inactivity_counter / 100),
                torch.ones_like(self.decay_mask)
            )
            features = features * decay_weights.view(1, -1)
        
        return features
```

##### F.2.2.4 GrowthFactorSignal

当神经元从冻结状态恢复时，给予2倍增益，模拟BDNF的神经营养作用：

```python
class GrowthFactorSignal(nn.Module):
    def forward(self, features, recover_events):
        if recover_events.sum() > 0:
            # 立即应用2倍增益
            boost_mask = torch.ones(self.dim, device=features.device)
            boost_mask[recover_events.bool()] = 2.0
            features = features * boost_mask.view(1, -1)
        
        # 30步内渐进衰减回1.0
        # (渐进衰减逻辑...)
        
        return features
```

---

### F.3 数学形式化

#### F.3.1 状态变量定义

设D为特征维度，定义以下状态变量：

| 变量 | 符号 | 定义 |
|------|------|------|
| 使用计数 | u[d] | 神经元d被使用的总次数 |
| 不活跃步数 | i[d] | 神经元d连续不活跃的步数 |
| 冻结状态 | f[d] | 0=活跃, 1=冻结 |
| 恢复计数 | r[d] | 神经元d恢复后的步数 |

#### F.3.2 状态转移方程

```
状态转移规则:
─────────────────────────────────────────────────
1. 冻结: f[t] = 1  if i[t-1] > θ_freeze (默认200)
         f[t] = 0  otherwise

2. 软衰减: w[t] = α^(i[t-1]/100)  if 0 < i[t-1] < θ_soft (默认100)
          w[t] = 1            otherwise

3. 恢复: f[t] = 0  if a[t] > θ_recovery (默认0.1) AND f[t-1] = 1
         i[t] = 0  if f[t] = 0
         i[t] = i[t-1] + 1  if f[t] = 1

其中:
  θ_freeze = hard_freeze_threshold = 200
  θ_soft = inactivity_threshold = 100
  α = soft_decay_factor = 0.9
```

#### F.3.3 输出计算

给定输入特征x，稀疏控制输出x'为：

```
x'[b,d] = s_f[b,d] × s_s[b,d] × s_r[b,d] × g[b,d] × x[b,d]

其中:
  s_f[d] = (1 - f[d])                    # 冻结掩码
  s_s[d] = w[d]                          # 软衰减权重
  s_r[d] = 2.0 if r[d] < 30 else 1.0     # 生长因子
  g[d]   ~ Bernoulli(1-p) if training    # 随机Dropout
```

#### F.3.4 有效参数量

定义有效参数量为未被冻结的神经元比例：

```
E有效 = (1/D) × Σ_d (1 - f[d])
     = 1 - frozen_ratio

考虑生长因子的有效增益:
E总增益 = E有效 × max(g[d]) = E有效 × 2.0
```

---

### F.4 配置参数

#### F.4.1 完整配置

```python
SPARSE_CONTROL_CONFIG = {
    # ============== 核心参数 ==============
    'dim': 1024,                    # 特征维度
    'inactivity_threshold': 100,     # 软冻结阈值 (步)
    'hard_freeze_threshold': 200,    # 硬冻结阈值 (步)
    'soft_decay_factor': 0.9,        # 软衰减系数 (每100步)
    'growth_factor_boost': 2.0,      # 恢复增益倍数
    'growth_recovery_steps': 30,    # 增益衰减步数
    
    # ============== 防过拟合增强 ==============
    'enable_random_dropout': True,  # 随机Dropout开关
    'random_dropout_rate': 0.15,    # 随机Dropout比例
    
    'enable_l2': True,             # L2正则化开关
    'l2_weight': 0.01,            # L2权重系数
    
    # ============== 其他选项 ==============
    'min_activity_to_track': 0.01, # 活跃度阈值
    'enable_dual_path': True,    # 启用双路径(硬+软)
}
```

#### F.4.2 参数调优建议

| 场景 | 推荐配置 |
|------|----------|
| 数据集小 (<1000样本) | threshold降低到100/150, dropout=0.2 |
| 数据集大 (>10000样本) | threshold默认, dropout=0.1 |
| 过拟合严重 | threshold=50/100, dropout=0.25 |
| 欠拟合 | threshold不生效, dropout=0 |

---

### F.5 状态机

#### F.5.1 完整状态转移图

```mermaid
stateDiagram-v2
    [*] --> ACTIVE : 初始化
    ACTIVE --> SOFT_DECAY : inactivity > 100
    SOFT_DECAY --> HARD_FROZEN : inactivity > 200
    HARD_FROZEN --> RECOVERY : high_activity > 0.1
    RECOVERY --> ACTIVE : 恢复完成
    SOFT_DECAY --> ACTIVE : activity恢复
```
                    ┌─────────────────────────────────────┐
                    │           ACTIVE                    │
                    │    (正常参与计算, 梯度流动)        │
                    └─────────────────┬───────────────┘
                                    │
                           inactivity > 100
                                    ▼
                    ┌─────────────────────────────────────┐
                    │         SOFT_DECAY                 │
                    │   (权重逐渐衰减: 0.9^(i/100))    │
                    └──────────��──────┬───────────────┘
                                    │
                           inactivity > 200
                                    ▼
                    ┌─────────────────────────────────────┐
                    │        HARD_FROZEN                  │
                    │    (输出=0, 梯度截断)             │
                    └─────────────────┬───────────────┘
                                    │ high_activity (a > 0.1)
     recovery                              │
     events                              │
     (gain=2x) ◄─────────────────────┘
```

#### F.5.2 状态说明

| 状态 | 输出 | 梯度 | 说明 |
|------|------|------|------|
| ACTIVE | x | 正常流动 | 正常参与计算 |
| SOFT_DECAY | x×0.9^(i/100) | 正常 | 权重逐渐衰减 |
| HARD_FROZEN | 0 | 截断=0 | 输出置零，梯度截断 |
| RECOVERY | x×2.0 | 正常 | 增益2x恢复 |

---

### F.6 应用位置

#### F.6.1 在Censor中的集成

本模块在Censor Pipeline中应用于5个位置：

```mermaid
flowchart TD
    A[输入 Video<br/>B×3×16×224×224] --> B[Stage 1<br/>Preprocessing]
    B --> C[Stage 2<br/>Dual-Pathway]
    C --> D[Stage 2.5<br/>Sparse Control]
    D --> E[Stage 3<br/>Attention]
    E --> F[Stage 4<br/>Fusion]
    F --> G[Stage 4.5<br/>Sparse Control]
    G --> H[Stage 5<br/>AU Decoder]
    H --> I[Stage 6<br/>MoE Head]
    I --> J[Stage 6.5<br/>Sparse Control]
    J --> K[Stage 7<br/>Emotion Reporter]
    
    subgraph "Stage 2: Dual-Pathway"
    C1[FastPath<br/>3D ResNet-18] --> C2[512-D]
    C3[SlowPath<br/>3D Swin-T] --> C4[768-D]
    end
    
    subgraph "Stage 2.5: Sparse Control"
    D1[fast_path<br/>512] & D2[slow_path<br/>768]
    end
    
    subgraph "Stage 4.5: Sparse Control"
    G1[fusion<br/>1024]
    end
    
    subgraph "Stage 6.5: Sparse Control"
    J1[moe_coarse<br/>3] & J2[moe_fine<br/>9]
    end
```

#### F.6.2 各位置详细

| 位置 | 维度 | 控制内容 | 主要作用 |
|------|------|--------|--------|
| fast_path | 512 | 快速通路输出 | 控制光流特征通道 |
| slow_path | 768 | 慢速通路输出 | 控制RGB+rPPG特征通道 |
| fusion | 1024 | 融合输出 | 控制混合特征 |
| moe_coarse | 3 | 粗粒度专家 | 控制情绪组选择 |
| moe_fine | 9 | 细粒度专家 | 控制具体表情类别 |

---

### F.7 防过拟合三重防御

#### F.7.1 防御层次

**第一层: 稀疏冻结**
- 机制: inactivity > 200 → 冻结
- 效果: 有效参数量减少
- 梯度: 冻结神经元梯度为0

**第二层: 随机Dropout**
- 机制: 额外15%随机屏蔽
- 效果: 模拟数据增强
- 梯度: 随机归零的神经元梯度为0

**第三层: L2正则化**
- 机制: 调用`get_l2_loss()`
- 效果: 限制权重幅度
- 梯度: 权重衰减项

#### F.7.2 使用方法

```python
# 1. 在模型定义中
from model.biomimetic_enhance import SparseControlWrapper

self.sparse_control = SparseControlWrapper({
    'fast_path': 512,
    'slow_path': 768,
    'fusion': 1024,
    'moe_coarse': 3,
    'moe_fine': 9,
})

# 2. 在前向传播中
def forward(self, x):
    # ...existing pipeline...
    
    # Stage 2.5: 应用稀疏控制
    pathway_feats, pathway_stats = self.sparse_control({
        'fast_path': fast_feat,
        'slow_path': slow_feat,
    })
    
    # Stage 4.5: 应用稀疏控制
    fusion_feats, fusion_stats = self.sparse_control({
        'fusion': fused_feat,
    })
    
    # 合并统计
    all_stats = {**pathway_stats, **fusion_stats}
    
    return output, all_stats

# 3. 在训练循环中获取损失
def compute_loss(output, target, model):
    # 交叉熵损失
    ce_loss = F.cross_entropy(output['logits'], target)
    
    # L2正则化损失
    l2_loss = 0
    for name, ctrl in model.sparse_control.sparse_controllers.items():
        l2_loss += ctrl.get_l2_loss()
    
    # 总损失 (可调整L2权重)
    total_loss = ce_loss + 0.01 * l2_loss
    
    return total_loss, {'ce': ce_loss, 'l2': l2_loss}
```

---

### F.8 实验验证

#### F.8.1 验证结果

| 测试 | 条件 | 冻结率 | 恢复率 |
|------|------|--------|--------|
| 零活动600步 | 输入全0 | 100% | - |
| 低活动200步 | 0.001x | 逐渐增加 | - |
| 高活动恢复 | 10x | - | 100% |

#### F.8.2 典型输出

```
[阶段1] 冻结: frozen_ratio=57.1%, freeze_events=586
[阶段2] 恢复: frozen_ratio=0.0%, recovery_events=18296
[统计] 有效参数量从2304降至~990后恢复至~2304+增益
```

---

### F.9 与其他防过拟合方法对比

| 方法 | 本模块 | Dropout | L2 | Label Smoothing |
|------|-------|--------|-----|---------------|
| 时序自适应 | ✅ | ❌ | ❌ | ❌ |
| 可恢复 | ✅ | ❌ | ❌ | ❌ |
| 多位置 | ✅ (5) | ❌ (1) | ❌ (1) | ❌ (1) |
| 增益机制 | ✅ | ❌ | ❌ | ❌ |
| 计算开销 | 中 | 低 | 低 | 低 |

---

### F.10 最佳实践与注意事项

#### F.10.1 训练建议

1. **初期禁用**: 训练前10%步骤禁用稀疏控制，让模型先学习基础特征
```python
def compute_loss(output, target, model, step, total_steps):
    if step < total_steps * 0.1:
        # 前10%步骤禁用稀疏控制
        return F.cross_entropy(output, target)
    
    # 之后启用
    return compute_sparse_loss(output, target, model)
```

2. **监控统计**: 定期检查冻结率
```python
for name, stats in model.sparse_control.get_all_stats().items():
    print(f"{name}: frozen={stats['frozen_ratio']:.1%}")
```

3. **调整阈值**: 根据验证集表现调整
```python
# 如果验证集准确率过低 → 过拟合
# → 降低hard_freeze_threshold 或 提高random_dropout_rate
```

#### F.10.2 推理建议

推理时可以选择禁用部分机制：
```python
# 推理时关闭随机Dropout
model.sparse_control.eval()  # 切换到eval模式

# 或者手动控制
for ctrl in model.sparse_control.sparse_controllers.values():
    ctrl.enable_random_dropout = False
```

#### F.10.3 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 冻结率过高 | 阈值太低 | 提高hard_freeze_threshold |
| 冻结率过低 | 数据太丰富 | 无需处理，正常 |
| 恢复失败 | 恢复阈值太高 | 降低recovery_threshold |
| L2损失波动 | batch_size变化 | 使用平均值 |

---

### F.11 API参考

#### F.11.1 核心类

```python
# 1. LongTermMemorySparseControl
from model.biomimetic_enhance import LongTermMemorySparseControl

ctrl = LongTermMemorySparseControl(config)
output, stats = ctrl(input_features)

# 2. SparseControlWrapper
from model.biomimetic_enhance import SparseControlWrapper

wrapper = SparseControlWrapper({
    'fast_path': 512,
    'slow_path': 768,
    'fusion': 1024,
})

controlled_feats, stats = wrapper(features_dict)
all_stats = wrapper.get_all_stats()

# 3. 获取L2损失
l2_loss = ctrl.get_l2_loss()  # 返回标量Tensor
```

#### F.11.2 返回值

**stats字典**:
```python
{
    'frozen_ratio': 0.0-1.0,      # 冻结比例
    'usage_ratio': 0.0-1.0,       # 使用比例
    'freeze_events': int,              # 冻结事件数
    'recovery_events': int,          # 恢复事件数
    'inactivity_mean': float,        # 平均不活跃步数
    'random_dropout_rate': float,   # 实际Dropout率
    'l2_contrib': float,         # L2贡献值
}
```

---

## 十二、引用

```bibtex
@article{censor2025,
  title={Censor: A Biomimetic Dual-Pathway Micro-Expression Recognition System with Fusiform-Amygdala Circuit and Mixture-of-Experts},
  author={},
  journal={},
  year={2025}
}
```