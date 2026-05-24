# FOMM-based Micro-Expression Generation Plan

## 1. 方案概述

### 1.1 核心思路
基于 First Order Motion Model (FOMM) 进行微表情微调，复用其成熟的人脸运动生成能力，注入微表情特有的精细控制。

### 1.2 优势分析

| 直接从头训练 | FOMM微调方案 |
|--------------|--------------|
| 数据需求：数千样本 | 数据需求：247样本 ✅ |
| 训练时间：数周 | 训练时间：数小时 ✅ |
| 质量不稳定 | 质量成熟稳定 ✅ |
| 创新性：中 | 创新性：应用创新 ✅ |

---

## 2. FOMM架构分析

### 2.1 FOMM原始架构

```
FOMM (First Order Motion Model) = 2个模块：

┌─────────────────────────────────────────────────────────────┐
│  Module 1: Motion Extractor (KE + Dense Motion)             │
│                                                              │
│  Driving Video → Keypoint Detector → Keypoint Positions     │
│                         ↓                                    │
│                   Dense Motion Network                       │
│                         ↓                                    │
│                   Motion Field (光流类似物)                   │
│                                                              │
│  Output: keypoints + motion_field                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Module 2: Generator (OC + G)                                │
│                                                              │
│  Source Image + Motion Field                                 │
│         ↓                                                    │
│  Occlusion Map Generator                                     │
│         ↓                                                    │
│  Image Generator (warping + inpainting)                      │
│         ↓                                                    │
│  Generated Video                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键点定义（FOMM默认10个关键点）

```
FOMM关键点分布：
0-1: 眉毛区域（左右）
2-3: 眼睛区域（左右）
4:   鼻子
5-6: 嘴角（左右）
7:   下巴
8-9: 脸部轮廓

这与AU有天然对应关系！
```

---

## 3. 微表情适配设计

### 3.1 AU → FOMM Keypoint 映射

```python
# 17 AU → 10 FOMM Keypoints 映射表

AU_MAPPING = {
    # 眉毛区域 (keypoints 0-1)
    'AU1': {'keypoint': 0, 'direction': 'up', 'region': 'inner_brow'},      # Inner Brow Raiser
    'AU2': {'keypoint': 1, 'direction': 'up', 'region': 'outer_brow'},      # Outer Brow Raiser
    'AU4': {'keypoints': [0, 1], 'direction': 'down_inward', 'region': 'brow_lowerer'},  # Brow Lowerer

    # 眼睛区域 (keypoints 2-3)
    'AU5': {'keypoints': [2, 3], 'direction': 'up', 'region': 'upper_lid'}, # Upper Lid Raiser
    'AU6': {'keypoints': [2, 3], 'direction': 'squeeze', 'region': 'cheek'}, # Cheek Raiser (眯眼)
    'AU7': {'keypoints': [2, 3], 'direction': 'down', 'region': 'lower_lid'}, # Lower Lid Depressor

    # 鼻子区域 (keypoint 4)
    'AU9': {'keypoint': 4, 'direction': 'up', 'region': 'nose'},            # Nose Wrinkler

    # 嘴部区域 (keypoints 5-6)
    'AU10': {'keypoint': 4, 'direction': 'up', 'region': 'upper_lip'},      # Upper Lip Raiser
    'AU12': {'keypoints': [5, 6], 'direction': 'outward_up', 'region': 'lip_corner'},  # Lip Corner Puller (微笑)
    'AU14': {'keypoints': [5, 6], 'direction': 'inward', 'region': 'dimpler'},  # Dimpler (嘴角收紧)
    'AU15': {'keypoints': [5, 6], 'direction': 'downward', 'region': 'lip_corner_down'},  # Lip Corner Depressor

    # 下巴区域 (keypoint 7)
    'AU17': {'keypoint': 7, 'direction': 'up', 'region': 'chin'},           # Chin Raiser Lower

    # 复合AU（微表情特有）
    'AU20': {'keypoints': [5, 6], 'direction': 'outward', 'region': 'lip_stretch'},  # Lip Stretch
    'AU25': {'keypoints': [5, 6, 7], 'direction': 'open', 'region': 'lips_part'},    # Lips Part
}

# 情感类别 → AU组合
EMOTION_AU_CONFIG = {
    'happiness': {
        'AU6': 0.6,   # Cheek Raiser (眯眼微笑)
        'AU12': 0.8,  # Lip Corner Puller (嘴角上扬)
        'AU25': 0.2,  # Lips Part (轻微张嘴)
    },
    'surprise': {
        'AU1': 0.7,   # Inner Brow Raiser
        'AU2': 0.7,   # Outer Brow Raiser
        'AU5': 0.8,   # Upper Lid Raiser (睁眼)
        'AU25': 0.5,  # Lips Part (张嘴)
    },
    'disgust': {
        'AU4': 0.5,   # Brow Lowerer
        'AU9': 0.7,   # Nose Wrinkler
        'AU10': 0.4,  # Upper Lip Raiser
        'AU17': 0.3,  # Chin Raiser
    },
    'repression': {
        'AU14': 0.6,  # Dimpler (嘴角收紧)
        'AU17': 0.4,  # Chin Raiser
        'AU4': 0.3,   # Brow Lowerer (轻微)
    },
}
```

### 3.2 微表情时间调制

```python
class MicroExpressionTemporalModulation:
    """
    微表情特有的时间动力学，调制FOMM的运动幅度。
    """

    def generate_curve(self, emotion_class, duration_frames, intensity=1.0):
        """
        生成onset-apex-offset时间曲线。

        Args:
            emotion_class: 情感类别
            duration_frames: 总帧数
            intensity: 微表情强度 (0.1-1.0)

        Returns:
            modulation_curve: 每帧的运动幅度调制系数
        """
        T = duration_frames

        # 微表情时间分布（比普通表情更短）
        onset_ratio = 0.3    # onset: 0-30%
        apex_ratio = 0.2     # apex: 30-50%
        offset_ratio = 0.5   # offset: 50-100%

        curve = torch.zeros(T)

        # Onset阶段：缓慢上升
        onset_end = int(T * onset_ratio)
        for t in range(onset_end):
            curve[t] = intensity * (t / onset_end) ** 0.5  # 平缓上升

        # Apex阶段：峰值保持
        apex_start = onset_end
        apex_end = int(T * (onset_ratio + apex_ratio))
        curve[apex_start:apex_end] = intensity

        # Offset阶段：缓慢下降
        for t in range(apex_end, T):
            progress = (t - apex_end) / (T - apex_end)
            curve[t] = intensity * (1 - progress) ** 0.7  # 平缓下降

        return curve
```

---

## 4. 微调策略

### 4.1 微调目标

保持FOMM的运动生成能力，注入微表情的：
1. **更小幅度的运动**（微表情强度低）
2. **更短的时间**（微表情持续时间短）
3. **特定AU组合**（微表情AU配置）

### 4.2 微调方法

```
微调策略：

┌─────────────────────────────────────────────────────────────┐
│  Step 1: 加载预训练FOMM                                      │
│                                                              │
│  - Motion Extractor: 保持冻结 ❄️                             │
│  - Generator: 微调 🔥                                        │
│                                                              │
│  原因：运动提取器已学会人脸关键点检测                         │
│        只需微调生成器适应微表情的细微运动                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Step 2: 添加AU条件注入                                      │
│                                                              │
│  在Generator中加入条件编码：                                  │
│  - AU_activation (17维) → 条件向量                           │
│  - emotion_class (4类) → 条件向量                            │
│  - intensity (1维) → 幅度调制                                │
│                                                              │
│  class AUConditionEncoder(nn.Module):                        │
│      def forward(self, au_activation, emotion, intensity):   │
│          # 融合条件                                          │
│          cond = self.au_encoder(au_activation)               │
│          cond += self.emotion_encoder(emotion)               │
│          cond *= intensity                                   │
│          return cond                                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Step 3: 微表情数据增强                                      │
│                                                              │
│  CASME2只有247样本 → 需要增强                                 │
│                                                              │
│  增强策略：                                                   │
│  1. 时间裁剪：提取onset-apex-offset片段                      │
│  2. 中性脸提取：首帧作为source image                         │
│  3. 强度变化：模拟不同强度的微表情                           │
│  4. AU抖动：轻微改变AU激活，增加多样性                        │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 损失函数设计

```python
class MicroExpressionLoss(nn.Module):
    """
    微表情生成损失函数。
    """

    def forward(self, generated, real, keypoints_gen, keypoints_real, au_pred, au_target):
        losses = {}

        # 1. 重建损失（像素级）
        losses['reconstruction'] = F.l1_loss(generated, real)

        # 2. 感知损失（特征级）
        losses['perceptual'] = self.perceptual_loss(generated, real)

        # 3. 关键点一致性（运动正确性）
        losses['keypoint'] = F.mse_loss(keypoints_gen, keypoints_real)

        # 4. AU一致性（肌肉正确性）
        losses['au'] = F.mse_loss(au_pred, au_target)

        # 5. 时间一致性（相邻帧平滑）
        losses['temporal'] = self.temporal_consistency_loss(generated)

        # 6. 微表情幅度约束（限制运动幅度）
        # 微表情运动应该比普通表情小
        motion_magnitude = keypoints_gen.abs().mean()
        losses['magnitude'] = torch.relu(motion_magnitude - 0.3)  # 限制幅度

        return losses
```

---

## 5. 整体架构

### 5.1 Censor-FOMM 联合架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Censor-FOMM: 识别 + 微表情生成                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────── 识别模块（我们的Censor）───────────────┐              │
│  │                                                         │              │
│  │  Input Video                                            │              │
│  │      ↓                                                  │              │
│  │  ┌─────────┐    ┌─────────┐                            │              │
│  │  │  Fast   │    │  Slow   │                            │              │
│  │  │ Encoder │    │ Encoder │                            │              │
│  │  └────┬────┘    └────┬────┘                            │              │
│  │       └───── FFA融合 ────┘                              │              │
│  │              ↓                                          │              │
│  │        ┌─────────┐                                      │              │
│  │        │  MoE    │ → 情感类别 (happiness/surprise/...)   │              │
│  │        │ Classifier│                                    │              │
│  │        └─────────┘                                      │              │
│  │              ↓                                          │              │
│  │        ┌─────────────┐                                  │              │
│  │        │ AU Predictor │ → AU激活向量 (17维)              │              │
│  │        │ (新增模块)   │                                  │              │
│  │        └─────────────┘                                  │              │
│  │              ↓                                          │              │
│  │        CASANet → Apex时间点                              │              │
│  │                                                         │              │
│  │  Output: 情感类别 + AU激活 + Apex时间                    │              │
│  │                                                         │              │
│  └─────────────────────────────────────────────────────────┘              │
│                          ↓ (条件输入)                                     │
│                                                                          │
│  ┌─────────────── 生成模块（FOMM微调）───────────────┐                  │
│  │                                                     │                  │
│  │  输入：                                              │                  │
│  │  - neutral_face: 中性脸（视频首帧）                  │                  │
│  │  - emotion_class: 情感类别                          │                  │
│  │  - au_activation: AU激活向量                        │                  │
│  │  - apex_time: Apex时间点                            │                  │
│  │  - intensity: 强度参数                              │                  │
│  │      ↓                                              │                  │
│  │  ┌─────────────────────────────────┐               │                  │
│  │  │ AU → Keypoint 映射               │               │                  │
│  │  │ (AU Controller)                  │               │                  │
│  │  │ 17 AU → 10 FOMM keypoints         │               │                  │
│  │  └─────────────────────────────────┘               │                  │
│  │      ↓                                              │                  │
│  │  ┌─────────────────────────────────┐               │                  │
│  │  │ Temporal Modulation              │               │                  │
│  │  │ (onset-apex-offset曲线)           │               │                  │
│  │  │ 调制每帧的运动幅度                │               │                  │
│  │  └─────────────────────────────────┘               │                  │
│  │      ↓                                              │                  │
│  │  ┌─────────────────────────────────┐               │                  │
│  │  │ FOMM Motion Extractor (冻结)     │               │                  │
│  │  │ keypoints → motion_field          │               │                  │
│  │  └─────────────────────────────────┘               │                  │
│  │      ↓                                              │                  │
│  │  ┌─────────────────────────────────┐               │                  │
│  │  │ FOMM Generator (微调)            │               │                  │
│  │  │ source + motion → generated video │               │                  │
│  │  │ + AU条件注入                      │               │                  │
│  │  └─────────────────────────────────┘               │                  │
│  │      ↓                                              │                  │
│  │  Output: 微表情视频序列                              │                  │
│  │                                                     │                  │
│  └─────────────────────────────────────────────────────┘                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 新增模块详解

#### 5.2.1 AU Predictor（识别模块新增）
```python
class AUPredictor(nn.Module):
    """
    从融合特征预测17个AU的激活强度。
    """

    def __init__(self, input_dim=1024, num_au=17):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_au),
            nn.Sigmoid()  # AU激活范围 0-1
        )

    def forward(self, fused_features):
        au_activation = self.fc(fused_features)  # (B, 17)
        return au_activation
```

#### 5.2.2 AU Controller（生成模块新增）
```python
class AUController(nn.Module):
    """
    AU激活 → FOMM关键点位移。
    """

    def __init__(self):
        super().__init__()
        # AU到关键点的映射矩阵
        # 17 AU → 10 keypoints × 2 (x, y位移)
        self.au_to_keypoint = nn.Linear(17, 20)

        # 情感类别嵌入
        self.emotion_embed = nn.Embedding(4, 10)  # 4类情感

    def forward(self, au_activation, emotion_class, intensity):
        """
        Args:
            au_activation: (B, 17) AU激活
            emotion_class: (B,) 情感类别索引
            intensity: (B, 1) 强度参数

        Returns:
            keypoint_displacement: (B, 10, 2) 关键点位移
        """
        # AU → 关键点位移
        kp_displacement = self.au_to_keypoint(au_activation)  # (B, 20)
        kp_displacement = kp_displacement.view(-1, 10, 2)     # (B, 10, 2)

        # 情感条件
        emotion_cond = self.emotion_embed(emotion_class)     # (B, 10)
        emotion_cond = emotion_cond.view(-1, 10, 1)          # (B, 10, 1)
        kp_displacement += emotion_cond * 0.1

        # 强度调制
        kp_displacement *= intensity.unsqueeze(-1).unsqueeze(-1)

        return kp_displacement
```

---

## 6. 代码修改计划

### 6.1 文件结构

```
D:\censor\
├── model\
│   ├── attention.py          # [复用] FFA, CASANet
│   ├── fast_pathway.py       # [复用] Fast Encoder
│   ├── slow_pathway.py       # [复用] Slow Encoder
│   ├── au_predictor.py       # [新增] AU预测器
│   ├── au_controller.py      # [新增] AU控制器
│   ├── temporal_modulation.py# [新增] 时间调制
│   └── censor_fomm.py        # [新增] Censor-FOMM联合类
│
├── generation\
│   ├── fomm_loader.py        # [新增] 加载预训练FOMM
│   ├── fomm_adapter.py       # [新增] FOMM适配器
│   ├── train_generation.py   # [新增] 生成训练脚本
│   └── evaluate_generation.py# [新增] 生成评估
│
├── losses\
│   └── generation_loss.py    # [新增] 生成损失函数
│
├── main.py                   # [修改] 添加CensorFOMM类
├── train_cross.py            # [修改] 添加generation模式
└── config\defaults.py        # [修改] 添加FOMM配置
```

### 6.2 新增代码量

| 文件 | 行数 |
|------|------|
| `au_predictor.py` | ~40 |
| `au_controller.py` | ~60 |
| `temporal_modulation.py` | ~50 |
| `censor_fomm.py` | ~80 |
| `fomm_loader.py` | ~30 |
| `fomm_adapter.py` | ~80 |
| `train_generation.py` | ~120 |
| `evaluate_generation.py` | ~60 |
| `generation_loss.py` | ~40 |

**总计新增**：~480行

### 6.3 修改代码量

| 文件 | 修改行数 |
|------|----------|
| `main.py` | ~30 |
| `train_cross.py` | ~50 |
| `config/defaults.py` | ~20 |

**总计修改**：~100行

---

## 7. 实验设计

### 7.1 实验列表

| 实验 | 目的 | 数据 | 时间 |
|------|------|------|------|
| **E1: AU预测验证** | 验证AUPredictor准确率 | CASME2 | 2h |
| **E2: FOMM微调** | 基础微表情生成 | CASME2 | 4h |
| **E3: 可控生成** | 强度/速度控制 | CASME2 | 2h |
| **E4: 识别反馈** | 生成样本的识别准确率 | CASME2 | 2h |
| **E5: VTuber集成** | 实时情感交互演示 | Demo | 1h |

### 7.2 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **AU-Acc** | AU预测准确率 | >70% |
| **FID** | Fréchet Inception Distance | <50 |
| **Keypoint-Error** | 关键点位移误差 | <5px |
| **Recognition-Acc** | 生成样本识别准确率 | >80% |
| **Temporal-Consistency** | 时间曲线一致性 | >0.8 |

### 7.3 实验流程

```python
# E1: AU预测验证
# 使用CASME2标注的AU数据验证AUPredictor

# E2: FOMM微调
# 1. 加载预训练FOMM
# 2. 在CASME2上微调Generator
# 3. 评估生成质量

# E3: 可控生成
# 1. 给定不同强度参数生成
# 2. 评估强度一致性

# E4: 识别反馈
# 1. 生成微表情样本
# 2. 用Censor识别模块识别
# 3. 验证类别一致性

# E5: VTuber集成
# 1. 实时识别用户表情
# 2. 生成VTuber微表情回应
# 3. 录制演示视频
```

---

## 8. 论文框架调整

### 8.1 新标题

**"Micro-Expression Generation via AU-Conditioned Motion Model Fine-tuning"**

或更简洁：

**"Censor-FOMM: AU-Driven Micro-Expression Generation"**

### 8.2 论文结构

```
1. Introduction (~800字)
   - 微表情生成的挑战
   - AU驱动的方法优势
   - FOMM微调的创新性

2. Related Work (~500字)
   - 微表情识别进展
   - 人脸动画生成（FOMM等）
   - AU建模

3. Method (~1000字)
   - 3.1 AU Predictor设计
   - 3.2 AU Controller设计
   - 3.3 FOMM微调策略
   - 3.4 时间调制机制

4. Experiments (~600字)
   - AU预测准确率
   - 生成质量评估
   - 可控生成验证
   - VTuber集成演示

5. Results (~500字)
   - 定量结果表格
   - 可视化对比

6. Discussion (~300字)
   - 与从头训练对比
   - VTuber应用前景

7. Conclusion (~200字)
```

---

## 9. 开发时间表

| Day | 任务 |
|-----|------|
| **Day 1** | AU Predictor + AU Controller |
| **Day 2** | FOMM集成 + 适配器 |
| **Day 3** | 时间调制 + 损失函数 |
| **Day 4** | 训练脚本 + 评估脚本 |
| **Day 5** | 实验运行（AU预测 + FOMM微调） |
| **Day 6** | 实验运行（可控生成 + 识别反馈） |
| **Day 7** | VTuber集成演示 |
| **Day 8-9** | 论文撰写 |

**总计**：~9天

---

## 10. 预训练FOMM资源

### 10.1 下载地址

```
# FOMM官方仓库
https://github.com/AliaksandrSiarohin/first-order-model

# 预训练权重
- VoxCeleb预训练：fomm_voxceleb.pth
- TaiChi预训练：fomm_taichi.pth

推荐使用VoxCeleb预训练（人脸对话场景）
```

### 10.2 集成方式

```python
# fomm_loader.py

def load_pretrained_fomm(checkpoint_path):
    """
    加载预训练FOMM模型。

    Returns:
        motion_extractor: 关键点检测器（冻结）
        generator: 图像生成器（微调）
    """
    checkpoint = torch.load(checkpoint_path)

    motion_extractor = MotionExtractor()
    motion_extractor.load_state_dict(checkpoint['motion_extractor'])
    motion_extractor.eval()  # 冻结

    generator = Generator()
    generator.load_state_dict(checkpoint['generator'])
    generator.train()  # 微调

    return motion_extractor, generator
```

---

## 11. 下一步行动

1. **立即**：下载FOMM预训练权重
2. **Day 1**：实现AU Predictor
3. **Day 2**：集成FOMM
4. **Day 3**：开始微调实验

准备开始？