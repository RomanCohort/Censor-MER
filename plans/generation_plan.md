# Micro-Expression Generation Project Plan

## 1. 项目定位

### 1.1 方向转型
| 原方向 | 新方向 |
|--------|--------|
| MER识别 | ME生成 + 识别联合系统 |
| "验证研究"（创新不足） | "生成创新"（创新充分） |

### 1.2 论文定位
**标题**："Dual-Pathway Micro-Expression Generation with Neural Dynamics Modeling"
**目标期刊**：Neural Networks (Elsevier)
**核心贡献**：
- 首个双通路微表情生成框架
- 时间动力学建模（onset-apex-offset）
- AU级别的可控生成
- 识别+生成联合系统

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Censor-G: 识别+生成联合系统                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────── 识别模块（条件提取）───────────────┐          │
│  │                                                    │          │
│  │  Input Video                                       │          │
│  │      ↓                                             │          │
│  │  ┌─────────┐    ┌─────────┐                       │          │
│  │  │  Fast   │    │  Slow   │                       │          │
│  │  │(光流)   │    │(RGB+rPPG)│                       │          │
│  │  │ Encoder │    │ Encoder │                       │          │
│  │  └────┬────┘    └────┬────┘                       │          │
│  │       │              │                             │          │
│  │       └─── FFA融合 ───┘                             │          │
│  │              ↓                                      │          │
│  │        ┌─────────┐                                 │          │
│  │        │  MoE    │ → 情感类别                       │          │
│  │        │ Classifier│                               │          │
│  │        └─────────┘                                 │          │
│  │              ↓                                      │          │
│  │        CASANet → Apex时间点                         │          │
│  │                                                    │          │
│  └────────────────────────────────────────────────────┘          │
│                         ↓                                        │
│  ┌─────────────── 生成模块（条件生成）───────────────┐          │
│  │                                                    │          │
│  │  输入：中性脸 + 情感类别 + Apex时间                 │          │
│  │      ↓                                             │          │
│  │  ┌─────────────────────────────────┐              │          │
│  │  │   Temporal Dynamics Generator   │              │          │
│  │  │   (onset-apex-offset曲线)        │              │          │
│  │  └─────────────────────────────────┘              │          │
│  │              ↓                                      │          │
│  │  ┌─────────────────────────────────┐              │          │
│  │  │    AU Controller (17 AUs)       │              │          │
│  │  │    AU激活强度随时间变化           │              │          │
│  │  └─────────────────────────────────┘              │          │
│  │              ↓                                      │          │
│  │  ┌─────────┐    ┌─────────┐                       │          │
│  │  │  Flow   │    │  RGB    │                       │          │
│  │  │ Decoder │───→│ Decoder │                       │          │
│  │  │(逆向Fast)│    │(逆向Slow)│                       │          │
│  │  └─────────┘    └─────────┘                       │          │
│  │              ↓                                      │          │
│  │        Output: 微表情视频序列                       │          │
│  │                                                    │          │
│  └────────────────────────────────────────────────────┘          │
│                         ↓                                        │
│  ┌─────────────── GAN训练 ───────────────────────────┐          │
│  │                                                    │          │
│  │  Generator ←─→ Discriminator                       │          │
│  │  (对抗训练)                                        │          │
│  │                                                    │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块详解

#### 2.2.1 Temporal Dynamics Generator
```python
功能：生成微表情的时间曲线
输入：情感类别, 强度参数
输出：时间曲线 t_curve ∈ ℝ^{T}

设计：
- onset阶段：缓慢上升（0-30%时间）
- apex阶段：峰值保持（30-50%时间）
- offset阶段：缓慢下降（50-100%时间）

数学：
t_curve(t) = onset(t) * I_onset + apex(t) * I_apex + offset(t) * I_offset
其中 I_x 是强度参数
```

#### 2.2.2 AU Controller
```python
功能：控制17个面部AU的激活
输入：情感类别, t_curve
输出：AU_activation ∈ ℝ^{17×T}

映射：
happiness → AU6(Cheek Raiser) + AU12(Lip Corner Puller)
surprise  → AU1(Inner Brow Raiser) + AU2(Outer Brow Raiser) + AU5(Upper Lid Raiser)
disgust   → AU9(Nose Wrinkler) + AU10(Upper Lip Raiser)
repression → AU14(Dimpler) + AU17(Chin Raiser Lower)

输出：每个AU在每帧的激活强度
```

#### 2.2.3 Flow Decoder
```python
功能：从AU激活生成光流序列
输入：AU_activation, t_curve
输出：flow_seq ∈ ℝ^{2×T×H×W}

设计：逆向Fast通路
- AU→光流映射（每个AU对应特定区域的运动）
- 时间调制（t_curve控制运动幅度）
```

#### 2.2.4 RGB Decoder
```python
功能：从光流和中性脸生成RGB序列
输入：neutral_face, flow_seq
输出：rgb_seq ∈ ℝ^{3×T×H×W}

设计：逆向Slow通路
- 光流引导的warping
- 残差生成（超出warping的细节变化）
```

---

## 3. 代码修改计划

### 3.1 新增文件

| 文件 | 内容 | 行数 |
|------|------|------|
| `model/generator.py` | Generator类 | ~200 |
| `model/discriminator.py` | Discriminator类 | ~50 |
| `model/temporal_dynamics.py` | 时间曲线生成 | ~80 |
| `model/au_controller.py` | AU控制 | ~100 |
| `train_generation.py` | GAN训练循环 | ~150 |
| `evaluate_generation.py` | 生成评估 | ~80 |
| `losses/gan_loss.py` | GAN损失 | ~30 |

**总计新增**：~690行

### 3.2 修改文件

| 文件 | 修改内容 | 行数 |
|------|----------|------|
| `main.py` | 添加CensorGenerator类 | ~50 |
| `train_cross.py` | 添加generation模式 | ~100 |
| `config/defaults.py` | 添加生成配置 | ~20 |
| `scripts/` | 生成实验脚本 | ~50 |

**总计修改**：~220行

### 3.3 可复用文件（直接复用）

| 文件 | 复用内容 |
|------|----------|
| `model/attention.py` | FFA, CASANet |
| `preprocessing/` | 光流、rPPG、人脸检测 |
| `model/fast_pathway.py` | Encoder架构 → Decoder参考 |
| `model/slow_pathway.py` | Encoder架构 → Decoder参考 |
| `data/` | 数据加载 |
| `utils/` | 评估工具 |

---

## 4. 实验计划

### 4.1 生成实验

| 实验 | 数据集 | 评估指标 | 时间 |
|------|--------|----------|------|
| 基础生成 | CASME2 | FID, AU-Acc | 3h |
| 可控生成 | CASME2 | 强度一致性 | 2h |
| 跨数据集 | CASME2→SAMM | FID | 2h |
| 识别+生成联合 | CASME2 | 识别准确率 | 4h |

### 4.2 评估指标

| 指标 | 说明 |
|------|------|
| **FID** | Fréchet Inception Distance（生成质量） |
| **AU-Acc** | AU激活一致性（与真实AU对比） |
| **Temporal-Consistency** | 时间曲线一致性 |
| **Recognition-Feedback** | 生成样本的识别准确率 |

### 4.3 可控生成实验

| 控制维度 | 实验 |
|----------|------|
| **强度** | 低/中/高强度生成 → AU幅度变化 |
| **速度** | 快/慢微表情 → 时间曲线变化 |
| **类别** | 4类情感 → 不同AU组合 |

---

## 5. 论文框架

### 5.1 标题与摘要

**标题**：
"Dual-Pathway Micro-Expression Generation with Neural Dynamics Modeling"

**摘要**：
微表情生成是情感计算的新挑战。本文提出Censor-G，首个双通路微表情生成框架。设计包括：(1) 时间动力学生成器建模onset-apex-offset曲线；(2) AU控制器实现17个面部肌肉的精细控制；(3) 双通路解码器从光流和RGB逆向生成视频。实验在CASME II上验证生成质量（FID=XX）和可控性。联合识别+生成系统实现情感交互闭环。

### 5.2 论文结构

```
1. Introduction (~1000字)
   - 微表情生成的重要性
   - 与识别的区别与联系
   - 创新点概述

2. Related Work (~600字)
   - 微表情识别
   - 视频生成方法
   - 面部动作单元(AU)建模

3. Method (~1200字)
   - 3.1 整体架构
   - 3.2 时间动力学生成
   - 3.3 AU控制器
   - 3.4 双通路解码器
   - 3.5 GAN训练策略

4. Experiments (~800字)
   - 4.1 数据集与评估
   - 4.2 基础生成实验
   - 4.3 可控生成实验
   - 4.4 识别+生成联合

5. Results (~700字)
   - 生成质量评估
   - 可控性验证
   - 与识别模块联动

6. Discussion (~400字)
   - 生成vs识别的对比
   - 应用前景（VTuber、情感交互）

7. Conclusion (~200字)
```

---

## 6. 时间规划

### 6.1 开发阶段

| 阶段 | 任务 | 时间 |
|------|------|------|
| **Phase 1** | Generator架构开发 | 2天 |
| **Phase 2** | Temporal Dynamics + AU Controller | 1天 |
| **Phase 3** | GAN训练集成 | 1天 |
| **Phase 4** | 评估指标实现 | 0.5天 |
| **Phase 5** | 实验运行 | 2天 |
| **Phase 6** | 论文撰写 | 3天 |

**总计**：~9.5天

### 6.2 关键里程碑

| 里程碑 | 目标 | 时间 |
|--------|------|------|
| M1 | Generator代码完成 | Day 2 |
| M2 | 首次生成实验成功 | Day 5 |
| M3 | 所有实验完成 | Day 7 |
| M4 | 论文初稿完成 | Day 10 |

---

## 7. 创新点总结

| 创新点 | 技术层面 | 实验验证 |
|--------|----------|----------|
| **时间动力学生成** | onset-apex-offset曲线建模 | 时间一致性评估 |
| **AU级别控制** | 17个AU的精细激活 | AU-Acc评估 |
| **双通路逆向解码** | Encoder→Decoder逆向设计 | FID评估 |
| **识别+生成联合** | 情感闭环系统 | 识别反馈评估 |

---

## 8. 与VTuber系统结合

当前Civis Lucri-Faber VTuber系统可以集成：

| VTuber组件 | 生成模块应用 |
|------------|--------------|
| 情感识别 | 识别用户情感 → 触发生成 |
| VTuber表情 | 生成微表情 → VTuber表达 |
| 情感交互 | 识别→生成→响应闭环 |

论文可以增加：
> "Application: Integration with VTuber emotional interaction system"

---

## 9. 风险与对策

| 风险 | 对策 |
|------|------|
| 数据稀缺（247样本） | 使用预训练 + AU先验 |
| 生成质量不稳定 | GAN稳定技巧（谱归一化、梯度惩罚） |
| 评估困难 | 多指标（FID + AU-Acc + 识别反馈） |
| 训练时间长 | 减少epoch + 早停 |

---

## 10. 下一步行动

1. **立即**：设计Generator架构
2. **Day 1**：实现Temporal Dynamics + AU Controller
3. **Day 2**：实现双通路Decoder
4. **Day 3**：GAN训练集成
5. **Day 5**：开始实验

准备开始代码开发？