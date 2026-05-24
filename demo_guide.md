# Censor 演示指南

## 快速开始

### 方式一：直接运行
```bash
cd D:/censor
python demo.py
```

### 方式二：Streamlit界面
```bash
cd D:/censor
streamlit run monitor/app.py
```

---

## 演示流程（约5分钟）

### 1. 系统概览（1分钟）
- 运行 `python demo.py` 展示参数量
- 说明双通道架构

### 2. 架构演示（2分钟）
- 打开 `docs/README_EN.md` 或 `docs/TECHNICAL_EN.md`
- 展示Mermaid流程图
- 解释各模块

### 3. 代码演示（1分钟）
```python
# 展示关键模块
python -c "
import torch
from model.biomimetic_enhance import Censor

model = Censor()
x = torch.randn(1, 3, 16, 224, 224)
out = model(x)
print('ME prediction:', out['me_logits'].argmax())
print('AU intensities:', out['au_intensities'].shape)
print('Apex frame:', out['apex_scores'])
"
```

### 4. 回答问题（1分钟）
- 准备回答的几个关键点
- 可打开文档对应位置

---

## 导师可能问的问题（快速参考）

| 问题 | 快速参考位置 |
|------|-----------|
| 双通道原理 | docs/TECHNICAL_EN.md 3.2节 |
| Bio-Gating | docs/TECHNICAL_EN.md 3.3节 |
| rPPG局限 | docs/TECHNICAL_EN.md 3.1.2节 |
| 消融实验 | docs/TECHNICAL_EN.md 4.1节 |
| 损失权重 | docs/TECHNICAL_EN.md 4.1节 |

---

## 文件位置

- 主代码: `D:/censor/`
- 英文文档: `D:/censor/docs/TECHNICAL_EN.md`
- 中文文档: `D:/censor/docs/TECHNICAL_CN.md`
- 演示脚本: `D:/censor/demo.py`
- 监控界面: `D:/censor/monitor/app.py`

---

## 注意事项

1. **GPU**: 如果没有GPU，会很慢（建议提前测试）
2. **依赖**: 确保torch已安装
3. **数据**: demo使用随机数据，不需要真实数据集