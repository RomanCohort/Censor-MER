# Censor: AutoDL 训练指南

## 一、AutoDL 环境选择

### 1.1 推荐镜像
选择 **PyTorch 2.0 + Python 3.10** 镜像：

| 镜像名称 | PyTorch | CUDA | 推荐 |
|---------|---------|------|------|
| PyTorch 2.0.1 | 2.0.1 | 11.8 | ✓ 推荐 |
| PyTorch 2.1.0 | 2.1.0 | 12.1 | ✓ 可用 |

### 1.2 GPU 选择

| GPU型号 | 显存 | batch_size | 推荐场景 |
|--------|------|-----------|---------|
| RTX 3090 | 24GB | 4-8 | **性价比最高** |
| RTX 4090 | 24GB | 8-16 | 训练速度最快 |
| A100 40G | 40GB | 16-32 | 大批量训练 |
| V100 16G | 16GB | 2-4 | 省钱但慢 |

**建议**: RTX 3090 或 RTX 4090，性价比最高。

---

## 二、环境配置

### 2.1 克隆项目

```bash
# 进入工作目录
cd /root/autodl-tmp

# 克隆项目
git clone https://github.com/RomanCohort/Censor-MER.git
cd Censor-MER
```

### 2.2 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装额外依赖（OpenCV optical flow）
pip install opencv-contrib-python

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2.3 验证模型

```bash
# 测试模型前向传播（合成数据）
python main.py

# 预期输出：
#   Total parameters: 68,353,230
#   ME Logits: torch.Size([2, 7])
#   AU Intensities: torch.Size([2, 16, 28])
```

---

## 三、数据集准备

### 3.1 下载微表情数据集

| 数据集 | 样本数 | 帧率 | 分辨率 | 下载地址 |
|--------|-------|------|--------|---------|
| **CASME II** | 247 | 200fps | 640×480 | [申请链接](http://casme.psych.ac.cn/casme/c2) |
| **SAMM** | 159 | 200fps | 2040×1088 | [申请链接](https://www.mmu.ac.uk) |
| **SMIC-HS** | 164 | 100fps | 640×480 | [申请链接](https://www.oulu.fi) |

> **注意**: 所有数据集需要签署许可协议后才能下载。

### 3.2 数据目录结构

将数据集解压到 `/root/autodl-tmp/data/`：

```
/root/autodl-tmp/data/
├── CASME_II/
│   ├── videos/
│   │   ├── sub01/
│   │   │   ├── EP01_01f.avi
│   │   │   ├── EP01_02f.avi
│   │   │   └── ...
│   │   ├── sub02/
│   │   └── ...
│   ├── label.csv
│   └── AU_label.csv
├── SAMM/
│   ├── videos/
│   ├── SAMM_label.csv
│   └── ...
└── SMIC/
    ├── videos/
    └── ...
```

### 3.3 使用合成数据快速测试

如果暂时没有真实数据集，可以使用合成数据测试训练流程：

```bash
python train.py --synthetic_data --epochs 10 --batch_size 4
```

---

## 四、训练命令

### 4.1 基础训练命令

```bash
# CASME II 数据集训练
python train.py \
    --dataset casme2 \
    --data_root /root/autodl-tmp/data/CASME_II \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --output_dir /root/autodl-tmp/checkpoints

# SAMM 数据集训练
python train.py \
    --dataset samm \
    --data_root /root/autodl-tmp/data/SAMM \
    --epochs 50 \
    --batch_size 2 \
    --lr 1e-4
```

### 4.2 调整显存占用

| 显存 | batch_size | 梯度累积 |
|------|-----------|---------|
| 16GB | 2 | accum_steps=4 |
| 24GB | 4-8 | 无需累积 |
| 40GB | 16 | 无需累积 |

```bash
# 显存不足时使用梯度累积
python train.py \
    --batch_size 2 \
    --accum_steps 4 \
    --epochs 50
```

### 4.3 多 GPU 训练

```bash
# 使用 train_multi_gpu.py（如果有多卡）
python train_multi_gpu.py \
    --data_root /root/autodl-tmp/data/CASME_II \
    --epochs 100 \
    --batch_size 8 \
    --num_workers 4
```

### 4.4 训练参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 2 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--au_loss_weight` | 0.5 | AU损失权重 (α) |
| `--moe_loss_weight` | 0.01 | MoE负载均衡权重 (β) |
| `--landmark_loss_weight` | 0.1 | OPD界标损失权重 (γ) |
| `--output_dir` | ./checkpoints | 模型保存目录 |
| `--val_every` | 1 | 验证频率（每N轮） |

---

## 五、监控与日志

### 5.1 训练日志

训练过程自动记录到：
- `checkpoints/metrics.csv` — 每轮指标
- `checkpoints/model_best.pt` — 最佳模型
- `checkpoints/model_epoch_*.pt` — 周期性保存

### 5.2 TensorBoard 可视化（可选）

```bash
# 安装 tensorboard
pip install tensorboard

# 启动（需要在 AutoDL 自定义服务中开放端口）
tensorboard --logdir checkpoints --port 6006
```

### 5.3 AutoDL 端口映射

在 AutoDL 控制台 → 自定义服务 → 开启端口映射：
- TensorBoard: 6006
- Jupyter: 8888（默认已有）

---

## 六、常见问题

### Q1: 显存不足 (OOM)

```bash
# 方案1: 减小 batch_size
--batch_size 1 --accum_steps 8

# 方案2: 使用混合精度（已默认启用）
# AMP 自动开启，无需额外配置

# 方案3: 减少输入帧数
# 修改 config/defaults.py: INPUT_CONFIG['temporal'] = 8
```

### Q2: OpenCV optical flow 报错

```bash
# 安装完整 OpenCV
pip uninstall opencv-python
pip install opencv-contrib-python
```

### Q3: 数据集格式不匹配

检查 `dataset.py` 中的数据加载逻辑，或使用 `prepare_data.py` 预处理：

```bash
python prepare_data.py --dataset casme2 --data_root /root/autodl-tmp/data/CASME_II
```

### Q4: 训练中断后恢复

```bash
python train.py \
    --resume /root/autodl-tmp/checkpoints/model_epoch_20.pt \
    --epochs 50
```

---

## 七、训练时间预估

| GPU | 数据集 | batch_size | 50 epochs 时间 |
|-----|--------|-----------|---------------|
| RTX 3090 | CASME II (247) | 4 | ~2-3 小时 |
| RTX 4090 | CASME II | 8 | ~1-1.5 小时 |
| V100 16G | CASME II | 2 | ~4-5 小时 |

---

## 八、保存模型

训练完成后，将模型保存到持久化目录：

```bash
# 复制到 AutoDL 数据盘（不会被清除）
cp checkpoints/model_best.pt /root/autodl-tmp/models/censor_best.pt

# 或下载到本地
# AutoDL 文件管理器 → 下载
```

---

## 九、快速开始脚本

创建训练脚本 `run_train.sh`：

```bash
#!/bin/bash

# AutoDL 快速训练脚本

cd /root/autodl-tmp/Censor-MER

# 检查数据
if [ ! -d "/root/autodl-tmp/data/CASME_II" ]; then
    echo "[WARNING] CASME II not found, using synthetic data"
    python train.py --synthetic_data --epochs 10 --batch_size 4
else
    echo "[INFO] Training on CASME II"
    python train.py \
        --dataset casme2 \
        --data_root /root/autodl-tmp/data/CASME_II \
        --epochs 50 \
        --batch_size 4 \
        --lr 1e-4 \
        --output_dir /root/autodl-tmp/checkpoints \
        --val_every 1
fi

echo "[DONE] Training completed!"
echo "Model saved to: /root/autodl-tmp/checkpoints/model_best.pt"
```

运行：
```bash
chmod +x run_train.sh
./run_train.sh
```

---

## 十、联系与支持

- GitHub: https://github.com/RomanCohort/Censor-MER
- 问题反馈: GitHub Issues