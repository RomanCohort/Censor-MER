#!/bin/bash
# 运行所有方案

cd /root/autodl-tmp/Censor-MER

echo "============================================================"
echo "运行所有方案"
echo "============================================================"

# 方案1
echo ""
echo "[1/4] 方案1: rPPG估计..."
bash scripts/run_method1.sh

# 方案2
echo ""
echo "[2/4] 方案2: 3通道识别器..."
bash scripts/run_method2.sh

# 方案3
echo ""
echo "[3/4] 方案3: 优化GAN/RLHF..."
bash scripts/run_method3.sh

# 完整版
echo ""
echo "[4/4] 完整升级版..."
bash scripts/run_full_upgrade.sh

echo ""
echo "============================================================"
echo "所有方案完成！"
echo "============================================================"