#!/bin/bash
# 监控并行实验进度

cd /root/autodl-tmp/Censor-MER

echo "============================================================"
echo "Parallel Experiments Monitor"
echo "============================================================"

# 检查进程
echo ""
echo "[Running Processes]"
ps aux | grep -E "train_hybrid|train_gan|train_rlhf" | grep -v grep

# 显示日志尾部
echo ""
echo "============================================================"
echo "[Hybrid Training - Latest Log]"
echo "============================================================"
tail -20 logs/parallel/hybrid.log 2>/dev/null || echo "  Log not found"

echo ""
echo "============================================================"
echo "[GAN Training - Latest Log]"
echo "============================================================"
tail -20 logs/parallel/gan.log 2>/dev/null || echo "  Log not found"

echo ""
echo "============================================================"
echo "[RLHF Training - Latest Log]"
echo "============================================================"
tail -20 logs/parallel/rlhf.log 2>/dev/null || echo "  Log not found"

echo ""
echo "============================================================"
echo "[GPU Usage]"
echo "============================================================"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

echo ""
echo "============================================================"