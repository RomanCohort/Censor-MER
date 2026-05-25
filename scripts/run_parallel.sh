#!/bin/bash
# 并行实验：同时训练Hybrid + GAN + RLHF

cd /root/autodl-tmp/Censor-MER

echo "============================================================"
echo "Starting Parallel Experiments"
echo "============================================================"

# 创建日志目录
mkdir -p logs/parallel

# 1. Hybrid训练（后台）
echo "[1] Starting Hybrid training..."
nohup bash scripts/run_hybrid.sh > logs/parallel/hybrid.log 2>&1 &
HYBRID_PID=$!
echo "  Hybrid PID: $HYBRID_PID"

# 2. GAN训练（后台）
echo "[2] Starting GAN training..."
nohup bash scripts/run_gan.sh > logs/parallel/gan.log 2>&1 &
GAN_PID=$!
echo "  GAN PID: $GAN_PID"

# 3. RLHF训练（后台）
echo "[3] Starting RLHF training..."
nohup bash scripts/run_rlhf.sh > logs/parallel/rlhf.log 2>&1 &
RLHF_PID=$!
echo "  RLHF PID: $RLHF_PID"

echo ""
echo "============================================================"
echo "All experiments started in background!"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  Hybrid: tail -f logs/parallel/hybrid.log"
echo "  GAN:    tail -f logs/parallel/gan.log"
echo "  RLHF:   tail -f logs/parallel/rlhf.log"
echo ""
echo "Check all processes:"
echo "  ps aux | grep train"
echo ""
echo "Kill all:"
echo "  kill $HYBRID_PID $GAN_PID $RLHF_PID"
echo "============================================================"