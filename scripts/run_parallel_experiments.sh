#!/bin/bash
# =============================================================================
# Censor: AutoDL Parallel Three Experiments Training
# =============================================================================
# 三个实验并行运行（需要多GPU或足够显存）
#
# Usage:
#   ./scripts/run_parallel_experiments.sh
# =============================================================================

set -e

PROJECT_DIR="/root/autodl-tmp/Censor-MER"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints"
LOG_DIR="/root/autodl-tmp/logs"

mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

cd $PROJECT_DIR

# GPU显存自适应
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -ge 40000 ]; then
    BATCH_SIZE=8
elif [ "$GPU_MEM" -ge 24000 ]; then
    BATCH_SIZE=4
else
    BATCH_SIZE=2
fi

echo "============================================================"
echo "Censor: Parallel Three Experiments"
echo "============================================================"
echo "GPU Memory: ${GPU_MEM}MB, Batch Size: $BATCH_SIZE"
echo ""

# 并行启动三个实验
echo "[Starting] Launching 3 experiments in parallel..."

# Exp1: SNN (GPU 0)
CUDA_VISIBLE_DEVICES=0 python generation/train_snn.py \
    --epochs 30 \
    --batch_size $BATCH_SIZE \
    --lr 1e-4 \
    --save_dir $CHECKPOINT_DIR/censor_g_snn \
    --log_dir $LOG_DIR/snn \
    2>&1 | tee $LOG_DIR/exp1_snn.log &
PID1=$!
echo "[Exp1] SNN training started (PID: $PID1, GPU 0)"

# Exp2: RLHF (GPU 1 或共享GPU 0)
if [ "$(nvidia-smi -L | wc -l)" -ge 2 ]; then
    GPU2=1
else
    GPU2=0
fi

CUDA_VISIBLE_DEVICES=$GPU2 python generation/train_rlhf.py \
    --epochs 30 \
    --batch_size $BATCH_SIZE \
    --lr 1e-5 \
    --save_dir $CHECKPOINT_DIR/rlhf_gen \
    --log_dir $LOG_DIR/rlhf \
    2>&1 | tee $LOG_DIR/exp2_rlhf.log &
PID2=$!
echo "[Exp2] RLHF training started (PID: $PID2, GPU $GPU2)"

# Exp3: Hybrid (GPU 2 或共享)
if [ "$(nvidia-smi -L | wc -l)" -ge 3 ]; then
    GPU3=2
else
    GPU3=0
fi

CUDA_VISIBLE_DEVICES=$GPU3 python generation/train_hybrid.py \
    --epochs 20 \
    --batch_size $BATCH_SIZE \
    --lr 1e-4 \
    --save_dir $CHECKPOINT_DIR/hybrid \
    --log_dir $LOG_DIR/hybrid \
    2>&1 | tee $LOG_DIR/exp3_hybrid.log &
PID3=$!
echo "[Exp3] Hybrid training started (PID: $PID3, GPU $GPU3)"

echo ""
echo "============================================================"
echo "All experiments running in parallel"
echo "============================================================"
echo "PIDs: Exp1=$PID1, Exp2=$PID2, Exp3=$PID3"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/exp1_snn.log"
echo "  tail -f $LOG_DIR/exp2_rlhf.log"
echo "  tail -f $LOG_DIR/exp3_hybrid.log"
echo ""
echo "Waiting for all experiments to complete..."

# 等待所有进程
wait $PID1
echo "[Exp1] SNN completed!"

wait $PID2
echo "[Exp2] RLHF completed!"

wait $PID3
echo "[Exp3] Hybrid completed!"

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "============================================================"
echo "Results in: $LOG_DIR/"
