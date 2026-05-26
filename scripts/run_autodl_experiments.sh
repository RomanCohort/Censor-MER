#!/bin/bash
# =============================================================================
# Censor: AutoDL Three Experiments Training Script
# =============================================================================
# 在AutoDL上运行三个微表情生成实验：
#   Experiment 1: SNN Generator Training
#   Experiment 2: RLHF Optimization Training
#   Experiment 3: Full Hybrid Training (SNN + GAN + RLHF)
#
# Usage:
#   cd /root/autodl-tmp/Censor-MER
#   chmod +x scripts/run_autodl_experiments.sh
#   ./scripts/run_autodl_experiments.sh
#
# 或者运行单个实验：
#   ./scripts/run_autodl_experiments.sh exp1    # 仅SNN
#   ./scripts/run_autodl_experiments.sh exp2    # 仅RLHF
#   ./scripts/run_autodl_experiments.sh exp3    # 仅Full
# =============================================================================

set -e

# 配置
PROJECT_DIR="/root/autodl-tmp/Censor-MER"
DATA_ROOT="/root/autodl-tmp/data"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints"
LOG_DIR="/root/autodl-tmp/logs"

# GPU显存自适应batch_size
detect_batch_size() {
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEM" -ge 40000 ]; then
        echo 16
    elif [ "$GPU_MEM" -ge 24000 ]; then
        echo 8
    elif [ "$GPU_MEM" -ge 16000 ]; then
        echo 4
    else
        echo 2
    fi
}

BATCH_SIZE=$(detect_batch_size)
echo "[Config] Batch size: $BATCH_SIZE (GPU memory adaptive)"

# 创建目录
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

cd $PROJECT_DIR

# =============================================================================
# Experiment 1: SNN Generator Training
# =============================================================================
run_exp1() {
    echo ""
    echo "============================================================"
    echo "Experiment 1: SNN Generator Training"
    echo "============================================================"
    echo "[Goal] Train Spiking Neural Network temporal dynamics"
    echo "[Expected] temporal_consistency > 0.75 (was 0.68)"
    echo ""

    python generation/train_snn.py \
        --epochs 30 \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --num_frames 16 \
        --image_size 224 \
        --save_dir $CHECKPOINT_DIR/censor_g_snn \
        --log_dir $LOG_DIR/snn \
        2>&1 | tee $LOG_DIR/exp1_snn.log

    echo "[Exp1] Complete! Check results in $LOG_DIR/exp1_snn.log"
}

# =============================================================================
# Experiment 2: RLHF Optimization Training
# =============================================================================
run_exp2() {
    echo ""
    echo "============================================================"
    echo "Experiment 2: RLHF Optimization Training"
    echo "============================================================"
    echo "[Goal] Optimize generator with recognition-based reward"
    echo "[Expected] reward > 0.3 (was 0.2)"
    echo ""

    # 检查识别器checkpoint是否存在
    RECOGNIZER_CKPT="$CHECKPOINT_DIR/cross_casme2/cross_src_casme2_best.pth"
    if [ ! -f "$RECOGNIZER_CKPT" ]; then
        echo "[Warning] Recognizer checkpoint not found, will use simplified version"
        RECOGNIZER_ARG=""
    else
        echo "[Info] Found recognizer checkpoint: $RECOGNIZER_CKPT"
        RECOGNIZER_ARG="--recognizer_checkpoint $RECOGNIZER_CKPT"
    fi

    # 检查生成器checkpoint
    GENERATOR_CKPT="$CHECKPOINT_DIR/censor_g_gen_v6/censor_g_gen_final.pth"
    if [ ! -f "$GENERATOR_CKPT" ]; then
        echo "[Warning] Generator checkpoint not found, training from scratch"
        GENERATOR_ARG=""
    else
        echo "[Info] Found generator checkpoint: $GENERATOR_CKPT"
        GENERATOR_ARG="--generator_checkpoint $GENERATOR_CKPT"
    fi

    python generation/train_rlhf.py \
        --epochs 30 \
        --batch_size $BATCH_SIZE \
        --lr 1e-5 \
        --clip_ratio 0.2 \
        --entropy_coef 0.01 \
        $GENERATOR_ARG \
        $RECOGNIZER_ARG \
        --save_dir $CHECKPOINT_DIR/rlhf_gen \
        --log_dir $LOG_DIR/rlhf \
        2>&1 | tee $LOG_DIR/exp2_rlhf.log

    echo "[Exp2] Complete! Check results in $LOG_DIR/exp2_rlhf.log"
}

# =============================================================================
# Experiment 3: Full Hybrid Training (SNN + GAN + RLHF)
# =============================================================================
run_exp3() {
    echo ""
    echo "============================================================"
    echo "Experiment 3: Full Hybrid Training (SNN + GAN + RLHF)"
    echo "============================================================"
    echo "[Goal] End-to-end training with all components"
    echo "[Expected] FID < 0.20, temporal > 0.75, reward > 0.3"
    echo ""

    # Step 1: Train SNN generator
    echo "[Exp3.1] Training SNN generator..."
    python generation/train_snn.py \
        --epochs 20 \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --save_dir $CHECKPOINT_DIR/hybrid_snn \
        --log_dir $LOG_DIR/hybrid \
        2>&1 | tee -a $LOG_DIR/exp3_hybrid.log

    # Step 2: Train with GAN
    echo "[Exp3.2] Training with GAN loss..."
    python generation/train_gan.py \
        --epochs 15 \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --generator_checkpoint $CHECKPOINT_DIR/hybrid_snn/censor_g_snn_final.pth \
        --use_gan \
        --save_dir $CHECKPOINT_DIR/hybrid_gan \
        --log_dir $LOG_DIR/hybrid \
        2>&1 | tee -a $LOG_DIR/exp3_hybrid.log

    # Step 3: RLHF fine-tuning
    echo "[Exp3.3] RLHF fine-tuning..."
    python generation/train_rlhf.py \
        --epochs 15 \
        --batch_size $BATCH_SIZE \
        --lr 1e-5 \
        --generator_checkpoint $CHECKPOINT_DIR/hybrid_gan/generator_final.pth \
        --save_dir $CHECKPOINT_DIR/hybrid_final \
        --log_dir $LOG_DIR/hybrid \
        2>&1 | tee -a $LOG_DIR/exp3_hybrid.log

    echo "[Exp3] Complete! Check results in $LOG_DIR/exp3_hybrid.log"
}

# =============================================================================
# Evaluation
# =============================================================================
run_evaluation() {
    echo ""
    echo "============================================================"
    echo "Running Evaluation"
    echo "============================================================"

    python -c "
import torch
import json
import os

results = {}

# Eval SNN
print('[Eval] SNN Generator...')
from model.censor_g_snn import CensorGSNN
snn = CensorGSNN()
snn_ckpt = '$CHECKPOINT_DIR/censor_g_snn/censor_g_snn_final.pth'
if os.path.exists(snn_ckpt):
    snn.load_state_dict(torch.load(snn_ckpt, map_location='cpu'))
au_temporal = snn.v3_temporal(torch.ones(1, 17))
diff1 = au_temporal[:,:,1:] - au_temporal[:,:,:-1]
diff2 = diff1[:,:,1:] - diff1[:,:,:-1]
results['snn_smoothness'] = diff2.abs().mean().item()
print(f'  SNN smoothness: {results[\"snn_smoothness\"]:.4f}')

# Eval RLHF reward
print('[Eval] RLHF Generator...')
rlhf_ckpt = '$CHECKPOINT_DIR/rlhf_gen/rlhf_final.pth'
if os.path.exists(rlhf_ckpt):
    log_file = '$LOG_DIR/rlhf/rlhf_training_log.json'
    if os.path.exists(log_file):
        with open(log_file) as f:
            log = json.load(f)
            if log['epochs']:
                results['rlhf_final_reward'] = log['epochs'][-1]['reward']
                print(f'  RLHF reward: {results[\"rlhf_final_reward\"]:.4f}')

# Eval Hybrid
print('[Eval] Hybrid Generator...')
hybrid_ckpt = '$CHECKPOINT_DIR/hybrid_final/rlhf_final.pth'
if os.path.exists(hybrid_ckpt):
    results['hybrid_exists'] = True
    print('  Hybrid model trained')

# Save results
with open('$LOG_DIR/experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print()
print('[Summary]')
print('='*50)
for k, v in results.items():
    print(f'  {k}: {v}')
print('='*50)
"
}

# =============================================================================
# Main
# =============================================================================
echo ""
echo "============================================================"
echo "Censor: AutoDL Three Experiments Training"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: $LOG_DIR"
echo ""

# 根据参数决定运行哪些实验
case "${1:-all}" in
    exp1)
        run_exp1
        ;;
    exp2)
        run_exp2
        ;;
    exp3)
        run_exp3
        ;;
    eval)
        run_evaluation
        ;;
    all|"")
        run_exp1
        run_exp2
        run_exp3
        run_evaluation
        ;;
    *)
        echo "Usage: $0 [exp1|exp2|exp3|eval|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo "Results saved to: $LOG_DIR/experiment_results.json"
echo ""
echo "To download checkpoints from AutoDL:"
echo "  scp -r user@autodl-instance:$CHECKPOINT_DIR ./local_checkpoints"
echo ""
