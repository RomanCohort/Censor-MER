#!/bin/bash
# 全面改进版：所有模型使用87%识别器作为判别器

cd /root/autodl-tmp/Censor-MER

echo "============================================================"
echo "全面改进版训练"
echo "所有模型使用87%识别器作为判别器"
echo "============================================================"

# 清理GPU
pkill -f train_gan
pkill -f train_rlhf
pkill -f train_hybrid

sleep 5

# 创建日志目录
mkdir -p logs/improved

# 1. 改进版Hybrid训练
echo ""
echo "[1] Training Improved Hybrid..."
nohup bash scripts/run_hybrid_improved.sh > logs/improved/hybrid.log 2>&1 &
HYBRID_PID=$!
echo "  PID: $HYBRID_PID"

# 等待一会再启动下一个
sleep 10

# 2. 改进版GAN训练
echo ""
echo "[2] Training Improved GAN..."
nohup python generation/train_gan_improved.py \
    --generator_checkpoint "./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth" \
    --recognizer_checkpoint "./checkpoints/cross_casme2/cross_src_casme2_best.pth" \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --epochs 30 \
    --batch_size 4 \
    --save_dir "./checkpoints/gan_improved" \
    > logs/improved/gan.log 2>&1 &
GAN_PID=$!
echo "  PID: $GAN_PID"

sleep 10

# 3. 改进版RLHF训练
echo ""
echo "[3] Training Improved RLHF..."
nohup python generation/train_rlhf_improved.py \
    --generator_checkpoint "./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth" \
    --recognizer_checkpoint "./checkpoints/cross_casme2/cross_src_casme2_best.pth" \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --epochs 30 \
    --batch_size 4 \
    --save_dir "./checkpoints/rlhf_improved" \
    > logs/improved/rlhf.log 2>&1 &
RLHF_PID=$!
echo "  PID: $RLHF_PID"

echo ""
echo "============================================================"
echo "全部改进版训练已启动！"
echo "============================================================"
echo ""
echo "监控命令："
echo "  tail -f logs/improved/hybrid.log"
echo "  tail -f logs/improved/gan.log"
echo "  tail -f logs/improved/rlhf.log"
echo ""
echo "进程PID："
echo "  Hybrid: $HYBRID_PID"
echo "  GAN: $GAN_PID"
echo "  RLHF: $RLHF_PID"
echo "============================================================"