#!/bin/bash
# 方案3：优化GAN和RLHF（当前最佳方案）

cd /root/autodl-tmp/Censor-MER

echo "============================================================"
echo "方案3：优化GAN和RLHF"
echo "当前最佳：GAN 30%, RLHF 0.5"
echo "============================================================"

# GAN优化（更多epochs）
echo ""
echo "[1] GAN优化..."
python generation/train_gan_improved.py \
    --generator_checkpoint "./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth" \
    --recognizer_checkpoint "./checkpoints/cross_casme2/cross_src_casme2_best.pth" \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --epochs 60 \
    --batch_size 8 \
    --lr 5e-5 \
    --save_dir "./checkpoints/gan_v2"

# RLHF优化（更多epochs）
echo ""
echo "[2] RLHF优化..."
python generation/train_rlhf_improved.py \
    --generator_checkpoint "./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth" \
    --recognizer_checkpoint "./checkpoints/cross_casme2/cross_src_casme2_best.pth" \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --epochs 60 \
    --batch_size 8 \
    --lr 5e-5 \
    --save_dir "./checkpoints/rlhf_v2"

echo ""
echo "============================================================"
echo "方案3完成！"
echo "============================================================"