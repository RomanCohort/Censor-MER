#!/bin/bash
# 方案2：3通道识别器

cd /root/autodl-tmp/Censor-MER

python generation/train_method2_rgb.py \
    --generator_checkpoint "./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth" \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --epochs 30 \
    --batch_size 8 \
    --save_dir "./checkpoints/method2"