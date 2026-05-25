#!/bin/bash
# 方案1：从生成视频估计rPPG信号（无零填充）

cd /root/autodl-tmp/Censor-MER

python generation/train_method1_rppg.py \
    --generator_checkpoint "./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth" \
    --recognizer_checkpoint "./checkpoints/cross_casme2/cross_src_casme2_best.pth" \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --epochs 30 \
    --batch_size 8 \
    --save_dir "./checkpoints/method1"