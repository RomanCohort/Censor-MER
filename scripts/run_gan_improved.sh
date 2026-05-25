#!/bin/bash
# 改进版GAN训练

cd /root/autodl-tmp/Censor-MER

python generation/train_gan_improved.py \
    --generator_checkpoint "./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth" \
    --recognizer_checkpoint "./checkpoints/cross_casme2/cross_src_casme2_best.pth" \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --epochs 30 \
    --batch_size 4 \
    --lr 1e-4 \
    --save_dir "./checkpoints/gan_improved"