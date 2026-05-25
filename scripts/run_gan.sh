#!/bin/bash
# GAN Training: Generator vs Recognition Discriminator

cd /root/autodl-tmp/Censor-MER

python generation/train_gan.py \
    --generator_checkpoint "./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth" \
    --recognizer_checkpoint "./checkpoints/cross_casme2/cross_src_casme2_best.pth" \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --epochs 50 \
    --batch_size 8 \
    --g_lr 1e-4 \
    --freeze_discriminator true \
    --save_dir "./checkpoints/gan_generator" \
    --log_dir "./logs/gan_generator" \
    --save_every 10