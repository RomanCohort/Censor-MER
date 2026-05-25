#!/bin/bash
# Hybrid Model Training: Diffusion + Blendshape + GAN

cd /root/autodl-tmp/Censor-MER

python generation/train_hybrid.py \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --stage1_epochs 10 \
    --stage2_epochs 10 \
    --stage3_epochs 10 \
    --batch_size 4 \
    --diffusion_lr 1e-4 \
    --refiner_lr 1e-4 \
    --discriminator_lr 1e-5 \
    --num_frames 16 \
    --image_size 64 \
    --save_dir "./checkpoints/hybrid_model" \
    --log_dir "./logs/hybrid_model"