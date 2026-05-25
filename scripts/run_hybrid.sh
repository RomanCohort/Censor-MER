#!/bin/bash
# Hybrid Model Training: Diffusion + Blendshape + GAN (Extended)

cd /root/autodl-tmp/Censor-MER

python generation/train_hybrid.py \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --stage1_epochs 20 \
    --stage2_epochs 20 \
    --stage3_epochs 20 \
    --batch_size 8 \
    --diffusion_lr 1e-4 \
    --refiner_lr 1e-4 \
    --discriminator_lr 1e-5 \
    --num_frames 16 \
    --image_size 64 \
    --save_dir "./checkpoints/hybrid_model_v2" \
    --log_dir "./logs/hybrid_model_v2"