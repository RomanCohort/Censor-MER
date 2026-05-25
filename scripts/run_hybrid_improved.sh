#!/bin/bash
# 调整版Hybrid训练：使用87%识别器作为判别器

cd /root/autodl-tmp/Censor-MER

python generation/train_hybrid_improved.py \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --diffusion_checkpoint "./checkpoints/hybrid_model_v2/hybrid_final.pth" \
    --recognizer_checkpoint "./checkpoints/cross_casme2/cross_src_casme2_best.pth" \
    --epochs 30 \
    --batch_size 8 \
    --save_dir "./checkpoints/hybrid_improved"