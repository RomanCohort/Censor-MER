#!/bin/bash
# Hybrid + rPPG估计（最有潜力突破70%的方案）

cd /root/autodl-tmp/Censor-MER

python generation/train_hybrid_rppg.py \
    --diffusion_checkpoint "./checkpoints/hybrid_model_v2/hybrid_final.pth" \
    --recognizer_checkpoint "./checkpoints/cross_casme2/cross_src_casme2_best.pth" \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --epochs 40 \
    --batch_size 6 \
    --lr 1e-4 \
    --save_dir "./checkpoints/hybrid_rppg"