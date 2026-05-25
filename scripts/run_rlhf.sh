#!/bin/bash
# RLHF Training Script (Extended)

cd /root/autodl-tmp/Censor-MER

python generation/train_rlhf.py \
    --generator_checkpoint "./checkpoints/censor_g_gen_v6/censor_g_gen_final.pth" \
    --recognizer_checkpoint "./checkpoints/cross_casme2/cross_src_casme2_best.pth" \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5 \
    --save_dir "./checkpoints/rlhf_gen_v2" \
    --log_dir "./logs/rlhf_gen_v2" \
    --save_every 10