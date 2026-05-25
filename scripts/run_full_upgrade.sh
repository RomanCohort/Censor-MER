#!/bin/bash
# 完整升级版训练

cd /root/autodl-tmp/Censor-MER

echo "============================================================"
echo "完整升级版训练"
echo "目标：70%+识别率"
echo "============================================================"

python generation/train_full_upgrade.py \
    --recognizer_checkpoint "./checkpoints/cross_casme2/cross_src_casme2_best.pth" \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --smic_root "/root/SMIC_all_cropped" \
    --samm_root "/root/data/SAMM" \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --pretrain_epochs 20 \
    --finetune_epochs 30 \
    --save_dir "./checkpoints/full_upgrade"