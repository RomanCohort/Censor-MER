#!/bin/bash
# 测试混合模型生成效果

cd /root/autodl-tmp/Censor-MER

python generation/test_hybrid.py \
    --checkpoint "./checkpoints/hybrid_model/hybrid_final.pth" \
    --casme2_root "/root/autodl-tmp/data/CASME2" \
    --output_dir "./results/hybrid_test" \
    --num_samples 10