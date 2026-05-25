#!/bin/bash
# 启动Web反馈收集界面

cd /root/autodl-tmp/Censor-MER

# 安装Gradio（如果没有）
pip install gradio -q

# 启动界面
python interface/feedback_interface.py \
    --checkpoint "./checkpoints/rlhf_gen_v2/rlhf_final.pth" \
    --api_key "" \
    --share \
    --port 7860