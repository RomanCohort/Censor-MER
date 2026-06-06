#!/bin/bash
# 启动Web反馈收集界面 (Streamlit版本)

cd /root/autodl-tmp/Censor-MER

# 安装依赖（如果没有）
pip install streamlit sqlalchemy pandas openpyxl plotly -q

# 启动界面
streamlit run interface/feedback_streamlit.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true