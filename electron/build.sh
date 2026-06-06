#!/bin/bash
# Electron 应用打包脚本
# 用于打包微表情反馈收集系统为桌面应用

echo "=========================================="
echo "Censor Electron 打包脚本"
echo "=========================================="

# 进入 electron 目录
cd "$(dirname "$0")"

# 检查 node_modules
if [ ! -d "node_modules" ]; then
    echo "安装 Electron 依赖..."
    npm install
fi

# 检查 Python/Streamlit 环境
echo "检查 Python 环境..."
python -c "import streamlit" 2>/dev/null || {
    echo "请先安装 Streamlit: pip install streamlit"
    exit 1
}

# 打包
echo "开始打包..."
npm run build

echo "=========================================="
echo "打包完成！"
echo "输出目录: dist/"
echo "=========================================="