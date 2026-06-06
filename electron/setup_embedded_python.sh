#!/bin/bash
# Linux/Mac 版本：准备嵌入 Python 环境

set -e

echo "=========================================="
echo "Censor - 准备嵌入 Python 环境"
echo "=========================================="

cd "$(dirname "$0")"

# 创建目录
mkdir -p embedded/packages

echo ""
echo "[1/3] 检查 Python..."

if ! command -v python3 &> /dev/null; then
    echo "Python3 未安装，请先安装 Python"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "找到 $PYTHON_VERSION"

echo ""
echo "[2/3] 创建虚拟环境..."

python3 -m venv embedded/venv

echo ""
echo "[3/3] 安装依赖..."

source embedded/venv/bin/activate

pip install --upgrade pip
pip install streamlit sqlalchemy pandas openpyxl plotly numpy pillow

# 创建启动脚本
cat > embedded/run_streamlit.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run ../interface/feedback_choice.py --server.port 7860 --server.headless true --server.address localhost
EOF

chmod +x embedded/run_streamlit.sh

echo ""
echo "=========================================="
echo "嵌入 Python 环境准备完成！"
echo "=========================================="
echo ""
echo "测试: ./embedded/run_streamlit.sh"