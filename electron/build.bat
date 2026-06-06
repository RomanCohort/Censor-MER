@echo off
REM Electron 应用打包脚本 (Windows)
REM 用于打包微表情反馈收集系统为桌面应用

echo ==========================================
echo Censor Electron 打包脚本
echo ==========================================

cd /d "%~dp0"

REM 检查 node_modules
if not exist "node_modules" (
    echo 安装 Electron 依赖...
    call npm install
)

REM 检查 Python/Streamlit 环境
echo 检查 Python 环境...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo 请先安装 Streamlit: pip install streamlit sqlalchemy pandas openpyxl
    pause
    exit /b 1
)

REM 打包
echo 开始打包...
call npm run build

echo ==========================================
echo 打包完成！
echo 输出目录: dist\
echo ==========================================

pause