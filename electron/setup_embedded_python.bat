@echo off
REM 准备嵌入 Python 环境
REM 下载 Python embeddable 包并安装必要依赖

setlocal enabledelayedexpansion

echo ==========================================
echo Censor - 准备嵌入 Python 环境
echo ==========================================

cd /d "%~dp0"

REM 设置版本
set PYTHON_VERSION=3.11.9
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip

REM 创建目录
if not exist "embedded" mkdir embedded
if not exist "embedded\python" mkdir embedded\python
if not exist "embedded\packages" mkdir embedded\packages

echo.
echo [1/5] 下载 Python %PYTHON_VERSION% embeddable...

REM 检查是否已下载
if exist "embedded\python\python.exe" (
    echo Python 已存在，跳过下载
) else (
    echo 正在下载...

    REM 使用 PowerShell 下载
    powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile 'embedded\python_embed.zip'"

    if exist "embedded\python_embed.zip" (
        echo 解压...
        powershell -Command "Expand-Archive -Path 'embedded\python_embed.zip' -DestinationPath 'embedded\python' -Force"
        del embedded\python_embed.zip
        echo 完成
    ) else (
        echo 下载失败！请手动下载: %PYTHON_URL%
        pause
        exit /b 1
    )
)

echo.
echo [2/5] 配置 Python 路径...

REM 启用 site 模块（允许安装包）
if exist "embedded\python\python311._pth" (
    echo 修改 _pth 文件...
    (
        echo python311.zip
        echo .
        echo Lib
        echo Lib\site-packages
        echo import site
    ) > embedded\python\python311._pth
)

echo.
echo [3/5] 安装 pip...

REM 检查 pip 是否存在
if exist "embedded\python\Scripts\pip.exe" (
    echo pip 已存在，跳过安装
) else (
    echo 正在安装 pip...

    REM 下载 get-pip.py
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'embedded\get-pip.py'"

    if exist "embedded\get-pip.py" (
        embedded\python\python.exe embedded\get-pip.py --no-warn-script-location
        del embedded\get-pip.py

        REM 创建 Scripts 目录
        if not exist "embedded\python\Scripts" mkdir embedded\python\Scripts

        echo 完成
    ) else (
        echo pip 安装失败！
        pause
        exit /b 1
    )
)

echo.
echo [4/5] 安装 Streamlit 及依赖...

set PIP_CMD=embedded\python\python.exe -m pip install --target=embedded\packages --no-warn-script-location

echo 正在安装 streamlit...
%PIP_CMD% streamlit

echo 正在安装 sqlalchemy...
%PIP_CMD% sqlalchemy

echo 正在安装 pandas...
%PIP_CMD% pandas

echo 正在安装 openpyxl...
%PIP_CMD% openpyxl

echo 正在安装 plotly...
%PIP_CMD% plotly

echo 正在安装 numpy...
%PIP_CMD% numpy

echo 正在安装 pillow...
%PIP_CMD% pillow

echo.
echo [5/5] 创建启动脚本...

(
    echo @echo off
    echo cd /d "%%~dp0"
    echo set PYTHONPATH=%%~dp0packages
    echo python.exe -m streamlit run %%~dp0..\interface\feedback_choice.py --server.port 7860 --server.headless true --server.address localhost
) > embedded\run_streamlit.bat

echo.
echo ==========================================
echo 嵌入 Python 环境准备完成！
echo ==========================================
echo.
echo 目录结构:
echo   embedded\python\     - Python 运行时
echo   embedded\packages\   - Python 包
echo   embedded\run_streamlit.bat - 启动脚本
echo.
echo 测试: embedded\run_streamlit.bat
echo.

pause