@echo off
REM 一键打包脚本 - 包含嵌入 Python 环境
REM 执行此脚本将生成完全独立的桌面应用

setlocal enabledelayedexpansion

echo ==========================================
echo Censor 微表情反馈系统 - 一键打包
echo ==========================================

cd /d "%~dp0"

echo.
echo 步骤概览:
echo   1. 检查 Electron 依赖
echo   2. 准备嵌入 Python 环境
echo   3. 打包 Electron 应用
echo.

REM 步骤 1: 检查 Electron
echo [步骤 1] 检查 Electron...

if not exist "node_modules" (
    echo 安装 Electron 依赖...
    call npm install
    if errorlevel 1 (
        echo npm install 失败！请确保 Node.js 已安装
        pause
        exit /b 1
    )
)

echo Electron 依赖已就绪.

REM 步骤 2: 准备嵌入 Python
echo.
echo [步骤 2] 准备嵌入 Python 环境...

if exist "embedded\python\python.exe" (
    echo 嵌入 Python 已存在，跳过准备
) else (
    echo 正在准备...
    call setup_embedded_python.bat
    if errorlevel 1 (
        echo Python 环境准备失败！
        pause
        exit /b 1
    )
)

echo Python 环境已就绪.

REM 步骤 3: 打包
echo.
echo [步骤 3] 打包 Electron 应用...

call npm run build
if errorlevel 1 (
    echo 打包失败！
    pause
    exit /b 1
)

echo.
echo ==========================================
echo 打包完成！
echo ==========================================
echo.
echo 输出目录: dist\
echo.
echo 文件列表:
dir dist\*.exe /b 2>nul
echo.
echo 安装包大小约:
for %%f in (dist\*.exe) do (
    echo %%f: %%~zf 字节
)
echo.
echo 用户使用:
echo   - 双击 Setup.exe 安装
echo   - 或直接运行便携版 exe
echo.
echo 注意: 此版本已嵌入 Python，无需用户安装 Python

pause