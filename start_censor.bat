@echo off
chcp 65001 >nul
title Censor - 微表情识别系统

echo ========================================
echo   Censor - 仿生双通道微表情识别系统
echo ========================================
echo.

cd /d D:\censor

echo [1/2] 启动 Streamlit 前端...
start "Censor-Streamlit" cmd /k "cd /d D:\censor && streamlit run frontend/app.py --server.port 8501"

echo [2/2] 打开浏览器...
timeout /t 3 /nobreak >nul
start http://localhost:8501

echo.
echo ========================================
echo   系统已启动！
echo   请访问: http://localhost:8501
echo ========================================
echo.
echo 按任意键退出...
pause >nul