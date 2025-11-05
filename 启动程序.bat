@echo off
chcp 65001 >nul
title 眼底血管图像分割系统
echo.
echo ========================================
echo   眼底血管图像分割系统 v2.0
echo ========================================
echo.
echo 正在启动程序...
echo.

python main.py

if errorlevel 1 (
    echo.
    echo 程序运行出错！
    echo 请确保已安装所有依赖包。
    echo.
    pause
)
