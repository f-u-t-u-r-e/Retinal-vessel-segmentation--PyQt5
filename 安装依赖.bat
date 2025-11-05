@echo off
chcp 65001 >nul
title 安装依赖包
echo.
echo ========================================
echo   眼底血管图像分割系统
echo   依赖包安装程序
echo ========================================
echo.
echo 正在安装依赖包，请稍候...
echo.

pip install torch torchvision opencv-python Pillow numpy scikit-learn matplotlib PyQt5 scipy

if errorlevel 1 (
    echo.
    echo 安装失败！请检查网络连接。
    pause
    exit /b 1
) else (
    echo.
    echo ========================================
    echo   所有依赖包安装完成！
    echo ========================================
    echo.
    pause
)
