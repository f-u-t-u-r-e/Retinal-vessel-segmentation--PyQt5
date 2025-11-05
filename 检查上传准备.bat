@echo off
chcp 65001 >nul
title GitHub上传检查

echo ╔═══════════════════════════════════════════════════════════╗
echo ║              GitHub 上传前安全检查                         ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.

echo [1] 检查敏感文件...
echo.

:: 检查users.json
if exist users.json (
    findstr /C:"SABC123456" users.json >nul
    if errorlevel 1 (
        echo   ✓ users.json 已清理个人账号
    ) else (
        echo   ✗ users.json 包含个人账号，请运行"准备GitHub上传.bat"
    )
) else (
    echo   ! users.json 不存在
)

:: 检查.gitignore
if exist .gitignore (
    echo   ✓ .gitignore 已创建
) else (
    echo   ✗ .gitignore 缺失！
)

:: 检查checkpoint.pth
if exist checkpoint.pth (
    echo   ! checkpoint.pth 存在（约400MB，会被.gitignore排除）
) else (
    echo   - checkpoint.pth 不存在（可选）
)

echo.
echo [2] 检查文档完整性...
echo.

if exist README.md (echo   ✓ README.md) else (echo   ✗ README.md 缺失)
if exist requirements.txt (echo   ✓ requirements.txt) else (echo   ✗ requirements.txt 缺失)
if exist GITHUB_UPLOAD_GUIDE.md (echo   ✓ GITHUB_UPLOAD_GUIDE.md) else (echo   - GITHUB_UPLOAD_GUIDE.md)
if exist data\README.md (echo   ✓ data\README.md) else (echo   ! data\README.md 缺失)

echo.
echo [3] 检查必要文件...
echo.

if exist main.py (echo   ✓ main.py) else (echo   ✗ main.py 缺失！)
if exist UNet.py (echo   ✓ UNet.py) else (echo   ✗ UNet.py 缺失！)
if exist train.py (echo   ✓ train.py) else (echo   ✗ train.py 缺失！)
if exist predict.py (echo   ✓ predict.py) else (echo   ✗ predict.py 缺失！)

echo.
echo ════════════════════════════════════════════════════════════
echo                       检查完成！
echo ════════════════════════════════════════════════════════════
echo.
echo [提示] 上传前请确认：
echo   1. users.json 已清理个人数据
echo   2. .gitignore 正确配置
echo   3. 所有核心文件存在
echo.
echo [下一步] 
echo   - 如有问题，运行"准备GitHub上传.bat"进行清理
echo   - 查看 UPLOAD_CHECKLIST.md 了解详细步骤
echo   - 查看 GITHUB_UPLOAD_GUIDE.md 了解上传方法
echo.
pause
