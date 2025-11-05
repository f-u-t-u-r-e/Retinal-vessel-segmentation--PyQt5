@echo off
chcp 65001 >nul
title 准备GitHub上传 - 清理个人数据

echo ╔═══════════════════════════════════════════════════════════╗
echo ║              准备GitHub上传 - 数据清理工具                ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.

echo [提示] 此脚本将清理所有个人隐私数据
echo [提示] 原始文件将备份到 .backup 文件
echo.
pause

:: 备份并重置用户数据
echo [1/5] 处理用户数据...
if exist users.json (
    copy users.json users.json.backup >nul
    echo [{"name":"admin123","username":"admin123","password":"admin123","values":"管理员"}] > users.json
    echo       ✓ users.json 已重置为默认管理员账号
) else (
    echo       ! users.json 不存在，跳过
)

:: 清理输出文件
echo.
echo [2/5] 清理输出文件...
if exist out_file (
    for %%f in (out_file\*) do del /q "%%f" 2>nul
    echo       ✓ out_file 目录已清空
)
if exist outputs (
    for %%f in (outputs\*) do del /q "%%f" 2>nul
    echo       ✓ outputs 目录已清空
)

:: 清理ROC曲线
echo.
echo [3/5] 清理ROC曲线...
if exist roc (
    for %%f in (roc\*.png) do del /q "%%f" 2>nul
    echo       ✓ roc 目录中的PNG文件已清除
)

:: 清理Python缓存
echo.
echo [4/5] 清理Python缓存...
if exist __pycache__ (
    rmdir /s /q __pycache__ 2>nul
    echo       ✓ __pycache__ 已删除
)

:: 清理测试输出
echo.
echo [5/5] 清理测试输出...
if exist "data\test\outputs" (
    for %%f in (data\test\outputs\*) do del /q "%%f" 2>nul
    echo       ✓ 测试输出已清空
)

echo.
echo ════════════════════════════════════════════════════════════
echo                         清理完成！
echo ════════════════════════════════════════════════════════════
echo.
echo [说明] 以下文件已准备好上传到GitHub：
echo        - 源代码文件 (*.py)
echo        - 配置文件 (.gitignore, requirements.txt)
echo        - 文档文件 (*.md)
echo        - 批处理脚本 (*.bat)
echo        - 示例用户数据 (users.json.example)
echo.
echo [注意] 以下文件已被 .gitignore 排除：
echo        - users.json.backup (个人数据备份)
echo        - checkpoint.pth (模型文件太大)
echo        - data/ 目录下的图像文件
echo        - 所有输出文件
echo.
echo [建议] 模型文件可以使用 Git LFS 或上传到网盘分享
echo.
pause
