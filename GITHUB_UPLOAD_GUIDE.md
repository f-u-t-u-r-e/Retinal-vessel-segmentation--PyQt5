# GitHub 上传指南

## 📋 上传前准备

### 1. 清理个人数据
双击运行 **`准备GitHub上传.bat`**，自动清理所有个人隐私数据。

### 2. 检查 .gitignore
已创建 `.gitignore` 文件，自动排除以下内容：
- ✓ 用户数据 (`users.json`)
- ✓ 模型文件 (`checkpoint.pth`)
- ✓ 输出文件 (`out_file/`, `outputs/`, `roc/`)
- ✓ 数据集图像 (`data/` 目录下的图像文件)
- ✓ Python 缓存 (`__pycache__/`)

## 🚀 上传步骤

### 方式1：使用 GitHub Desktop（推荐新手）

1. 下载安装 [GitHub Desktop](https://desktop.github.com/)
2. 登录你的 GitHub 账号
3. 点击 `File` → `Add Local Repository`
4. 选择项目文件夹：`C:\Users\exexex6661\Desktop\eyeax`
5. 如果提示"not a git repository"，点击 `create a repository`
6. 填写仓库信息：
   - Name: `eyeax` 或 `retinal-vessel-segmentation`
   - Description: `基于U-Net的眼底血管图像分割系统`
   - 勾选 `Initialize this repository with a README`（如果没有README.md）
7. 点击 `Publish repository`
8. 选择是否公开（Public/Private）
9. 点击 `Publish Repository`

### 方式2：使用 Git 命令行

```bash
# 1. 初始化 Git 仓库
cd C:\Users\exexex6661\Desktop\eyeax
git init

# 2. 添加所有文件
git add .

# 3. 提交
git commit -m "Initial commit: 眼底血管图像分割系统"

# 4. 在 GitHub 网站创建新仓库
# 访问 https://github.com/new
# 创建名为 eyeax 的仓库

# 5. 关联远程仓库（替换 YOUR_USERNAME）
git remote add origin https://github.com/YOUR_USERNAME/eyeax.git

# 6. 推送到 GitHub
git branch -M main
git push -u origin main
```

## 📦 模型文件处理

由于 `checkpoint.pth` 文件过大（>100MB），GitHub 不允许直接上传。

### 选项1：使用 Git LFS（推荐）

```bash
# 安装 Git LFS
git lfs install

# 追踪大文件
git lfs track "*.pth"

# 添加 .gitattributes
git add .gitattributes

# 正常提交
git add checkpoint.pth
git commit -m "Add model checkpoint"
git push
```

### 选项2：网盘分享
1. 上传 `checkpoint.pth` 到网盘（百度网盘、阿里云盘等）
2. 在 README.md 中添加下载链接
3. 用户下载后放到项目根目录

### 选项3：GitHub Release
1. 将代码推送到 GitHub
2. 在仓库页面点击 `Releases` → `Create a new release`
3. 上传 `checkpoint.pth` 作为附件
4. 在 README.md 中说明下载方式

## 📝 建议的仓库描述

**中文：**
```
基于U-Net深度学习网络的眼底血管图像分割系统，使用PyTorch实现，集成PyQt5图形界面。支持用户管理、图像分割、模型训练和性能评估。在DRIVE数据集上准确率>95%。
```

**English:**
```
Retinal vessel segmentation system based on U-Net deep learning network, implemented with PyTorch and PyQt5 GUI. Features user management, image segmentation, model training, and performance evaluation. Achieves >95% accuracy on DRIVE dataset.
```

## 🏷️ 推荐的标签（Tags）

- `deep-learning`
- `pytorch`
- `image-segmentation`
- `u-net`
- `medical-imaging`
- `retinal-vessel`
- `pyqt5`
- `computer-vision`

## 📄 README 更新建议

在 README.md 中添加以下内容：

### 徽章（Badges）
```markdown
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.7%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)
```

### 模型下载说明
```markdown
## 模型下载

由于模型文件较大，请从以下链接下载：
- [百度网盘](链接) 提取码: xxxx
- [GitHub Release](https://github.com/YOUR_USERNAME/eyeax/releases)

下载后将 `checkpoint.pth` 放到项目根目录。
```

### 数据集说明
```markdown
## 数据集

本项目使用 DRIVE 数据集，请自行下载：
- 官方网站: https://drive.grand-challenge.org/

详细说明请查看 [data/README.md](data/README.md)
```

## ⚠️ 注意事项

1. **不要上传个人数据**：检查 `users.json` 是否已重置
2. **检查敏感信息**：确保代码中没有硬编码的密码、API密钥等
3. **版权说明**：确保有权分享所有代码和资源
4. **License**：建议添加 MIT 或 Apache 2.0 许可证
5. **数据集版权**：DRIVE 数据集仅供研究使用，注意版权声明

## 📮 上传后操作

1. 添加仓库描述和标签
2. 启用 Issues 和 Discussions（可选）
3. 添加 Topics 方便搜索
4. 在 README 中添加演示截图
5. 编写详细的使用文档

## 🔗 有用的链接

- [GitHub 新手指南](https://docs.github.com/cn/get-started)
- [Git LFS 文档](https://git-lfs.github.com/)
- [.gitignore 生成器](https://www.toptal.com/developers/gitignore)
- [README 模板](https://github.com/othneildrew/Best-README-Template)

---

如有问题，请参考 GitHub 官方文档或在项目中创建 Issue。
