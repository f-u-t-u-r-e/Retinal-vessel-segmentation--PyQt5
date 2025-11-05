# 基于U-Net的眼底血管图像分割系统

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.7%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 项目简介

本项目是一个基于深度学习U-Net网络的眼底血管图像分割系统，集成了PyQt5图形界面，提供了完整的用户管理、图像分割、模型训练和评估功能。

## 主要功能

- **用户管理系统**：支持用户注册、登录、密码修改和权限管理
- **图像分割**：基于U-Net网络对眼底血管图像进行自动分割
- **模型训练**：支持自定义数据集训练模型
- **性能评估**：提供多种评估指标（精准率、敏感度、F1-Score、Dice系数、IoU等）
- **ROC曲线**：生成ROC曲线评估模型性能

## 技术栈

- **深度学习框架**：PyTorch
- **图像处理**：OpenCV, PIL
- **GUI框架**：PyQt5
- **数据处理**：NumPy, scikit-learn
- **可视化**：Matplotlib

## 模型下载

⚠️ **重要提示**：由于模型文件较大（>100MB），未包含在仓库中。

请从以下方式获取训练好的模型：

1. **自行训练**：运行 `python train.py` 训练新模型
2. **GitHub Release**：从 [Releases](../../releases) 下载预训练模型
3. **联系作者**：获取预训练模型文件

下载后将 `checkpoint.pth` 放到项目根目录。

## 数据集

本项目使用 **DRIVE 数据集**（Digital Retinal Images for Vessel Extraction）。

由于版权原因，数据集未包含在仓库中，请自行下载：
- 官方网站: https://drive.grand-challenge.org/

详细说明请查看 [data/README.md](data/README.md)

## 项目结构

```
eyeax/
├── main.py              # 主程序（GUI界面）
├── train.py             # 模型训练脚本
├── predict.py           # 图像预测脚本
├── evaluate.py          # 模型评估脚本
├── UNet.py              # U-Net网络模型定义
├── dataloader.py        # 数据加载器
├── loss.py              # 损失函数定义
├── 登录页面.py          # 登录界面UI
├── 注册.py              # 注册界面UI
├── 用户管理.py          # 用户管理界面UI
├── untitled.py          # 主界面UI
├── users.json           # 用户数据文件
├── checkpoint.pth       # 训练好的模型权重
├── data/                # 数据目录
│   ├── training/        # 训练数据
│   │   ├── images/      # 训练图像
│   │   └── 1st_manual/  # 训练标签
│   └── test/            # 测试数据
│       ├── images/      # 测试图像
│       ├── 1st_manual/  # 第一组测试标签
│       ├── 2nd_manual/  # 第二组测试标签
│       └── outputs/     # 预测输出
├── images/              # GUI界面图像资源
├── out_file/            # 分割结果输出目录
└── roc/                 # ROC曲线保存目录
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

### 1. 训练模型

```bash
python train.py
```

训练参数可在 `train.py` 中修改：
- `epoch`: 训练轮数（默认500）
- `img_size`: 图像尺寸（默认512x512）
- `batch_size`: 批次大小（默认1）

### 2. 测试预测

单张图像预测：
```python
from predict import predict_single
result = predict_single("path/to/image.tif")
result.save("output.png")
```

批量预测：
```bash
python predict.py
```

### 3. 模型评估

```bash
python evaluate.py
```

评估结果包括：
- 精准率（Sp）
- 敏感度（Se）
- 准确率（Acc）
- F1 Score
- F2 Score
- Dice系数
- IoU（前景和背景）
- ROC曲线和AUC值

### 4. 启动GUI系统

**方式1：使用批处理文件（推荐）**
```bash
# 双击运行 启动程序.bat
```

**方式2：使用Python命令**
```bash
python main.py
```

默认管理员账号：
- 用户名：admin123
- 密码：admin123

## 网络结构

本项目使用改进的U-Net网络，包含以下特点：

1. **编码器-解码器结构**：5层下采样和4层上采样
2. **跳跃连接**：保留多尺度特征信息
3. **金字塔池化模块**：增强多尺度特征提取能力
4. **可变形卷积**：在输出层使用DCN提高分割精度
5. **LeakyReLU激活函数**：避免死神经元问题

## 损失函数

使用混合损失函数：
- **Dice Loss**：优化分割区域重叠度
- **Focal Loss**：处理类别不平衡问题
- **混合权重**：lambda=0.5（可调整）

## 数据集

本项目使用DRIVE数据集（Digital Retinal Images for Vessel Extraction）：
- 训练集：16张（按8:2分为训练和验证）
- 测试集：20张
- 图像尺寸：原始565x584，训练时resize为512x512
- 标注类型：两组专家标注（1st_manual和2nd_manual）

## 性能指标

模型在DRIVE测试集上的表现：
- 准确率 > 95%
- Dice系数 > 0.80
- AUC > 0.97

（具体数值请运行evaluate.py获取）

## 注意事项

1. 程序会自动检测GPU，如果没有GPU则使用CPU运行
2. 确保数据集路径正确
3. 首次运行会自动创建users.json文件
4. 图像格式支持：.tif, .png, .jpg, .jpeg

## 开发环境

- Python 3.7+
- PyTorch 1.7+
- CUDA 10.2+（可选，用于GPU加速）

## 作者

眼底血管图像分割系统开发团队

## 注意事项

- DRIVE 数据集仅供研究使用，请遵守其使用协议
- 本软件仅供学习和研究目的，不用于商业用途
- 医疗影像分析结果仅供参考，不能作为临床诊断依据

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 [Issue](../../issues)
- 发起 [Discussion](../../discussions)

## 更新日志

### v2.0 (2025-11)
- 重构所有代码，删除无用内容
- 优化代码结构和注释
- 统一命名规范
- 改进错误处理
- 添加完整文档

### v1.0
- 初始版本
- 实现基本功能
