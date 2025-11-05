# 数据集说明

## 目录结构

```
data/
├── training/           # 训练数据
│   ├── images/        # 训练图像（.tif格式）
│   └── 1st_manual/    # 训练标签（.gif格式）
└── test/              # 测试数据
    ├── images/        # 测试图像（.tif格式）
    ├── 1st_manual/    # 第一组测试标签（.gif格式）
    ├── 2nd_manual/    # 第二组测试标签（.gif格式）
    ├── mask/          # 掩码文件（.gif格式）
    └── outputs/       # 预测输出（程序自动生成）
```

## 数据集获取

本项目使用 **DRIVE 数据集**（Digital Retinal Images for Vessel Extraction）

### 下载地址
- 官方网站: https://drive.grand-challenge.org/
- 或其他公开来源

### 数据集说明
- **训练集**: 20张眼底图像及对应的血管标注
- **测试集**: 20张眼底图像及两组专家标注
- **图像尺寸**: 565 x 584 像素
- **图像格式**: TIF（原图）, GIF（标注）

### 使用方法
1. 下载 DRIVE 数据集
2. 解压后将文件放置到对应目录：
   - 训练图像 → `training/images/`
   - 训练标注 → `training/1st_manual/`
   - 测试图像 → `test/images/`
   - 测试标注 → `test/1st_manual/` 和 `test/2nd_manual/`
   - 测试掩码 → `test/mask/`

## 注意事项

- 请确保图像文件名与标注文件名对应
- 本项目仅供学习研究使用
- 请遵守数据集的使用协议和版权规定
