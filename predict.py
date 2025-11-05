import os
import torch
import numpy as np
from PIL import Image
from UNet import UNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def predict_single(filepath, weight_path="./checkpoint.pth", img_size=(512, 512)):
    """
    对单张图片进行预测
    
    Args:
        filepath: 输入图像路径
        weight_path: 模型权重路径
        img_size: 图像尺寸
        
    Returns:
        PIL.Image: 预测结果图像
    """
    network = UNet().to(device)
    network.load_state_dict(torch.load(weight_path, map_location=device))
    network.eval()
    
    img = Image.open(filepath)
    img = img.resize(img_size, Image.BILINEAR)
    img = np.array(img)
    
    img = np.expand_dims(img, axis=0)
    img = img.transpose((0, 3, 1, 2))
    img = img.astype(np.float32)
    
    with torch.no_grad():
        pred = network(torch.from_numpy(img).to(device)).cpu().numpy()
    
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    
    return Image.fromarray(pred[0, 0, :, :])


def predict_batch(test_dir, weight_path, outputs_dir, img_size=(512, 512)):
    """
    批量预测图像
    
    Args:
        test_dir: 测试图像目录
        weight_path: 模型权重路径
        outputs_dir: 输出目录
        img_size: 图像尺寸
    """
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    network = UNet().to(device)
    network.load_state_dict(torch.load(weight_path, map_location=device))
    network.eval()
    
    file_list = os.listdir(test_dir)
    
    for f in file_list:
        filepath = os.path.join(test_dir, f)
        img = np.array(Image.open(filepath).resize(img_size, Image.BILINEAR))
        
        img = np.expand_dims(img, axis=0)
        img = img.transpose((0, 3, 1, 2))
        img = img.astype(np.float32)
        
        with torch.no_grad():
            pred = network(torch.from_numpy(img).to(device)).cpu().numpy()
        
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        
        Image.fromarray(pred[0, 0, :, :]).save(os.path.join(outputs_dir, f))
        print(f"已处理: {f}")


if __name__ == "__main__":
    predict_batch("./data/test/images", "./checkpoint.pth", "./data/test/outputs")
