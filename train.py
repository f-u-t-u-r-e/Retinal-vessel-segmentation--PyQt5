import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataloader import DRIVE_Loader
from UNet import UNet
from loss import Hybrid_loss, DiceLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def train():
    """训练U-Net模型"""
    # 训练参数
    epoch = 500
    img_dir = "./data/training/images"
    mask_dir = "./data/training/1st_manual"
    img_size = (512, 512)
    
    # 创建数据加载器
    tr_loader = DataLoader(
        DRIVE_Loader(img_dir, mask_dir, img_size, 'train'),
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        DRIVE_Loader(img_dir, mask_dir, img_size, 'val'),
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    
    # 初始化模型和优化器
    criterion = Hybrid_loss()
    network = UNet().to(device)
    optimizer = Adam(network.parameters(), weight_decay=0.0001)
    best_score = 1.0
    
    # 训练循环
    for i in range(epoch):
        network.train()
        train_step = 0
        train_loss = 0
        val_loss = 0
        val_step = 0
        
        # 训练阶段
        for batch in tr_loader:
            imgs, mask = batch
            imgs = imgs.to(device)
            mask = mask.to(device)
            
            mask_pred = network(imgs)
            loss = criterion(mask_pred, mask)
            
            train_loss += loss.item()
            train_step += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        network.eval()
        with torch.no_grad():
            for batch in val_loader:
                imgs, mask = batch
                imgs = imgs.to(device)
                mask = mask.to(device)
                val_loss += DiceLoss()(network(imgs), mask).item()
                val_step += 1
        
        # 计算平均损失
        train_loss /= train_step
        val_loss /= val_step
        
        # 保存最佳模型
        if val_loss < best_score:
            best_score = val_loss
            torch.save(network.state_dict(), "./checkpoint.pth")
        
        print(f"Epoch {i}: train_loss={train_loss:.4f}, val_dice={val_loss:.4f}")


if __name__ == "__main__":
    train()



