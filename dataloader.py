import numpy as np
from PIL import Image
import torch
import os
from torch.utils.data import Dataset


class DRIVE_Loader(Dataset):
    """DRIVE眼底血管图像数据集加载器"""
    
    def __init__(self, img_dir, mask_dir, img_size=(512, 512), mode='train'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.mode = mode
        self.file_list = os.listdir(img_dir)
        self.split_dataset(0.8)

    def split_dataset(self, ratio):
        """按比例分割训练集和验证集"""
        train_len = int(ratio * len(self.file_list))
        if self.mode == 'train':
            self.file_list = self.file_list[:train_len]
        else:
            self.file_list = self.file_list[train_len:]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        """获取单个数据样本"""
        img_file = os.path.join(self.img_dir, self.file_list[item])
        mask_file = os.path.join(self.mask_dir, self.file_list[item].replace("tif", "gif"))
        
        img = np.array(Image.open(img_file).resize(self.img_size, Image.BILINEAR))
        mask = np.array(Image.open(mask_file).resize(self.img_size, Image.BILINEAR))
        
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        
        # HWC to CHW
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        mask = mask / 255.0
        
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        
        return torch.from_numpy(img), torch.from_numpy(mask)