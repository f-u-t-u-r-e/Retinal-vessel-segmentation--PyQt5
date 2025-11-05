import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class conv_block(nn.Module):
    """卷积块：Conv->BN->ReLU->Conv->BN->ReLU"""
    
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    """下采样模块：MaxPool + ConvBlock"""
    
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        return self.max_pool_conv(x)


class up(nn.Module):
    """上采样模块：ConvTranspose + Skip Connection + ConvBlock"""
    
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class outconv(nn.Module):
    """输出卷积块：使用可变形卷积"""
    
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.offset1 = nn.Conv2d(in_ch, 18, kernel_size=3, stride=1, padding=1)
        self.DCN1 = DeformConv2d(in_ch, in_ch, kernel_size=3, padding=1)
        
        self.offset2 = nn.Conv2d(in_ch, 18, kernel_size=3, stride=1, padding=1)
        self.DCN2 = DeformConv2d(in_ch, in_ch, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        offset_1 = self.offset1(x)
        x = self.DCN1(x, offset_1)
        x = self.bn1(x)
        x = self.relu(x)
        
        offset_2 = self.offset2(x)
        x = self.DCN2(x, offset_2)
        x = self.bn2(x)
        x = self.relu(x)
        
        return self.conv(x)


class PyramidPooling(nn.Module):
    """金字塔池化模块"""
    
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') 
                           for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class UNet(nn.Module):
    """U-Net网络结构"""
    
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = conv_block(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.pyramidpooling = PyramidPooling(64, 64, scales=(4, 8, 16, 32), ct_channels=64 // 4)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.pyramidpooling(x)
        x = self.outc(x)
        
        return F.sigmoid(x)
