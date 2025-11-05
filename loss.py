import torch
from torch import nn


class DiceLoss(nn.Module):
    """Dice Loss损失函数"""
    
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target, smooth=1e-5):
        y_pd = output.view(-1)
        y_gt = target.view(-1)
        intersection = torch.sum(y_pd * y_gt)
        score = (2. * intersection + smooth) / (torch.sum(y_pd) + torch.sum(y_gt) + smooth)
        loss = 1 - score
        return loss


class FocalLoss(nn.Module):
    """Focal Loss损失函数"""
    
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class Hybrid_loss(nn.Module):
    """混合损失函数：Dice Loss + Focal Loss"""
    
    def __init__(self, lambdaa=0.5, classes=2):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.lambdaa = lambdaa
        self.classes = classes

    def forward(self, pred, true):
        loss1 = self.dice_loss(pred, true)
        loss2 = self.focal_loss(pred, true)
        total_loss = (loss1 + self.lambdaa * loss2) * self.classes
        return total_loss