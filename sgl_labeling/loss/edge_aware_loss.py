import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2


class EdgeAwareLoss(nn.Module):
    """
    边缘感知损失函数
    对血管边缘区域施加更高的权重
    """
    def __init__(self, edge_weight=5.0):
        super(EdgeAwareLoss, self).__init__()
        self.edge_weight = edge_weight
        self.bce = nn.BCELoss(reduction='none')
        
    def extract_edge(self, mask):
        """
        提取边缘区域
        Args:
            mask: [B, C, H, W] 的ground truth
        Returns:
            edge_map: [B, C, H, W] 的边缘权重图
        """
        B, C, H, W = mask.shape
        edge_maps = []
        
        for b in range(B):
            # 转换为numpy进行边缘检测
            mask_np = mask[b, 0].cpu().numpy()
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            
            # 使用Canny边缘检测
            edges = cv2.Canny((mask_binary * 255).astype(np.uint8), 50, 150)
            
            # 膨胀边缘区域（5x5核）
            kernel = np.ones((5, 5), np.uint8)
            edge_zone = cv2.dilate(edges, kernel, iterations=1)
            
            # 创建权重图：边缘区域权重高，其他区域权重为1
            weight_map = np.ones_like(mask_np, dtype=np.float32)
            weight_map[edge_zone > 0] = self.edge_weight
            
            edge_maps.append(weight_map)
        
        # 转换回tensor
        edge_maps = np.stack(edge_maps, axis=0)
        edge_maps = torch.from_numpy(edge_maps).unsqueeze(1).to(mask.device)
        
        return edge_maps
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] 预测概率图
            target: [B, C, H, W] ground truth
        """
        # 提取边缘权重图
        edge_weights = self.extract_edge(target)
        
        # 计算加权BCE损失
        bce_loss = self.bce(pred, target)
        weighted_loss = bce_loss * edge_weights
        
        return weighted_loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss - 用于处理困难样本（如细血管）
    FL(pt) = -α(1-pt)^γ * log(pt)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] 预测概率
            target: [B, C, H, W] ground truth (0 or 1)
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # pt = p if target = 1, else 1-p
        pt = torch.exp(-bce_loss)
        
        # Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ThinVesselLoss(nn.Module):
    """
    细血管加权损失
    根据距离变换识别细血管区域并加权
    """
    def __init__(self, thin_threshold=3.0, thin_weight=3.0):
        super(ThinVesselLoss, self).__init__()
        self.thin_threshold = thin_threshold
        self.thin_weight = thin_weight
        self.bce = nn.BCELoss(reduction='none')
        
    def get_vessel_width_map(self, mask):
        """
        计算血管宽度图
        """
        B, C, H, W = mask.shape
        width_maps = []
        
        for b in range(B):
            mask_np = mask[b, 0].cpu().numpy()
            binary = (mask_np > 0.5).astype(np.uint8)
            
            # 距离变换 - 计算到最近背景的距离
            if binary.sum() > 0:
                dist = distance_transform_edt(binary)
                width = dist * 2  # 直径 = 半径 * 2
            else:
                width = np.zeros_like(mask_np)
            
            width_maps.append(width)
        
        width_maps = np.stack(width_maps, axis=0)
        return torch.from_numpy(width_maps).unsqueeze(1).to(mask.device)
    
    def forward(self, pred, target):
        """
        对细血管区域加权
        """
        # 获取血管宽度图
        width_map = self.get_vessel_width_map(target)
        
        # 创建细血管权重图
        weight_map = torch.ones_like(width_map)
        thin_mask = (width_map > 0) & (width_map < self.thin_threshold)
        weight_map[thin_mask] = self.thin_weight
        
        # 加权BCE
        bce_loss = self.bce(pred, target)
        weighted_loss = bce_loss * weight_map
        
        return weighted_loss.mean()


class MultiScaleLoss(nn.Module):
    """
    多尺度组合损失函数
    Total Loss = α·BCE + β·Dice + γ·Edge + δ·Focal + ε·ThinVessel
    """
    def __init__(self, 
                 bce_weight=1.0,
                 dice_weight=1.0, 
                 edge_weight=2.0,
                 focal_weight=1.0,
                 thin_weight=1.5,
                 edge_enhance=5.0,
                 thin_enhance=3.0):
        super(MultiScaleLoss, self).__init__()
        
        # 损失权重
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        self.focal_weight = focal_weight
        self.thin_weight = thin_weight
        
        # 各个损失函数
        self.bce_loss = nn.BCELoss()
        self.edge_loss = EdgeAwareLoss(edge_weight=edge_enhance)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.thin_loss = ThinVesselLoss(thin_threshold=3.0, thin_weight=thin_enhance)
        
    def dice_loss(self, pred, target, smooth=1e-5):
        """Dice Loss"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: [B, C, H, W] 预测
            target: [B, C, H, W] ground truth
            mask: [B, C, H, W] 可选的mask
        """
        # 应用mask
        if mask is not None:
            pred = pred * mask
            target = target * mask
        
        # 1. BCE Loss
        loss_bce = self.bce_loss(pred, target)
        
        # 2. Dice Loss
        loss_dice = self.dice_loss(pred, target)
        
        # 3. Edge-Aware Loss
        loss_edge = self.edge_loss(pred, target)
        
        # 4. Focal Loss (困难样本)
        loss_focal = self.focal_loss(pred, target)
        
        # 5. Thin Vessel Loss (细血管)
        loss_thin = self.thin_loss(pred, target)
        
        # 组合损失
        total_loss = (self.bce_weight * loss_bce + 
                     self.dice_weight * loss_dice + 
                     self.edge_weight * loss_edge + 
                     self.focal_weight * loss_focal + 
                     self.thin_weight * loss_thin)
        
        # 返回总损失和各个损失的详细信息（用于监控）
        loss_dict = {
            'total': total_loss,
            'bce': loss_bce,
            'dice': loss_dice,
            'edge': loss_edge,
            'focal': loss_focal,
            'thin': loss_thin
        }
        
        return total_loss, loss_dict


class AdaptiveMultiScaleLoss(nn.Module):
    """
    自适应多尺度损失 - 训练过程中动态调整权重
    """
    def __init__(self, total_epochs=100):
        super(AdaptiveMultiScaleLoss, self).__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        self.base_loss = MultiScaleLoss()
        
    def set_epoch(self, epoch):
        """设置当前epoch，用于动态调整权重"""
        self.current_epoch = epoch
        
        # 训练前期：注重整体分割
        # 训练后期：注重细节（边缘、细血管）
        progress = epoch / self.total_epochs
        
        self.base_loss.bce_weight = 1.0
        self.base_loss.dice_weight = 1.0
        self.base_loss.edge_weight = 1.0 + progress * 2.0  # 1.0 -> 3.0
        self.base_loss.focal_weight = 0.5 + progress * 1.5  # 0.5 -> 2.0
        self.base_loss.thin_weight = 1.0 + progress * 2.0   # 1.0 -> 3.0
        
    def forward(self, pred, target, mask=None):
        return self.base_loss(pred, target, mask)