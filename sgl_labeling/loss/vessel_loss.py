"""
血管分割专用损失函数
保存为: loss/vessel_loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VesselSegmentationLoss(nn.Module):
    """专门为血管分割优化的综合损失函数"""
    
    def __init__(self, 
                 dice_weight=0.3,
                 tversky_weight=0.3,
                 connectivity_weight=0.2,
                 bce_weight=0.2,
                 alpha=0.7,  # Tversky参数 - 控制假阴性惩罚
                 beta=0.3,   # Tversky参数 - 控制假阳性惩罚
                 thin_vessel_boost=2.0):  # 细血管增强系数
        super(VesselSegmentationLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.connectivity_weight = connectivity_weight
        self.bce_weight = bce_weight
        self.alpha = alpha
        self.beta = beta
        self.thin_vessel_boost = thin_vessel_boost
        
    def dice_loss(self, pred, target, smooth=1e-5):
        """Dice loss for handling imbalanced data"""
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice
    
    def tversky_loss(self, pred, target, smooth=1e-5):
        """Tversky loss - 可以调整对假阳性和假阴性的惩罚权重"""
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        true_pos = (pred_flat * target_flat).sum()
        false_neg = ((1 - pred_flat) * target_flat).sum()
        false_pos = (pred_flat * (1 - target_flat)).sum()
        
        tversky = (true_pos + smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + smooth)
        return 1 - tversky
    
    def connectivity_loss(self, pred, target):
        """连续性损失 - 鼓励连续的血管结构"""
        # 使用形态学操作评估连续性
        kernel = torch.ones(1, 1, 3, 3, device=pred.device)
        
        # 保证输入是4D张量
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        # 计算预测和目标的连通性
        pred_dilated = F.conv2d(pred, kernel, padding=1)
        target_dilated = F.conv2d(target, kernel, padding=1)
        
        # 连续性度量
        pred_conn = torch.clamp(pred_dilated / 9.0, 0, 1)  # 归一化
        target_conn = torch.clamp(target_dilated / 9.0, 0, 1)
        
        return F.mse_loss(pred_conn, target_conn)
    
    def compute_thin_vessel_weight(self, target):
        """计算细血管权重图"""
        kernel = torch.ones(1, 1, 3, 3, device=target.device)
        
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        # 膨胀操作
        dilated = F.conv2d(target, kernel, padding=1)
        dilated = (dilated > 0.5).float()
        
        # 腐蚀操作
        eroded = F.conv2d(target, kernel, padding=1)
        eroded = (eroded >= 8.5).float()
        
        # 细血管区域：膨胀后的边缘
        thin_vessels = dilated - eroded
        
        # 生成权重图：细血管区域权重更高
        weight = torch.ones_like(target)
        weight = weight + thin_vessels * (self.thin_vessel_boost - 1)
        
        return weight.squeeze(1) if target.dim() == 4 else weight
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: 预测输出 (B, 1, H, W) 或 (B, H, W)
            target: 真实标签 (B, 1, H, W) 或 (B, H, W)
            mask: 有效区域掩码 (B, 1, H, W) 或 (B, H, W)
        """
        # 确保维度一致
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        if mask is not None and mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
            
        # 应用掩码
        if mask is not None:
            pred = pred * mask
            target = target * mask
            
        # 计算细血管权重
        weight = self.compute_thin_vessel_weight(target)
        
        # 各项损失
        dice = self.dice_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        connectivity = self.connectivity_loss(pred, target)
        
        # 加权BCE
        bce = F.binary_cross_entropy(pred, target, weight=weight, reduction='mean')
        
        # 组合损失
        total_loss = (self.dice_weight * dice + 
                     self.tversky_weight * tversky + 
                     self.connectivity_weight * connectivity + 
                     self.bce_weight * bce)
        
        # 返回总损失和各分量（用于监控）
        loss_dict = {
            'total': total_loss,
            'dice': dice,
            'tversky': tversky,
            'connectivity': connectivity,
            'bce': bce
        }
        
        return total_loss, loss_dict


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss - 结合Focal Loss和Tversky Loss"""
    
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, pred, target, smooth=1e-5):
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        true_pos = (pred_flat * target_flat).sum()
        false_neg = ((1 - pred_flat) * target_flat).sum()
        false_pos = (pred_flat * (1 - target_flat)).sum()
        
        tversky = (true_pos + smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


class BoundaryLoss(nn.Module):
    """边界损失 - 专门优化血管边缘"""
    
    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def forward(self, pred, target):
        """
        计算边界损失
        """
        n, c, h, w = pred.shape
        
        # 计算边界
        kernel = torch.ones(1, 1, 3, 3, device=pred.device)
        
        # 目标边界
        target_dilated = F.conv2d(target, kernel, padding=1)
        target_eroded = F.conv2d(1 - target, kernel, padding=1)
        target_boundary = torch.clamp(target_dilated - (9 - target_eroded), 0, 1)
        
        # 预测边界
        pred_dilated = F.conv2d(pred, kernel, padding=1)
        pred_eroded = F.conv2d(1 - pred, kernel, padding=1)
        pred_boundary = torch.clamp(pred_dilated - (9 - pred_eroded), 0, 1)
        
        # 边界损失
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        
        return boundary_loss


class CombinedVesselLoss(nn.Module):
    """组合多个损失函数的综合损失"""
    
    def __init__(self, 
                 vessel_weight=0.6,
                 focal_tversky_weight=0.2,
                 boundary_weight=0.2):
        super(CombinedVesselLoss, self).__init__()
        
        # 主损失
        self.vessel_loss = VesselSegmentationLoss(
            dice_weight=0.3,
            tversky_weight=0.3,
            connectivity_weight=0.2,
            bce_weight=0.2,
            alpha=0.7,
            beta=0.3,
            thin_vessel_boost=2.0
        )
        
        # 辅助损失
        self.focal_tversky = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=2.0)
        self.boundary_loss = BoundaryLoss()
        
        # 权重
        self.vessel_weight = vessel_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.boundary_weight = boundary_weight
        
    def forward(self, pred, target, mask=None):
        """
        计算组合损失
        """
        # 主损失
        vessel_loss, loss_dict = self.vessel_loss(pred, target, mask)
        
        # 辅助损失
        focal_tversky_loss = self.focal_tversky(pred, target)
        boundary_loss = self.boundary_loss(pred, target)
        
        # 总损失
        total_loss = (self.vessel_weight * vessel_loss + 
                     self.focal_tversky_weight * focal_tversky_loss +
                     self.boundary_weight * boundary_loss)
        
        # 更新损失字典
        loss_dict['focal_tversky'] = focal_tversky_loss
        loss_dict['boundary'] = boundary_loss
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict