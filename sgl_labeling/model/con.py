import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def make_model(args, parent=False):
    return MainNet()

class ChannelAttention(nn.Module):
    """通道注意力模块 - 轻量级版本"""
    def __init__(self, in_channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out

class ImprovedConvBlock(nn.Module):
    """改进的卷积块 - 添加残差连接和注意力"""
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(ImprovedConvBlock, self).__init__()
        
        # 基础卷积
        if in_channels == 3:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.PReLU()
            )
        else:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.PReLU(),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
        
        # 残差连接
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        
        # 轻量级注意力
        self.use_attention = use_attention
        if use_attention:
            self.channel_att = ChannelAttention(out_channels)
            self.spatial_att = SpatialAttention()
        
        self.relu = nn.PReLU()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        
        if self.use_attention:
            out = self.channel_att(out)
            out = self.spatial_att(out)
        
        out = out + residual
        return self.relu(out)

class MultiScaleBlock(nn.Module):
    """多尺度特征提取块 - 轻量级ASPP"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels//4, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels//4, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(in_channels, out_channels//4, 3, padding=8, dilation=8)
        
        self.fusion = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        
    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        out = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        return self.fusion(out)

class GatedFusion(nn.Module):
    """门控融合模块 - 用于跳跃连接"""
    def __init__(self, channels):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
    def forward(self, x_low, x_high):
        concat = torch.cat([x_low, x_high], dim=1)
        gate = self.gate(concat)
        return x_low * gate + x_high * (1 - gate)

class SegNet(nn.Module):
    """轻量级改进的血管分割网络"""
    def __init__(self):
        super(SegNet, self).__init__()
        
        # 编码器 - 使用改进的卷积块但保持原始深度
        self.conv1 = ImprovedConvBlock(3, 32, use_attention=False)
        self.conv2 = ImprovedConvBlock(32, 64, use_attention=False)
        self.conv3 = ImprovedConvBlock(64, 128, use_attention=True)
        self.conv4 = ImprovedConvBlock(128, 256, use_attention=True)
        
        # 瓶颈层 - 使用多尺度块
        self.conv5 = nn.Sequential(
            ImprovedConvBlock(256, 512, use_attention=True),
            MultiScaleBlock(512, 512)
        )
        
        self.pool = nn.MaxPool2d(2)
        
        # 解码器 - 使用门控融合
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        
        # 门控融合模块
        self.fusion4 = GatedFusion(256)
        self.fusion3 = GatedFusion(128)
        self.fusion2 = GatedFusion(64)
        self.fusion1 = GatedFusion(32)
        
        # 解码卷积
        self.conv6 = ImprovedConvBlock(512, 256, use_attention=False)
        self.conv7 = ImprovedConvBlock(256, 128, use_attention=False)
        self.conv8 = ImprovedConvBlock(128, 64, use_attention=False)
        self.conv9 = ImprovedConvBlock(64, 32, use_attention=False)
        
        # 边缘细化模块
        self.edge_refine = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU()
        )
        
        # 最终输出 - 保持与原网络兼容
        self.conv11 = nn.Sequential(
            nn.Conv2d(35 + 8, 16, 3, padding=1),  # 32+3+8=43
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Conv2d(8, 1, 3, padding=1)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        # 编码路径
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)
        x4 = self.conv4(x4)
        x5 = self.pool(x4)
        x5 = self.conv5(x5)
        
        # 解码路径 with 门控融合
        u4 = self.upconv4(x5)
        u4 = self.fusion4(x4, u4)
        u4 = torch.cat([u4, x4], 1)
        u4 = self.conv6(u4)
        
        u3 = self.upconv3(u4)
        u3 = self.fusion3(x3, u3)
        u3 = torch.cat([u3, x3], 1)
        u3 = self.conv7(u3)
        
        u2 = self.upconv2(u3)
        u2 = self.fusion2(x2, u2)
        u2 = torch.cat([u2, x2], 1)
        u2 = self.conv8(u2)
        
        u1 = self.upconv1(u2)
        u1 = self.fusion1(x1, u1)
        u1 = torch.cat([u1, x1], 1)
        u1 = self.conv9(u1)
        
        # 边缘细化
        edge_feat = self.edge_refine(u1)
        
        # 应用dropout
        u1 = self.dropout(u1)
        
        # 与原始输入和边缘特征拼接
        u1 = torch.cat([u1, x, edge_feat], 1)
        
        # 最终预测
        out_pred = torch.sigmoid(self.conv11(u1))
        
        return out_pred
    
    def load_state_dict(self, state_dict, strict=True):
        """保持与原SegNet相同的加载逻辑"""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.s2 = SegNet()
    
    def forward(self, x):
        out = self.s2(x)
        return out, out

# 改进的损失函数 - 专门针对血管分割
class VesselSegmentationLoss(nn.Module):
    """专门为血管分割优化的损失函数"""
    def __init__(self):
        super(VesselSegmentationLoss, self).__init__()
        
    def dice_loss(self, pred, target, smooth=1e-5):
        """Dice loss for handling imbalanced data"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice
    
    def tversky_loss(self, pred, target, alpha=0.7, beta=0.3, smooth=1e-5):
        """Tversky loss - 可以调整对假阳性和假阴性的惩罚权重"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        true_pos = (pred * target).sum()
        false_neg = ((1 - pred) * target).sum()
        false_pos = (pred * (1 - target)).sum()
        
        tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
        return 1 - tversky
    
    def connectivity_loss(self, pred, target):
        """连续性损失 - 鼓励连续的血管结构"""
        # 使用形态学操作评估连续性
        kernel = torch.ones(1, 1, 3, 3).to(pred.device)
        
        # 计算预测和目标的连通性
        pred_dilated = F.conv2d(pred.unsqueeze(1) if pred.dim() == 3 else pred, 
                                kernel, padding=1)
        target_dilated = F.conv2d(target.unsqueeze(1) if target.dim() == 3 else target, 
                                  kernel, padding=1)
        
        # 连续性度量
        pred_conn = torch.clamp(pred_dilated, 0, 1)
        target_conn = torch.clamp(target_dilated, 0, 1)
        
        return F.mse_loss(pred_conn, target_conn)
    
    def forward(self, pred, target):
        """组合损失"""
        # 主要损失
        dice = self.dice_loss(pred, target)
        tversky = self.tversky_loss(pred, target, alpha=0.7, beta=0.3)  # 更多惩罚假阴性
        
        # 连续性损失
        connectivity = self.connectivity_loss(pred, target)
        
        # BCE for general supervision
        bce = F.binary_cross_entropy(pred, target)
        
        # 加权组合
        total_loss = 0.3 * dice + 0.3 * tversky + 0.2 * connectivity + 0.2 * bce
        
        return total_loss