#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 保留两阶段架构，只改进SegNet部分

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return MainNet()

# ============ 保持原始的EnhanceNet不变 ============
class EnhanceNet(nn.Module):
    def __init__(self):
        super(EnhanceNet, self).__init__()
        self.conv1 = self.conv_block(3,32)
        self.conv2 = self.conv_block(32,64)
        self.conv3 = self.conv_block(64,128)
        self.conv4 = self.conv_block(128,128*2)
        self.conv5 = self.conv_block(128*2,128*4)
        self.pool = torch.nn.MaxPool2d(2)
        self.upconv1 = self.upconv(64,32)
        self.upconv2 = self.upconv(128,64)
        self.upconv3 = self.upconv(128*2,128)
        self.upconv4 = self.upconv(128*4,128*2)
        self.conv6 = self.conv_block(128*4,128*2)
        self.conv7 = self.conv_block(128*2,128)
        self.conv8 = self.conv_block(128,64)
        self.conv9 = self.conv_block(64,32)
        self.conv11 = self.conv_block(35,3)
        self.last_act = nn.PReLU()

    def conv_block(self, channel_in, channel_out):
        if channel_in==3:
            return nn.Sequential(
                nn.Conv2d(channel_in, channel_out, 3, 1, 1),
                nn.PReLU(),
                nn.BatchNorm2d(channel_out),
                nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            )
        else:
            return nn.Sequential(
                nn.PReLU(),
                nn.BatchNorm2d(channel_in),
                nn.Conv2d(channel_in, channel_out, 3, 1, 1),
                nn.PReLU(),
                nn.BatchNorm2d(channel_out),
                nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            )

    def upconv(self, channel_in, channel_out):
        return nn.ConvTranspose2d(channel_in,channel_out,kernel_size=2,stride=2)

    def forward(self, x):
        x = x / 255.  # 关键的归一化！
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)
        x4 = self.conv4(x4)
        x5 = self.pool(x4)
        x5 = self.conv5(x5)
        u4 = self.upconv4(x5)
        u4 = torch.cat([u4, x4], 1)
        u4 = self.conv6(u4)
        u3 = self.upconv3(u4)
        u3 = torch.cat([u3, x3], 1)
        u3 = self.conv7(u3)
        u2 = self.upconv2(u3)
        u2 = torch.cat([u2, x2], 1)
        u2 = self.conv8(u2)
        u1 = self.upconv1(u2)
        u1 = torch.cat([u1, x1], 1)
        u1 = self.conv9(u1)
        u1 = torch.cat([u1, x], 1)
        out_pred = torch.sigmoid(self.conv11(u1))
        return out_pred

# ============ 轻量级改进的SegNet ============
class LightweightAttention(nn.Module):
    """轻量级注意力 - 减少计算开销"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(self.avg_pool(x))

class ImprovedConvBlock(nn.Module):
    """改进的卷积块 - 添加残差和轻量级注意力"""
    def __init__(self, in_ch, out_ch, use_attention=False):
        super().__init__()
        
        if in_ch == 3:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.PReLU(),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )
        else:
            self.conv = nn.Sequential(
                nn.PReLU(),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.PReLU(),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )
        
        # 残差连接
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        # 选择性添加注意力
        self.attention = LightweightAttention(out_ch) if use_attention else nn.Identity()
        
    def forward(self, x):
        return self.attention(self.conv(x) + self.shortcut(x))

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        
        # 编码器 - 只在深层使用注意力
        self.conv1 = ImprovedConvBlock(3, 32, use_attention=False)
        self.conv2 = ImprovedConvBlock(32, 64, use_attention=False)
        self.conv3 = ImprovedConvBlock(64, 128, use_attention=True)
        self.conv4 = ImprovedConvBlock(128, 256, use_attention=True)
        self.conv5 = ImprovedConvBlock(256, 512, use_attention=True)
        
        self.pool = nn.MaxPool2d(2)
        
        # 解码器
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        
        self.conv6 = ImprovedConvBlock(512, 256, use_attention=False)
        self.conv7 = ImprovedConvBlock(256, 128, use_attention=False)
        self.conv8 = ImprovedConvBlock(128, 64, use_attention=False)
        self.conv9 = ImprovedConvBlock(64, 32, use_attention=False)
        
        # 输出层 - 保持与baseline相同
        self.conv11 = nn.Sequential(
            nn.Conv2d(35, 16, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )
        
        # 轻量级Dropout
        self.dropout = nn.Dropout2d(0.05)
        
    def forward(self, x):
        # 编码
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)
        x4 = self.conv4(x4)
        x5 = self.pool(x4)
        x5 = self.conv5(x5)
        
        # 解码
        u4 = self.upconv4(x5)
        u4 = torch.cat([u4, x4], 1)
        u4 = self.conv6(u4)
        
        u3 = self.upconv3(u4)
        u3 = torch.cat([u3, x3], 1)
        u3 = self.conv7(u3)
        
        u2 = self.upconv2(u3)
        u2 = torch.cat([u2, x2], 1)
        u2 = self.conv8(u2)
        
        u1 = self.upconv1(u2)
        u1 = torch.cat([u1, x1], 1)
        u1 = self.conv9(u1)
        
        # 应用dropout
        u1 = self.dropout(u1)
        
        # 与原始输入拼接（保持与baseline一致）
        u1 = torch.cat([u1, x], 1)
        
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
                        raise RuntimeError(f'While copying {name}, '
                                         f'model: {own_state[name].size()}, '
                                         f'checkpoint: {param.size()}')
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(f'unexpected key "{name}" in state_dict')

# ============ 两阶段主网络 ============
class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.s1 = EnhanceNet()  # 保留图像增强
        self.s2 = SegNet()       # 使用改进的分割网络
    
    def forward(self, x):
        x1 = self.s1(x)    # 先增强
        out = self.s2(x1)  # 再分割
        return x1, out     # 返回格式与baseline一致