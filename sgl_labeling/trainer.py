#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/10/19 18:49
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : trainer2.py
# @Description   : 



import os
import math
from decimal import Decimal

import data
import utility
import numpy as np
import torch
import torch.nn.utils as utils
import torch.nn as nn
import torch.nn.functional as F
from loss.bceloss import dice_bce_loss as DICE
from loss.bceloss import penalty_bce_loss as PBCE
from loss.tv import TVLoss

# 新增：导入边缘感知多尺度损失
from loss.edge_aware_loss import MultiScaleLoss, AdaptiveMultiScaleLoss

# ========== 新增：导入血管分割专用损失 ==========
from loss.vessel_loss import (
    VesselSegmentationLoss, 
    FocalTverskyLoss,
    BoundaryLoss,
    CombinedVesselLoss
)
# ===============================================

from tqdm import tqdm
import time
torch.autograd.set_detect_anomaly(True)

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.ploss = nn.L1Loss().cuda()
        self.bce_loss = nn.BCELoss().cuda()
        self.dice_bce_loss = DICE().cuda()
        self.pbce_loss = PBCE().cuda()
        self.tv_loss = TVLoss().cuda()
        
        # ========== 配置损失函数选项 ==========
        # 选择使用哪种损失函数：
        # 'original': 原始损失
        # 'multiscale': 多尺度边缘感知损失
        # 'vessel': 血管分割专用损失
        # 'combined': 组合血管损失（包含多种损失）
        self.loss_mode = 'vessel'  # 修改这里选择损失函数
        
        # 多尺度边缘感知损失
        if self.loss_mode == 'multiscale':
            self.multiscale_loss = MultiScaleLoss(
                bce_weight=1.0,
                dice_weight=1.0,
                edge_weight=2.0,
                focal_weight=1.0,
                thin_weight=1.5,
                edge_enhance=5.0,
                thin_enhance=3.0
            ).cuda()
            
        # ========== 血管分割专用损失 ==========
        elif self.loss_mode == 'vessel':
            self.vessel_loss = VesselSegmentationLoss(
                dice_weight=0.3,
                tversky_weight=0.3,      # Tversky损失权重（控制假阳性/假阴性）
                connectivity_weight=0.2,  # 连续性损失权重
                bce_weight=0.2,
                alpha=0.7,               # Tversky α参数（假阴性惩罚）
                beta=0.3,                # Tversky β参数（假阳性惩罚）
                thin_vessel_boost=2.0    # 细血管区域增强系数
            ).cuda()
            
        # ========== 组合血管损失 ==========
        elif self.loss_mode == 'combined':
            self.combined_loss = CombinedVesselLoss(
                vessel_weight=0.6,           # 主损失权重
                focal_tversky_weight=0.2,    # Focal Tversky权重
                boundary_weight=0.2          # 边界损失权重
            ).cuda()
        # =====================================
        
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        
        # 如果使用自适应损失，更新epoch
        if self.loss_mode == 'multiscale' and hasattr(self.multiscale_loss, 'set_epoch'):
            self.multiscale_loss.set_epoch(epoch)
        
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        
        # 用于记录各个损失分量
        loss_components = {
            'total': 0, 'bce': 0, 'dice': 0, 
            'tversky': 0, 'connectivity': 0,
            'edge': 0, 'focal': 0, 'thin': 0,
            'focal_tversky': 0, 'boundary': 0
        }
        
        for batch, (data_pack, _) in enumerate(self.loader_train):
            data_pack = self.prepare(data_pack)
            timer_data.hold()
            timer_model.tic()
            hr, ve, ma, pm, _, _ = data_pack
            self.optimizer.zero_grad()
            enh, estimation = self.model(hr, 1)
            
            # ========== 根据选择的损失模式计算损失 ==========
            if self.loss_mode == 'original':
                # 使用原始损失函数
                loss = self.loss(estimation, ve, pm*ma)
                
            elif self.loss_mode == 'multiscale':
                # 使用多尺度边缘感知损失
                loss, loss_dict = self.multiscale_loss(estimation, ve, pm * ma)
                
                # 记录损失分量
                for key in ['bce', 'dice', 'edge', 'focal', 'thin']:
                    if key in loss_dict:
                        loss_components[key] += loss_dict[key].item()
                loss_components['total'] += loss.item()
                
            elif self.loss_mode == 'vessel':
                # 使用血管分割专用损失
                loss, loss_dict = self.vessel_loss(estimation, ve, pm * ma)
                
                # 记录损失分量
                for key in loss_dict.keys():
                    if key in loss_components:
                        loss_components[key] += loss_dict[key].item()
                        
            elif self.loss_mode == 'combined':
                # 使用组合血管损失
                loss, loss_dict = self.combined_loss(estimation, ve, pm * ma)
                
                # 记录损失分量
                for key in loss_dict.keys():
                    if key in loss_components:
                        loss_components[key] += loss_dict[key].item()
            # =============================================
            
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                # ========== 显示详细损失信息 ==========
                if self.loss_mode != 'original' and batch > 0:
                    # 计算平均损失
                    avg_losses = {k: v / (batch + 1) for k, v in loss_components.items() if v > 0}
                    
                    # 根据损失模式显示不同的信息
                    if self.loss_mode == 'vessel':
                        loss_info = '[Total: {:.4f} | Dice: {:.4f} | Tversky: {:.4f} | Conn: {:.4f} | BCE: {:.4f}]'.format(
                            avg_losses.get('total', 0), 
                            avg_losses.get('dice', 0),
                            avg_losses.get('tversky', 0),
                            avg_losses.get('connectivity', 0),
                            avg_losses.get('bce', 0)
                        )
                    elif self.loss_mode == 'combined':
                        loss_info = '[Total: {:.4f} | Vessel: {:.4f} | F-Tversky: {:.4f} | Boundary: {:.4f}]'.format(
                            avg_losses.get('total', 0),
                            avg_losses.get('dice', 0) + avg_losses.get('tversky', 0),
                            avg_losses.get('focal_tversky', 0),
                            avg_losses.get('boundary', 0)
                        )
                    else:  # multiscale
                        loss_info = '[Total: {:.4f} | BCE: {:.4f} | Dice: {:.4f} | Edge: {:.4f}]'.format(
                            avg_losses.get('total', 0),
                            avg_losses.get('bce', 0),
                            avg_losses.get('dice', 0),
                            avg_losses.get('edge', 0)
                        )
                else:
                    loss_info = self.loss.display_loss(batch)
                
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    loss_info,
                    timer_model.release(),
                    timer_data.release()))
                # ==========================================

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        Background_IOU = []
        Vessel_IOU = []
        ACC = []
        SE = []
        SP = []
        AUC = []
        BIOU = []
       
        thin_Background_IOU = []
        thin_Vessel_IOU = []
        thin_ACC = []
        thin_SE = []
        thin_SP = []
        thin_DICE = []
       
        thick_Background_IOU = []
        thick_Vessel_IOU = []
        thick_ACC = []
        thick_SE = []
        thick_SP = []
        thick_DICE = []
        
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for data_pack, filename in tqdm(d, ncols=80):
                    data_pack = self.prepare(data_pack)
                    hr, ve, ma, pm, ve_thin, ve_thick = data_pack
                    enhance, est_o = self.model(hr, idx_scale) 
                    
                    est_o = est_o * ma
                    ve = ve * ma
                    dmap = utility.visualize_dmap(est_o, ve) * 255
                    enhance = enhance * ma * 255
                    est = est_o * 255.
                    pm = pm[:, :, 0:584, 0:565]
                    pm[pm>=0.99] = 1
                    pm[pm<0.99] = 0  #define the regions
                    
                    hr = hr
                    ve = ve * ma * 255.
                    ve_thin = ve_thin * ma * 255.
                    ve_thick = ve_thick * ma * 255.
                    est[est>100] = 255
                    est[est<=100] = 0
                    est = est[:, :, 0:584, 0:565]
                    dmap = dmap[:, :, 0:584, 0:565]
                    est_o = est_o[:, :, 0:584, 0:565]
                    enhance = enhance[:, :, 0:584, 0:565]
                    ve = ve[:, :, 0:584, 0:565]
                    ve_thin = ve_thin[:, :, 0:584, 0:565]
                    ve_thick = ve_thick[:, :, 0:584, 0:565]
                    hr = hr[:, :, 0:584, 0:565]
                    est = utility.quantize(est, self.args.rgb_range)
                    estnp = np.transpose(est[0].cpu().numpy(), (1,2,0)) / 255.
                    est_thin, est_thick = data.common.ttsep(estnp)
                    vis_vessel = utility.visualize(est, ve, False)
                    vis_thin = utility.visualize(est_thin*255., ve_thin, True)                    
                    vis_thick = utility.visualize(est_thick*255., ve_thick, True)                    
                    save_list = [est_o*255]
                   
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_dice(est, ve, False)
                    ve = utility.quantize(ve, self.args.rgb_range)
                    Acc,Se,Sp,Auc,IU0, IU1 = utility.calc_metrics(est_o, est, ve, False)
                    BIOU.append(utility.calc_boundiou(est_o, ve/255., pm))
                    AUC.append(Auc)
                    Background_IOU.append(IU0)
                    Vessel_IOU.append(IU1)
                    ACC.append(Acc)
                    SE.append(Se)
                    SP.append(Sp)
                    #Thin vessel
                    Acc,Se,Sp,_,IU0, IU1 = utility.calc_metrics(est_o, est_thin, ve_thin, True)
                    thin_Background_IOU.append(IU0)
                    thin_Vessel_IOU.append(IU1)
                    thin_ACC.append(Acc)
                    thin_SE.append(Se)
                    thin_SP.append(Sp)
                    DICE = utility.calc_dice(est_thin, ve_thin, True)
                    thin_DICE.append(DICE)
                    #Thick vessel
                    Acc,Se,Sp,_,IU0, IU1 = utility.calc_metrics(est_o, est_thick, ve_thick, True)
                    thick_Background_IOU.append(IU0)
                    thick_Vessel_IOU.append(IU1)
                    thick_ACC.append(Acc)
                    thick_SE.append(Se)
                    thick_SP.append(Sp)
                    DICE = utility.calc_dice(est_thick, ve_thick, True)
                    thick_DICE.append(DICE)

                    if self.args.save_gt:
                        save_list.extend([hr, pm*255])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                        
                print(np.mean(np.stack(BIOU)))
                print('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(ACC))),str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),str(np.mean(np.stack(AUC))),str(np.mean(np.stack(Background_IOU))),str(np.mean(np.stack(Vessel_IOU)))))
                #Thin Vessel
                print('Thin Vessel: Acc: %s  |  Se: %s |  Sp: %s | DICE: %s | Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(thin_ACC))),str(np.mean(np.stack(thin_SE))), str(np.mean(np.stack(thin_SP))), str(np.mean(np.stack(thin_DICE))), str(np.mean(np.stack(thin_Background_IOU))),str(np.mean(np.stack(thin_Vessel_IOU)))))
                #Thick Vessel
                print('Thick Vessel: Acc: %s  |  Se: %s |  Sp: %s |  DICE: %s |  Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(thick_ACC))),str(np.mean(np.stack(thick_SE))), str(np.mean(np.stack(thick_SP))),str(np.mean(np.stack(thick_DICE))),str(np.mean(np.stack(thick_Background_IOU))),str(np.mean(np.stack(thick_Vessel_IOU)))))
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR Score: {:.6f} (Best: {:.6f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs