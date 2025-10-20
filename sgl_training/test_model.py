#!/usr/bin/env python3
"""
增强的血管分割模型测试脚本 - 集成定性和定量分析
"""
import argparse
import numpy as np
import torch
from tqdm import tqdm
import cv2
from scipy import ndimage
from skimage import morphology
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
import data
import utility
import model as model_module


class QualitativeAnalyzer:
    """定性分析器 - 集成所有定性分析功能"""
    
    def __init__(self):
        self.analysis_cache = []
    
    def analyze_vessel_width_distribution(self, vessel_img):
        """分析血管宽度分布"""
        binary = (vessel_img > 127).astype(np.uint8)
        if np.sum(binary) == 0:
            return None, None
        
        dist_transform = ndimage.distance_transform_edt(binary)
        vessel_widths = dist_transform[binary > 0] * 2
        
        if len(vessel_widths) == 0:
            return None, None
        
        width_stats = {
            'mean': float(np.mean(vessel_widths)),
            'std': float(np.std(vessel_widths)),
            'min': float(np.min(vessel_widths)),
            'max': float(np.max(vessel_widths)),
            'median': float(np.median(vessel_widths)),
            'thin_vessels_ratio': float(np.sum(vessel_widths < 3) / len(vessel_widths)),
            'medium_vessels_ratio': float(np.sum((vessel_widths >= 3) & (vessel_widths < 8)) / len(vessel_widths)),
            'thick_vessels_ratio': float(np.sum(vessel_widths >= 8) / len(vessel_widths))
        }
        
        return width_stats, vessel_widths
    
    def analyze_connectivity(self, vessel_img):
        """分析血管连通性"""
        binary = (vessel_img > 127).astype(np.uint8)
        if np.sum(binary) == 0:
            return None, None
        
        # 连通域分析
        labeled, num_components = ndimage.label(binary)
        component_sizes = ndimage.sum(binary, labeled, range(1, num_components + 1))
        
        # 骨架分析
        skeleton = morphology.skeletonize(binary)
        skeleton_length = np.sum(skeleton)
        
        connectivity_stats = {
            'num_components': int(num_components),
            'avg_component_size': float(np.mean(component_sizes)) if len(component_sizes) > 0 else 0.0,
            'largest_component_size': float(np.max(component_sizes)) if len(component_sizes) > 0 else 0.0,
            'skeleton_length': float(skeleton_length),
            'fragmentation_index': float(num_components / (skeleton_length + 1)),
            'largest_component_ratio': float(np.max(component_sizes) / skeleton_length) if skeleton_length > 0 and len(component_sizes) > 0 else 0.0
        }
        
        return connectivity_stats, skeleton
    
    def analyze_edge_performance(self, vessel_img, prob_img):
        """分析边缘区域的性能"""
        binary = (vessel_img > 127).astype(np.uint8)
        
        # 提取边缘
        edges = cv2.Canny(binary * 255, 50, 150)
        
        # 膨胀边缘区域
        kernel = np.ones((5, 5), np.uint8)
        edge_zone = cv2.dilate(edges, kernel, iterations=1)
        
        # 分析边缘区域的概率值
        edge_probs = prob_img[edge_zone > 0]
        
        if len(edge_probs) == 0:
            return None, edges
        
        edge_stats = {
            'edge_length': int(np.sum(edges > 0)),
            'avg_edge_prob': float(np.mean(edge_probs)),
            'std_edge_prob': float(np.std(edge_probs)),
            'low_confidence_edges': float(np.sum(edge_probs < 150) / len(edge_probs)),
            'edge_confidence': float(np.mean(edge_probs))
        }
        
        return edge_stats, edges
    
    def analyze_probability_distribution(self, prob_img, vessel_img):
        """分析概率分布"""
        binary = (vessel_img > 127).astype(np.uint8)
        
        vessel_probs = prob_img[binary > 0]
        background_probs = prob_img[binary == 0]
        
        if len(vessel_probs) == 0 or len(background_probs) == 0:
            return None
        
        prob_stats = {
            'vessel_prob_mean': float(np.mean(vessel_probs)),
            'vessel_prob_std': float(np.std(vessel_probs)),
            'background_prob_mean': float(np.mean(background_probs)),
            'background_prob_std': float(np.std(background_probs)),
            'separation_score': float(np.mean(vessel_probs) - np.mean(background_probs)),
            'uncertain_vessel_ratio': float(np.sum((vessel_probs > 100) & (vessel_probs < 200)) / len(vessel_probs))
        }
        
        return prob_stats
    
    def analyze_image_quality(self, raw_img, enhance_img):
        """分析图像质量"""
        raw_contrast = np.std(raw_img)
        enhance_contrast = np.std(enhance_img)
        
        def local_contrast(img, window_size=15):
            kernel = np.ones((window_size, window_size)) / (window_size ** 2)
            local_mean = cv2.filter2D(img.astype(float), -1, kernel)
            local_var = cv2.filter2D((img.astype(float) - local_mean) ** 2, -1, kernel)
            return np.mean(np.sqrt(local_var))
        
        quality_stats = {
            'raw_contrast': float(raw_contrast),
            'enhance_contrast': float(enhance_contrast),
            'contrast_improvement': float((enhance_contrast - raw_contrast) / raw_contrast) if raw_contrast > 0 else 0.0,
            'raw_local_contrast': float(local_contrast(raw_img)),
            'enhance_local_contrast': float(local_contrast(enhance_img)),
            'brightness_mean': float(np.mean(raw_img)),
            'brightness_std': float(np.std(raw_img))
        }
        
        return quality_stats
    
    def analyze_difficult_regions(self, penalty_map, vessel_img):
        """分析困难区域"""
        binary = (vessel_img > 127).astype(np.uint8)
        high_penalty = penalty_map > 150
        
        vessel_pixels = np.sum(binary > 0)
        if vessel_pixels == 0:
            return None, high_penalty
        
        difficult_stats = {
            'difficult_area_ratio': float(np.sum(high_penalty) / high_penalty.size),
            'difficult_vessel_ratio': float(np.sum(high_penalty & (binary > 0)) / vessel_pixels),
            'avg_penalty_score': float(np.mean(penalty_map)),
            'max_penalty_score': float(np.max(penalty_map))
        }
        
        return difficult_stats, high_penalty
    
    def find_false_positives_negatives(self, vessel_img, prob_img):
        """寻找可能的假阳性和假阴性"""
        binary = (vessel_img > 127).astype(np.uint8)
        
        potential_fp = (binary > 0) & (prob_img < 180)
        potential_fn = (binary == 0) & (prob_img > 100)
        
        vessel_pixels = np.sum(binary > 0)
        background_pixels = np.sum(binary == 0)
        
        fp_fn_stats = {
            'potential_fp_ratio': float(np.sum(potential_fp) / vessel_pixels) if vessel_pixels > 0 else 0.0,
            'potential_fn_pixels': int(np.sum(potential_fn)),
            'potential_fn_ratio': float(np.sum(potential_fn) / background_pixels) if background_pixels > 0 else 0.0,
            'false_positive_rate': float(np.sum(potential_fp) / vessel_pixels) if vessel_pixels > 0 else 0.0
        }
        
        return fp_fn_stats, potential_fp, potential_fn


class ModelTester:
    """增强的模型测试类 - 集成定量和定性分析"""
    
    def __init__(self, args, loader_test, model, ckp):
        self.args = args
        self.scale = args.scale
        self.loader_test = loader_test
        self.model = model
        self.ckp = ckp
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        
        # 初始化定性分析器
        self.qualitative_analyzer = QualitativeAnalyzer()
        
        # 结果保存路径
        self.analysis_dir = Path(ckp.dir) / 'qualitative_analysis'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare(self, tensors):
        """准备数据，移动到设备"""
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        return [_prepare(t) for t in tensors]
    
    def postprocess(self, hr, ve, ma, est_o, enhance, pm):
        """后处理模型输出"""
        est_o = est_o * ma
        ve = ve * ma
        enhance = enhance * ma * 255
        
        est = est_o * 255.0
        
        crop_h, crop_w = self.args.crop_size
        pm = pm[:, :, :crop_h, :crop_w]
        pm[pm >= 0.99] = 1
        pm[pm < 0.99] = 0
        
        ve = ve * ma * 255.
        
        est[est > 100] = 255
        est[est <= 100] = 0
        
        est = est[:, :, :crop_h, :crop_w]
        est_o = est_o[:, :, :crop_h, :crop_w]
        enhance = enhance[:, :, :crop_h, :crop_w]
        ve = ve[:, :, :crop_h, :crop_w]
        hr = hr[:, :, :crop_h, :crop_w]
        
        return hr, ve, est, est_o, enhance, pm
    
    def compute_quantitative_metrics(self, est, est_o, ve, pm):
        """计算定量评估指标"""
        metrics = {}
        
        # DICE分数
        metrics['DICE'] = utility.calc_dice(est, ve, False)
        
        # 其他指标
        acc, se, sp, auc, iou_bg, iou_vessel = utility.calc_metrics(
            est_o, est, ve, False
        )
        metrics['ACC'] = acc
        metrics['SE'] = se
        metrics['SP'] = sp
        metrics['AUC'] = auc
        metrics['Background_IOU'] = iou_bg
        metrics['Vessel_IOU'] = iou_vessel
        
        # 边界IOU
        metrics['BIOU'] = utility.calc_boundiou(est_o, ve / 255., pm)
        
        return metrics
    
    def compute_qualitative_metrics(self, est, est_o, ve, hr, enhance, pm):
        """计算定性分析指标"""
        # 转换为numpy数组
        est_np = est.squeeze().cpu().numpy()
        est_o_np = est_o.squeeze().cpu().numpy()
        ve_np = ve.squeeze().cpu().numpy()
        hr_np = hr.squeeze().cpu().numpy()
        enhance_np = enhance.squeeze().cpu().numpy()
        pm_np = pm.squeeze().cpu().numpy()
        
        # 确保是2D数组
        if len(est_np.shape) > 2:
            est_np = est_np[0]
        if len(est_o_np.shape) > 2:
            est_o_np = est_o_np[0]
        if len(ve_np.shape) > 2:
            ve_np = ve_np[0]
        if len(hr_np.shape) > 2:
            hr_np = hr_np.mean(axis=0)  # RGB转灰度
        if len(enhance_np.shape) > 2:
            enhance_np = enhance_np.mean(axis=0)
        if len(pm_np.shape) > 2:
            pm_np = pm_np[0]
        
        # 概率图
        prob_np = (est_o_np * 255).astype(np.uint8)
        
        qualitative_metrics = {}
        
        # 1. 血管宽度分析
        width_stats, widths = self.qualitative_analyzer.analyze_vessel_width_distribution(est_np)
        if width_stats:
            qualitative_metrics['width_analysis'] = width_stats
        
        # 2. 连通性分析
        conn_stats, skeleton = self.qualitative_analyzer.analyze_connectivity(est_np)
        if conn_stats:
            qualitative_metrics['connectivity_analysis'] = conn_stats
        
        # 3. 边缘性能分析
        edge_stats, edges = self.qualitative_analyzer.analyze_edge_performance(est_np, prob_np)
        if edge_stats:
            qualitative_metrics['edge_analysis'] = edge_stats
        
        # 4. 概率分布分析
        prob_stats = self.qualitative_analyzer.analyze_probability_distribution(prob_np, est_np)
        if prob_stats:
            qualitative_metrics['probability_analysis'] = prob_stats
        
        # 5. 图像质量分析
        quality_stats = self.qualitative_analyzer.analyze_image_quality(hr_np, enhance_np)
        if quality_stats:
            qualitative_metrics['quality_analysis'] = quality_stats
        
        # 6. 困难区域分析
        diff_stats, high_penalty = self.qualitative_analyzer.analyze_difficult_regions(pm_np, est_np)
        if diff_stats:
            qualitative_metrics['difficult_regions'] = diff_stats
        
        # 7. 假阳性/假阴性分析
        fp_fn_stats, potential_fp, potential_fn = self.qualitative_analyzer.find_false_positives_negatives(est_np, prob_np)
        if fp_fn_stats:
            qualitative_metrics['error_analysis'] = fp_fn_stats
        
        # 保存可视化数据（可选）
        if self.args.save_qualitative_viz:
            viz_data = {
                'widths': widths,
                'skeleton': skeleton,
                'edges': edges,
                'potential_fp': potential_fp,
                'potential_fn': potential_fn,
                'high_penalty': high_penalty
            }
            return qualitative_metrics, viz_data
        
        return qualitative_metrics, None
    
    def visualize_qualitative_analysis(self, filename, images_dict, viz_data):
        """可视化定性分析结果"""
        if viz_data is None:
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle(f'Qualitative Analysis - {filename}', fontsize=16, fontweight='bold')
        
        # 原始图像
        axes[0, 0].imshow(images_dict['raw'], cmap='gray')
        axes[0, 0].set_title('Raw Image')
        axes[0, 0].axis('off')
        
        # 增强图像
        axes[0, 1].imshow(images_dict['enhance'], cmap='gray')
        axes[0, 1].set_title('Enhanced Image')
        axes[0, 1].axis('off')
        
        # 分割结果
        axes[0, 2].imshow(images_dict['segmentation'], cmap='gray')
        axes[0, 2].set_title('Segmentation Result')
        axes[0, 2].axis('off')
        
        # 概率图
        axes[1, 0].imshow(images_dict['probability'], cmap='jet')
        axes[1, 0].set_title('Probability Map')
        axes[1, 0].axis('off')
        
        # 血管宽度分布
        if viz_data['widths'] is not None and len(viz_data['widths']) > 0:
            axes[1, 1].hist(viz_data['widths'], bins=50, color='blue', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Vessel Width Distribution')
            axes[1, 1].set_xlabel('Width (pixels)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 骨架
        if viz_data['skeleton'] is not None:
            axes[1, 2].imshow(viz_data['skeleton'], cmap='gray')
            axes[1, 2].set_title('Vessel Skeleton')
            axes[1, 2].axis('off')
        
        # 边缘检测
        if viz_data['edges'] is not None:
            axes[2, 0].imshow(viz_data['edges'], cmap='gray')
            axes[2, 0].set_title('Vessel Edges')
            axes[2, 0].axis('off')
        
        # 潜在假阳性
        if viz_data['potential_fp'] is not None:
            fp_overlay = cv2.cvtColor(images_dict['raw'].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            fp_overlay[viz_data['potential_fp']] = [255, 0, 0]
            axes[2, 1].imshow(fp_overlay)
            axes[2, 1].set_title('Potential False Positives (Red)')
            axes[2, 1].axis('off')
        
        # 潜在假阴性
        if viz_data['potential_fn'] is not None:
            fn_overlay = cv2.cvtColor(images_dict['raw'].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            fn_overlay[viz_data['potential_fn']] = [0, 255, 0]
            axes[2, 2].imshow(fn_overlay)
            axes[2, 2].set_title('Potential False Negatives (Green)')
            axes[2, 2].axis('off')
        
        plt.tight_layout()
        save_path = self.analysis_dir / f'viz_{filename}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def test(self):
        """执行完整的模型测试（定量+定性）"""
        torch.set_grad_enabled(False)
        self.model.eval()
        
        self.ckp.write_log('\n' + '='*80)
        self.ckp.write_log('Starting Comprehensive Evaluation (Quantitative + Qualitative)')
        self.ckp.write_log('='*80)
        
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        
        # 初始化指标收集器
        quantitative_collector = {
            'DICE': [], 'ACC': [], 'SE': [], 'SP': [], 
            'AUC': [], 'Background_IOU': [], 'Vessel_IOU': [], 'BIOU': []
        }
        
        qualitative_collector = []
        
        timer_test = utility.timer()
        if self.args.save_results: 
            self.ckp.begin_background()
        
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                
                for data_pack, filename in tqdm(d, ncols=80, desc=f'Testing {d.dataset.name}'):
                    # 准备数据
                    hr, ve, ma, te, pm, ve_thin, ve_thick = self.prepare(data_pack)
                    
                    # 模型推理
                    enhance, est_o = self.model(hr, idx_scale)
                    
                    # 后处理
                    hr, ve, est, est_o, enhance, pm = self.postprocess(
                        hr, ve, ma, est_o, enhance, pm
                    )
                    
                    # 量化处理
                    est = utility.quantize(est, self.args.rgb_range)
                    vis_vessel = utility.visualize(est, ve, False)
                    
                    # ========== 定量分析 ==========
                    quantitative_metrics = self.compute_quantitative_metrics(est, est_o, ve, pm)
                    for key, value in quantitative_metrics.items():
                        quantitative_collector[key].append(value)
                    
                    # ========== 定性分析 ==========
                    qualitative_metrics, viz_data = self.compute_qualitative_metrics(
                        est, est_o, ve, hr, enhance, pm
                    )
                    
                    # 合并所有指标
                    complete_metrics = {
                        'image_id': filename[0],
                        'quantitative': quantitative_metrics,
                        'qualitative': qualitative_metrics
                    }
                    qualitative_collector.append(complete_metrics)
                    
                    # 可视化（如果启用）
                    if self.args.save_qualitative_viz and viz_data:
                        images_dict = {
                            'raw': hr.squeeze().cpu().numpy(),
                            'enhance': enhance.squeeze().cpu().numpy(),
                            'segmentation': est.squeeze().cpu().numpy(),
                            'probability': (est_o.squeeze().cpu().numpy() * 255).astype(np.uint8)
                        }
                        # 处理多通道图像
                        if len(images_dict['raw'].shape) > 2:
                            images_dict['raw'] = images_dict['raw'].mean(axis=0)
                        if len(images_dict['enhance'].shape) > 2:
                            images_dict['enhance'] = images_dict['enhance'].mean(axis=0)
                        if len(images_dict['segmentation'].shape) > 2:
                            images_dict['segmentation'] = images_dict['segmentation'][0]
                        
                        self.visualize_qualitative_analysis(filename[0], images_dict, viz_data)
                    
                    # 保存结果
                    if self.args.save_results:
                        save_list = [enhance, vis_vessel, est_o * 255]
                        if self.args.save_gt:
                            save_list.extend([hr, pm * 255])
                        self.ckp.save_results(d, filename[0], save_list, scale)
                    
                    # 更新DICE日志
                    self.ckp.log[-1, idx_data, idx_scale] += quantitative_metrics['DICE']
                
                # ========== 计算平均指标 ==========
                avg_quantitative = {k: np.mean(v) for k, v in quantitative_collector.items()}
                
                # 计算定性指标平均值
                avg_qualitative = self._compute_average_qualitative_metrics(qualitative_collector)
                
                # ========== 打印结果 ==========
                self._print_comprehensive_results(avg_quantitative, avg_qualitative, d.dataset.name, scale)
                
                # 记录DICE分数
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tDICE Score: {:.6f} (Best: {:.6f} @epoch {})'.format(
                        d.dataset.name, scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
        
        # ========== 保存完整分析报告 ==========
        self._save_analysis_report(qualitative_collector, avg_quantitative, avg_qualitative)
        
        # 计时信息
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        
        if self.args.save_results:
            self.ckp.end_background()
        
        self.ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)
        torch.set_grad_enabled(True)
        
        return {
            'quantitative': avg_quantitative,
            'qualitative': avg_qualitative,
            'detailed': qualitative_collector
        }
    
    def _compute_average_qualitative_metrics(self, collector):
        """计算定性指标的平均值"""
        if not collector:
            return {}
        
        avg_metrics = {}
        metric_categories = ['width_analysis', 'connectivity_analysis', 'edge_analysis',
                           'probability_analysis', 'quality_analysis', 'difficult_regions',
                           'error_analysis']
        
        for category in metric_categories:
            category_data = {}
            for item in collector:
                if 'qualitative' in item and category in item['qualitative']:
                    for key, value in item['qualitative'][category].items():
                        if key not in category_data:
                            category_data[key] = []
                        category_data[key].append(value)
            
            if category_data:
                avg_metrics[category] = {k: np.mean(v) for k, v in category_data.items()}
        
        return avg_metrics
    
    def _print_comprehensive_results(self, quant_metrics, qual_metrics, dataset_name, scale):
        """打印综合评估结果"""
        self.ckp.write_log('\n' + '='*80)
        self.ckp.write_log(f'Comprehensive Results for {dataset_name} x{scale}')
        self.ckp.write_log('='*80)
        
        # 定量指标
        self.ckp.write_log('\n【Quantitative Metrics】')
        self.ckp.write_log(f"  DICE: {quant_metrics['DICE']:.6f}")
        self.ckp.write_log(f"  Acc:  {quant_metrics['ACC']:.6f}  |  Se:  {quant_metrics['SE']:.6f}  |  Sp:  {quant_metrics['SP']:.6f}")
        self.ckp.write_log(f"  AUC:  {quant_metrics['AUC']:.6f}  |  BIOU: {quant_metrics['BIOU']:.6f}")
        self.ckp.write_log(f"  Vessel_IOU: {quant_metrics['Vessel_IOU']:.6f}  |  Background_IOU: {quant_metrics['Background_IOU']:.6f}")
        
        # 定性指标
        self.ckp.write_log('\n【Qualitative Metrics】')
        
        if 'width_analysis' in qual_metrics:
            w = qual_metrics['width_analysis']
            self.ckp.write_log(f"  Vessel Width: mean={w['mean']:.2f}px, thin={w['thin_vessels_ratio']*100:.1f}%, medium={w['medium_vessels_ratio']*100:.1f}%, thick={w['thick_vessels_ratio']*100:.1f}%")
        
        if 'connectivity_analysis' in qual_metrics:
            c = qual_metrics['connectivity_analysis']
            self.ckp.write_log(f"  Connectivity: components={c['num_components']:.1f}, fragmentation={c['fragmentation_index']:.4f}, largest_ratio={c['largest_component_ratio']*100:.1f}%")
        
        if 'edge_analysis' in qual_metrics:
            e = qual_metrics['edge_analysis']
            self.ckp.write_log(f"  Edge Performance: avg_prob={e['avg_edge_prob']:.2f}, low_conf={e['low_confidence_edges']*100:.1f}%")
        
        if 'probability_analysis' in qual_metrics:
            p = qual_metrics['probability_analysis']
            self.ckp.write_log(f"  Probability: vessel={p['vessel_prob_mean']:.2f}, background={p['background_prob_mean']:.2f}, separation={p['separation_score']:.2f}")
        
        if 'error_analysis' in qual_metrics:
            er = qual_metrics['error_analysis']
            self.ckp.write_log(f"  Error Analysis: FP_ratio={er['potential_fp_ratio']*100:.1f}%, FN_pixels={er['potential_fn_pixels']:.0f}")
        
        self.ckp.write_log('='*80 + '\n')
    
    def _convert_to_serializable(self, obj):
        """将numpy类型转换为Python原生类型以便JSON序列化"""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    def _save_analysis_report(self, detailed_results, avg_quant, avg_qual):
        """保存详细分析报告"""
        report = {
            'summary': {
                'quantitative': avg_quant,
                'qualitative': avg_qual
            },
            'detailed_results': detailed_results,
            'recommendations': self._generate_recommendations(avg_qual)
        }
        
        # 转换为可序列化格式
        serializable_report = self._convert_to_serializable(report)
        
        # 保存JSON
        report_path = self.analysis_dir / 'comprehensive_analysis.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        self.ckp.write_log(f'\n💾 Comprehensive analysis saved to: {report_path}')
        
        # 生成可读性报告
        self._generate_readable_report(avg_quant, avg_qual)
    
    def _generate_recommendations(self, qual_metrics):
        """基于定性分析生成改进建议"""
        recommendations = []
        
        # 检查细血管检测
        if 'width_analysis' in qual_metrics:
            thin_ratio = qual_metrics['width_analysis'].get('thin_vessels_ratio', 0)
            if thin_ratio > 0.3:
                recommendations.append({
                    'priority': 'HIGH',
                    'issue': 'High proportion of thin vessels',
                    'suggestion': 'Use multi-scale feature fusion and increase weight for thin vessel samples'
                })
        
        # 检查连续性
        if 'connectivity_analysis' in qual_metrics:
            frag_index = qual_metrics['connectivity_analysis'].get('fragmentation_index', 0)
            if frag_index > 0.01:
                recommendations.append({
                    'priority': 'HIGH',
                    'issue': 'High vessel fragmentation',
                    'suggestion': 'Use connectivity loss and morphological post-processing'
                })
        
        # 检查边缘精度
        if 'edge_analysis' in qual_metrics:
            edge_conf = qual_metrics['edge_analysis'].get('avg_edge_prob', 255)
            if edge_conf < 180:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'issue': 'Low edge confidence',
                    'suggestion': 'Use edge-enhanced loss function and increase edge region samples'
                })
        
        # 检查类别分离度
        if 'probability_analysis' in qual_metrics:
            sep_score = qual_metrics['probability_analysis'].get('separation_score', 255)
            if sep_score < 150:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'issue': 'Low class separation',
                    'suggestion': 'Improve feature extraction or use contrastive learning'
                })
        
        return recommendations
    
    def _generate_readable_report(self, avg_quant, avg_qual):
        """生成可读性报告"""
        report_path = self.analysis_dir / 'analysis_summary.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('='*80 + '\n')
            f.write('COMPREHENSIVE EVALUATION REPORT\n')
            f.write('='*80 + '\n\n')
            
            # 定量指标
            f.write('【QUANTITATIVE METRICS】\n')
            f.write('-'*80 + '\n')
            for metric, value in avg_quant.items():
                f.write(f'{metric:20s}: {value:.6f}\n')
            
            # 定性指标
            f.write('\n【QUALITATIVE METRICS】\n')
            f.write('-'*80 + '\n')
            
            for category, metrics in avg_qual.items():
                f.write(f'\n{category.replace("_", " ").title()}:\n')
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f'  {metric:30s}: {value:.4f}\n')
                    else:
                        f.write(f'  {metric:30s}: {value}\n')
            
            # 改进建议
            f.write('\n【IMPROVEMENT RECOMMENDATIONS】\n')
            f.write('-'*80 + '\n')
            recommendations = self._generate_recommendations(avg_qual)
            for i, rec in enumerate(recommendations, 1):
                f.write(f'\n{i}. [{rec["priority"]}] {rec["issue"]}\n')
                f.write(f'   Suggestion: {rec["suggestion"]}\n')
            
            f.write('\n' + '='*80 + '\n')
        
        self.ckp.write_log(f'📄 Readable report saved to: {report_path}')


def parse_args():
    """解析测试参数"""
    parser = argparse.ArgumentParser(description='Vessel Segmentation Comprehensive Testing')
    
    # 硬件设置
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--n_threads', type=int, default=6,
                        help='number of threads for data loading')
    
    # 数据设置
    parser.add_argument('--dir_data', type=str, default='../dataset',
                        help='dataset directory')
    parser.add_argument('--data_train', type=str, default='DRIVE',
                        help='train dataset name')
    parser.add_argument('--data_test', type=str, default='DRIVE',
                        help='test dataset name')
    parser.add_argument('--dataset', type=str, default='DRIVE',
                        help='dataset name: DRIVE or CHASE')
    parser.add_argument('--data_range', type=str, default='1-20/1-20',
                        help='train/test data range')
    parser.add_argument('--scale', type=str, default='1',
                        help='super resolution scale')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[584, 565],
                        help='crop size [height, width]')
    parser.add_argument('--ext', type=str, default='sep',
                        help='dataset file extension')
    parser.add_argument('--no_augment', action='store_true',
                        help='do not use data augmentation')
    parser.add_argument('--patch_size', type=int, default=192,
                        help='output patch size')
    parser.add_argument('--mark', type=str, default='',
                        help='which file to train')
    parser.add_argument('--raw2rgb', action='store_true',
                        help='use raw images as input')
    parser.add_argument('--poled', action='store_true',
                        help='use raw images as input')
    parser.add_argument('--syn', action='store_true',
                        help='testing on synthesis data')
    
    # 模型设置
    parser.add_argument('--model', default='EDSR', help='model name')
    parser.add_argument('--pre_train', type=str, default='',
                        help='pre-trained model directory')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test')
    parser.add_argument('--n_resblocks', type=int, default=16,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--n_resgroups', type=int, default=10,
                        help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')
    parser.add_argument('--act', type=str, default='relu',
                        help='activation function')
    parser.add_argument('--extend', type=str, default='.',
                        help='pre-trained model directory')
    parser.add_argument('--dilation', action='store_true',
                        help='use dilated convolution')
    
    # 测试设置
    parser.add_argument('--test_only', action='store_true',
                        help='set this option to test the model')
    parser.add_argument('--self_ensemble', action='store_true',
                        help='use self-ensemble method for test')
    parser.add_argument('--save_results', action='store_true',
                        help='save output results')
    parser.add_argument('--save_gt', action='store_true',
                        help='save ground truth images')
    parser.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')
    
    # 定性分析设置
    parser.add_argument('--save_qualitative_viz', action='store_true',
                        help='save qualitative analysis visualizations')
    parser.add_argument('--enable_qualitative', action='store_true', default=True,
                        help='enable qualitative analysis')
    
    # 日志设置
    parser.add_argument('--save', type=str, default='test',
                        help='file name to save')
    parser.add_argument('--load', type=str, default='',
                        help='file name to load')
    parser.add_argument('--resume', type=int, default=0,
                        help='resume from specific checkpoint')
    parser.add_argument('--reset', action='store_true',
                        help='reset the training')
    parser.add_argument('--print_every', type=int, default=100,
                        help='how many batches to wait before logging')
    
    # 其他兼容性参数
    parser.add_argument('--debug', action='store_true',
                        help='Enables debug mode')
    parser.add_argument('--template', default='.',
                        help='You can set various templates')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--test_every', type=int, default=1000,
                        help='do test per every N batches')
    parser.add_argument('--loss', type=str, default='1*L1',
                        help='loss function configuration')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--decay', type=str, default='200',
                        help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='learning rate decay factor')
    parser.add_argument('--optimizer', default='ADAM',
                        help='optimizer to use')
    parser.add_argument('--save_models', action='store_true',
                        help='save all intermediate models')
    
    args = parser.parse_args()
    
    # 处理参数
    args.scale = [int(x) for x in args.scale.split('+')]
    args.data_train = args.data_train.split('+')
    args.data_test = args.data_test.split('+')
    args.decay = args.decay.split('+')
    
    return args


def main():
    """主测试函数"""
    # 解析参数
    args = parse_args()
    
    # 创建checkpoint
    checkpoint = utility.checkpoint(args)
    
    # 加载数据
    print("Loading test data...")
    loader = data.Data(args)
    
    # 加载模型
    print("Loading model...")
    model = model_module.Model(args, checkpoint)
    
    # 创建测试器并执行测试
    print("Starting comprehensive testing...")
    print("  - Quantitative metrics: DICE, ACC, SE, SP, AUC, IOU, BIOU")
    print("  - Qualitative analysis: Width, Connectivity, Edge, Probability, Quality, Errors")
    
    tester = ModelTester(args, loader.loader_test, model, checkpoint)
    results = tester.test()
    
    # 打印最终结果
    print("\n" + "="*80)
    print("FINAL COMPREHENSIVE RESULTS")
    print("="*80)
    
    print("\n【Quantitative Metrics】")
    for metric, value in results['quantitative'].items():
        print(f"{metric:20s}: {value:.6f}")
    
    print("\n【Qualitative Metrics Summary】")
    for category, metrics in results['qualitative'].items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric:30s}: {value:.4f}")
    
    print("\n" + "="*80)
    print("All results saved to:", tester.analysis_dir)
    print("="*80)


if __name__ == '__main__':
    main()