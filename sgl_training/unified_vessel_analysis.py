import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage import morphology
from pathlib import Path
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class VesselSegmentationAnalyzer:
    """血管分割单方法分析器"""
    
    def __init__(self, results_dir: str, method_name: str = None):
        """
        初始化分析器
        
        Args:
            results_dir: 结果图像目录
            method_name: 方法名称（可选，用于标识）
        """
        self.results_dir = Path(results_dir)
        self.method_name = method_name or self.results_dir.parent.name
        self.analysis_results = []
        
    def load_image_set(self, image_id: str) -> Dict:
        """
        加载一组图像（同一个ID的所有类型）
        
        支持的文件命名模式：
        - {id}_training_x1_HR.png (原始图像)
        - {id}_training_x1_teacher.png (分割结果/Ground Truth)
        - {id}_training_x1_PenaltyMapHard.png (困难区域图)
        """
        images = {}
        
        # 定义文件后缀映射
        suffix_mapping = {
            'HR': 'Raw',  # 原始高分辨率图像
            'teacher': 'Vessel',  # 分割结果
            'PenaltyMapHard': 'PenaltyMapHard',  # 惩罚图
        }
        
        # 尝试加载每种类型的图像
        for file_suffix, internal_name in suffix_mapping.items():
            pattern = f"{image_id}_*_{file_suffix}.png"
            files = list(self.results_dir.glob(pattern))
            
            if files:
                img = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images[internal_name] = img
        
        # 如果没有找到teacher，尝试找Vessel或其他可能的分割结果
        if 'Vessel' not in images:
            for pattern in [f"{image_id}_*_Vessel.png", f"{image_id}_*_vessel.png"]:
                files = list(self.results_dir.glob(pattern))
                if files:
                    img = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images['Vessel'] = img
                        break
        
        return images
    
    def analyze_vessel_width_distribution(self, vessel_img: np.ndarray) -> tuple:
        """分析血管宽度分布"""
        binary = (vessel_img > 127).astype(np.uint8)
        
        if np.sum(binary) == 0:
            return {
                'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                'thin_vessels_ratio': 0, 'medium_vessels_ratio': 0, 'thick_vessels_ratio': 0
            }, np.array([])
        
        dist_transform = ndimage.distance_transform_edt(binary)
        vessel_widths = dist_transform[binary > 0] * 2
        
        width_stats = {
            'mean': float(np.mean(vessel_widths)),
            'std': float(np.std(vessel_widths)),
            'min': float(np.min(vessel_widths)),
            'max': float(np.max(vessel_widths)),
            'thin_vessels_ratio': float(np.sum(vessel_widths < 3) / len(vessel_widths)),
            'medium_vessels_ratio': float(np.sum((vessel_widths >= 3) & (vessel_widths < 8)) / len(vessel_widths)),
            'thick_vessels_ratio': float(np.sum(vessel_widths >= 8) / len(vessel_widths))
        }
        
        return width_stats, vessel_widths
    
    def analyze_connectivity(self, vessel_img: np.ndarray) -> tuple:
        """分析血管连通性"""
        binary = (vessel_img > 127).astype(np.uint8)
        
        labeled, num_components = ndimage.label(binary)
        component_sizes = ndimage.sum(binary, labeled, range(1, num_components + 1))
        
        skeleton = morphology.skeletonize(binary)
        skeleton_length = int(np.sum(skeleton))
        
        connectivity_stats = {
            'num_components': int(num_components),
            'avg_component_size': float(np.mean(component_sizes)) if len(component_sizes) > 0 else 0,
            'largest_component_size': int(np.max(component_sizes)) if len(component_sizes) > 0 else 0,
            'skeleton_length': skeleton_length,
            'fragmentation_index': float(num_components / (skeleton_length + 1))
        }
        
        return connectivity_stats, skeleton
    
    def analyze_edge_performance(self, vessel_img: np.ndarray) -> tuple:
        """分析边缘区域的性能"""
        binary = (vessel_img > 127).astype(np.uint8)
        edges = cv2.Canny(binary * 255, 50, 150)
        
        # 创建伪概率图（基于距离变换）
        dist = ndimage.distance_transform_edt(binary)
        pseudo_prob = np.clip(dist * 50, 0, 255).astype(np.uint8)
        
        kernel = np.ones((5, 5), np.uint8)
        edge_zone = cv2.dilate(edges, kernel, iterations=1)
        edge_probs = pseudo_prob[edge_zone > 0]
        
        edge_stats = {
            'edge_length': int(np.sum(edges > 0)),
            'avg_edge_prob': float(np.mean(edge_probs)) if len(edge_probs) > 0 else 0,
            'std_edge_prob': float(np.std(edge_probs)) if len(edge_probs) > 0 else 0,
            'low_confidence_edges': float(np.sum(edge_probs < 150) / len(edge_probs)) if len(edge_probs) > 0 else 0
        }
        
        return edge_stats, edges
    
    def analyze_probability_distribution(self, vessel_img: np.ndarray) -> Dict:
        """分析概率分布（基于距离变换的伪概率）"""
        binary = (vessel_img > 127).astype(np.uint8)
        
        # 创建伪概率图
        dist_vessel = ndimage.distance_transform_edt(binary)
        dist_background = ndimage.distance_transform_edt(1 - binary)
        
        prob_img = np.zeros_like(vessel_img, dtype=np.float32)
        prob_img[binary > 0] = np.clip(200 + dist_vessel[binary > 0] * 10, 0, 255)
        prob_img[binary == 0] = np.clip(100 - dist_background[binary == 0] * 20, 0, 100)
        
        vessel_probs = prob_img[binary > 0]
        background_probs = prob_img[binary == 0]
        
        prob_stats = {
            'vessel_prob_mean': float(np.mean(vessel_probs)) if len(vessel_probs) > 0 else 0,
            'vessel_prob_std': float(np.std(vessel_probs)) if len(vessel_probs) > 0 else 0,
            'background_prob_mean': float(np.mean(background_probs)) if len(background_probs) > 0 else 0,
            'background_prob_std': float(np.std(background_probs)) if len(background_probs) > 0 else 0,
            'separation_score': float(np.mean(vessel_probs) - np.mean(background_probs)) if len(vessel_probs) > 0 and len(background_probs) > 0 else 0,
            'uncertain_vessel_ratio': float(np.sum((vessel_probs > 100) & (vessel_probs < 200)) / len(vessel_probs)) if len(vessel_probs) > 0 else 0
        }
        
        return prob_stats
    
    def find_false_positives_negatives(self, vessel_img: np.ndarray) -> tuple:
        """寻找可能的假阳性和假阴性区域"""
        binary = (vessel_img > 127).astype(np.uint8)
        
        # 创建伪概率用于分析
        dist_vessel = ndimage.distance_transform_edt(binary)
        pseudo_prob = np.clip(dist_vessel * 50, 0, 255).astype(np.uint8)
        
        potential_fp = (binary > 0) & (pseudo_prob < 180)
        potential_fn = (binary == 0) & (pseudo_prob > 100)
        
        vessel_pixels = np.sum(binary > 0)
        background_pixels = np.sum(binary == 0)
        
        fp_fn_stats = {
            'potential_fp_ratio': float(np.sum(potential_fp) / vessel_pixels) if vessel_pixels > 0 else 0,
            'potential_fn_pixels': int(np.sum(potential_fn)),
            'potential_fn_ratio': float(np.sum(potential_fn) / background_pixels) if background_pixels > 0 else 0
        }
        
        return fp_fn_stats, potential_fp, potential_fn
    
    def analyze_single_image(self, image_id: str, verbose: bool = True) -> Optional[Dict]:
        """分析单张图像"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"分析图像: {image_id}")
            print(f"{'='*60}")
        
        images = self.load_image_set(image_id)
        
        if 'Vessel' not in images:
            print(f"⚠️  图像 {image_id} 缺少分割结果文件")
            return None
        
        results = {'image_id': image_id}
        
        # 1. 血管宽度分析
        width_stats, widths = self.analyze_vessel_width_distribution(images['Vessel'])
        results['width_analysis'] = width_stats
        if verbose:
            print(f"\n1. 血管宽度分布:")
            print(f"   平均宽度: {width_stats['mean']:.2f} pixels")
            print(f"   细血管比例: {width_stats['thin_vessels_ratio']*100:.1f}%")
        
        # 2. 连通性分析
        conn_stats, skeleton = self.analyze_connectivity(images['Vessel'])
        results['connectivity_analysis'] = conn_stats
        if verbose:
            print(f"\n2. 血管连通性:")
            print(f"   连通分量: {conn_stats['num_components']}")
            print(f"   碎片化指数: {conn_stats['fragmentation_index']:.4f}")
        
        # 3. 边缘性能分析
        edge_stats, edges = self.analyze_edge_performance(images['Vessel'])
        results['edge_analysis'] = edge_stats
        if verbose:
            print(f"\n3. 边缘性能:")
            print(f"   边缘长度: {edge_stats['edge_length']}")
            print(f"   平均置信度: {edge_stats['avg_edge_prob']:.2f}")
        
        # 4. 概率分布分析
        prob_stats = self.analyze_probability_distribution(images['Vessel'])
        results['probability_analysis'] = prob_stats
        if verbose:
            print(f"\n4. 概率分布:")
            print(f"   分离度得分: {prob_stats['separation_score']:.2f}")
        
        # 5. 假阳性/假阴性分析
        fp_fn_stats, potential_fp, potential_fn = self.find_false_positives_negatives(images['Vessel'])
        results['error_analysis'] = fp_fn_stats
        if verbose:
            print(f"\n5. 错误分析:")
            print(f"   潜在假阳性: {fp_fn_stats['potential_fp_ratio']*100:.1f}%")
        
        return results
    
    def analyze_all_images(self, verbose: bool = True) -> List[Dict]:
        """分析所有图像"""
        # 获取所有teacher文件作为分割结果
        all_files = list(self.results_dir.glob("*_teacher.png"))
        
        if len(all_files) == 0:
            all_files = list(self.results_dir.glob("*_Vessel.png"))
        
        # 提取图像ID
        image_ids = []
        for f in all_files:
            # 从 "21_training_x1_teacher.png" 中提取 "21"
            parts = f.stem.split('_')
            if len(parts) > 0:
                image_ids.append(parts[0])
        
        image_ids = sorted(set(image_ids))
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"方法: {self.method_name}")
            print(f"找到 {len(image_ids)} 张图像待分析")
            print(f"{'='*60}")
        
        self.analysis_results = []
        for img_id in image_ids:
            result = self.analyze_single_image(img_id, verbose=False)
            if result:
                self.analysis_results.append(result)
        
        if verbose:
            print(f"✓ 成功分析 {len(self.analysis_results)} 张图像")
        
        return self.analysis_results
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """保存分析结果为JSON"""
        if output_path is None:
            output_path = self.results_dir / 'analysis_summary.json'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 结果已保存: {output_path}")
        return str(output_path)


class VesselMethodComparison:
    """多方法对比分析器"""
    
    def __init__(self):
        self.methods_data = {}
        self.metrics_config = {
            'fragmentation_index': {
                'name': 'Fragmentation Index',
                'name_zh': '碎片化指数',
                'lower_is_better': True,
                'weight': 0.20
            },
            'edge_confidence': {
                'name': 'Edge Confidence',
                'name_zh': '边缘置信度',
                'lower_is_better': False,
                'weight': 0.15
            },
            'separation_score': {
                'name': 'Separation Score',
                'name_zh': '分离度得分',
                'lower_is_better': False,
                'weight': 0.20
            },
            'thin_vessels_ratio': {
                'name': 'Thin Vessels Ratio',
                'name_zh': '细血管占比',
                'lower_is_better': False,
                'weight': 0.15
            },
            'num_components': {
                'name': 'Components Count',
                'name_zh': '连通组件数',
                'lower_is_better': True,
                'weight': 0.10
            },
            'largest_component_ratio': {
                'name': 'Largest Component',
                'name_zh': '最大组件占比',
                'lower_is_better': False,
                'weight': 0.10
            },
            'false_positive_rate': {
                'name': 'False Positive Rate',
                'name_zh': '假阳性率',
                'lower_is_better': True,
                'weight': 0.10
            }
        }
    
    def load_method_results(self, method_name: str, results_path: str) -> bool:
        """加载单个方法的结果"""
        print(f"📂 加载方法: {method_name}")
        
        results_path = Path(results_path)
        
        if results_path.is_file():
            json_path = results_path
        else:
            json_path = results_path / 'analysis_summary.json'
        
        if not json_path.exists():
            print(f"  ⚠️  文件不存在: {json_path}")
            return False
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list) or len(data) == 0:
                print(f"  ⚠️  数据格式错误或为空")
                return False
            
            self.methods_data[method_name] = data
            print(f"  ✓ 成功加载 {len(data)} 张图像的数据")
            return True
            
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            return False
    
    def extract_metrics(self, method_data: List[Dict]) -> Dict:
        """提取关键指标的统计信息"""
        metrics = {}
        
        # 碎片化指数
        frag_indices = [img['connectivity_analysis']['fragmentation_index'] for img in method_data]
        metrics['fragmentation_index'] = {
            'mean': np.mean(frag_indices), 'std': np.std(frag_indices),
            'min': np.min(frag_indices), 'max': np.max(frag_indices)
        }
        
        # 边缘置信度
        edge_confs = [img['edge_analysis']['avg_edge_prob'] for img in method_data]
        metrics['edge_confidence'] = {
            'mean': np.mean(edge_confs), 'std': np.std(edge_confs),
            'min': np.min(edge_confs), 'max': np.max(edge_confs)
        }
        
        # 分离度得分
        sep_scores = [img['probability_analysis']['separation_score'] for img in method_data]
        metrics['separation_score'] = {
            'mean': np.mean(sep_scores), 'std': np.std(sep_scores),
            'min': np.min(sep_scores), 'max': np.max(sep_scores)
        }
        
        # 细血管占比
        thin_ratios = [img['width_analysis']['thin_vessels_ratio'] for img in method_data]
        metrics['thin_vessels_ratio'] = {
            'mean': np.mean(thin_ratios), 'std': np.std(thin_ratios),
            'min': np.min(thin_ratios), 'max': np.max(thin_ratios)
        }
        
        # 连通组件数
        num_comps = [img['connectivity_analysis']['num_components'] for img in method_data]
        metrics['num_components'] = {
            'mean': np.mean(num_comps), 'std': np.std(num_comps),
            'min': np.min(num_comps), 'max': np.max(num_comps)
        }
        
        # 最大组件占比
        largest_ratios = []
        for img in method_data:
            largest = img['connectivity_analysis']['largest_component_size']
            total = img['connectivity_analysis']['skeleton_length']
            ratio = largest / total if total > 0 else 0
            largest_ratios.append(ratio)
        metrics['largest_component_ratio'] = {
            'mean': np.mean(largest_ratios), 'std': np.std(largest_ratios),
            'min': np.min(largest_ratios), 'max': np.max(largest_ratios)
        }
        
        # 假阳性率
        fp_rates = [img['error_analysis']['potential_fp_ratio'] for img in method_data]
        metrics['false_positive_rate'] = {
            'mean': np.mean(fp_rates), 'std': np.std(fp_rates),
            'min': np.min(fp_rates), 'max': np.max(fp_rates)
        }
        
        return metrics
    
    def calculate_overall_score(self, metrics: Dict) -> float:
        """计算综合评分"""
        all_values = {metric: [] for metric in self.metrics_config.keys()}
        
        for method_data in self.methods_data.values():
            method_metrics = self.extract_metrics(method_data)
            for metric in self.metrics_config.keys():
                all_values[metric].append(method_metrics[metric]['mean'])
        
        score = 0
        for metric, config in self.metrics_config.items():
            values = all_values[metric]
            min_val, max_val = min(values), max(values)
            
            if max_val - min_val < 1e-10:
                normalized = 0.5
            else:
                current_val = metrics[metric]['mean']
                if config['lower_is_better']:
                    normalized = 1 - (current_val - min_val) / (max_val - min_val)
                else:
                    normalized = (current_val - min_val) / (max_val - min_val)
            
            score += normalized * config['weight']
        
        return score * 100
    
    def generate_leaderboard(self) -> tuple:
        """生成排行榜"""
        print("\n" + "="*100)
        print("🏆  METHOD PERFORMANCE LEADERBOARD  🏆".center(100))
        print("="*100)
        
        rows = []
        for method_name, method_data in self.methods_data.items():
            metrics = self.extract_metrics(method_data)
            overall_score = self.calculate_overall_score(metrics)
            
            row = {
                'Method': method_name,
                'Score': f"{overall_score:.1f}",
                'Frag.Idx': f"{metrics['fragmentation_index']['mean']:.4f}",
                'Edge': f"{metrics['edge_confidence']['mean']:.1f}",
                'Sep.': f"{metrics['separation_score']['mean']:.1f}",
                'Thin%': f"{metrics['thin_vessels_ratio']['mean']*100:.1f}",
                'Comps': f"{metrics['num_components']['mean']:.0f}",
                'FP%': f"{metrics['false_positive_rate']['mean']*100:.1f}",
                '_score': overall_score
            }
            rows.append(row)
        
        df = pd.DataFrame(rows).sort_values('_score', ascending=False)
        df = df.drop('_score', axis=1).reset_index(drop=True)
        df.index = df.index + 1
        
        print("\n" + df.to_string())
        print("\n" + "="*100)
        
        return df
    
    def visualize_comparison(self, save_path: Optional[str] = None):
        """可视化对比结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Methods Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_list = list(self.metrics_config.keys())[:6]
        
        for idx, metric in enumerate(metrics_list):
            ax = axes[idx // 3, idx % 3]
            config = self.metrics_config[metric]
            
            values, names = [], []
            for method_name, method_data in self.methods_data.items():
                metrics = self.extract_metrics(method_data)
                values.append(metrics[metric]['mean'])
                names.append(method_name)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
            bars = ax.bar(range(len(names)), values, color=colors, edgecolor='black', alpha=0.8)
            
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
            ax.set_title(config['name'], fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            best_idx = np.argmin(values) if config['lower_is_better'] else np.argmax(values)
            ax.axhline(values[best_idx], color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 可视化已保存: {save_path}")
        
        plt.show()


def main():
    """统一分析流程"""
    print("="*100)
    print("🔬 Retinal Vessel Segmentation Unified Analysis Tool 🔬".center(100))
    print("="*100)
    
    # ==================== 配置区 ====================
    # 定义要分析的方法及其结果目录
    methods_config = {
        'Baseline': '../experiment/test_drive_1_split8_baseline/results-DRIVE',
        'Baseline2': '../experiment/test_drive_1_split8_baseline/results-DRIVE',
        # 'Method-2': '../experiment/method2/results-DRIVE',
        # 添加更多方法...
    }
    
    output_dir = Path('../experiment/unified_analysis_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    # ===============================================
    
    print(f"\n📋 将分析 {len(methods_config)} 个方法\n")
    
    # 步骤1: 对每个方法进行单独分析
    print("\n" + "="*100)
    print("STEP 1: Individual Method Analysis".center(100))
    print("="*100)
    
    json_paths = {}
    for method_name, results_dir in methods_config.items():
        print(f"\n{'─'*100}")
        print(f"分析方法: {method_name}")
        print(f"{'─'*100}")
        
        analyzer = VesselSegmentationAnalyzer(results_dir, method_name)
        analyzer.analyze_all_images(verbose=True)
        
        json_path = output_dir / f'{method_name}_analysis.json'
        analyzer.save_results(json_path)
        json_paths[method_name] = json_path
    
    # 步骤2: 多方法对比分析
    if len(methods_config) >= 2:
        print("\n" + "="*100)
        print("STEP 2: Methods Comparison".center(100))
        print("="*100)
        
        comparator = VesselMethodComparison()
        
        for method_name, json_path in json_paths.items():
            comparator.load_method_results(method_name, str(json_path))
        
        # 生成排行榜
        leaderboard = comparator.generate_leaderboard()
        leaderboard.to_csv(output_dir / 'leaderboard.csv')
        
        # 生成可视化
        comparator.visualize_comparison(str(output_dir / 'comparison_chart.png'))
    
    print("\n" + "="*100)
    print("✅ 分析完成！".center(100))
    print(f"结果保存在: {output_dir}".center(100))
    print("="*100)


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
    main()