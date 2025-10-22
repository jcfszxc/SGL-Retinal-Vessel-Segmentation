import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pandas as pd

class DifficultyAnalyzer:
    """分析视网膜血管分割任务中的困难样本"""
    
    def __init__(self, result_dir):
        """
        Args:
            result_dir: 结果目录路径，例如 'experiment/test_drive_1_split8/results-DRIVE/'
        """
        self.result_dir = Path(result_dir)
        self.samples = self._parse_samples()
        
    def _parse_samples(self):
        """解析目录中的样本"""
        samples = defaultdict(dict)
        
        for img_path in self.result_dir.glob('*.png'):
            parts = img_path.stem.split('_')
            sample_id = parts[0]  # 例如 '21'
            img_type = parts[-1]  # 'HR', 'PenaltyMapHard', 'teacher'
            
            samples[sample_id][img_type] = img_path
            
        return dict(samples)
    
    def calculate_difficulty_metrics(self):
        """计算每个样本的难度指标"""
        results = []
        
        for sample_id, paths in self.samples.items():
            metrics = {'sample_id': sample_id}
            
            # 1. 惩罚图的平均强度（越高越困难）
            if 'PenaltyMapHard' in paths:
                penalty_map = cv2.imread(str(paths['PenaltyMapHard']), cv2.IMREAD_GRAYSCALE)
                metrics['penalty_mean'] = penalty_map.mean()
                metrics['penalty_std'] = penalty_map.std()
                metrics['penalty_max'] = penalty_map.max()
                # 高惩罚区域的占比
                metrics['hard_region_ratio'] = (penalty_map > 128).sum() / penalty_map.size
            
            # 2. 教师预测与GT的差异（如果有HR作为参考）
            if 'teacher' in paths and 'HR' in paths:
                teacher_pred = cv2.imread(str(paths['teacher']), cv2.IMREAD_GRAYSCALE)
                hr_img = cv2.imread(str(paths['HR']), cv2.IMREAD_GRAYSCALE)
                
                # 计算预测误差
                diff = np.abs(teacher_pred.astype(float) - hr_img.astype(float))
                metrics['pred_error_mean'] = diff.mean()
                metrics['pred_error_std'] = diff.std()
                # 大误差区域占比
                metrics['large_error_ratio'] = (diff > 50).sum() / diff.size
            
            # 3. 边缘复杂度（基于HR图像）
            if 'HR' in paths:
                hr_img = cv2.imread(str(paths['HR']), cv2.IMREAD_GRAYSCALE)
                # Sobel边缘检测
                sobelx = cv2.Sobel(hr_img, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(hr_img, cv2.CV_64F, 0, 1, ksize=3)
                edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
                metrics['edge_complexity'] = edge_magnitude.mean()
            
            results.append(metrics)
        
        return pd.DataFrame(results).sort_values('sample_id')
    
    def rank_by_difficulty(self, df, method='composite'):
        """
        根据不同方法对样本难度排序
        
        Args:
            df: 包含难度指标的DataFrame
            method: 'penalty' - 仅基于惩罚图
                   'error' - 仅基于预测误差
                   'composite' - 综合评分
        """
        if method == 'penalty':
            df['difficulty_score'] = df['penalty_mean']
        elif method == 'error':
            df['difficulty_score'] = df['pred_error_mean']
        elif method == 'composite':
            # 归一化各指标后加权求和
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            features = ['penalty_mean', 'hard_region_ratio', 
                       'pred_error_mean', 'edge_complexity']
            available_features = [f for f in features if f in df.columns]
            
            normalized = scaler.fit_transform(df[available_features])
            # 等权重综合
            df['difficulty_score'] = normalized.mean(axis=1)
        
        return df.sort_values('difficulty_score', ascending=False)
    
    def visualize_difficulty_distribution(self, df, save_path=None):
        """可视化难度分布"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Sample Difficulty Analysis', fontsize=16)
        
        # 1. 综合难度得分排名
        ax = axes[0, 0]
        df_sorted = df.sort_values('difficulty_score', ascending=False)
        ax.barh(df_sorted['sample_id'], df_sorted['difficulty_score'])
        ax.set_xlabel('Difficulty Score')
        ax.set_ylabel('Sample ID')
        ax.set_title('Overall Difficulty Ranking')
        ax.invert_yaxis()
        
        # 2. 惩罚图均值分布
        if 'penalty_mean' in df.columns:
            ax = axes[0, 1]
            ax.hist(df['penalty_mean'], bins=15, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Penalty Mean')
            ax.set_ylabel('Count')
            ax.set_title('Penalty Map Distribution')
        
        # 3. 预测误差分布
        if 'pred_error_mean' in df.columns:
            ax = axes[0, 2]
            ax.hist(df['pred_error_mean'], bins=15, edgecolor='black', 
                   alpha=0.7, color='orange')
            ax.set_xlabel('Prediction Error Mean')
            ax.set_ylabel('Count')
            ax.set_title('Prediction Error Distribution')
        
        # 4. 困难区域占比 vs 预测误差
        if 'hard_region_ratio' in df.columns and 'pred_error_mean' in df.columns:
            ax = axes[1, 0]
            ax.scatter(df['hard_region_ratio'], df['pred_error_mean'], alpha=0.6)
            for _, row in df.iterrows():
                ax.annotate(row['sample_id'], 
                          (row['hard_region_ratio'], row['pred_error_mean']),
                          fontsize=8, alpha=0.7)
            ax.set_xlabel('Hard Region Ratio')
            ax.set_ylabel('Prediction Error Mean')
            ax.set_title('Hard Regions vs Prediction Error')
        
        # 5. 边缘复杂度
        if 'edge_complexity' in df.columns:
            ax = axes[1, 1]
            df_sorted_edge = df.sort_values('edge_complexity', ascending=False)
            ax.barh(df_sorted_edge['sample_id'].head(10), 
                   df_sorted_edge['edge_complexity'].head(10))
            ax.set_xlabel('Edge Complexity')
            ax.set_ylabel('Sample ID')
            ax.set_title('Top 10 Edge Complexity')
            ax.invert_yaxis()
        
        # 6. 相关性热图
        ax = axes[1, 2]
        metric_cols = [col for col in df.columns if col not in ['sample_id', 'difficulty_score']]
        if len(metric_cols) > 1:
            corr = df[metric_cols].corr()
            im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(metric_cols)))
            ax.set_yticks(range(len(metric_cols)))
            ax.set_xticklabels(metric_cols, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(metric_cols, fontsize=8)
            plt.colorbar(im, ax=ax)
            ax.set_title('Metric Correlation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def generate_difficulty_report(self, top_k=5):
        """生成困难样本报告"""
        df = self.calculate_difficulty_metrics()
        df_ranked = self.rank_by_difficulty(df, method='composite')
        
        print("=" * 80)
        print("困难样本分析报告".center(80))
        print("=" * 80)
        print(f"\n总样本数: {len(df)}")
        print(f"\n{'='*80}")
        print(f"Top {top_k} 最困难样本:")
        print("="*80)
        
        for i, (_, row) in enumerate(df_ranked.head(top_k).iterrows(), 1):
            print(f"\n排名 #{i}: 样本 {row['sample_id']}")
            print(f"  综合难度得分: {row['difficulty_score']:.4f}")
            if 'penalty_mean' in row:
                print(f"  惩罚图均值: {row['penalty_mean']:.2f}")
            if 'hard_region_ratio' in row:
                print(f"  困难区域占比: {row['hard_region_ratio']:.2%}")
            if 'pred_error_mean' in row:
                print(f"  预测误差均值: {row['pred_error_mean']:.2f}")
            if 'edge_complexity' in row:
                print(f"  边缘复杂度: {row['edge_complexity']:.2f}")
        
        print(f"\n{'='*80}")
        print(f"Top {top_k} 最简单样本:")
        print("="*80)
        
        for i, (_, row) in enumerate(df_ranked.tail(top_k).iterrows(), 1):
            print(f"\n排名 #{i}: 样本 {row['sample_id']}")
            print(f"  综合难度得分: {row['difficulty_score']:.4f}")
        
        print("\n" + "="*80)
        print("统计摘要:")
        print("="*80)
        print(df_ranked.describe())
        
        return df_ranked
    
    def visualize_samples(self, sample_ids, save_dir=None):
        """可视化指定样本的各类图像"""
        n_samples = len(sample_ids)
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample_id in enumerate(sample_ids):
            if sample_id not in self.samples:
                print(f"警告: 样本 {sample_id} 不存在")
                continue
            
            paths = self.samples[sample_id]
            
            # HR图像
            if 'HR' in paths:
                img = cv2.imread(str(paths['HR']), cv2.IMREAD_GRAYSCALE)
                axes[i, 0].imshow(img, cmap='gray')
                axes[i, 0].set_title(f'Sample {sample_id} - HR/GT')
                axes[i, 0].axis('off')
            
            # Teacher预测
            if 'teacher' in paths:
                img = cv2.imread(str(paths['teacher']), cv2.IMREAD_GRAYSCALE)
                axes[i, 1].imshow(img, cmap='gray')
                axes[i, 1].set_title(f'Sample {sample_id} - Teacher Pred')
                axes[i, 1].axis('off')
            
            # 惩罚图
            if 'PenaltyMapHard' in paths:
                img = cv2.imread(str(paths['PenaltyMapHard']), cv2.IMREAD_GRAYSCALE)
                axes[i, 2].imshow(img, cmap='hot')
                axes[i, 2].set_title(f'Sample {sample_id} - Penalty Map')
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'sample_visualization.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"样本可视化已保存到: {save_path}")
        
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 设置结果目录路径
    result_dir = "experiment/test_drive_1_split8/results-DRIVE/"
    
    # 创建分析器
    analyzer = DifficultyAnalyzer(result_dir)
    
    # 生成报告
    df_ranked = analyzer.generate_difficulty_report(top_k=5)
    
    # 可视化分布
    analyzer.visualize_difficulty_distribution(df_ranked, 
                                               save_path='difficulty_analysis.png')
    
    # 可视化最困难的3个样本
    top_difficult = df_ranked.head(3)['sample_id'].tolist()
    analyzer.visualize_samples(top_difficult, save_dir='.')
    
    # 保存详细结果
    df_ranked.to_csv('difficulty_ranking.csv', index=False)
    print("\n详细排名已保存到: difficulty_ranking.csv")