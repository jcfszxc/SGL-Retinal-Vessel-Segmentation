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
            result_dir: 结果目录路径，例如 '../experiment/test_drive_1_split8/results-DRIVE/'
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
    
    def visualize_difficulty_distribution(self, df):
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
    
    def visualize_samples(self, sample_ids):
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


class DifficultyAwareSGL:
    """
    基于样本难度的Study Group Learning改进策略
    核心思想：困难样本在多个split间共享，简单样本固定在某个split
    """
    
    def __init__(self, df_ranked):
        """
        Args:
            df_ranked: 包含难度排名的DataFrame
        """
        self.df = df_ranked.sort_values('sample_id').reset_index(drop=True)
        # 创建ID映射：从原始sample_id映射到1-N的索引
        self.df['index_id'] = range(1, len(self.df) + 1)
        self.id_mapping = dict(zip(self.df['sample_id'], self.df['index_id']))
        
    @staticmethod
    def _list_to_range_str(sample_list):
        """
        将样本列表转换为范围字符串
        例如 [1,2,3,5,6,8] -> '1-3-5-6-8-8'
        每两个数字表示一个连续范围的起始和结束
        """
        if not sample_list:
            return ""
        
        sorted_list = sorted([int(s) for s in sample_list])
        
        # 找出所有连续的数字段
        ranges = []
        start = sorted_list[0]
        end = sorted_list[0]
        
        for i in range(1, len(sorted_list)):
            if sorted_list[i] == end + 1:
                # 连续
                end = sorted_list[i]
            else:
                # 不连续，保存当前范围
                ranges.append((start, end))
                start = sorted_list[i]
                end = sorted_list[i]
        
        # 保存最后一个范围
        ranges.append((start, end))
        
        # 将范围转换为字符串格式
        range_strs = []
        for start, end in ranges:
            range_strs.extend([str(start), str(end)])
        
        return '-'.join(range_strs)
        
    def fuzzy_membership_assignment(self, num_splits=7, overlap_ratio=0.3):
        """
        为每个样本分配模糊隶属度
        
        Args:
            num_splits: SGL的split数量
            overlap_ratio: 困难样本的重叠比例（0-1）
        
        Returns:
            membership_dict: {index_id: [split1_weight, split2_weight, ...]}
        """
        membership = {}
        
        for idx, row in self.df.iterrows():
            index_id = str(int(row['index_id']))
            diff_score = row['difficulty_score']
            
            # 基础策略：每个样本主要属于一个split
            primary_split = int(idx) % num_splits
            weights = np.zeros(num_splits)
            weights[primary_split] = 1.0
            
            # 困难样本：增加到相邻split的隶属度
            if diff_score > self.df['difficulty_score'].quantile(0.7):  # Top 30%困难样本
                # 向前后各一个split扩展
                neighbor_splits = [
                    (primary_split - 1) % num_splits,
                    (primary_split + 1) % num_splits
                ]
                
                for neighbor in neighbor_splits:
                    weights[neighbor] = overlap_ratio
                
                # 重新归一化
                weights = weights / weights.sum()
            
            membership[index_id] = weights.tolist()
        
        return membership
    
    def adaptive_curriculum_strategy(self, num_splits=7):
        """
        自适应课程学习策略
        前期训练：使用简单样本
        后期训练：逐渐引入困难样本
        
        Returns:
            curriculum_plan: {split_idx: {'easy': [...], 'medium': [...], 'hard': [...]}}
        """
        # 按难度分为三档（使用index_id）
        quantiles = self.df['difficulty_score'].quantile([0.33, 0.67])
        
        easy_samples = self.df[self.df['difficulty_score'] <= quantiles[0.33]]['index_id'].tolist()
        medium_samples = self.df[
            (self.df['difficulty_score'] > quantiles[0.33]) & 
            (self.df['difficulty_score'] <= quantiles[0.67])
        ]['index_id'].tolist()
        hard_samples = self.df[self.df['difficulty_score'] > quantiles[0.67]]['index_id'].tolist()
        
        curriculum = {}
        for split_idx in range(num_splits):
            phase_ratio = split_idx / num_splits  # 0 -> 1
            
            # 早期split：主要用简单样本
            # 后期split：包含所有样本
            if phase_ratio < 0.3:  # 前30%的split
                curriculum[split_idx] = {
                    'easy': easy_samples,
                    'medium': medium_samples[:len(medium_samples)//2],
                    'hard': []
                }
            elif phase_ratio < 0.7:  # 中间split
                curriculum[split_idx] = {
                    'easy': easy_samples,
                    'medium': medium_samples,
                    'hard': hard_samples[:len(hard_samples)//2]
                }
            else:  # 后期split
                curriculum[split_idx] = {
                    'easy': easy_samples,
                    'medium': medium_samples,
                    'hard': hard_samples
                }
        
        return curriculum
    
    def difficulty_balanced_split(self, num_splits=7, test_size=3):
        """
        难度平衡的数据划分
        确保每个split的测试集难度分布相似
        
        Returns:
            splits: [{train: [...], test: [...]}, ...]
        """
        # 按难度排序（使用index_id）
        sorted_samples = self.df.sort_values('difficulty_score')['index_id'].tolist()
        
        splits = []
        for i in range(num_splits):
            # 分层抽样：从难度分布中均匀抽取测试样本
            test_indices = list(range(i, len(sorted_samples), num_splits))[:test_size]
            test_samples = [sorted_samples[idx] for idx in test_indices]
            
            train_samples = [s for s in sorted_samples if s not in test_samples]
            
            # 计算测试集难度（需要找回对应的difficulty_score）
            test_df = self.df[self.df['index_id'].isin(test_samples)]
            
            splits.append({
                'train': [int(s) for s in train_samples],
                'test': [int(s) for s in test_samples],
                'test_difficulty_mean': test_df['difficulty_score'].mean(),
                'test_difficulty_std': test_df['difficulty_score'].std()
            })
        
        return splits
    
    def visualize_strategy_comparison(self):
        """可视化不同策略的效果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 原始分布
        ax = axes[0, 0]
        ax.scatter(self.df['index_id'], self.df['difficulty_score'], 
                  c=self.df['difficulty_score'], cmap='RdYlGn_r', s=100)
        ax.set_xlabel('Sample Index ID')
        ax.set_ylabel('Difficulty Score')
        ax.set_title('Original Difficulty Distribution')
        ax.grid(True, alpha=0.3)
        
        # 2. 模糊隶属度热图
        ax = axes[0, 1]
        membership = self.fuzzy_membership_assignment(num_splits=7)
        membership_matrix = np.array([membership[str(int(iid))] for iid in self.df['index_id']])
        im = ax.imshow(membership_matrix.T, aspect='auto', cmap='YlOrRd')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Split Index')
        ax.set_title('Fuzzy Membership Heatmap')
        plt.colorbar(im, ax=ax, label='Membership Weight')
        
        # 3. 课程学习样本分配
        ax = axes[1, 0]
        curriculum = self.adaptive_curriculum_strategy(num_splits=7)
        easy_counts = [len(curriculum[i]['easy']) for i in range(7)]
        medium_counts = [len(curriculum[i]['medium']) for i in range(7)]
        hard_counts = [len(curriculum[i]['hard']) for i in range(7)]
        
        x = np.arange(7)
        ax.bar(x, easy_counts, label='Easy', color='green', alpha=0.7)
        ax.bar(x, medium_counts, bottom=easy_counts, label='Medium', color='orange', alpha=0.7)
        ax.bar(x, hard_counts, bottom=np.array(easy_counts)+np.array(medium_counts), 
               label='Hard', color='red', alpha=0.7)
        ax.set_xlabel('Split Index')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Curriculum Learning Strategy')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 平衡划分的测试集难度
        ax = axes[1, 1]
        splits = self.difficulty_balanced_split(num_splits=7)
        test_means = [s['test_difficulty_mean'] for s in splits]
        test_stds = [s['test_difficulty_std'] for s in splits]
        
        ax.bar(range(7), test_means, yerr=test_stds, capsize=5, alpha=0.7, color='steelblue')
        ax.axhline(self.df['difficulty_score'].mean(), color='red', 
                  linestyle='--', label='Overall Mean')
        ax.set_xlabel('Split Index')
        ax.set_ylabel('Test Set Difficulty Mean')
        ax.set_title('Balanced Split Test Set Difficulty')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
    
    def generate_training_script(self, num_splits=7, test_size=3):
        """
        生成完整的训练脚本，包含三种策略
        
        Args:
            num_splits: split数量
            test_size: 每个split的测试样本数
            
        Returns:
            script_content: 脚本内容字符串
        """
        script_lines = ["#!/bin/bash", ""]
        script_lines.append("# ========================================")
        script_lines.append("# 基于样本难度的SGL训练脚本")
        script_lines.append("# 包含三种策略: 课程学习、难度平衡、困难样本共享")
        script_lines.append("# ========================================")
        script_lines.append("")
        
        # 策略1: 课程学习 (Curriculum Learning)
        script_lines.append("# ========================================")
        script_lines.append("# 策略1: 课程学习 - 从简单到困难逐步训练")
        script_lines.append("# ========================================")
        script_lines.append("")
        curriculum_splits = self._generate_curriculum_splits(num_splits, test_size)
        script_lines.extend(self._format_training_commands(curriculum_splits, "curriculum"))
        
        # 策略2: 难度平衡 (Difficulty Balanced)
        script_lines.append("")
        script_lines.append("# ========================================")
        script_lines.append("# 策略2: 难度平衡 - 每个split测试集难度相似")
        script_lines.append("# ========================================")
        script_lines.append("")
        balanced_splits = self._generate_balanced_splits(num_splits, test_size)
        script_lines.extend(self._format_training_commands(balanced_splits, "balanced"))
        
        # 策略3: 困难样本共享 (Hard Sample Sharing)
        script_lines.append("")
        script_lines.append("# ========================================")
        script_lines.append("# 策略3: 困难样本共享 - 困难样本在多个split中训练")
        script_lines.append("# ========================================")
        script_lines.append("")
        sharing_splits = self._generate_sharing_splits(num_splits, test_size)
        script_lines.extend(self._format_training_commands(sharing_splits, "sharing"))
        
        return '\n'.join(script_lines)
    
    def _generate_curriculum_splits(self, num_splits, test_size):
        """生成课程学习策略的数据划分"""
        curriculum = self.adaptive_curriculum_strategy(num_splits)
        splits = []
        
        for split_idx in range(num_splits):
            all_train = (curriculum[split_idx]['easy'] + 
                        curriculum[split_idx]['medium'] + 
                        curriculum[split_idx]['hard'])
            
            # 从训练集中取最后test_size个作为测试
            test_samples = all_train[-test_size:] if len(all_train) >= test_size else all_train[-len(all_train):]
            train_samples = [s for s in all_train if s not in test_samples]
            
            splits.append({
                'train': train_samples,
                'test': test_samples,
                'info': f"Easy={len(curriculum[split_idx]['easy'])}, Medium={len(curriculum[split_idx]['medium'])}, Hard={len(curriculum[split_idx]['hard'])}"
            })
        
        return splits
    
    def _generate_balanced_splits(self, num_splits, test_size):
        """生成难度平衡策略的数据划分"""
        return self.difficulty_balanced_split(num_splits, test_size)
    
    def _generate_sharing_splits(self, num_splits, test_size):
        """生成困难样本共享策略的数据划分"""
        # 按难度分为三档（使用index_id）
        quantiles = self.df['difficulty_score'].quantile([0.33, 0.67])
        
        easy_samples = self.df[self.df['difficulty_score'] <= quantiles[0.33]]['index_id'].tolist()
        medium_samples = self.df[
            (self.df['difficulty_score'] > quantiles[0.33]) & 
            (self.df['difficulty_score'] <= quantiles[0.67])
        ]['index_id'].tolist()
        hard_samples = self.df[self.df['difficulty_score'] > quantiles[0.67]]['index_id'].tolist()
        
        all_samples = self.df['index_id'].tolist()
        
        splits = []
        for i in range(num_splits):
            # 测试集: 从所有样本中均匀抽取
            test_indices = list(range(i, len(all_samples), num_splits))[:test_size]
            test_samples = [all_samples[idx] for idx in test_indices]
            
            # 训练集: 简单样本正常分配，困难样本添加到相邻split
            base_train = [s for s in all_samples if s not in test_samples]
            
            # 困难样本共享：添加前后split的困难样本
            prev_split = (i - 1) % num_splits
            next_split = (i + 1) % num_splits
            
            # 从困难样本中选择属于相邻split的样本
            shared_hard = []
            for hard_sample in hard_samples:
                sample_idx = all_samples.index(hard_sample)
                primary_split = sample_idx % num_splits
                if primary_split in [prev_split, next_split]:
                    shared_hard.append(hard_sample)
            
            train_samples = list(set(base_train + shared_hard))
            
            splits.append({
                'train': train_samples,
                'test': test_samples,
                'info': f"Base={len(base_train)}, +SharedHard={len(shared_hard)}"
            })
        
        return splits
    
    def _format_training_commands(self, splits, strategy_name):
        """格式化训练命令"""
        lines = []
        
        for idx, split in enumerate(splits):
            train_range = self._list_to_range_str(split['train'])
            test_range = self._list_to_range_str(split['test'])
            
            info = split.get('info', '')
            if 'test_difficulty_mean' in split:
                info = f"TestDiff: mean={split['test_difficulty_mean']:.3f}, std={split['test_difficulty_std']:.3f}"
            
            # 训练命令
            lines.append(f"# Split {idx+1}: {info}")
            lines.append(f"python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \\")
            lines.append(f"  --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \\")
            lines.append(f"  --data_range '{train_range}/{test_range}' \\")
            lines.append(f"  --save drive_{strategy_name}_split{idx+1} \\")
            lines.append(f"  --scale 1 --patch_size 256 --reset")
            lines.append("")
            
        # 测试命令（生成伪标签）
        lines.append(f"# Testing to obtain pseudo labels for {strategy_name} strategy")
        for idx, split in enumerate(splits):
            train_range = self._list_to_range_str(split['train'])
            test_range = self._list_to_range_str(split['test'])
            
            lines.append(f"python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \\")
            lines.append(f"  --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \\")
            lines.append(f"  --data_range '{train_range}/{test_range}' \\")
            lines.append(f"  --save test_drive_{strategy_name} \\")
            lines.append(f"  --pre_train '../experiment/drive_{strategy_name}_split{idx+1}/model/model_latest.pt' \\")
            lines.append(f"  --scale 1 --patch_size 256 --save_gt --save_results")
            
        return lines


# ========================= 主程序 =========================
if __name__ == "__main__":
    # 设置结果目录路径
    result_dir = "../experiment/test_drive_1_split8/results-DRIVE/"
    
    # 创建分析器
    analyzer = DifficultyAnalyzer(result_dir)
    
    # 计算难度指标
    df = analyzer.calculate_difficulty_metrics()
    df_ranked = analyzer.rank_by_difficulty(df, method='composite')
    
    print("="*80)
    print("样本难度分析完成")
    print("="*80)
    print(f"总样本数: {len(df_ranked)}")
    print("\n前5个最困难样本 (原始文件编号):")
    print(df_ranked.head(5)[['sample_id', 'difficulty_score']])
    print("\n前5个最简单样本 (原始文件编号):")
    print(df_ranked.tail(5)[['sample_id', 'difficulty_score']])
    
    # 生成SGL训练脚本
    print("\n" + "="*80)
    print("生成训练脚本")
    print("="*80)
    
    sgl = DifficultyAwareSGL(df_ranked)
    
    # 显示ID映射信息
    print(f"\n样本ID映射说明:")
    print(f"DRIVE数据集文件编号为 21-40.tif (共20个文件)")
    print(f"训练脚本中 --data_range 使用数组索引 1-20")
    print(f"\n前10个样本的映射关系:")
    print(f"{'文件名':<20} {'训练脚本索引':<15} {'难度得分'}")
    print("-" * 50)
    for orig_id, index_id in sorted(sgl.id_mapping.items(), key=lambda x: x[1])[:10]:
        diff_score = df_ranked[df_ranked['sample_id'] == orig_id]['difficulty_score'].values[0]
        print(f"{orig_id}_training.tif {index_id:>8} {diff_score:>20.4f}")
    if len(sgl.id_mapping) > 10:
        print(f"... (共 {len(sgl.id_mapping)} 个样本)")
    
    script_content = sgl.generate_training_script(num_splits=7, test_size=3)
    
    # 保存到文件
    script_path = "run_drive_k8_new.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\n" + "="*80)
    print(f"训练脚本已保存到: {script_path}")
    print("="*80)
    print("\n使用说明:")
    print(f"  bash {script_path}")
    print("\n脚本包含三种基于难度的SGL改进策略:")
    print("  1. 课程学习 (curriculum) - 从简单到困难逐步训练")
    print("  2. 难度平衡 (balanced) - 每个split测试集难度分布相似") 
    print("  3. 困难样本共享 (sharing) - 困难样本在多个split中训练")
    print("\n注意: 训练脚本中的数字1-20对应文件21-40.tif")