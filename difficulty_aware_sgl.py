import numpy as np
import pandas as pd
from pathlib import Path
import json

class DifficultyAwareSGL:
    """
    基于样本难度的Study Group Learning改进策略
    核心思想：困难样本在多个split间共享，简单样本固定在某个split
    """
    
    def __init__(self, difficulty_csv_path):
        """
        Args:
            difficulty_csv_path: 困难度分析结果的CSV路径
        """
        self.df = pd.read_csv(difficulty_csv_path)
        self.df = self.df.sort_values('sample_id')
        
    def fuzzy_membership_assignment(self, num_splits=7, overlap_ratio=0.3):
        """
        为每个样本分配模糊隶属度
        
        Args:
            num_splits: SGL的split数量
            overlap_ratio: 困难样本的重叠比例（0-1）
        
        Returns:
            membership_dict: {sample_id: [split1_weight, split2_weight, ...]}
        """
        # 归一化难度得分到[0,1]
        difficulty_scores = self.df['difficulty_score'].values
        
        membership = {}
        
        for idx, row in self.df.iterrows():
            sample_id = str(int(row['sample_id']))
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
            
            membership[sample_id] = weights.tolist()
        
        return membership
    
    def adaptive_curriculum_strategy(self, num_splits=7):
        """
        自适应课程学习策略
        前期训练：使用简单样本
        后期训练：逐渐引入困难样本
        
        Returns:
            curriculum_plan: {split_idx: {'easy': [...], 'medium': [...], 'hard': [...]}}
        """
        # 按难度分为三档
        quantiles = self.df['difficulty_score'].quantile([0.33, 0.67])
        
        easy_samples = self.df[self.df['difficulty_score'] <= quantiles[0.33]]['sample_id'].tolist()
        medium_samples = self.df[
            (self.df['difficulty_score'] > quantiles[0.33]) & 
            (self.df['difficulty_score'] <= quantiles[0.67])
        ]['sample_id'].tolist()
        hard_samples = self.df[self.df['difficulty_score'] > quantiles[0.67]]['sample_id'].tolist()
        
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
        # 按难度排序
        sorted_samples = self.df.sort_values('difficulty_score')['sample_id'].tolist()
        
        splits = []
        for i in range(num_splits):
            # 分层抽样：从难度分布中均匀抽取测试样本
            test_indices = list(range(i, len(sorted_samples), num_splits))[:test_size]
            test_samples = [sorted_samples[idx] for idx in test_indices]
            
            train_samples = [s for s in sorted_samples if s not in test_samples]
            
            splits.append({
                'train': [int(s) for s in train_samples],
                'test': [int(s) for s in test_samples],
                'test_difficulty_mean': self.df[self.df['sample_id'].isin(test_samples)]['difficulty_score'].mean(),
                'test_difficulty_std': self.df[self.df['sample_id'].isin(test_samples)]['difficulty_score'].std()
            })
        
        return splits
    
    def generate_training_commands(self, strategy='fuzzy', output_dir='./scripts'):
        """
        生成训练命令脚本
        
        Args:
            strategy: 'fuzzy' - 模糊隶属度
                     'curriculum' - 课程学习
                     'balanced' - 难度平衡
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if strategy == 'fuzzy':
            membership = self.fuzzy_membership_assignment()
            self._save_fuzzy_config(membership, output_dir)
            
        elif strategy == 'curriculum':
            curriculum = self.adaptive_curriculum_strategy()
            self._save_curriculum_scripts(curriculum, output_dir)
            
        elif strategy == 'balanced':
            splits = self.difficulty_balanced_split()
            self._save_balanced_scripts(splits, output_dir)
    
    def _save_fuzzy_config(self, membership, output_dir):
        """保存模糊隶属度配置"""
        config_path = output_dir / 'fuzzy_membership.json'
        with open(config_path, 'w') as f:
            json.dump(membership, f, indent=2)
        
        print(f"模糊隶属度配置已保存到: {config_path}")
        print("\n使用说明：在训练循环中根据membership权重对loss加权")
        print("例如：loss = base_loss * membership_weight[sample_id][split_idx]")
    
    def _save_curriculum_scripts(self, curriculum, output_dir):
        """保存课程学习训练脚本"""
        script_path = output_dir / 'curriculum_training.sh'
        
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# 基于课程学习的SGL训练脚本\n")
            f.write("# 早期split使用简单样本，后期逐渐引入困难样本\n\n")
            
            for split_idx, samples in curriculum.items():
                all_train = samples['easy'] + samples['medium'] + samples['hard']
                train_range = self._list_to_range_str(all_train)
                
                # 固定使用后3个样本作为测试（可根据需要调整）
                test_samples = all_train[-3:]
                test_range = self._list_to_range_str(test_samples)
                
                f.write(f"\n# Split {split_idx+1}: ")
                f.write(f"Easy={len(samples['easy'])}, Medium={len(samples['medium'])}, Hard={len(samples['hard'])}\n")
                f.write(f"python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \\\n")
                f.write(f"  --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \\\n")
                f.write(f"  --data_range '{train_range}/{test_range}' \\\n")
                f.write(f"  --save drive_curriculum_split{split_idx+1} \\\n")
                f.write(f"  --scale 1 --patch_size 256 --reset\n")
        
        print(f"\n课程学习训练脚本已保存到: {script_path}")
        print("运行: bash scripts/curriculum_training.sh")
    
    def _save_balanced_scripts(self, splits, output_dir):
        """保存难度平衡的训练脚本"""
        script_path = output_dir / 'balanced_training.sh'
        
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# 难度平衡的SGL训练脚本\n")
            f.write("# 每个split的测试集难度分布相似\n\n")
            
            for idx, split in enumerate(splits):
                train_range = self._list_to_range_str(split['train'])
                test_range = self._list_to_range_str(split['test'])
                
                f.write(f"\n# Split {idx+1}: 测试集难度均值={split['test_difficulty_mean']:.3f}, 标准差={split['test_difficulty_std']:.3f}\n")
                f.write(f"python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \\\n")
                f.write(f"  --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \\\n")
                f.write(f"  --data_range '{train_range}/{test_range}' \\\n")
                f.write(f"  --save drive_balanced_split{idx+1} \\\n")
                f.write(f"  --scale 1 --patch_size 256 --reset\n")
        
        print(f"\n难度平衡训练脚本已保存到: {script_path}")
        print("运行: bash scripts/balanced_training.sh")
    
    def _list_to_range_str(self, sample_list):
        """将样本列表转换为范围字符串，例如 [21,22,23,25,26] -> '21-23-25-26'"""
        if not sample_list:
            return ""
        
        sorted_list = sorted([int(s) for s in sample_list])
        
        # 简化版：直接用连字符连接
        # 更复杂的实现可以识别连续区间
        return '-'.join(map(str, sorted_list))
    
    def visualize_strategy_comparison(self):
        """可视化不同策略的效果"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 原始分布
        ax = axes[0, 0]
        self.df['sample_index'] = range(len(self.df))
        ax.scatter(self.df['sample_index'], self.df['difficulty_score'], 
                  c=self.df['difficulty_score'], cmap='RdYlGn_r', s=100)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Difficulty Score')
        ax.set_title('Original Difficulty Distribution')
        ax.grid(True, alpha=0.3)
        
        # 2. 模糊隶属度热图
        ax = axes[0, 1]
        membership = self.fuzzy_membership_assignment(num_splits=7)
        membership_matrix = np.array([membership[str(int(sid))] for sid in self.df['sample_id']])
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
        plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
        print("\n策略对比图已保存到: strategy_comparison.png")
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 加载之前分析的结果
    sgl = DifficultyAwareSGL('difficulty_ranking.csv')
    
    print("="*80)
    print("SGL策略改进方案生成器")
    print("="*80)
    
    # 1. 可视化不同策略
    sgl.visualize_strategy_comparison()
    
    # 2. 生成三种策略的配置
    print("\n正在生成训练配置...")
    sgl.generate_training_commands(strategy='fuzzy', output_dir='./scripts')
    sgl.generate_training_commands(strategy='curriculum', output_dir='./scripts')
    sgl.generate_training_commands(strategy='balanced', output_dir='./scripts')
    
    print("\n" + "="*80)
    print("配置生成完成！")
    print("="*80)
    print("\n可选方案：")
    print("1. 模糊隶属度 (Fuzzy): 需要修改训练代码支持样本加权")
    print("2. 课程学习 (Curriculum): 直接运行 bash scripts/curriculum_training.sh")
    print("3. 难度平衡 (Balanced): 直接运行 bash scripts/balanced_training.sh")
    print("\n推荐：先尝试方案2或3，它们无需修改原始代码")