#!/bin/bash

# ========================================
# 基于样本难度的SGL训练脚本
# 包含三种策略: 课程学习、难度平衡、困难样本共享
# ========================================

# # ========================================
# # 策略1: 课程学习 - 从简单到困难逐步训练
# # ========================================

# # Split 1: Easy=7, Medium=3, Hard=0
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '3-3-5-6-9-11-20-20/1-1-12-12-15-15' \
#   --save drive_curriculum_split1 \
#   --scale 1 --patch_size 256 --reset

# # Split 2: Easy=7, Medium=3, Hard=0
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '3-3-5-6-9-11-20-20/1-1-12-12-15-15' \
#   --save drive_curriculum_split2 \
#   --scale 1 --patch_size 256 --reset

# # Split 3: Easy=7, Medium=3, Hard=0
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '3-3-5-6-9-11-20-20/1-1-12-12-15-15' \
#   --save drive_curriculum_split3 \
#   --scale 1 --patch_size 256 --reset

# # Split 4: Easy=7, Medium=6, Hard=3
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-1-3-3-5-6-9-12-15-15-17-20/2-2-4-4-7-7' \
#   --save drive_curriculum_split4 \
#   --scale 1 --patch_size 256 --reset

# # Split 5: Easy=7, Medium=6, Hard=3
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-1-3-3-5-6-9-12-15-15-17-20/2-2-4-4-7-7' \
#   --save drive_curriculum_split5 \
#   --scale 1 --patch_size 256 --reset

# # Split 6: Easy=7, Medium=6, Hard=7
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-12-15-15-17-20/13-14-16-16' \
#   --save drive_curriculum_split6 \
#   --scale 1 --patch_size 256 --reset

# # Split 7: Easy=7, Medium=6, Hard=7
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-12-15-15-17-20/13-14-16-16' \
#   --save drive_curriculum_split7 \
#   --scale 1 --patch_size 256 --reset

# # Testing to obtain pseudo labels for curriculum strategy
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '3-3-5-6-9-11-20-20/1-1-12-12-15-15' \
#   --save test_drive_curriculum \
#   --pre_train '../experiment/drive_curriculum_split1/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '3-3-5-6-9-11-20-20/1-1-12-12-15-15' \
#   --save test_drive_curriculum \
#   --pre_train '../experiment/drive_curriculum_split2/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '3-3-5-6-9-11-20-20/1-1-12-12-15-15' \
#   --save test_drive_curriculum \
#   --pre_train '../experiment/drive_curriculum_split3/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-1-3-3-5-6-9-12-15-15-17-20/2-2-4-4-7-7' \
#   --save test_drive_curriculum \
#   --pre_train '../experiment/drive_curriculum_split4/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-1-3-3-5-6-9-12-15-15-17-20/2-2-4-4-7-7' \
#   --save test_drive_curriculum \
#   --pre_train '../experiment/drive_curriculum_split5/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-12-15-15-17-20/13-14-16-16' \
#   --save test_drive_curriculum \
#   --pre_train '../experiment/drive_curriculum_split6/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-12-15-15-17-20/13-14-16-16' \
#   --save test_drive_curriculum \
#   --pre_train '../experiment/drive_curriculum_split7/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results

# # ========================================
# # 策略2: 难度平衡 - 每个split测试集难度相似
# # ========================================

# # Split 1: TestDiff: mean=0.487, std=0.244
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-5-7-15-17-18-20-20/6-6-16-16-19-19' \
#   --save drive_balanced_split1 \
#   --scale 1 --patch_size 256 --reset

# # Split 2: TestDiff: mean=0.527, std=0.232
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-1-3-9-11-11-13-20/2-2-10-10-12-12' \
#   --save drive_balanced_split2 \
#   --scale 1 --patch_size 256 --reset

# # Split 3: TestDiff: mean=0.559, std=0.236
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '2-13-15-19/1-1-14-14-20-20' \
#   --save drive_balanced_split3 \
#   --scale 1 --patch_size 256 --reset

# # Split 4: TestDiff: mean=0.595, std=0.186
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-7-10-14-16-20/8-9-15-15' \
#   --save drive_balanced_split4 \
#   --scale 1 --patch_size 256 --reset

# # Split 5: TestDiff: mean=0.615, std=0.194
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-2-4-6-8-16-18-20/3-3-7-7-17-17' \
#   --save drive_balanced_split5 \
#   --scale 1 --patch_size 256 --reset

# # Split 6: TestDiff: mean=0.692, std=0.244
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-3-6-17-19-20/4-5-18-18' \
#   --save drive_balanced_split6 \
#   --scale 1 --patch_size 256 --reset

# # Split 7: TestDiff: mean=0.583, std=0.152
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-10-12-12-14-20/11-11-13-13' \
#   --save drive_balanced_split7 \
#   --scale 1 --patch_size 256 --reset

# # Testing to obtain pseudo labels for balanced strategy
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-5-7-15-17-18-20-20/6-6-16-16-19-19' \
#   --save test_drive_balanced \
#   --pre_train '../experiment/drive_balanced_split1/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-1-3-9-11-11-13-20/2-2-10-10-12-12' \
#   --save test_drive_balanced \
#   --pre_train '../experiment/drive_balanced_split2/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '2-13-15-19/1-1-14-14-20-20' \
#   --save test_drive_balanced \
#   --pre_train '../experiment/drive_balanced_split3/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-7-10-14-16-20/8-9-15-15' \
#   --save test_drive_balanced \
#   --pre_train '../experiment/drive_balanced_split4/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-2-4-6-8-16-18-20/3-3-7-7-17-17' \
#   --save test_drive_balanced \
#   --pre_train '../experiment/drive_balanced_split5/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-3-6-17-19-20/4-5-18-18' \
#   --save test_drive_balanced \
#   --pre_train '../experiment/drive_balanced_split6/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-10-12-12-14-20/11-11-13-13' \
#   --save test_drive_balanced \
#   --pre_train '../experiment/drive_balanced_split7/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results

# # ========================================
# # 策略3: 困难样本共享 - 困难样本在多个split中训练
# # ========================================

# # Split 1: Base=17, +SharedHard=4
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '2-7-9-14-16-20/1-1-8-8-15-15' \
#   --save drive_sharing_split1 \
#   --scale 1 --patch_size 256 --reset

# # Split 2: Base=17, +SharedHard=1
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-1-3-8-10-15-17-20/2-2-9-9-16-16' \
#   --save drive_sharing_split2 \
#   --scale 1 --patch_size 256 --reset

# # Split 3: Base=17, +SharedHard=3
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-2-4-9-11-16-18-20/3-3-10-10-17-17' \
#   --save drive_sharing_split3 \
#   --scale 1 --patch_size 256 --reset

# # Split 4: Base=17, +SharedHard=0
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-3-5-10-12-17-19-20/4-4-11-11-18-18' \
#   --save drive_sharing_split4 \
#   --scale 1 --patch_size 256 --reset

# # Split 5: Base=17, +SharedHard=2
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-4-6-11-13-18-20-20/5-5-12-12-19-19' \
#   --save drive_sharing_split5 \
#   --scale 1 --patch_size 256 --reset

# # Split 6: Base=17, +SharedHard=2
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-5-7-12-14-19/6-6-13-13-20-20' \
#   --save drive_sharing_split6 \
#   --scale 1 --patch_size 256 --reset

# # Split 7: Base=18, +SharedHard=2
# python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-6-8-13-15-20/7-7-14-14' \
#   --save drive_sharing_split7 \
#   --scale 1 --patch_size 256 --reset

# # Testing to obtain pseudo labels for sharing strategy
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '2-7-9-14-16-20/1-1-8-8-15-15' \
#   --save test_drive_sharing \
#   --pre_train '../experiment/drive_sharing_split1/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-1-3-8-10-15-17-20/2-2-9-9-16-16' \
#   --save test_drive_sharing \
#   --pre_train '../experiment/drive_sharing_split2/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-2-4-9-11-16-18-20/3-3-10-10-17-17' \
#   --save test_drive_sharing \
#   --pre_train '../experiment/drive_sharing_split3/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-3-5-10-12-17-19-20/4-4-11-11-18-18' \
#   --save test_drive_sharing \
#   --pre_train '../experiment/drive_sharing_split4/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-4-6-11-13-18-20-20/5-5-12-12-19-19' \
#   --save test_drive_sharing \
#   --pre_train '../experiment/drive_sharing_split5/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-5-7-12-14-19/6-6-13-13-20-20' \
#   --save test_drive_sharing \
#   --pre_train '../experiment/drive_sharing_split6/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results
# python main.py --self_ensemble --test_only --patch_size 256 --model CON --loss 1*BCE \
#   --data_train DRIVE --data_test DRIVE --n_GPUs 1 --epochs 30 \
#   --data_range '1-6-8-13-15-20/7-7-14-14' \
#   --save test_drive_sharing \
#   --pre_train '../experiment/drive_sharing_split7/model/model_latest.pt' \
#   --scale 1 --patch_size 256 --save_gt --save_results