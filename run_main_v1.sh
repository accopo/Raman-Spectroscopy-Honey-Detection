#!/bin/bash
#SBATCH -p zjhu                 # 使用 zjhu 队列
#SBATCH -N 1                    # 申请 1 个节点
#SBATCH --gres=gpu:1            # 申请 1 张显卡 (3090出战！)
#SBATCH -J dual_head_cnn        # 给任务起个炫酷的名字
#SBATCH -o train_log_v1.txt        # 所有的 print 输出都会实时保存在这里

# 1. 唤醒环境
source ~/.bashrc
conda activate raman_env

# 2. 运行
python -u trian_v1_baseline.py