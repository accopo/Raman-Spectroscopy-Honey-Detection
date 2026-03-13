#!/bin/bash
#SBATCH -p zjhu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -J test_torch
#SBATCH -o gpu_test_result.txt

# 激活你的专属环境
source ~/.bashrc
conda activate raman_env

# 运行测试代码
python check_gpu.py