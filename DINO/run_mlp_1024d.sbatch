#!/bin/bash
#SBATCH --job-name=MLP_1024D
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00      
#SBATCH --mem=240G          
#SBATCH --output=logs/mlp_1024d_%j.log
#SBATCH --error=logs/mlp_1024d_%j.err
#SBATCH --exclusive         

mkdir -p logs

echo "=========================================================="
echo "🚀 Starting 1024D DINO Prompt Distillation (MLP) on $(hostname)"
echo "🕒 Start Time: $(date)"
echo "=========================================================="

# 1. 唤醒 Conda 并激活你的 pointcept 环境
eval "$(conda shell.bash hook)"
conda activate pointcept

# 2. 进入当前提交任务的目录
cd $SLURM_SUBMIT_DIR

# 3. 运行全新的 1024D MLP 蒸馏与数据集打包脚本
python -u generate_dino_prompt_1024d.py

echo "=========================================================="
echo "✅ Job finished at $(date)"
echo "=========================================================="