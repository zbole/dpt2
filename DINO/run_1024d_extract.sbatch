#!/bin/bash
#SBATCH --job-name=Sensat_1024D
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00      
#SBATCH --mem=240G          
#SBATCH --output=logs/ext_1024d_%j.log
#SBATCH --error=logs/ext_1024d_%j.err
#SBATCH --exclusive         

mkdir -p logs

echo "=========================================================="
echo "🚀 Starting 1024D ViT-Large Extraction on $(hostname)"
echo "🕒 Start Time: $(date)"
echo "=========================================================="

# 1. 唤醒 Conda 并激活你的 pointcept 环境
eval "$(conda shell.bash hook)"
conda activate pointcept

# 2. 进入当前提交任务的目录（确保能找到 python 脚本和 weights 文件夹）
cd $SLURM_SUBMIT_DIR

# 3. 运行提取脚本 (-u 保证日志实时输出到 .log 文件)
python -u extract_dino_1024d_hpc.py

echo "=========================================================="
echo "✅ Job finished at $(date)"
echo "=========================================================="