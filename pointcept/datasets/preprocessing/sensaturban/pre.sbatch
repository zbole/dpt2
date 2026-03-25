#!/bin/bash
#SBATCH --job-name=Pre_Sensat_DINO
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00      # 设为 48 小时比较保险，防止 I/O 拥塞导致超时
#SBATCH --mem=240G          # 满血 240G 内存，KDTree 随便跑
#SBATCH --output=logs/pre_sensat_%j.log
#SBATCH --error=logs/pre_sensat_%j.err
#SBATCH --exclusive         # 独占节点，防止别人挤占 CPU 和硬盘 I/O

echo "=========================================================="
echo "🚀 Starting Preprocessing Job $SLURM_JOB_ID on $(hostname)"
echo "🕒 Start Time: $(date)"
echo "=========================================================="

# 1. 唤醒 Conda 并激活你的 pointcept 环境
eval "$(conda shell.bash hook)"
conda activate pointcept

# 2. 确保进入正确的工作目录
cd ~/SuperPointcept_GCDM/pointcept/datasets/preprocessing/sensaturban

# 3. 运行断点续传脚本 (-u 保证日志实时输出到 .log 文件)
python -u resume_process.py

echo "=========================================================="
echo "✅ Job finished at $(date)"
echo "=========================================================="