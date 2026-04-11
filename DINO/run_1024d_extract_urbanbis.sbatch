#!/bin/bash
#SBATCH --job-name=UrbanBIS_Pre
#SBATCH --partition=workq
#SBATCH --nodes=1
# 🚀 申请 1 张 GPU（极其关键，DINO 提特征全靠它）
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00      
#SBATCH --mem=240G          
#SBATCH --output=logs/prep_urbanbis_%j.log
#SBATCH --error=logs/prep_urbanbis_%j.err       

mkdir -p logs

echo "=========================================================="
echo "🚀 Starting UrbanBIS DINOv3 1024D Extraction on $(hostname)"
echo "🕒 Start Time: $(date)"
echo "=========================================================="

# 🚀 预检：打印分配到的 GPU 信息，确保显卡就位！
nvidia-smi

# 1. 唤醒 Conda 并激活你的 pointcept 环境
eval "$(conda shell.bash hook)"
conda activate pointcept

# 2. 进入当前提交任务的目录
cd $SLURM_SUBMIT_DIR

# 3. 运行提取脚本 (-u 保证日志实时输出到 .log 文件，方便你用 tail -f 查看进度)
python -u extract_urbanbis.py

echo "=========================================================="
echo "✅ Job finished at $(date)"
echo "=========================================================="