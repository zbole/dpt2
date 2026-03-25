#!/bin/bash
#SBATCH --job-name=Reprocess_Test
#SBATCH --partition=workq        # ⚠️ 如果你们学校跑 GPU 任务有专属的 partition (比如 gpuq), 请改一下这里
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1             # 🚀 极其关键！向集群申请 1 张 GPU 给 DINOv3 用
#SBATCH --time=04:00:00          # 稍微多给点时间以防万一
#SBATCH --output=logs/reprocess_test_%j.log
#SBATCH --error=logs/reprocess_test_%j.err

# 环境路径
export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif

# 🚀 你的 conda 环境 python 绝对路径
PYTHON_EXEC=/home/b6ae/bolezhang.b6ae/miniforge3/envs/pointcept/bin/python

echo "🚀 Starting 1035D Test Set Reprocessing with DINOv3..."

# --nv 参数非常重要，它把宿主机的 Nvidia GPU 映射进 Apptainer 容器里
apptainer exec --nv \
  --cleanenv \
  --containall \
  -B /home/b6ae/bolezhang.b6ae:/home/b6ae/bolezhang.b6ae \
  -B /lus:/lus \
  $SIF_FILE \
  bash -c "
    cd /home/b6ae/bolezhang.b6ae/SuperPointcept_GCDM/pointcept/datasets/preprocessing/sensaturban
    
    # 🚀 执行我们刚刚写好的 Test 专属预处理脚本
    $PYTHON_EXEC reprocess_test_only.py
  "
  
echo "🎉 Reprocessing Job Finished!"