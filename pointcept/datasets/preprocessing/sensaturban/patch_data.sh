#!/bin/bash
#SBATCH --job-name=Patch_1035D
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/patch_1035D_%j.log
#SBATCH --error=logs/patch_1035D_%j.err

# 环境路径
export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif
# 🚀 你的 conda 环境 python 绝对路径
PYTHON_EXEC=/home/b6ae/bolezhang.b6ae/miniforge3/envs/pointcept/bin/python

echo "🚀 Starting 1035D Data Patching..."

apptainer exec --nv \
  --cleanenv \
  --containall \
  -B /home/b6ae/bolezhang.b6ae:/home/b6ae/bolezhang.b6ae \
  -B /lus:/lus \
  $SIF_FILE \
  bash -c "
    cd /home/b6ae/bolezhang.b6ae/SuperPointcept_GCDM/pointcept/datasets/preprocessing/sensaturban
    
    # 🚀 直接调用绝对路径，并加上 python 字样
    $PYTHON_EXEC patch_global_z.py
  "