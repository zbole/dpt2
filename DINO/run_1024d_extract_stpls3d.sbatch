#!/bin/bash
# 🚀 根据你要跑的数据集修改 Job Name
#SBATCH --job-name=Data_Prep
#SBATCH --partition=workq
#SBATCH --nodes=1
# 🚀 申请 1 张 GPU（DINO 提特征全靠它）
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00      
#SBATCH --mem=240G          
# 🚀 可以根据数据集改一下 log 的名字
#SBATCH --output=logs/prep_dataset_%j.log
#SBATCH --error=logs/prep_dataset_%j.err       

mkdir -p logs

# 1. 定义你的环境和代码路径 (与你的训练脚本保持一致)
export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export CODE_DIR=$SLURM_SUBMIT_DIR
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif

# 创建临时目录
export TMPDIR=/dev/shm/tmp_${SLURM_JOB_ID}
mkdir -p $TMPDIR

echo "=========================================================="
echo "🚀 Starting DINOv3 Extraction on $(hostname)"
echo "🕒 Start Time: $(date)"
echo "=========================================================="

# 预检：打印分配到的 GPU 信息
nvidia-smi

# 2. 使用 Apptainer 容器执行 Python 脚本
apptainer exec --nv \
  --cleanenv \
  --containall \
  -B $CODE_DIR:/workspace \
  -B $ENV_DIR:$ENV_DIR \
  -B /lus:/lus \
  -B $TMPDIR:/tmp \
  $SIF_FILE \
  bash -c "
    cd /workspace
    # 配置 Python 环境变量指向你的 .pip 依赖库
    export PYTHONUSERBASE=$ENV_DIR/.pip
    export PATH=$ENV_DIR/.pip/bin:\$PATH
    export PYTHONPATH=/workspace:$ENV_DIR/cumm:$ENV_DIR/spconv:\$PYTHONPATH
    
    echo '📦 Environment loaded. Starting python script...'
    
    python -u extract_stpls3d_2.py
  "

# 清理临时文件夹
rm -rf $TMPDIR

echo "=========================================================="
echo "✅ Job finished at $(date)"
echo "=========================================================="