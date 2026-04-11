#!/bin/bash
# 🚀 任务名称改为 Sensat_Pre
#SBATCH --job-name=Sensat_Pre
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00      
#SBATCH --mem=240G          
# 🚀 日志文件名也同步更新，方便 tail -f 监控
#SBATCH --output=logs/prep_sensat_%j.log
#SBATCH --error=logs/prep_sensat_%j.err      

mkdir -p logs

# 1. 定义环境路径（保持与 Isambard-AI 集群配置一致）
export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export CODE_DIR=$SLURM_SUBMIT_DIR
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif

# 创建临时目录
export TMPDIR=/dev/shm/tmp_${SLURM_JOB_ID}
mkdir -p $TMPDIR

echo "=========================================================="
echo "🚀 Starting SensatUrban DINOv3 Extraction on $(hostname)"
echo "🕒 Start Time: $(date)"
echo "=========================================================="

# 预检 GPU 状态
nvidia-smi

# 2. 使用 Apptainer 容器执行 SensatUrban 提取脚本
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
    
    echo '📦 Environment loaded. Processing SensatUrban scenes...'
    
    # 🚀 核心：运行针对 SensatUrban 优化后的提取脚本
    # 确保你的 Python 文件名与此处一致
    python -u extract_sensaturban.py
  "

# 清理
rm -rf $TMPDIR

echo "=========================================================="
echo "✅ SensatUrban Extraction Job finished at $(date)"
echo "=========================================================="