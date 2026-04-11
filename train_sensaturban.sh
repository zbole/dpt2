#!/bin/bash
#SBATCH --job-name=Resume_PTV3
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=444G
# 🚀 移除了 Array 相关参数，恢复常规日志命名
#SBATCH --output=logs/resume_sensaturban_%j.log
#SBATCH --error=logs/resume_sensaturban_%j.err

export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export CODE_DIR=$SLURM_SUBMIT_DIR
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif

export WANDB_API_KEY=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
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_DIR=/workspace/wandb_${SLURM_JOB_ID}
export WANDB_START_METHOD=thread

export PYTHONUSERBASE=$ENV_DIR/.pip
export PYTHONPATH=$CODE_DIR:$ENV_DIR/cumm:$ENV_DIR/spconv:$PYTHONPATH
export PATH=$ENV_DIR/.pip/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0a"

# 🛡️ 防 OOM 底层护盾
export MALLOC_ARENA_MAX=1
export OMP_NUM_THREADS=4

export TMPDIR=/tmp/slurm_tmp_${SLURM_JOB_ID}
mkdir -p $TMPDIR

# 🚀 指向 Lustre 上的数据源
export DATA_DIR=/lus/lfs1aip2/projects/b6ae/datasets/sensaturban/processed_1025D_SP-PT

# ==========================================================
# 🎯 极其关键：精准对齐之前存活的那个任务文件夹！
# 根据你发给我的 Config，你之前存活任务的 save_path 是 'exp/sensaturban/DSGG-PT_Final'
# 所以这里的 EXP_NAME 必须是 DSGG-PT_Final，Pointcept 才能找到里面的 .pth 文件。
# ==========================================================
CONFIG_NAME="semseg-pt-v3m1-0-base"
EXP_NAME="SensatUrban_DSGG-PT_FinalEXP"

echo "=========================================================="
echo "🚀 Resuming Job $SLURM_JOB_ID on $(hostname)"
echo "📂 Using Config: $CONFIG_NAME"
echo "🎯 Target Experiment: $EXP_NAME"
echo "=========================================================="

# 🚀 预检：打印分配到的 GPU 信息，确保显卡就位！
echo "📊 Checking GPU Status..."
nvidia-smi
echo "=========================================================="

apptainer exec --nv \
  --cleanenv \
  --containall \
  -B $CODE_DIR:/workspace \
  -B $ENV_DIR:$ENV_DIR \
  -B $DATA_DIR:/datasets/sensaturban/processed_1025D_SP-PT \
  -B /lus:/lus \
  -B $TMPDIR:/tmp \
  $SIF_FILE \
  bash -c "
    cd /workspace
    export PYTHONPATH=/workspace:$ENV_DIR/cumm:$ENV_DIR/spconv:\$PYTHONPATH
    export PYTHONUSERBASE=$ENV_DIR/.pip
    export PATH=$ENV_DIR/.pip/bin:\$PATH
    export WANDB_API_KEY=$WANDB_API_KEY
    
    # 🚀 核心修改：-r true 开启断点续传！
    sh scripts/test.sh -g 1 -d sensaturban -c $CONFIG_NAME -n $EXP_NAME -r true
  "

rm -rf $TMPDIR