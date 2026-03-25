#!/bin/bash
#SBATCH --job-name=PTV3_GodView
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=300G
#SBATCH --output=logs/train_1035D_%j.log
#SBATCH --error=logs/train_1035D_%j.err
#SBATCH --exclusive

export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export CODE_DIR=$SLURM_SUBMIT_DIR
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif
export DATA_DIR=/lus/lfs1aip2/projects/b6ae/datasets
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ⚠️ 强烈建议：明早睡醒后，第一件事去 W&B 后台重置一下你的 API Key
export WANDB_API_KEY=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
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_DIR=/workspace/wandb_$SLURM_JOB_ID
export WANDB_START_METHOD=thread

export PYTHONUSERBASE=$ENV_DIR/.pip
export PYTHONPATH=$CODE_DIR:$ENV_DIR/cumm:$ENV_DIR/spconv:$PYTHONPATH
export PATH=$ENV_DIR/.pip/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0a"
export TMPDIR=/dev/shm/tmp_$SLURM_JOB_ID

mkdir -p $TMPDIR

# 指向你刚刚修改好的那个 Config 文件名
CONFIG_NAME="semseg-pt-v3m1-0-base" 

# 🚀 霸气的新名字：1035D特征 + GodView(上帝高度) + DINO1024 + 动态门控
EXP_NAME="PTV3_1035D_z_DINO1024_DynGCDM_v10"

echo "=========================================================="
echo "🚀 Starting Single Job $SLURM_JOB_ID on $(hostname)"
echo "📂 Using Config: $CONFIG_NAME"
echo "🏷️  Experiment Name: $EXP_NAME"
echo "=========================================================="

apptainer exec --nv \
  --cleanenv \
  --containall \
  -B $CODE_DIR:/workspace \
  -B $ENV_DIR:$ENV_DIR \
  -B $DATA_DIR:/datasets \
  -B /lus:/lus \
  -B $TMPDIR:/tmp \
  $SIF_FILE \
  bash -c "
    cd /workspace
    export PYTHONPATH=/workspace:$ENV_DIR/cumm:$ENV_DIR/spconv:\$PYTHONPATH
    export PYTHONUSERBASE=$ENV_DIR/.pip
    export PATH=$ENV_DIR/.pip/bin:\$PATH
    export WANDB_API_KEY=$WANDB_API_KEY
    
    # 🚀 -r false 确保从头随机初始化这只全新的 1035D 性能巨兽！
    sh scripts/test.sh -g 1 -d sensaturban -c $CONFIG_NAME -n $EXP_NAME
  "

rm -rf $TMPDIR