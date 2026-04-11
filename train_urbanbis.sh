#!/bin/bash
#SBATCH --job-name=PTV3_UrbanBIS
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=280G
#SBATCH --output=logs/train_urbanbis_%j.log
#SBATCH --error=logs/train_urbanbis_%j.err

export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export CODE_DIR=$SLURM_SUBMIT_DIR
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif

export DATA_DIR=/lus/lfs1aip2/projects/b6ae/datasets/UrbanBIS/processed_1025D_Pure

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

export TMPDIR=/dev/shm/tmp_${SLURM_JOB_ID}
mkdir -p $TMPDIR

# 🚀 修改 2：确认配置名。你之前在 test 的时候用的是 configs/UrbanBIS/default.yaml
# 如果你的 yaml 文件名就叫 default.yaml，这里就填 default
CONFIG_NAME="semseg-pt-v3m1-0-base" 
# 🚀 修改 3：统一成你之前报错日志里的那个实验名称
EXP_NAME="UrbanBIS_DSGG-PT_Exp"

echo "=========================================================="
echo "🚀 Starting Job $SLURM_JOB_ID on $(hostname)"
echo "📂 Using Config: $CONFIG_NAME"
echo "🏷️  Experiment Name: $EXP_NAME"
echo "=========================================================="

apptainer exec --nv \
  --cleanenv \
  --containall \
  -B $CODE_DIR:/workspace \
  -B $ENV_DIR:$ENV_DIR \
  -B $DATA_DIR:/datasets/UrbanBIS/processed_1025D_Pure \
  -B /lus:/lus \
  -B $TMPDIR:/tmp \
  $SIF_FILE \
  bash -c "
    cd /workspace
    export PYTHONPATH=/workspace:$ENV_DIR/cumm:$ENV_DIR/spconv:\$PYTHONPATH
    export PYTHONUSERBASE=$ENV_DIR/.pip
    export PATH=$ENV_DIR/.pip/bin:\$PATH
    export WANDB_API_KEY=$WANDB_API_KEY
    
    # 🚀 修改 4：将 -d 后的数据集参数改为 UrbanBIS，对应 configs 里的文件夹名
    sh scripts/train.sh -g 1 -d UrbanBIS -c $CONFIG_NAME -n $EXP_NAME -r false
  "
rm -rf $TMPDIR