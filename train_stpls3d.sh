#!/bin/bash
#SBATCH --job-name=PTV3_STPLS3D_SynEval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=444G
#SBATCH --output=logs/train_stpls3d_SynEval_%j.log
#SBATCH --error=logs/train_stpls3d_SynEval_%j.err

export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export CODE_DIR=$SLURM_SUBMIT_DIR
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif

# 🚀 物理路径指向全新的 Synthetic_Eval 文件夹
export DATA_DIR=/lus/lfs1aip2/projects/b6ae/datasets/STPLS3D/processed_1025D_Synthetic_Eval

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

# 🛡️ 新增：防 OOM 底层护盾（限制内存碎片和多线程抢占）
export MALLOC_ARENA_MAX=1
export OMP_NUM_THREADS=4

export TMPDIR=/dev/shm/tmp_${SLURM_JOB_ID}
mkdir -p $TMPDIR

CONFIG_NAME="semseg-pt-v3m1-0-base"
EXP_NAME="STPLS3D_DSGG-PT_Final"

echo "=========================================================="
echo "🚀 Starting Job $SLURM_JOB_ID on $(hostname)"
echo "📂 Using Config: $CONFIG_NAME"
echo "🏷️  Experiment Name: $EXP_NAME"
echo "=========================================================="

# 🚀 新增：预检 GPU，看看这次分到的是什么神仙显卡
echo "📊 Checking GPU Status..."
nvidia-smi
echo "=========================================================="

apptainer exec --nv \
  --cleanenv \
  --containall \
  -B $CODE_DIR:/workspace \
  -B $ENV_DIR:$ENV_DIR \
  -B $DATA_DIR:/datasets/STPLS3D/processed_1025D_Synthetic_Eval \
  -B /lus:/lus \
  -B $TMPDIR:/tmp \
  $SIF_FILE \
  bash -c "
    cd /workspace
    export PYTHONPATH=/workspace:$ENV_DIR/cumm:$ENV_DIR/spconv:\$PYTHONPATH
    export PYTHONUSERBASE=$ENV_DIR/.pip
    export PATH=$ENV_DIR/.pip/bin:\$PATH
    export WANDB_API_KEY=$WANDB_API_KEY
    
    sh scripts/train.sh -g 1 -d stpls3d -c $CONFIG_NAME -n $EXP_NAME -r true
  "
rm -rf $TMPDIR