#!/bin/bash
#SBATCH --job-name=PTV3_GodView
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=280G
# 🚀 日志文件名改用 %A (数组主ID) 和 %a (子任务ID) 区分
#SBATCH --output=logs/train_1035D_%A_%a.log
#SBATCH --error=logs/train_1035D_%A_%a.err
#SBATCH --exclusive
# 🚀 核心魔法：启动 0, 1, 2 三个并行任务
#SBATCH --array=0-2 

export ENV_DIR=/home/b6ae/bolezhang.b6ae/Pointcept
export CODE_DIR=$SLURM_SUBMIT_DIR
export SIF_FILE=$ENV_DIR/pytorch_24.08.sif
export DATA_DIR=/lus/lfs1aip2/projects/b6ae/datasets

export WANDB_API_KEY=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
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
# 为了防止三个任务的 WandB 缓存冲突，加入任务 ID 隔离
export WANDB_DIR=/workspace/wandb_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export WANDB_START_METHOD=thread

export PYTHONUSERBASE=$ENV_DIR/.pip
export PYTHONPATH=$CODE_DIR:$ENV_DIR/cumm:$ENV_DIR/spconv:$PYTHONPATH
export PATH=$ENV_DIR/.pip/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0a"

# 临时目录也加入任务 ID 隔离
export TMPDIR=/dev/shm/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $TMPDIR

# ==========================================================
# 🚀 配置数组：在这里定义你的三组不同实验！
# ==========================================================
# 假设你要跑三个不同的 Config (如果 Config 相同，写三遍一样的即可)
CONFIG_NAMES=(
    "semseg-pt-v3m1-0-base" 
    "semseg-pt-v3m1-0-base" 
    "semseg-pt-v3m1-0-base"
)

# 假设你要给三组实验起不同的名字 (比如测三次取平均，或者三组不同消融)
EXP_NAMES=(
    "DSGG-PT_Final_Exp1"
    "DSGG-PT_Final_Exp2"
    "DSGG-PT_Final_Exp3"
)

# SLURM 会根据当前的任务 ID (0, 1, 2) 自动抽取对应的配置
CONFIG_NAME=${CONFIG_NAMES[$SLURM_ARRAY_TASK_ID]}
EXP_NAME=${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================================="
echo "🚀 Starting Array Task $SLURM_ARRAY_TASK_ID (Job $SLURM_JOB_ID) on $(hostname)"
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
    
    # 🚀 注意：我帮你把 -r true 改成 -r false 了，确保从头训练！
    sh scripts/train.sh -g 1 -d sensaturban -c $CONFIG_NAME -n $EXP_NAME -r false
  "

rm -rf $TMPDIR