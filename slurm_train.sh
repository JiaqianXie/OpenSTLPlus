#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=60gb
#SBATCH --nodes=2
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 # Number of GPUs (per node)

module load cuda/12.1.0
module load cudnn/8.9.5-cu12
module load miniconda3/23.5.2
module load gcc/12.3.0
module load ninja/1.11.1
module load sqlite/3.43.1

source activate /home/li309/scratch3_envs/OpenSTL
cd /home/li309/yang_seg/video_pred/OpenSTLPlus

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=12580
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export WANDB_API_KEY=d364b735626a90b09a932b7b2f5ded75f2f613ef

WANDB_MODE=offline python tools/train.py mmnist configs/mmnist/SwinLSTM-D.py --ex_name mmnist_SwinLSTM --dist --nnodes $SLURM_NNODES
