#!/bin/sh
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH -o /mnt/nfs/scratch1/kfaria/slurm-output/slurm-%j.out

export XDG_RUNTIME_DIR=""
conda activate default
which python
module list

sweep_id=$1

CUDA_VISIBLE_DEVICES=0 wandb agent $sweep_id --project t5-finetuning --entity kfaria &
CUDA_VISIBLE_DEVICES=1 wandb agent $sweep_id --project t5-finetuning --entity kfaria
wandb sync --clean
