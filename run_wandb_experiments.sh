#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --partition=m40-short
#SBATCH -o /mnt/nfs/scratch1/kfaria/slurm-output/slurm-%j.out


export XDG_RUNTIME_DIR=""
conda activate default
which python
module list

sweep_id=$1

wandb agent $sweep_id --project t5-baselines --entity kfaria
wandb sync --clean
