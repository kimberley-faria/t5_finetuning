#!/bin/sh
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH -o /mnt/nfs/scratch1/kfaria/slurm-output/slurm-%j-%a.out
#SBATCH --array=1-6

export XDG_RUNTIME_DIR=""
conda activate default
which python
module list

sweep_id=$1

wandb agent $sweep_id --project t5-baselines --entity kfaria
wandb sync --clean
