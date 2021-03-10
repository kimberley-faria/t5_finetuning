#!/bin/sh
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH -d singleton

export XDG_RUNTIME_DIR=""
conda activate default
which python
module list

sweep_id=$1

wandb agent --count 1 $sweep_id --project t5-finetuning --entity kfaria
wandb sync --clean