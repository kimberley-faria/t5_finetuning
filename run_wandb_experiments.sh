#!/bin/sh
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB

export XDG_RUNTIME_DIR=""
conda activate default
which python
module list

sweep_id=$1

wandb agent $sweep_id --project t5-finetuning --entity kfaria
wandb sync --clean
