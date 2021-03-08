#!/bin/sh
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH -d singleton

export XDG_RUNTIME_DIR=""
conda activate default
which python
module list

sweep_file=$1

wandb sweep $sweep_file | grep -oP '(?<=ID: ).*' | xargs -I{} wandb agent --count 1 {} --project t5-finetuning --entity kfaria
wandb sync --clean
