#!/bin/sh
#SBATCH --partition=m40-short
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH -d singleton

export XDG_RUNTIME_DIR=""
conda activate default
which python
module list

batch=$1
max_len=$2
n_train=$3
lr=$4
session_num=$5

python run_model.py $batch $max_len $n_train $lr $session_num