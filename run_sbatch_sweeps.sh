#!/bin/sh

number_of_jobs=$1
sweep_id=$2
partition=$3

for var_name in $(seq 1 $number_of_jobs); do
  sbatch run_wandb_experiments.sh $sweep_id --partition=$partition
done
