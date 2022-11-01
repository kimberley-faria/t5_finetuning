#!/bin/sh

while true
do 
	cd /mnt/nfs/work1/mccallum/kfaria/t5_finetuning/experiment_logs2/scitail_b/sentiment/
	ls -a | grep json | wc -l 
	# cat /mnt/nfs/scratch1/kfaria/slurm-output/*7555995*.out | grep Epoch | grep 50 | tail -1
	sleep 300
done
