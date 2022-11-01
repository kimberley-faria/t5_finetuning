#!/bin/sh

while true
do 
	cd /mnt/nfs/work1/mccallum/kfaria/t5_finetuning/experiment_logs/scitail_b/pos_neg/
	ls -a | grep json | wc -l 
	sleep 300
done
