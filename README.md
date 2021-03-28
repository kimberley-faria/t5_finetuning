# t5_finetuning

Todo:

Logging - write to sys.out -> redirect to out file in sbatch
WandB
3 training datasets
try the emotion dataset

Dataset:
Using Large Movie Review Dataset from https://ai.stanford.edu/~amaas/data/sentiment/

Download dataset
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

Extract dataset
tar -xvf aclImdb_v1.tar.gz

mkdir aclImdb/val aclImdb/val/pos aclImdb/val/neg

Seperate out val files

> import os  
> import glob  
> import shutil  
> import random  
>
> train_pos_files = glob.glob('aclImdb/train/pos/*.txt')  
> train_neg_files = glob.glob('aclImdb/train/neg/*.txt')  
>
> random.shuffle(train_pos_files)  
> random.shuffle(train_neg_files)  
>
> val_pos_files = train_pos_files[:2500]  
> val_neg_files = train_neg_files[:2500]  
>
> for f in val_pos_files:  
> &nbsp;&nbsp;&nbsp;&nbsp;shutil.move(f,  'aclImdb/val/pos')  
> for f in val_neg_files:  
> &nbsp;&nbsp;&nbsp;&nbsp;shutil.move(f,  'aclImdb/val/neg')`  

Running Sweeps and sbatch

wandb sweep sweep.yaml
> wandb: Creating sweep from: sweep.yaml  
> wandb: Created sweep with ID: 0cs77brz  
> wandb: View sweep at: https://wandb.ai/kfaria/t5-finetuning/sweeps/0cs77brz  
> wandb: Run sweep agent with: wandb agent kfaria/t5-finetuning/0cs77brz  

sbatch run_wandb_experiments.sh 0cs77brz  
> Submitted batch job 7397411  