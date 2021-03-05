#!/usr/bin/env python
# coding: utf-8

import sys
from subprocess import Popen

import tensorflow as tf
from tensorboard import program
from tensorboard.plugins.hparams import api as hp

DATA_DIR = "./data"
LOG_DIR = f"{DATA_DIR}/experiments/t5/logs"
SAVE_PATH = f"{DATA_DIR}/experiments/t5/models"
CACHE_PATH_TRAIN = f"{DATA_DIR}/cache/t5.train"
CACHE_PATH_TEST = f"{DATA_DIR}/cache/t5.test"

train_dataset_subsets = [2 ** n for n in range(15)[1:]] + [20000]

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([2]))
HP_N_TRAIN = hp.HParam('ntrain', hp.Discrete([4]))
HP_ENCODER_MAX_LEN = hp.HParam('encoder_max_len', hp.Discrete([256]))
HP_LEARNING_RATE = hp.HParam('lr', hp.Discrete([0.001, 0.0001, 0.00001]))

METRIC_ACCURACY = 'accuracy'
METRIC_LOSS = 'loss'
METRIC_VAL_ACCURACY = 'accuracy'
METRIC_VAL_LOSS = 'loss'

with tf.summary.create_file_writer(f"{LOG_DIR}/hparam_tuning").as_default():
    hp.hparams_config(
        hparams=[HP_BATCH_SIZE, HP_N_TRAIN, HP_ENCODER_MAX_LEN, HP_LEARNING_RATE],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
                 hp.Metric(METRIC_LOSS, display_name='Loss'),
                 hp.Metric(METRIC_VAL_ACCURACY, display_name='Val. Accuracy'),
                 hp.Metric(METRIC_VAL_LOSS, display_name='Val. Loss')],
    )



session_num = 0

for batch_size in HP_BATCH_SIZE.domain.values:
    for max_len in HP_ENCODER_MAX_LEN.domain.values:
        for ntrain in HP_N_TRAIN.domain.values:
            for lr in HP_LEARNING_RATE.domain.values:
                if ntrain >= batch_size:
                    hparams = {
                        HP_BATCH_SIZE: batch_size,
                        HP_ENCODER_MAX_LEN: max_len,
                        HP_N_TRAIN: ntrain,
                        HP_LEARNING_RATE: lr,
                    }
                    run_name = f"run-{session_num}"
                    print(f'--- Starting trial: {run_name}')
                    print({h.name: hparams[h] for h in hparams})
                    command = f"sbatch --job-name=run_model_{session_num}.run --output=out/run_model_{session_num}.out run_model.sh {batch_size} {max_len} {ntrain} {lr} {session_num}"
                    proc = Popen(command, shell=True,
                                 stdin=None, stdout=None, stderr=None, close_fds=True)
                    session_num += 1
log_paths = []
for i in range(session_num):
    log_paths.append(f"run{session_num}:{LOG_DIR}/hparam_tuning/run{session_num}")


tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', f"{LOG_DIR}/hparam_tuning", '--host', '0.0.0.0', '--port', '9001'])

# Note: tb.launch() create a daemon thread that will die automatically when your process is finished
url = tb.launch()
sys.stdout.write('TensorBoard at %s \n' % url)