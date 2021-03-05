import datetime
import glob
import os
import re
import sys

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from transformers import TFT5ForConditionalGeneration, AutoTokenizer


def _build(pos_files, neg_files, tokenizer, max_len=512):
    pos_inputs, pos_targets = _buil_examples_from_files(pos_files, 'positive', tokenizer, max_len=max_len)
    neg_inputs, neg_targets = _buil_examples_from_files(neg_files, 'negative', tokenizer, max_len=max_len)
    return pos_inputs + neg_inputs, pos_targets + neg_targets


def _buil_examples_from_files(files, sentiment, tokenizer, max_len=512):
    inputs = []
    targets = []

    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    for path in files:
        with open(path, 'r') as f:
            text = f.read()

        line = text.strip()
        line = REPLACE_NO_SPACE.sub("", line)
        line = REPLACE_WITH_SPACE.sub("", line)
        line = f"{line}"

        target = f"{sentiment}"

        # tokenize inputs
        tokenized_inputs = tokenizer(
            line, max_length=max_len, padding='max_length', return_tensors="tf", truncation=True
        )
        # tokenize targets
        tokenized_targets = tokenizer(
            target, max_length=2, padding='max_length', return_tensors="tf", truncation=True
        )

        inputs.append(tokenized_inputs)
        targets.append(tokenized_targets)

    return inputs, targets


def get_dataset(tokenizer, type_path, max_len=512, subset=None):
    data_dir = 'aclImdb'
    pos_file_path = os.path.join(data_dir, type_path, 'pos')
    neg_file_path = os.path.join(data_dir, type_path, 'neg')

    pos_files = glob.glob("%s/*.txt" % pos_file_path)
    neg_files = glob.glob("%s/*.txt" % neg_file_path)

    if subset:
        inputs, targets = _build(pos_files[:(subset // 2)], neg_files[:(subset // 2)], tokenizer, max_len=max_len)
    else:
        inputs, targets = _build(pos_files, neg_files, tokenizer, max_len=max_len)

    return [get_ids_and_masks(inputs, targets, i) for i in range(len(inputs))]


def get_ids_and_masks(inputs, targets, index):
    source_ids = inputs[index]["input_ids"][0]
    target_ids = targets[index]["input_ids"][0]

    src_mask = inputs[index]["attention_mask"][0]  # might need to squeeze
    target_mask = targets[index]["attention_mask"][0]  # might need to squeeze

    return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids,
            "decoder_attention_mask": target_mask}


def to_tf_dataset(dataset):
    columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
    return_types = {'input_ids': tf.int32, 'attention_mask': tf.int32,
                    'labels': tf.int32, 'decoder_attention_mask': tf.int32, }
    return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]),
                     'labels': tf.TensorShape([None]), 'decoder_attention_mask': tf.TensorShape([None])}
    ds = tf.data.Dataset.from_generator(lambda: dataset, return_types, return_shapes)
    return ds


def create_dataset(dataset, cache_path=None, batch_size=4, buffer_size=1000, shuffling=True):
    if cache_path is not None:
        dataset = dataset.cache(cache_path)
    if shuffling:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class SnapthatT5(TFT5ForConditionalGeneration):
    def __init__(self, *args, log_dir=None, cache_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    @tf.function
    def train_step(self, data):
        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = outputs[0]
            logits = outputs[1]
            loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        lr = self.optimizer._decayed_lr(tf.float32)

        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({'lr': lr})

        return metrics

    def test_step(self, data):
        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        output = self(x, training=False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]

        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}


def setup_tensorboard(log_dir, steps, run_name):
    start_profile_batch = steps + 10
    stop_profile_batch = start_profile_batch + 100
    profile_range = f"{start_profile_batch},{stop_profile_batch}"
    log_path = log_dir + f"/{run_name}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1,
                                                          update_freq=20, profile_batch=profile_range)
    checkpoint_filepath = SAVE_PATH + "/" + "T5-{epoch:04d}-{val_loss:.4f}.ckpt"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    callbacks = [tensorboard_callback, ]
    metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy')]

    return callbacks, metrics


def train_test_model(hparams, run_name):
    # For Tensorboard

    steps = hparams['ntrain'] // hparams['batch_size']

    optimizer = tf.keras.optimizers.Adam(hparams['lr'])

    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    callbacks, metrics = setup_tensorboard(f"{LOG_DIR}/hparam_tuning", steps, run_name)
    callbacks.append(hp.KerasCallback(f"{LOG_DIR}/hparam_tuning", hparams))

    # Test for 2, 4, 8, 16, 32, 64, etc.. till 20k
    tf_train_ds = to_tf_dataset(
        get_dataset(tokenizer, 'train', max_len=hparams['encoder_max_len'], subset=hparams['ntrain']))

    # Always run on entire validation set (5000)
    tf_valid_ds = to_tf_dataset(
        get_dataset(tokenizer, 'val', max_len=hparams['encoder_max_len'], subset=hparams['ntrain']))

    tf_test_ds = to_tf_dataset(
        get_dataset(tokenizer, 'test', max_len=hparams['encoder_max_len'], subset=hparams['ntrain']))

    train_ds = create_dataset(tf_train_ds, batch_size=hparams['batch_size'], shuffling=True, cache_path=None)
    valid_ds = create_dataset(tf_valid_ds, batch_size=hparams['batch_size'], shuffling=False, cache_path=None)
    test_ds = create_dataset(tf_test_ds, batch_size=hparams['batch_size'], shuffling=True, cache_path=None)

    model = SnapthatT5.from_pretrained("t5-small")

    model.compile(optimizer=optimizer, metrics=metrics)

    r = model.fit(train_ds, epochs=25, batch_size=hparams['batch_size'], callbacks=callbacks,
                  validation_data=valid_ds, validation_batch_size=hparams['batch_size'])

    # test_accuracy, test_loss = model.evaluate(test_ds)

    return [r.history[x][0] for x in ['accuracy', 'loss', 'val_accuracy', 'val_loss']]


def run(run_dir, hparams, run_name):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        train_accuracy, train_loss, val_accuracy, val_loss = train_test_model(hparams, run_name)
        tf.summary.scalar('accuracy', train_accuracy, step=1)
        tf.summary.scalar('loss', train_loss, step=1)
        tf.summary.scalar('val_accuracy', val_accuracy, step=1)
        tf.summary.scalar('val_loss', val_loss, step=1)


DATA_DIR = "./data"
LOG_DIR = f"{DATA_DIR}/experiments/t5/logs"
SAVE_PATH = f"{DATA_DIR}/experiments/t5/models"

if __name__ == '__main__':
    hparams = {
        'batch_size': int(sys.argv[1]),
        'encoder_max_len': int(sys.argv[2]),
        'ntrain': int(sys.argv[3]),
        'lr': float(sys.argv[4]),
    }
    run_name = f"run-{sys.argv[5]}"
    run(f"{LOG_DIR}/hparam_tuning/" + run_name, hparams, run_name)
