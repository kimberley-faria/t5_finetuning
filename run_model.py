import os

import pandas as pd
import tensorflow as tf
import wandb
from transformers import TFT5ForConditionalGeneration, AutoTokenizer

from config import SETTINGS

tokenizer = AutoTokenizer.from_pretrained('t5-base')


def get_dataset(tokenizer, data_dir, type_path, max_len=512):
    path = os.path.join(data_dir, type_path + '.txt')
    data_column = "text"
    class_column = "emotion"
    data = pd.read_csv(path, sep=";", header=None, names=[data_column, class_column],
                       engine="python")

    inputs, targets = _build(data, tokenizer, max_len=max_len)

    return [get_ids_and_masks(inputs, targets, i) for i in range(len(inputs))]


def get_ids_and_masks(inputs, targets, index):
    source_ids = inputs[index]["input_ids"][0]
    target_ids = targets[index]["input_ids"][0]

    src_mask = inputs[index]["attention_mask"][0]  # might need to squeeze
    target_mask = targets[index]["attention_mask"][0]  # might need to squeeze

    return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids,
            "decoder_attention_mask": target_mask}


def _build(data, tokenizer, max_len=512):
    inputs = []
    targets = []
    for idx in range(len(data)):
        data_column = "text"
        class_column = "emotion"

        input_, target = data.loc[idx, data_column], data.loc[idx, class_column]

        # tokenize inputs
        tokenized_inputs = tokenizer.batch_encode_plus(
            [input_], max_length=max_len, padding='max_length', return_tensors="tf", truncation=True
        )
        # tokenize targets
        tokenized_targets = tokenizer.batch_encode_plus(
            [target], max_length=2, padding='max_length', return_tensors="tf", truncation=True
        )

        inputs.append(tokenized_inputs)
        targets.append(tokenized_targets)

    return inputs, targets


class FinetunedT5(TFT5ForConditionalGeneration):
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

    @tf.function
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


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, step=None):
        self.epoch = 0
        self.step = step
        self.train_batch = 0
        self.valid_batch = 0

    def on_epoch_end(self, epoch, logs=None):
        wandb.log(
            {
                'epoch': epoch,
                'accuracy': logs['accuracy'],
                'loss': logs['loss']
            }
        )
        self.epoch += 1

    def on_train_batch_end(self, batch, logs=None):
        wandb.log(
            {
                'train_batch': self.train_batch,
                'train_accuracy': logs['accuracy'],
                'train_loss': logs['loss']
            }
        )
        self.train_batch += 1

    def on_test_batch_end(self, batch, logs=None):
        step = self.step if self.step else 'val'

        wandb.log(
            {
                'valid_batch': self.valid_batch,
                f'{step}_accuracy': logs['accuracy'],
                f'{step}_loss': logs['loss']
            }
        )
        self.valid_batch += 1


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


def train_test_model():
    optimizer = tf.keras.optimizers.Adam(config.lr)

    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy')]
    callbacks = [CustomCallback()]

    # Test for 2, 4, 8, 16, 32, 64, etc.. till 20k
    tf_train_ds = to_tf_dataset(get_dataset(tokenizer, 'emotion_data', 'train', 512))

    # Always run on entire validation set (5000)
    tf_valid_ds = to_tf_dataset(get_dataset(tokenizer, 'emotion_data', 'val', 512))

    train_ds = create_dataset(tf_train_ds, batch_size=config.batch_size, shuffling=True,
                              cache_path=None, buffer_size=2500)
    valid_ds = create_dataset(tf_valid_ds, batch_size=config.batch_size, shuffling=True,
                              cache_path=None, buffer_size=2500)

    model = FinetunedT5.from_pretrained("t5-small")

    model.compile(optimizer=optimizer, metrics=metrics)

    model.fit(train_ds, epochs=config.epochs, batch_size=config.batch_size, callbacks=callbacks,
              validation_data=valid_ds, validation_batch_size=config.batch_size)

    model.save_weights(os.path.join(wandb.run.dir, "model.h5"))

    # Evaluate on Test Dataset
    if config.evaluate:
        tf_test_ds = to_tf_dataset(get_dataset(tokenizer, 'emotion_data', 'test', 512))
        test_ds = create_dataset(tf_test_ds, batch_size=config.batch_size, shuffling=True, cache_path=None)
        model.evaluate(test_ds, callbacks=[CustomCallback(step='test')])


if __name__ == '__main__':
    hparams = {
        'batch_size': 32,
        'encoder_max_len': 256,
        'lr': 0.0001,
        'epochs': 5,
        'ntrain': 64,
        'nvalid': 32,
        'ntest': 100
    }

    if not os.path.exists(SETTINGS.get('data')):
        os.mkdir(SETTINGS.get('data'))

    wandb.init(project='t5-emotion-finetuning', config=hparams)
    config = wandb.config
    train_test_model()
