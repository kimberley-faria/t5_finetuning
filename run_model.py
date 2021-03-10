import glob
import logging
import os
import re
import sys
import textwrap

import tensorflow as tf
import wandb
from transformers import TFT5ForConditionalGeneration, AutoTokenizer

from config import SETTINGS

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def _build(pos_files, neg_files, tokenizer, max_len=512):
    pos_inputs, pos_targets = _buil_examples_from_files(pos_files, 'positive', tokenizer, max_len=max_len)
    neg_inputs, neg_targets = _buil_examples_from_files(neg_files, 'negative', tokenizer, max_len=max_len)
    logger.info("Return all inputs and targets")
    return pos_inputs + neg_inputs, pos_targets + neg_targets


def _buil_examples_from_files(files, sentiment, tokenizer, max_len=512):
    inputs = []
    targets = []

    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    for i, path in enumerate(files):
        with open(path, 'r', encoding="utf8") as f:
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
        logger.debug(f"Done processing file {i} of {len(files)} for label - {sentiment}")

    logger.info(f"Return tokenized inputs and targets for label: {sentiment}")
    return inputs, targets


def get_dataset(tokenizer, type_path, max_len=512, subset=None):
    logger.info(f"Get {type_path} dataset of size {subset} (If None, get entire dataset)")
    data_dir = os.path.join(SETTINGS.get('data'), 'aclImdb')
    pos_file_path = os.path.join(data_dir, type_path, 'pos')
    neg_file_path = os.path.join(data_dir, type_path, 'neg')

    pos_files = glob.glob("%s/*.txt" % pos_file_path)
    neg_files = glob.glob("%s/*.txt" % neg_file_path)

    if subset:
        inputs, targets = _build(pos_files[:(subset // 2)], neg_files[:(subset // 2)], tokenizer, max_len=max_len)
    else:
        inputs, targets = _build(pos_files, neg_files, tokenizer, max_len=max_len)

    logger.info(f"Process ids and masks for dataset: {type_path}")
    return [get_ids_and_masks(inputs, targets, i) for i in range(len(inputs))]


def get_ids_and_masks(inputs, targets, index):
    logger.debug(f"Split up the ids and attention_masks for inputs and targets for index {index}")
    source_ids = inputs[index]["input_ids"][0]
    target_ids = targets[index]["input_ids"][0]

    src_mask = inputs[index]["attention_mask"][0]  # might need to squeeze
    target_mask = targets[index]["attention_mask"][0]  # might need to squeeze

    return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids,
            "decoder_attention_mask": target_mask}


def to_tf_dataset(dataset):
    logger.info("Get a tf dataset")
    columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
    return_types = {'input_ids': tf.int32, 'attention_mask': tf.int32,
                    'labels': tf.int32, 'decoder_attention_mask': tf.int32, }
    return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]),
                     'labels': tf.TensorShape([None]), 'decoder_attention_mask': tf.TensorShape([None])}
    ds = tf.data.Dataset.from_generator(lambda: dataset, return_types, return_shapes)
    logger.info(f"TF Dataset {ds.__str__()} returned")
    return ds


def create_dataset(dataset, cache_path=None, batch_size=4, buffer_size=1000, shuffling=True):
    logger.info(f"Create dataset of batch:{batch_size}, shuffle: {shuffling}, buffer: {buffer_size}")
    if cache_path is not None:
        dataset = dataset.cache(cache_path)
    if shuffling:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class FinetunedT5(TFT5ForConditionalGeneration):
    def __init__(self, *args, log_dir=None, cache_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.accuracy_1st_token = tf.keras.metrics.Accuracy(name='accuracy_1st_token')
        self.accuracy_all_tokens = tf.keras.metrics.Accuracy(name='accuracy_all_tokens')

    @tf.function
    def train_step(self, data):
        x = data
        y = x["labels"]
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = outputs[0]
            logits = outputs[1]
            softmaxed_output = tf.nn.softmax(logits, axis=-1)
            y_pred = tf.argmax(softmaxed_output, axis=-1)
            y_no_eos = tf.gather(y, [0], axis=1)
            y_pred_no_eos = tf.gather(y_pred, [0], axis=1)
            # tf.print(y_no_eos, y_pred_no_eos)
            loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        lr = self.optimizer._decayed_lr(tf.float32)

        self.loss_tracker.update_state(loss)
        self.accuracy_all_tokens.update_state(y, y_pred)
        self.accuracy_1st_token.update_state(y_no_eos, y_pred_no_eos)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({'lr': lr})

        return metrics

    @tf.function
    def test_step(self, data):
        x = data
        y = x["labels"]
        output = self(x, training=False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]
        softmaxed_output = tf.nn.softmax(logits, axis=-1)
        y_pred = tf.argmax(softmaxed_output, axis=-1)
        y_no_eos = tf.gather(y, [0], axis=1)
        y_pred_no_eos = tf.gather(y_pred, [0], axis=1)
        # tf.print(y_no_eos, y_pred_no_eos)

        self.loss_tracker.update_state(loss)
        self.accuracy_all_tokens.update_state(y, y_pred)
        self.accuracy_1st_token.update_state(y_no_eos, y_pred_no_eos)
        return {m.name: m.result() for m in self.metrics}


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, step=None):
        self.step = step
        self.train_batch = 0
        self.validation_batch = 0
        self.epoch = 0
        self.val_acc_all_tokens = 0
        self.val_acc_1st_token = 0
        logger.info(f"Initialized CustomCallback to step: {step} ")

    def on_epoch_end(self, epoch, logs=None):
        logger.debug(f"End of epoch {epoch} ; got log keys: {list(logs.keys())}")
        wandb.log(
            {
                'epoch': epoch,
                'accuracy_1st_token': logs['accuracy_1st_token'],
                'accuracy_all_tokens': logs['accuracy_all_tokens'],
                'loss': logs['loss'],
                'val_accuracy_1st_token': self.val_acc_1st_token,
                'val_accuracy_all_tokens': self.val_acc_all_tokens,
                'val_loss': logs['val_loss'],
            }
        )
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        logger.debug(
            f"({self.train_batch}) End of batch {batch} of epoch {self.epoch} of train; got log keys: {list(logs.keys())}")

        wandb.log(
            {
                'training_batch_accuracy_1st_token': logs['accuracy_1st_token'],
                'training_batch_accuracy_all_tokens': logs['accuracy_all_tokens'],
                'training_batch_loss': logs['loss'],
                'train_batch': self.train_batch,
            }
        )
        self.train_batch += 1

    def on_test_batch_end(self, batch, logs=None):
        step = self.step if self.step else 'val'

        logger.debug(
            f"{self.validation_batch}) End of batch {batch} of epoch {self.epoch} of {step}; got log keys: {list(logs.keys())}")

        wandb.log(
            {
                f'{step}_batch_accuracy_1st_token': logs['accuracy_1st_token'],
                f'{step}_batch_accuracy_all_tokens': logs['accuracy_all_tokens'],
                f'{step}_batch_loss': logs['loss'],
                'val_batch': self.validation_batch,
            }
        )
        self.validation_batch += 1
        self.val_acc_all_tokens = logs['accuracy_all_tokens']
        self.val_acc_1st_token = logs['accuracy_1st_token']


def train_test_model():
    optimizer = tf.keras.optimizers.Adam(config.lr)

    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    callbacks = [CustomCallback()]

    # Test for 2, 4, 8, 16, 32, 64, etc.. till 20k
    tf_train_ds = to_tf_dataset(
        get_dataset(tokenizer, 'train', max_len=config.encoder_max_len, subset=config.ntrain))

    # Always run on entire validation set (5000)
    tf_valid_ds = to_tf_dataset(
        get_dataset(tokenizer, 'val', max_len=config.encoder_max_len, subset=config.nvalid))

    train_ds = create_dataset(tf_train_ds, batch_size=config.batch_size, shuffling=True,
                              cache_path=None, buffer_size=2500)
    valid_ds = create_dataset(tf_valid_ds, batch_size=config.batch_size, shuffling=True,
                              cache_path=None, buffer_size=2500)

    model = FinetunedT5.from_pretrained("t5-small")

    model.compile(optimizer=optimizer)

    model.fit(train_ds, epochs=config.epochs, batch_size=config.batch_size, callbacks=callbacks,
              validation_data=valid_ds, validation_batch_size=config.batch_size)

    model.save_weights(os.path.join(wandb.run.dir, "model.h5"))

    # Few Predictions on Val Set

    loader = tf_valid_ds.shuffle(25000).batch(32)
    it = iter(loader)
    batch = next(it)

    outs = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                          max_length=config.encoder_max_len)

    dec = [tokenizer.decode(ids) for ids in outs]

    texts = [tokenizer.decode(ids) for ids in batch['input_ids']]
    targets = [tokenizer.decode(ids) for ids in batch['labels']]

    for i in range(32):
        lines = textwrap.wrap("Review:\n%s\n" % texts[i], width=100)
        print("\n".join(lines))
        print("\nActual sentiment: %s" % targets[i])
        print("\nPredicted sentiment: %s" % dec[i])
        print(f"\nPredicted id: {outs[i]}")
        print("=====================================================================\n")

    # Evaluate on Test Dataset
    if config.evaluate:
        tf_test_ds = to_tf_dataset(
            get_dataset(tokenizer, 'test', max_len=config.encoder_max_len))
        test_ds = create_dataset(tf_test_ds, batch_size=config.batch_size, shuffling=True, cache_path=None)
        model.evaluate(test_ds, callbacks=[CustomCallback(step='test')])


if __name__ == '__main__':
    hparams = {
        'batch_size': 2,
        'encoder_max_len': 256,
        'ntrain': 8,
        'nvalid': 1000,
        'lr': 0.001,
        'epochs': 20,
        'evaluate': False
    }

    if not os.path.exists(SETTINGS.get('data')):
        os.mkdir(SETTINGS.get('data'))

    wandb.init(project='t5-finetuning', config=hparams, dir=f"{SETTINGS.get('data')}", tags=["rev-6", "gypsum"])
    config = wandb.config
    train_test_model()
