import json
import logging
import os
import sys
import re

import jsonlines
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, AutoTokenizer

import wandb
from config import SETTINGS, TRAINING_DATASET_FNAME, VALIDATION_DATASET_FNAME, LABELS_TYPE, SYSTEM, WANDB_ENTITY

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_ids_and_masks(inputs, targets, index):
    logger.debug(f"Split up the ids and attention_masks for inputs and targets for index {index}")
    source_ids = inputs[index]["input_ids"][0]
    target_ids = targets[index]["input_ids"][0]

    src_mask = inputs[index]["attention_mask"][0]  # might need to squeeze
    target_mask = targets[index]["attention_mask"][0]  # might need to squeeze

    return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids,
            "decoder_attention_mask": target_mask}


def t5_tokenized_examples(fname, max_len=128):
    inputs = []
    targets = []

    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    regex = re.compile(r"\[unused([\d]+)\]")

    with jsonlines.open(fname) as dataset:
        for task in dataset:
            for data in task['text'][:config.batch_size]:
                input_text = f"{data['sentence']}"
                label = f"{data['label']}"
                # label = regex.sub(r"<extra_id_\g<1>>", f"{data['label']}")

                logger.info(f"********** Task **********")
                logger.info(f"Tokens: {data['masked_tokens']}")
                logger.info(f"Sentence: {input_text}")
                logger.info(f"Output: {label}")

                tokenized_inputs = tokenizer(
                    input_text, max_length=max_len, padding='max_length', return_tensors="tf", truncation=True
                )
                tokenized_targets = tokenizer(
                    label, max_length=max_len, padding='max_length', return_tensors="tf", truncation=True
                )

                inputs.append(tokenized_inputs)
                targets.append(tokenized_targets)

    logger.info("{}, {}".format(len(inputs), len(targets)))
    return [get_ids_and_masks(inputs, targets, i) for i in range(len(inputs))]


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


class AllTokensAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='custom_accuracy', **kwargs):
        super(AllTokensAccuracy, self).__init__(name=name, **kwargs)
        self.total_labels = self.add_weight(name='total', initializer='zeros')
        self.correct_labels = self.add_weight(name='correct', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_labels.assign_add(tf.cast(tf.shape(y_pred)[0], 'float32'))
        is_equal = tf.equal(tf.cast(y_true, 'int32'), tf.cast(y_pred, 'int32'))
        self.correct_labels.assign_add(tf.reduce_sum(tf.cast(tf.math.reduce_all(is_equal, axis=1), 'float32')))

    def result(self):
        if tf.equal(self.total_labels, 0.0):
            return 0.0
        return tf.divide(self.correct_labels, self.total_labels)

    def reset_states(self):
        self.total_labels.assign(0.)
        self.correct_labels.assign(0.)


class MultipleTokensAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='custom_accuracy', **kwargs):
        super(MultipleTokensAccuracy, self).__init__(name=name, **kwargs)
        self.total_labels = self.add_weight(name='total', initializer='zeros')
        self.correct_labels = self.add_weight(name='correct', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_labels.assign_add(tf.cast(tf.shape(y_pred)[0], 'float32'))

        ones = tf.ones(tf.shape(y_true))
        lengths = tf.cast(tf.add(tf.argmax(tf.equal(ones, tf.cast(y_true, tf.float32)), axis=-1), 1), tf.int32)

        i = tf.constant(0)

        def cond(i):
            return tf.less(i, tf.shape(lengths)[0])

        def body(i):
            self.correct_labels.assign_add(tf.cast(tf.math.reduce_all(
                tf.equal(tf.cast(y_true[i, :lengths[i]], tf.int32), tf.cast(y_pred[i, :lengths[i]], tf.int32))),
                'float32'))
            return tf.add(i, 1)

        # tf.print(self.correct_labels)

        r = tf.while_loop(cond, body, [i])

    def result(self):
        if tf.equal(self.total_labels, 0.0):
            return 0.0
        return tf.divide(self.correct_labels, self.total_labels)

    def reset_states(self):
        self.total_labels.assign(0.)
        self.correct_labels.assign(0.)


class FinetunedT5(TFT5ForConditionalGeneration):
    def __init__(self, *args, log_dir=None, cache_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.accuracy_1st_token = tf.keras.metrics.Accuracy(name='accuracy_1st_token')
        self.accuracy_all_tokens = MultipleTokensAccuracy(name='accuracy_all_tokens')

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

            loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        lr = self.optimizer._decayed_lr(tf.float32)

        self.loss_tracker.update_state(loss)

        tf.print(y)
        tf.print(y_pred)
        tf.print("*****************")
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

        self.loss_tracker.update_state(loss)
        self.accuracy_all_tokens.update_state(y, y_pred)
        self.accuracy_1st_token.update_state(y_no_eos, y_pred_no_eos)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def predict_step(self, data):
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

        self.loss_tracker.update_state(loss)
        self.accuracy_all_tokens.update_state(y, y_pred)
        self.accuracy_1st_token.update_state(y_no_eos, y_pred_no_eos)

        result = {m.name: m.result() for m in self.metrics}
        result['y_true'] = y
        result['y_pred'] = y_pred

        return result


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
        self.train_batch += 1

    def on_test_batch_end(self, batch, logs=None):
        step = self.step if self.step else 'val'

        logger.debug(
            f"{self.validation_batch}) End of batch {batch} of epoch {self.epoch} of {step}; got log keys: {list(logs.keys())}")

        self.validation_batch += 1
        self.val_acc_all_tokens = logs['accuracy_all_tokens']
        self.val_acc_1st_token = logs['accuracy_1st_token']


def train_test_model(training_ds_fpath, val_ds_fpath):
    optimizer = tf.keras.optimizers.Adam(config.lr)

    callbacks = [CustomCallback()]

    tf_train_ds = to_tf_dataset(
        t5_tokenized_examples(training_ds_fpath, max_len=config.encoder_max_len))

    tf_valid_ds = to_tf_dataset(
        t5_tokenized_examples(val_ds_fpath, max_len=config.encoder_max_len))

    train_ds = create_dataset(tf_train_ds, batch_size=config.batch_size, shuffling=True,
                              cache_path=None, buffer_size=2500)
    valid_ds = create_dataset(tf_valid_ds, batch_size=config.batch_size, shuffling=True,
                              cache_path=None, buffer_size=2500)

    model = FinetunedT5.from_pretrained("t5-base")

    model.compile(optimizer=optimizer)

    logger.info("{}, {}".format(training_ds_fpath, val_ds_fpath))
    hist = model.fit(train_ds, epochs=config.epochs, batch_size=config.batch_size, callbacks=callbacks,
                     validation_data=valid_ds, validation_batch_size=config.batch_size)

    # model.save_weights(os.path.join(wandb.run.dir, "model.h5"))

    model.save(os.path.join(wandb.run.dir, "model.tf"))
    model.save(os.path.join(SETTINGS.get('model_path'), "model.tf"))

    predictions = model.predict(valid_ds, batch_size=config.batch_size, callbacks=callbacks)
    # print(predictions)
    print(predictions['y_pred'].shape)

    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    for i in range(8):

        print(tokenizer.batch_decode(predictions['y_true']))
        print(tokenizer.batch_decode(predictions['y_pred']))

    return hist.history


def run_model():
    global config, history
    if not os.path.exists(SETTINGS.get('data')):
        os.mkdir(SETTINGS.get('data'))

    wandb_params = {
        "project": f"{SETTINGS.get('project')}",
        "dir": f"{SETTINGS.get('data')}",
        "tags": [SYSTEM, "t5-base"],
    }

    if SYSTEM == 'local':
        hparams = {
            'batch_size': 2,
            'encoder_max_len': 128,
            'lr': 0.0001,
            'epochs': 1,
            'dataset': 'tasks_n2_1'
        }
        wandb_params["config"] = hparams

    wandb.init(**wandb_params)
    config = wandb.config
    run = wandb.Api().run(
        "{entity}/{project}/{run_id}".format(entity=WANDB_ENTITY, project=SETTINGS.get('project'), run_id=wandb.run.id))
    run.tags.append(config.dataset)
    run.tags.append(LABELS_TYPE)
    run.update()

    logger.info(
        f"RUN_ID: {wandb.run.id}, DATASET: {config.dataset}, LABELS_TYPE: {LABELS_TYPE}, "
        f"# of EPOCHS: {config.epochs}, LR: {config.lr}")

    training_ds_fpath = TRAINING_DATASET_FNAME.format(dataset_name=config.dataset)
    _, _, a = training_ds_fpath.partition(f"{config.dataset}")
    train_ds = a.split(".")[0]
    first_token_val_accuracies = []
    all_token_val_accuracies = []
    history = train_test_model(training_ds_fpath, VALIDATION_DATASET_FNAME.format(dataset_name=config.dataset))
    first_token_val_accuracies.append(history['val_accuracy_1st_token'])
    all_token_val_accuracies.append(history['val_accuracy_all_tokens'])
    findings = {
        'num_of_epochs': config.epochs,
        'learning_rate': config.lr,
        'training_ds_fpath': train_ds,
        'first_token_val_accuracy': history['val_accuracy_1st_token'],
        'all_token_val_accuracy': history['val_accuracy_all_tokens'],
    }
    wandb.log(findings)
    training_dataset = training_ds_fpath.split('.')[0].split("/")[-1]
    experiment_output = {
        training_dataset: findings
    }
    dataset_dir = f'{SETTINGS.get("root")}/experiment_logs3/{config.dataset}/{LABELS_TYPE}'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    print(dataset_dir)
    experiment_result = f'{dataset_dir}\{train_ds}_{config.epochs}_{config.lr}.json'
    print(experiment_result)
    with open(experiment_result, 'w') as fp:
        json.dump(experiment_output, fp)
    logger.info(f"RUN {wandb.run.id} COMPLETED! - Saved results of {training_dataset} to {experiment_result}")


if __name__ == '__main__':
    run_model()
