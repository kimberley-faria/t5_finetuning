import json
import logging
import os
import sys

import tensorflow as tf
import wandb
from transformers import TFT5ForConditionalGeneration, AutoTokenizer, BertTokenizer

from config import SETTINGS, TRAINING_DATASET_FNAME, VALIDATION_DATASET_FNAME, LABELS_TYPE, SYSTEM, WANDB_ENTITY

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_dataset_from_tf_records(fname, seq_length=128):
    name_to_features = {
        # "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        '''Decodes a record to a TensorFlow example.'''
        example = tf.io.parse_single_example(serialized=record, features=name_to_features)
        return example

    dataset = tf.data.TFRecordDataset(fname).map(
        lambda x: _decode_record(x, name_to_features))
    return dataset


def get_ids_and_masks(inputs, targets, index):
    logger.debug(f"Split up the ids and attention_masks for inputs and targets for index {index}")
    source_ids = inputs[index]["input_ids"][0]
    target_ids = targets[index]["input_ids"][0]

    src_mask = inputs[index]["attention_mask"][0]  # might need to squeeze
    target_mask = targets[index]["attention_mask"][0]  # might need to squeeze

    return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids,
            "decoder_attention_mask": target_mask}


def clean_data(data: str):
    return data.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()

def t5_tokenized_examples(fname, max_len=128):
    dataset = get_dataset_from_tf_records(fname)

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    inputs = []
    targets = []

    for data in dataset:
        bert_decoded_input = bert_tokenizer.decode(data['input_ids'])
        input_text = clean_data(bert_decoded_input)

        label = {
            0: "Amenity",
            1: "Cuisine",
            2: "Dish",
            3: "Hours",
            4: "Location",
            5: "Price",
            6: "Rating",
            7: "Restaurant_Name"
        }.get(data['label_ids'].numpy())

        tokenized_inputs = tokenizer(
            input_text, max_length=max_len, padding='max_length', return_tensors="tf", truncation=True
        )
        tokenized_targets = tokenizer(
            label, max_length=5, padding='max_length', return_tensors="tf", truncation=True
        )

        inputs.append(tokenized_inputs)
        targets.append(tokenized_targets)

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
        tf.print(y_true)
        tf.print(y_pred)


        ones = tf.ones(tf.shape(y_true))
        lengths = tf.cast(tf.add(tf.argmax(tf.equal(ones, tf.cast(y_true, tf.float32)), axis=-1), 1), tf.int32)
        tf.print(lengths)
        i = tf.constant(0)

        def cond(i):
            return tf.less(i, 2)

        def body(i):
            self.correct_labels.assign_add(tf.cast(tf.math.reduce_all(
                tf.equal(tf.cast(y_true[i, :lengths[i]], tf.int32), tf.cast(y_pred[i, :lengths[i]], tf.int32))),
                'float32'))
            return tf.add(i, 1)

        tf.print(self.correct_labels)

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

    model = FinetunedT5.from_pretrained("t5-small")

    model.compile(optimizer=optimizer)

    hist = model.fit(train_ds, epochs=config.epochs, batch_size=config.batch_size, callbacks=callbacks,
                     validation_data=valid_ds, validation_batch_size=config.batch_size)

    # model.save_weights(os.path.join(wandb.run.dir, "model.h5"))

    return hist.history


def run_model():
    global config, history
    if not os.path.exists(SETTINGS.get('data')):
        os.mkdir(SETTINGS.get('data'))

    wandb_params = {
        "project": f"{SETTINGS.get('project')}",
        "dir": f"{SETTINGS.get('data')}",
        "tags": [SYSTEM],
    }

    if SYSTEM == 'local':
        hparams = {
            'batch_size': 2,
            'encoder_max_len': 128,
            'lr': 0.00001,
            'epochs': 5,
            'training_ds_number': 0,
            'training_ds_size': 4,
            'dataset': 'restaurant'
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
        f"DATASET #: {config.training_ds_number}, DATASET_SIZE: {config.training_ds_size}, "
        f"# of EPOCHS: {config.epochs}, LR: {config.lr}")

    training_ds_fpath = TRAINING_DATASET_FNAME.format(dataset_name=config.dataset,
                                                      dataset_number=config.training_ds_number,
                                                      dataset_size=config.training_ds_size)
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
    dataset_dir = f'{SETTINGS.get("root")}/experiment_logs2/{config.dataset}/{LABELS_TYPE}'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    experiment_result = f'{dataset_dir}/{training_dataset}_{config.epochs}_{config.lr}.json'
    with open(experiment_result, 'w') as fp:
        json.dump(experiment_output, fp)
    logger.info(f"RUN {wandb.run.id} COMPLETED! - Saved results of {training_dataset} to {experiment_result}")


if __name__ == '__main__':
    run_model()
