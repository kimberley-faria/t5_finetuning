from transformers import BertTokenizer, AutoTokenizer
import tensorflow as tf

from config import TRAINING_DATASET_FNAME

tokenizer2 = BertTokenizer.from_pretrained("bert-base-cased")  # Check this!!!
tokenizer = AutoTokenizer.from_pretrained('t5-small')


def get_dataset(fname, seq_length=128):
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
    source_ids = inputs[index]["input_ids"][0]
    target_ids = targets[index]["input_ids"][0]

    src_mask = inputs[index]["attention_mask"][0]  # might need to squeeze
    target_mask = targets[index]["attention_mask"][0]  # might need to squeeze

    return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids,
            "decoder_attention_mask": target_mask}


def t5_tokenized_examples(fname, max_len=128):
    dataset = get_dataset(fname)

    inputs = []
    targets = []

    count = 0
    count_labels = {}
    for data in dataset:
        bert_decoded_input = tokenizer2.decode(data['input_ids'])

        label = {
            0: "national",
            1: "constituency",
        }.get(data['label_ids'].numpy())

        print(bert_decoded_input)
        print(data['label_ids'])
        print(label)
        print(
            "*******************************************************************************************************")
        count += 1
        if data['label_ids'].numpy() not in count_labels:
            count_labels[data['label_ids'].numpy()] = 0
        count_labels[data['label_ids'].numpy()] += 1
    print("Dataset count:", count)
    print("classes count:", count_labels)
    #
    # tokenized_inputs = tokenizer(
    #     bert_decoded_input, max_length=max_len, padding='max_length', return_tensors="tf", truncation=True
    # )
    # tokenized_targets = tokenizer(
    #     label, max_length=2, padding='max_length', return_tensors="tf", truncation=True
    # )
    #
    # inputs.append(tokenized_inputs)
    # targets.append(tokenized_targets)

    # return [get_ids_and_masks(inputs, targets, i) for i in range(len(inputs))]


if __name__ == '__main__':
    dataset = "pa_bnew"
    training_ds_fpath = TRAINING_DATASET_FNAME.format(dataset_name=dataset, dataset_number=0, dataset_size=4)
    _, _, a = training_ds_fpath.partition(f"{dataset}")
    t5_tokenized_examples(training_ds_fpath)
