from transformers import BertTokenizer, AutoTokenizer
import tensorflow as tf

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

    for data in dataset:
        bert_decoded_input = tokenizer2.decode(data['input_ids'])
        print("*******************************************************************************************************")
        print(bert_decoded_input)
        print(data['label_ids'])
        print("*******************************************************************************************************")

        label = "positive" if data['label_ids'] else "negative"

        tokenized_inputs = tokenizer(
            bert_decoded_input, max_length=max_len, padding='max_length', return_tensors="tf", truncation=True
        )
        tokenized_targets = tokenizer(
            label, max_length=2, padding='max_length', return_tensors="tf", truncation=True
        )

        inputs.append(tokenized_inputs)
        targets.append(tokenized_targets)

    return [get_ids_and_masks(inputs, targets, i) for i in range(len(inputs))]


if __name__ == '__main__':
    # record = r"C:\Users\faria\PycharmProjects\t5_finetuning\datasets\scitail\scitail_b_train_0_4.tf_record"
    record = r"C:\Users\faria\PycharmProjects\t5_finetuning\datasets\amazon\amazon_electronics_c_train_0_4.tf_record"
    t5_tokenized_examples(record)