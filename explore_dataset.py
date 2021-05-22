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


def clean_data(data: str):
    return data.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()


def t5_tokenized_examples(fname, max_len=256):
    dataset = get_dataset(fname)

    inputs = []
    targets = []

    count = 0
    count_labels = {}
    for data in dataset:
        bert_decoded_input = tokenizer2.decode(data['input_ids'])
        bert_input_text = clean_data(bert_decoded_input)
        input_text = f"sentence: {bert_input_text}"

        print("Raw Target:", data['label_ids'].numpy())
        # Restaurant

        target = {
            0: "Organization",
            1: "Other",
            2: "Person",
            3: "Location"
        }.get(data['label_ids'].numpy())

        label = f"entity: {target}"

        # label = {
        #     0: "Relevant",
        #     1: "Not Relevant"
        # }.get(data['label_ids'].numpy())

        count += 1
        if data['label_ids'].numpy() not in count_labels:
            count_labels[data['label_ids'].numpy()] = 0
        count_labels[data['label_ids'].numpy()] += 1

        print("Cleaned input:", input_text)
        print("Target:", label)

        tokenized_inputs = tokenizer(
            input_text, max_length=max_len, padding='max_length', return_tensors="tf", truncation=True
        )
        tokenized_targets = tokenizer(
            label, max_length=5, padding='max_length', return_tensors="tf", truncation=True
        )

        # print("T5 decoded input:", tokenizer.decode(tokenized_inputs['input_ids'][0]))
        # print("T5 decoded target", tokenizer.decode(tokenized_targets['input_ids'][0]))
        print("Target idx:", tokenized_targets['input_ids'][0])

        inputs.append(tokenized_inputs)
        targets.append(tokenized_targets)

        print("****************************************************************************************************")

    print("Dataset count:", count)
    print("classes count:", count_labels)

    return [get_ids_and_masks(inputs, targets, i) for i in range(len(inputs))]


if __name__ == '__main__':
    dataset = "conll_c"
    training_ds_fpath = TRAINING_DATASET_FNAME.format(dataset_name=dataset, dataset_number=1, dataset_size=4)
    _, _, a = training_ds_fpath.partition(f"{dataset}")
    t5_tokenized_examples(training_ds_fpath)
