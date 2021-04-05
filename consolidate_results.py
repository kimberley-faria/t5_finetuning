import json
from statistics import mean

from config import SETTINGS


def js_r(filename):
    with open(filename) as f_in:
        return json.load(f_in)


if __name__ == "__main__":
    training_ds_size = [4, 8, 16, 32]
    training_ds_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    epochs = [5, 10, 20, 30, 40, 50]
    learning_rate = [0.001, 0.0001, 0.0005, 0.00001]

    consolidated_results = {}
    for ds_size in training_ds_size:
        for e in epochs:
            for lr in learning_rate:
                consolidated_acc = 0
                exp_results = [js_r(
                    f"{SETTINGS.get('root')}/experiment_logs/amazon_electronics_c_train_{ds_no}_{ds_size}_{e}_{lr}.json")
                               for ds_no in training_ds_number]
                exp_accs = [res["all_token_val_accuracy"][-1] for res in exp_results]
                consolidated_results[(ds_size, e, lr)] = mean(exp_accs)

    with open(f'{SETTINGS.get("root")}/experiment_logs/consolidated_results.json', 'w') as fp:
        json.dump(consolidated_results, fp)
