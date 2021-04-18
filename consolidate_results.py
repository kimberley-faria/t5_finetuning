import json
import sys
from statistics import mean, stdev

from config import SETTINGS


def js_r(filename):
    with open(filename) as f_in:
        return json.load(f_in)


if __name__ == "__main__":
    training_dataset = sys.argv[1]
    labels_type = sys.argv[2]
    training_ds_size = [4, 8, 16, 32]
    training_ds_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    epochs = [5, 10, 20, 30, 40, 50]
    learning_rate = [0.001, 0.0001, 0.0005, "1e-05"]

    consolidated_results = []
    for ds_size in training_ds_size:
        for e in epochs:
            for lr in learning_rate:
                consolidated_acc = 0
                exp_results = [js_r(
                    f"{SETTINGS.get('root')}\\experiment_logs\\{training_dataset}\\{labels_type}\\{training_dataset}_train_{ds_no}_{ds_size}_{e}_{lr}.json")[
                                   f"{training_dataset}_train_{ds_no}_{ds_size}"]
                               for ds_no in training_ds_number]
                exp_accs = [res["all_token_val_accuracy"][-1] for res in exp_results]
                consolidated_results.append({
                    "training_dataset_size": ds_size,
                    "epochs": e,
                    "lr": lr,
                    "avg_val_acc": mean(exp_accs),
                    "sd": stdev(exp_accs)
                })

    with open(f'{SETTINGS.get("root")}\\experiment_logs\\{training_dataset}\\{labels_type}\\consolidated_results_{training_dataset}.json', 'w') as fp:
        json.dump(consolidated_results, fp)
