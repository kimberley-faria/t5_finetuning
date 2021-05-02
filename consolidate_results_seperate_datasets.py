import json
import os
import sys
from statistics import mean

from config import SETTINGS


def js_r(filename):
    with open(filename) as f_in:
        return json.load(f_in)


if __name__ == "__main__":
    training_dataset_1 = sys.argv[1]
    training_dataset_2 = sys.argv[2]
    label_types = sys.argv[3]

    dataset1_results = js_r(os.path.join(f'{SETTINGS.get("root")}', 'experiment_logs2', training_dataset_1, label_types,
                                         f'consolidated_results_{training_dataset_1}.json'))

    dataset2_results = js_r(os.path.join(f'{SETTINGS.get("root")}', 'experiment_logs2', training_dataset_2, label_types,
                                         f'consolidated_results_{training_dataset_2}.json'))

    consolidated_results = []
    for ds1_result, ds2_result in zip(dataset1_results, dataset2_results):
        if ds1_result["training_dataset_size"] != ds2_result["training_dataset_size"] and ds1_result["epochs"] != \
                ds2_result["epochs"] and ds1_result["lr"] != ds2_result["lr"]:
            raise Exception("H-Params don't match!")

        consolidated_results.append({
            "training_dataset_size": ds1_result["training_dataset_size"],
            "epochs": ds1_result["epochs"],
            "lr": ds1_result["lr"],
            "avg_val_acc": mean([ds1_result["avg_val_acc"], ds2_result["avg_val_acc"]]),
        })

    with open(
            os.path.join(f'{SETTINGS.get("root")}', 'experiment_logs2',
                         f'consolidated_results_{training_dataset_1}_{training_dataset_2}_{label_types}.json'),
            'w') as fp:
        json.dump(consolidated_results, fp)
