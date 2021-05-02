import json
import os
import sys
from statistics import mean, stdev

from config import SETTINGS


def js_r(filename):
    try:
        with open(filename) as f_in:
            return json.load(f_in)
    except:
        print(f"Unable to load file {filename}.")
        return None


if __name__ == "__main__":
    baseline_hparam_settings = {
        4: {
            "epochs": 20,
            "lr": 0.0005
        },
        8: {
            "epochs": 30,
            "lr": 0.0005
        },
        16: {
            "epochs": 40,
            "lr": 0.0001
        },
        32: {
            "epochs": 40,
            "lr": 0.0001
        },

    }

    experiment_logs = r"C:\Users\faria\PycharmProjects\t5_finetuning\t5_baseline_results\experiment_logs"
    base_name, dataset_dirs, _ = next(os.walk(experiment_logs))
    consolidated_results = {}
    for dataset_dir in dataset_dirs:
        print(f"Consolidating {dataset_dir}...")
        consolidated_results[dataset_dir] = []
        for root, dir, files in os.walk(os.path.join(base_name, dataset_dir)):
            if not files:
                continue
            for ds_size, hparams in baseline_hparam_settings.items():
                print(f"Consolidating for size: {ds_size} and hparams: {hparams}.")
                e = hparams["epochs"]
                lr = hparams["lr"]
                results_files = [os.path.join(root, f"{dataset_dir}_train_{ds_no}_{ds_size}_{e}_{lr}.json") for ds_no in
                                 range(10)]
                print(f"Received {len(results_files)} files.")
                # print("-----------------------------------------------------------------------------------------------")
                exp_results = []
                for file in results_files:
                    json_result = js_r(file)
                    if json_result:
                        exp_results.append(json_result)

                print(f"Received {len(exp_results)} result sets.")
                exp_accs = []

                for idx, ds_no in enumerate(range(len(exp_results))):
                    if "disaster" in dataset_dir and ds_size == 16:
                        print("Disaster Dataset, 0 file is corrupted, incrementing ds_no by 1")
                        ds_no += 1
                    exp_accs.append(
                        exp_results[idx][f"{dataset_dir}_train_{ds_no}_{ds_size}"]["all_token_val_accuracy"][-1])

                print(f"Received {len(exp_accs)} accuracies for sub-datasets.")
                results_dict = {
                    "training_dataset_size": ds_size,
                    "epochs": e,
                    "lr": lr,
                    "avg_val_acc": mean(exp_accs),
                    "sd": stdev(exp_accs)
                }
                if "scitail" in dataset_dir:
                    results_dict["labels"] = os.path.basename(root)
                consolidated_results[dataset_dir].append(results_dict)

    final_results_file = os.path.join(f'{SETTINGS.get("root")}', 'experiment_logs', f'consolidated_baselines.json')
    with open(final_results_file, 'w') as fp:
        json.dump(consolidated_results, fp)
    print(f"written results to {final_results_file}.")
