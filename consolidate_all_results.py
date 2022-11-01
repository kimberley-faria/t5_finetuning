import glob
import json

import pandas as pd


def js_r(filename):
    try:
        with open(filename) as f_in:
            return json.load(f_in)
    except:
        print(f"Unable to load file {filename}.")
        return None


if __name__ == "__main__":
    experiment_logs2 = r"C:\Users\faria\PycharmProjects\t5_finetuning\experiment_logs2"
    baseline_hparam_settings = {
        4: {
            "epochs": 20,
            "lr": 0.0005
        },
        8: {
            "epochs": 50,
            "lr": 0.0001
        },
        16: {
            "epochs": 50,
            "lr": 0.0001
        },
        32: {
            "epochs": 50,
            "lr": 0.0001
        },

    }

    rows = []
    for f in glob.glob(f"{experiment_logs2}\*\*\*.json"):
        split_file_name = f.split("\\")
        json_result = js_r(f)
        data = next(iter(json_result.values()))
        # print(data)
        split_training_ds_fp = data['training_ds_fpath'].split("_")
        result = {}
        result['dataset'] = split_file_name[-3]
        result['task_type'] = split_file_name[-2]
        result['size'] = split_training_ds_fp[-1]
        result['dataset_number'] = split_training_ds_fp[-2]
        result['num_epochs'] = data['num_of_epochs']
        result['learning_rate'] = data['learning_rate']
        result['first_token_val_accuracy'] = data['first_token_val_accuracy'][-1]
        result['all_token_val_accuracy'] = data['all_token_val_accuracy'][-1]


        hparam_settings = baseline_hparam_settings[int(result['size'])]
        print(hparam_settings)
        print(result['num_epochs'], hparam_settings['lr'])
        if hparam_settings['epochs'] == result['num_epochs'] and hparam_settings['lr'] == result['learning_rate']:

            rows.append(result)


    df = pd.DataFrame(rows)
    print(df.head)
    df.to_excel("output_all.xlsx")
