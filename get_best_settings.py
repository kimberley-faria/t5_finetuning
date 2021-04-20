import json


def js_r(filename):
    with open(filename) as f_in:
        return json.load(f_in)


if __name__ == "__main__":
    results = js_r(
        r"C:\Users\faria\PycharmProjects\t5_finetuning\experiment_logs"
        r"\consolidated_results_amazon_electronics_c_scitail_b_pos_neg.json")

    best = {}

    for result in results:
        if result['training_dataset_size'] in best:
            if result['avg_val_acc'] > best[result['training_dataset_size']]['avg_val_acc']:
                best[result['training_dataset_size']] = result
            continue
        best[result['training_dataset_size']] = result
    print(best)

    # Results:
    # best = {4: {'training_dataset_size': 4, 'epochs': 20, 'lr': 0.0005, 'avg_val_acc': 0.649075736105442},
    #         8: {'training_dataset_size': 8, 'epochs': 30, 'lr': 0.0005, 'avg_val_acc': 0.6687650725245475},
    #         16: {'training_dataset_size': 16, 'epochs': 40, 'lr': 0.0001, 'avg_val_acc': 0.6987329989671707},
    #         32: {'training_dataset_size': 32, 'epochs': 40, 'lr': 0.0001, 'avg_val_acc': 0.7477426558732987}}
