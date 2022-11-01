import json

from config import SYSTEM, LABELS_TYPE, SETTINGS

if __name__ == '__main__':
    print("****************************************************************************************************")
    settings_json = json.dumps(SETTINGS, indent=4)
    print(f"\nConfig set to: \nSETTINGS: {settings_json}, \nSYSTEM: {SYSTEM}, \nLABELS_TYPE: {LABELS_TYPE}\n")
    print("****************************************************************************************************")
    print("\nrun_wandb_experiments.sh contents:")
    print("----------------------------------")
    print(open(f'{SETTINGS.get("root")}/run_wandb_experiments.sh', "r").read())
    print("****************************************************************************************************")
    for size in [4, 8, 16, 32]:
        print(f"\nsweep{size}.yaml contents:")
        print("----------------------")
        print(open(f'{SETTINGS.get("root")}/sweep{size}.yaml', "r").read())
        print("****************************************************************************************************")

    print(f"\nsweep.yaml contents:")
    print("----------------------")
    print(open(f'{SETTINGS.get("root")}/sweep.yaml', "r").read())
    print("****************************************************************************************************")
