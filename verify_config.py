import json

from config import SYSTEM, DATASET, EVALUATION_METHOD, SETTINGS

if __name__ == '__main__':
    print("****************************************************************************************************")
    settings_json = json.dumps(SETTINGS, indent=4)
    print(
        f"\nConfig set to: \nSETTINGS: {settings_json}, \nSYSTEM: {SYSTEM}, \nDATASET: {DATASET}, \nEVAL_METHOD: {EVALUATION_METHOD}\n")

    print("****************************************************************************************************")
    print("\nrun_wandb_experiments.sh contents:")
    print("----------------------------------")
    print(open(f'{SETTINGS.get("root")}/run_wandb_experiments.sh', "r").read())
    print("****************************************************************************************************")
