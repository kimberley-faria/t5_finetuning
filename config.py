import os

SYSTEM = 'gypsum'
# SYSTEM = 'local'

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
WANDB_ENTITY = 'kfaria'

# For data, not backed up
SCRATCH_DIR = "/mnt/nfs/scratch1/kfaria"

# For code, imp stuff, backed up
WORKING_DIR = "/mnt/nfs/work1/mccallum/kfaria/t5_finetuning"

DATASET_DIR = '/mnt/nfs/scratch1/tbansal/fewshot'

LABELS_TYPE = "entailment_swap"

SETTINGS_DICT = {
    # Unix-style
    'gypsum': {
        'data': SCRATCH_DIR,
        'root': WORKING_DIR,
        'training_dataset': '/mnt/nfs/scratch1/tbansal/fewshot/{dataset_name}_train_{dataset_number}_{'
                            'dataset_size}.tf_record',
        'val_dataset': '/mnt/nfs/scratch1/tbansal/fewshot/{dataset_name}_eval.tf_record',
        'project': 't5-baselines'
    },
    # Windows-style
    'local': {
        'data': os.path.join(BASE_DIR, "data"),
        'root': BASE_DIR,
        'training_dataset': os.path.join(BASE_DIR,
                                         "datasets\\{dataset_name}\\{dataset_name}_train_{dataset_number}_{"
                                         "dataset_size}.tf_record"),
        'val_dataset': os.path.join(BASE_DIR,
                                    "datasets\\{dataset_name}\\{dataset_name}_eval.tf_record"),
        'project': 't5-finetuning'
    }
}

SETTINGS = SETTINGS_DICT.get(SYSTEM)

# DATASET = {
#     "amazon": "amazon_electronics_c",
#     "amazon_t": "amazon_electronics_t",
#     "scitail": "scitail_b",
#     "amazon_books_t": "amazon_books_t",
#     "amazon_kitchen_t": "amazon_kitchen_t",
#     "amazon_dvd_t": "amazon_dvd_t",
#     "conll_c": "conll_c",
#     "restaurant": "restaurant",
#     "airline": "airline",
#     "pa_bnew": "pa_bnew",
#     "pb_bnew":"pb_bnew",
#     "disaster_new": "disaster_new",
#     "amazonr_electronics": "amazonr_electronics",
#     "amazonr_books": "amazonr_books",
#     "amazonr_kitchen": "amazonr_kitchen",
#     "amazonr_dvd": "amazonr_dvd",
#     "emotion_new": "emotion_new",
# }.get(DATASET_NAME)

TRAINING_DATASET_FNAME = SETTINGS.get('training_dataset')
VALIDATION_DATASET_FNAME = SETTINGS.get('val_dataset')
