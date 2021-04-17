SYSTEM = 'gypsum'
# SYSTEM = 'local'
WANDB_ENTITY = 'kfaria'

# For data, not backed up
SCRATCH_DIR = "/mnt/nfs/scratch1/kfaria"

# For code, imp stuff, backed up
WORKING_DIR = "/mnt/nfs/work1/mccallum/kfaria/t5_finetuning"

DATASET_DIR = '/mnt/nfs/scratch1/tbansal/fewshot'

LABELS_TYPE = "pos_neg"

SETTINGS_DICT = {
    # Unix-style
    'gypsum': {
        'data': SCRATCH_DIR,
        'root': WORKING_DIR,
        'training_dataset': '/mnt/nfs/scratch1/tbansal/fewshot/{dataset_name}_train_{dataset_number}_{dataset_size}.tf_record',
        'val_dataset': '/mnt/nfs/scratch1/tbansal/fewshot/{dataset_name}_eval.tf_record',
        'project': 't5-baselines'
    },
    # Windows-style
    'local': {
        'data': ".\\data",
        'root': ".\\",
        'training_dataset': 'C:/Users/faria/PycharmProjects/t5_finetuning/datasets/{dataset_name}/{dataset_name}_train_{dataset_number}_{dataset_size}.tf_record',
        'val_dataset': 'C:/Users/faria/PycharmProjects/t5_finetuning/datasets/{dataset_name}/{dataset_name}_eval.tf_record',
        'project': 't5-finetuning'
    }
}

SETTINGS = SETTINGS_DICT.get(SYSTEM)

DATASET = {
    "amazon": "amazon_electronics_c",
    "scitail": "scitail_b"
}.get(DATASET_NAME)

TRAINING_DATASET_FNAME = SETTINGS.get('training_dataset')
VALIDATION_DATASET_FNAME = SETTINGS.get('val_dataset')
