SYSTEM = 'gypsum'
# SYSTEM = 'local'

# For data, not backed up
SCRATCH_DIR = "/mnt/nfs/scratch1/kfaria"

# For code, imp stuff, backed up
WORKING_DIR = "/mnt/nfs/work1/mccallum/kfaria/t5_finetuning"

SETTINGS_DICT = {
    # Unix-style
    'gypsum': {
        'data': SCRATCH_DIR,
        'root': WORKING_DIR
    },
    # Windows-style
    'local': {
        'data': ".\\data",
        'root': ".\\"
    }
}

SETTINGS = SETTINGS_DICT.get(SYSTEM)

DATASET_NAME = "scitail"
DATASET = {
    "amazon": "amazon_electronics_c",
    "scitail": "scitail_b"
}.get(DATASET_NAME)

TRAINING_DATASET_FNAME = '/mnt/nfs/scratch1/tbansal/fewshot/{dataset_name}_train_{dataset_number}_{dataset_size}.tf_record'
VALIDATION_DATASET_FNAME = '/mnt/nfs/scratch1/tbansal/fewshot/{dataset_name}_eval.tf_record'

DEBUG = True