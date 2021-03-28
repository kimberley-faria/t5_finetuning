SYSTEM = 'gypsum'
# SYSTEM = 'local'

# For data, not backed up
SCRATCH_DIR = "/mnt/nfs/scratch1/kfaria"

# For code, imp stuff, backed up
WORKING_DIR = "/mnt/nfs/work1/mccallum/kfaria"

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

AMZN_TRAINING_DATASETS = '/mnt/nfs/scratch1/tbansal/fewshot/amazon_electronics_c_train_{}_4.tf_record'
AMZN_VALIDATION_DATASET = '/mnt/nfs/scratch1/tbansal/fewshot/amazon_electronics_c_eval.tf_record'
