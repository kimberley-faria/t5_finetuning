SYSTEM = 'local'

# For data, not backed up
SCRATCH_DIR = "/mnt/nfs/scratch1/kfaria"

# For code, imp stuff, backed up
WORKING_DIR = "/mnt/nfs/work1/mccallum/kfaria"

SETTINGS_DICT = {
    'gypsum': {
        'data': SCRATCH_DIR,
        'root': WORKING_DIR
    },
    'local': {
        'data': "../data",
        'root': "../"
    }
}

SETTINGS = SETTINGS_DICT.get(SYSTEM)
