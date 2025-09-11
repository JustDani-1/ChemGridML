# env.py
import time

# Environment
TIMESTAMP = int(time.time())
DEFAULT_FP_SIZE = 1024
TEST_SIZE = 0.2
DEVICE = 'cpu'

# Study parameters
N_TESTS = 10
N_FOLDS = 5
N_TRIALS = 15

# Datasets
DATASETS = [
    'Caco2_Wang', 
    'PPBR_AZ', 
    'Lipophilicity_AstraZeneca', 
    'BBB_Martins', 
    'PAMPA_NCATS', 
    'Pgp_Broccatelli'
]
