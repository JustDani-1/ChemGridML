import torch, time

# Environment
TIMESTAMP = int(time.time())
DEFAULT_FP_SIZE = 1024
TEST_SIZE = 0.2
PATIENCE = 15
DEVICE = 'cpu'

# Study
N_TESTS = 10
N_FOLDS = 5
N_TRIALS = 15
# Fingerprints
FINGERPRINTS = ['ECFP', 'AtomPair', 'MACCS', 'RDKitFP', 'TOPOTOR', 'MOL2VEC']
MODELS = ['FNN', 'RF', 'XGBoost', 'SVM', 'ElasticNet', 'KNN']
# Learnables
LEARNABLES = ['GCN', 'GAT']
# Pre-trained
PRETRAINEDS = ['MOLBERT', 'GROVER']
# Datasets
DATASETS = ['Caco2_Wang', 'PPBR_AZ', 'Lipophilicity_AstraZeneca', 'BBB_Martins', 'PAMPA_NCATS', 'Pgp_Broccatelli']

