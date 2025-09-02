import torch, time

# Environment
TIMESTAMP = int(time.time())
DEFAULT_FP_SIZE = 2048
TEST_SIZE = 0.2
N_TESTS = 5
N_FOLDS = 5
PATIENCE = 15
DEVICE = 'cpu'
# if torch.backends.mps.is_available():
#    DEVICE = 'mps'
#if torch.cuda.is_available():
#    DEVICE = 'cuda'

# Study
FINGERPRINTS = ['ECFP', 'AtomPair', 'MACCS', 'RDKitFP', 'TOPOTOR', 'MOL2VEC']
MODELS = ['FNN', 'RF', 'XGBoost', 'SVM', 'ElasticNet', 'KNN']
DATASETS = ['Caco2_Wang', 'PPBR_AZ', 'Lipophilicity_AstraZeneca', 'BBB_Martins', 'PAMPA_NCATS', 'Pgp_Broccatelli']
N_TRIALS = 10
