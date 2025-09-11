import time

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
FPS: list[str] = []#['ECFP', 'AtomPair', 'MACCS', 'RDKitFP', 'TOPOTOR', 'MOL2VEC']
MDLS: list[str] = []#['FNN', 'RF', 'XGBoost', 'SVM', 'ElasticNet', 'KNN']
# Learnables
LEARNABLES = ['GCN', 'GAT']
# Pre-trained
FTRS = ['GRAPH']
PRETRAINEDS = []#['MOLBERT', 'GROVER']

# Combinations
FEATURES = FPS + FTRS
MODELS = MDLS + LEARNABLES
DATASETS = ['Caco2_Wang', 'PPBR_AZ', 'Lipophilicity_AstraZeneca', 'BBB_Martins', 'PAMPA_NCATS', 'Pgp_Broccatelli']

