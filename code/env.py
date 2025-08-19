import torch

# Environment
DEFAULT_FP_SIZE = 1024
TEST_SIZE = 0.2
N_FOLDS = 5
PATIENCE = 15
DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
if torch.cuda.is_available():
    DEVICE = 'cuda'

# Study
FINGERPRINTS =  ['ECFP', 'AtomPair', 'MACCS', 'RDKitFP', 'TOPOTOR', 'MOL2VEC']
MODELS = ['FNN']
DATASETS = ['Caco2_Wang', 'BBB_Martins']
N_TRIALS = 5