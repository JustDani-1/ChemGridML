import torch

# Environment
DEFAULT_FP_SIZE = 256
TEST_SIZE = 0.2
N_FOLDS = 5
PATIENCE = 15
DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
if torch.cuda.is_available():
    DEVICE = 'cuda'

# Study
FINGERPRINTS = ['ECFP', 'AtomPair', 'MACCS', 'RDKitFP', 'TOPOTOR', 'MOL2VEC'] #['ECFP', 'AtomPair'] #
MODELS = ['FNN']
DATASETS = ['Caco2_Wang', 'BBB_Martins'] #['Caco2_Wang', 'BBB_Martins', 'PPBR_AZ', 'Lipophilicity_AstraZeneca']
N_TRIALS = 3