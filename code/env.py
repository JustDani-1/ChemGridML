# env.py
import time, torch

# Environment
DEFAULT_FP_SIZE = 1024
TEST_SIZE = 0.2
DEVICE = 'cpu' #'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(DEVICE)

# Study parameters
N_TESTS = 10
N_FOLDS = 5
N_TRIALS = 15

