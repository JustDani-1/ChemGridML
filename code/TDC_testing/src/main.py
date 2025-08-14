from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import torch, optuna, joblib
import optuna.visualization as vis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters
EPOCHS = 200
PATIENCE = 25
LEARN_RATE = 0.001
BATCH_SIZE = 16
N_FOLDS = 5
VAL_SIZE = 0.15
HIDDEN_SIZE = 128
DROPOUT = 0.3
TEST_SIZE = 0.2

# Set device
DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
if torch.cuda.is_available():
    DEVICE = 'cuda'
print(DEVICE)

# Define network
class FNN(torch.nn.Module):
    def __init__(self, input_size=512, hidden_size=128, dropout_rate=0.3):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size // 2, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)
    
# Get data
data = ADME(name = 'Caco2_Wang')
df = data.get_data()
smiles = df['Drug']
mols = [Chem.MolFromSmiles(x) for x in smiles]
morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(includeChirality=True, radius=2, fpSize=512)

#Labels
labels = torch.tensor(df['Y'], dtype=torch.float32).unsqueeze(1)

# Features
morgans = np.stack([np.array(morgan_fp_gen.GetFingerprint(x)) for x in mols])
X_train, X_test, Y_train, Y_test = train_test_split(morgans, labels, test_size=TEST_SIZE, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)

def train_model(model, loss_fn, optimizer, train_loader, val_loader, epochs):
    """Train the model for given epochs using train & val data"""

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for e in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Patience for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch: {e}")
            break

        if e % 10 == 0:
            print(f"Epoch: {e+1}, Train: {train_loss:.3f}, Val: {val_loss:.3f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def kfold_cv(X, Y, epochs=EPOCHS, hidden_size=HIDDEN_SIZE, dropout=DROPOUT, lr=LEARN_RATE, batch_size=BATCH_SIZE):
    """K-fold CV"""
    
    kfold = KFold(N_FOLDS, shuffle=True, random_state=42)

    avg_val_loss = 0.0
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold+1}/{N_FOLDS}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = FNN(hidden_size=hidden_size, dropout_rate=dropout).to(DEVICE)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model = train_model(model, loss_fn, optimizer, train_loader, val_loader, epochs)

        model.eval()
        X_val = X_val.to(DEVICE)
        with torch.no_grad():
            outputs = model(X_val).cpu()
        val_loss = mean_squared_error(outputs, Y_val)

        avg_val_loss += val_loss

    return avg_val_loss
    
def test_model(model, X, Y):
    model.eval()
    X = X.to(DEVICE)
    with torch.no_grad(): 
        outputs = model(X).cpu()
    return mean_squared_error(outputs, Y)

def objective(trial):
    """Optuna hyperparameter optimization with respect to average validation loss accross folds"""
    # Possible hyperparameters
    epochs = trial.suggest_categorical('epochs', [100, 120, 150, 180, 200])
    hidden_size = trial.suggest_int('hidden_size', 64, 512)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    
    # Run your k-fold CV with these params
    avg_val_loss = kfold_cv(X_train, Y_train, epochs=epochs, hidden_size=hidden_size, lr=lr)
    return avg_val_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

joblib.dump(study, 'optuna_study_04.pkl')


        