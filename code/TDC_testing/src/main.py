from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import torch, optuna, joblib
import optuna.visualization as vis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(DEVICE)

class NN(torch.nn.Module):
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

# Features and Labels
morgans = np.stack([np.array(morgan_fp_gen.GetFingerprint(x)) for x in mols])
labels = torch.tensor(df['Y'], dtype=torch.float32).unsqueeze(1)
    
# Hyperparameters
EPOCHS = 200
PATIENCE = 25
LEARN_RATE = 0.001
BATCH_SIZE = 16
N_FOLDS = 5
VAL_SIZE = 0.15
HIDDEN_SIZE = 128
DROPOUT = 0.3

def train_model(model, train_loader, val_loader, epochs, lr):
    """Train the model for given epochs using train & val data"""
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
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
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch: {e}")
            break

        if e % 10 == 0:
            print(f"Epoch: {e+1}, Train: {train_loss:.3f}, Val: {val_loss:.3f}")

    return model, best_val_loss


def kfold_cv(X, Y, epochs=EPOCHS, hidden_size=HIDDEN_SIZE, dropout=DROPOUT, lr=LEARN_RATE, batch_size=BATCH_SIZE):
    """Train using kFold Cross Validation to avoid overfitting"""
    kfold = KFold(N_FOLDS, shuffle=True, random_state=42)

    avg_test_loss = 0.0

    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold+1}/{N_FOLDS}")
        # Split data into train, val and test
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        Y_train_val, Y_test = Y[train_val_idx], Y[test_idx]

        val_idx = int(len(X_train_val) * (1 - VAL_SIZE))
        X_train, X_val = X_train_val[:val_idx], X_train_val[val_idx:]
        Y_train, Y_val = Y_train_val[:val_idx], Y_train_val[val_idx:]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
        X_val_scaled = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
        X_test_scaled = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_scaled, Y_train)
        val_dataset = TensorDataset(X_val_scaled, Y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = NN(hidden_size=hidden_size, dropout_rate=dropout).to(DEVICE)
        model, best_val_loss = train_model(model, train_loader, val_loader, epochs, lr)

        model.eval()
        with torch.no_grad():
            X_test_scaled = X_test_scaled.to(DEVICE)
            outputs = model(X_test_scaled).cpu()
            test_loss = mean_squared_error(outputs, Y_test)

        avg_test_loss += test_loss
        
        print(f"Fold Result: Best Val: {best_val_loss:.3f}, Test Loss: {test_loss:.3f}")

    return avg_test_loss / N_FOLDS


def objective(trial):
    # Possible hyperparameters
    epochs = trial.suggest_categorical('epochs', [100, 120, 150, 180, 200])
    hidden_size = trial.suggest_int('hidden_size', 64, 512)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    
    # Run your k-fold CV with these params
    avg_train_loss = kfold_cv(morgans, labels, epochs=epochs, hidden_size=hidden_size, lr=lr)
    return avg_train_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

joblib.dump(study, 'optuna_study_02.pkl')

        