import torch
from sklearn.metrics import mean_squared_error
import env
import numpy as np

def evaluate(target, prediction, task_type):
    if task_type == 'regression':
        return mean_squared_error(target, prediction)
    elif task_type == 'classification':
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
        prediction = torch.tensor(prediction, dtype=torch.float32).unsqueeze(1)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        return loss_fn(prediction, target).item()

def rrmse(target, prediction):
    rmse = np.sqrt(mean_squared_error(target, prediction))
    mean_target = np.mean(target)
    
    # Edge case where mean is zero
    if abs(mean_target) < 1e-10:
        # If targets are near zero, use standard deviation as denominator
        std_target = np.std(target)
        if std_target < 1e-10:
            return 0.0  # Perfect prediction case
        rrmse = rmse / std_target
    else:
        rrmse = rmse / abs(mean_target)
    
    return rrmse

def train_with_val(model, optimizer, train_loader, val_loader, epochs):
    """Train the model for given epochs using train with early stopping via patience on val"""

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for e in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(env.DEVICE)
            batch_y = batch_y.to(env.DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = model.loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(env.DEVICE)
                batch_y = batch_y.to(env.DEVICE)
                outputs = model(batch_x)
                loss = model.loss_fn(outputs, batch_y)
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

        if patience_counter >= env.PATIENCE:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def train_without_val(model, optimizer, train_loader, epochs):
    """Train the model for given epochs using train, no early stopping"""

    model.train()
    for e in range(epochs):
        # Training
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(env.DEVICE)
            batch_y = batch_y.to(env.DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = model.loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

def get_task_type(Y):
    return 'classification' if Y[0] == 0 or Y[0] == 1 else 'regression'