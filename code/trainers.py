import torch
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, roc_auc_score
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

    return model

class BaseTrainer(ABC):
    
    @abstractmethod
    def evaluate(self, model, loader):
        """Determine the value of the loss function on the given set"""
        pass

    @abstractmethod
    def metric(self, model, test_loader):
        """Determine the final performance metric on the given set"""
        pass

class RegressionTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()
    
    def evaluate(self, model, loader):
        model.eval()
        prediction, target = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(env.DEVICE), batch_y.to(env.DEVICE)
                outputs = model(batch_x)
                prediction.extend(outputs.cpu().numpy())
                target.extend(batch_y.cpu().numpy())
        
        return mean_squared_error(target, prediction)
    
    def metric(self, model, test_loader):
        model.eval()
        prediction, target = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(env.DEVICE), batch_y.to(env.DEVICE)
                outputs = model(batch_x)
                prediction.extend(outputs.cpu().numpy())
                target.extend(batch_y.cpu().numpy())
        
        prediction = np.array(prediction)
        target = np.array(target)
        
        return {'RRMSE': rrmse(target, prediction)}

class ClassificationTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def evaluate(self, model, loader):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(env.DEVICE), batch_y.to(env.DEVICE)
                outputs = model(batch_x)
                loss = self.loss_fn(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def metric(self, model, test_loader):
        model.eval()
        prediction, target = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(env.DEVICE), batch_y.to(env.DEVICE)
                outputs = torch.sigmoid(model(batch_x))
                prediction.extend(outputs.cpu().numpy())
                target.extend(batch_y.cpu().numpy())
        
        return {'AUROC': roc_auc_score(target, prediction)}
    

def get_trainer(task_type):
    return RegressionTrainer() if task_type == 'regression' else ClassificationTrainer()

def get_task_type(Y):
    return 'classification' if Y[0] == 0 or Y[0] == 1 else 'regression'