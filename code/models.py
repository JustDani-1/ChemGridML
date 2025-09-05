import torch
import xgboost as xgb
from abc import ABC, abstractmethod
import env
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from torch.utils.data import DataLoader, TensorDataset

def get_activation(activation):
    activations = {
        'relu': torch.nn.ReLU(),
        'tanh': torch.nn.Tanh(),
        'sigmoid': torch.nn.Sigmoid()
    }
    return activations[activation]

class ModelRegistry:
    models = {}
    
    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls.models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name):
        return cls.models[name]
    

class ModelBase(ABC):
    def __init__(self, task_type, **kwargs):
        self.task_type = task_type
        self.hyperparams = kwargs
        self.model = None

    @abstractmethod
    def preprocess(self, X_train, X_val=None, Y_train=None, Y_val=None):
        """Perform framework- and model-specific preprocessing of the data"""
        pass

    @abstractmethod
    def fit(self, X_train, Y_train, X_val=None, Y_val=None):
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X):
        """Get the prediction of the model for the given data"""
        pass

    @staticmethod
    @abstractmethod
    def get_hyperparameter_space(trial):
        """Return model-specific hyperparameters for Optuna trial"""
        pass

class PytorchBase(ModelBase):
    def __init__(self, task_type, **kwargs):
        super().__init__(task_type, **kwargs)

    def preprocess(self, X_train, X_val, Y_train, Y_val):
        """Apply scaling and convert to tensors"""
        # Variance thresholding
        threshold = VarianceThreshold(threshold=0.0)
        threshold.fit(X_train)
        X_train = threshold.transform(X_train)
        X_val = threshold.transform(X_val)

        # Scaling
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
        Y_val = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)

        return X_train, X_val, Y_train, Y_val

    def fit(self, X_train, Y_train):
        # Create actual model
        input_size = X_train.shape[1]
        self.model = self._create_model(input_size).to(env.DEVICE)
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(X_train, Y_train),
            batch_size=self.hyperparams['batch_size'],
            shuffle=True
        )
        
        # Train model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])
        
        from util import train_without_val
        self.model = train_without_val(self.model, optimizer, train_loader, self.hyperparams['epochs'])


    def predict(self, X):
        """Predict using the trained model"""
        
        # Predict in batches
        self.model.eval()
        predictions = []
        batch_size = self.hyperparams.get('batch_size', 64)
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                batch_pred = self.model(batch).cpu().detach().numpy()
                predictions.append(batch_pred)
        
        return np.concatenate(predictions, axis=0).flatten()

    @abstractmethod
    def _create_model(self, input_size):
        """Create the actual PyTorch model"""
        pass

class SklearnBase(ModelBase):
    def __init__(self, task_type, **kwargs):
        super().__init__(task_type, **kwargs)

    def preprocess(self, X_train, X_val, Y_train, Y_val):
        """Apply scaling for sklearn models"""
        
        # Variance thresholding
        threshold = VarianceThreshold(threshold=0.0)
        threshold.fit(X_train)
        X_train = threshold.transform(X_train)
        X_val = threshold.transform(X_val)

        # Scaling (use different scalers based on model type)
        scaler_class = self._get_scaler_class()
        scaler = scaler_class()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        return X_train, X_val, Y_train, Y_val

    def fit(self, X_train, Y_train):
        """Train the sklearn model"""
        
        # Create actual model
        self.model = self._create_model()
        
        # Fit model
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        """Predict using the trained model"""
        return self.model.predict(X).flatten()

    def _get_scaler_class(self):
        """Return the appropriate scaler class for this model"""
        return StandardScaler

    @abstractmethod
    def _create_model(self):
        """Create the actual sklearn model"""
        pass

class DeepchemBase(ModelBase):
    def __init__(self, task_type, **kwargs):
        super().__init__(task_type, **kwargs)
        # TODO: Implement deepchem base class

    def preprocess(self, X_train, X_val, Y_train, Y_val):
        # TODO: Implement deepchem preprocessing
        pass

    def fit(self, X_train, Y_train):
        # TODO: Implement deepchem fitting
        pass

    def predict(self, X):
        # TODO: Implement deepchem prediction
        pass


# Concrete model implementations

@ModelRegistry.register('FNN')
class FNN(PytorchBase):
    def _create_model(self, input_size):
        """Create the FeedForward Neural Network"""
        class FNNModel(torch.nn.Module):
            def __init__(self, input_size, hidden_sizes, dropout_rate, activation, task_type):
                super().__init__()
                
                if task_type == 'regression':
                    self.loss_fn = torch.nn.MSELoss()
                else:
                    self.loss_fn = torch.nn.BCEWithLogitsLoss()
                
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_sizes:
                    layers.extend([
                        torch.nn.Linear(prev_size, hidden_size),
                        get_activation(activation),
                        torch.nn.Dropout(dropout_rate)
                    ])
                    prev_size = hidden_size
                    
                layers.append(torch.nn.Linear(prev_size, 1))
                self.network = torch.nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)
        
        return FNNModel(
            input_size=input_size,
            hidden_sizes=self.hyperparams.get('hidden_sizes', [128, 64]),
            dropout_rate=self.hyperparams.get('dropout_rate', 0.3),
            activation=self.hyperparams.get('activation', 'relu'),
            task_type=self.task_type
        )

    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            # Model hyperparameters
            'hidden_sizes': trial.suggest_categorical('hidden_sizes', [[128], [128, 64], [256, 128, 64]]),
            'dropout_rate': trial.suggest_categorical('dropout_rate', [0.0, 0.1, 0.25, 0.4, 0.5]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            
            # Training hyperparameters
            'epochs': trial.suggest_categorical('epochs', [50, 100, 150, 200]),
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
        }

@ModelRegistry.register('RF')
class RandomForestModel(SklearnBase):
    def _create_model(self):
        if self.task_type == 'regression':
            return RandomForestRegressor(
                n_estimators=self.hyperparams.get('n_estimators', 100),
                max_depth=self.hyperparams.get('max_depth', None),
                min_samples_split=self.hyperparams.get('min_samples_split', 2),
                min_samples_leaf=self.hyperparams.get('min_samples_leaf', 1),
                random_state=42,
                n_jobs=-1
            )
        else:
            return RandomForestClassifier(
                n_estimators=self.hyperparams.get('n_estimators', 100),
                max_depth=self.hyperparams.get('max_depth', None),
                min_samples_split=self.hyperparams.get('min_samples_split', 2),
                min_samples_leaf=self.hyperparams.get('min_samples_leaf', 1),
                random_state=42,
                n_jobs=-1
            )
    
    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200, 300]),
            'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20]),
            'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),
        }

@ModelRegistry.register('XGBoost')
class XGBoostModel(SklearnBase):
    def _create_model(self):
        if self.task_type == 'regression':
            return xgb.XGBRegressor(
                n_estimators=self.hyperparams.get('n_estimators', 100),
                max_depth=self.hyperparams.get('max_depth', 6),
                learning_rate=self.hyperparams.get('learning_rate', 0.1),
                subsample=self.hyperparams.get('subsample', 1.0),
                colsample_bytree=self.hyperparams.get('colsample_bytree', 1.0),
                random_state=42,
                n_jobs=-1
            )
        else:
            return xgb.XGBClassifier(
                n_estimators=self.hyperparams.get('n_estimators', 100),
                max_depth=self.hyperparams.get('max_depth', 6),
                learning_rate=self.hyperparams.get('learning_rate', 0.1),
                subsample=self.hyperparams.get('subsample', 1.0),
                colsample_bytree=self.hyperparams.get('colsample_bytree', 1.0),
                random_state=42,
                n_jobs=-1
            )
    
    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200, 300]),
            'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7, 8]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }

@ModelRegistry.register('SVM')
class SVMModel(SklearnBase):
    def _get_scaler_class(self):
        """SVM uses RobustScaler"""
        return RobustScaler
    
    def _create_model(self):
        if self.task_type == 'regression':
            return SVR(
                C=self.hyperparams.get('C', 1.0),
                gamma=self.hyperparams.get('gamma', 'scale'),
                kernel=self.hyperparams.get('kernel', 'rbf')
            )
        else:
            return SVC(
                C=self.hyperparams.get('C', 1.0),
                gamma=self.hyperparams.get('gamma', 'scale'),
                kernel=self.hyperparams.get('kernel', 'rbf'),
                probability=True
            )
    
    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            'C': trial.suggest_float('C', 0.001, 10, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
        }

@ModelRegistry.register('ElasticNet')
class ElasticNetModel(SklearnBase):
    def _create_model(self):
        if self.task_type == 'regression':
            return ElasticNet(
                alpha=self.hyperparams.get('alpha', 1.0),
                l1_ratio=self.hyperparams.get('l1_ratio', 0.5),
                random_state=42
            )
        else:
            return LogisticRegression(
                C=1.0/self.hyperparams.get('alpha', 1.0),
                l1_ratio=self.hyperparams.get('l1_ratio', 0.5),
                penalty='elasticnet',
                solver='saga',
                random_state=42,
                max_iter=1000
            )
    
    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            'alpha': trial.suggest_float('alpha', 0.001, 10, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
        }

@ModelRegistry.register('KNN')
class KNNModel(SklearnBase):
    def _create_model(self):
        if self.task_type == 'regression':
            return KNeighborsRegressor(
                n_neighbors=self.hyperparams.get('n_neighbors', 5),
                weights=self.hyperparams.get('weights', 'uniform'),
                metric=self.hyperparams.get('metric', 'minkowski')
            )
        else:
            return KNeighborsClassifier(
                n_neighbors=self.hyperparams.get('n_neighbors', 5),
                weights=self.hyperparams.get('weights', 'uniform'),
                metric=self.hyperparams.get('metric', 'minkowski')
            )
    
    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            'n_neighbors': trial.suggest_categorical('n_neighbors', [3, 5, 7, 9, 11]),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['minkowski', 'manhattan', 'chebyshev']),
        }