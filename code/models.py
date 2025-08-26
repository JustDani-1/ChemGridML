import torch
import xgboost as xgb
from abc import ABC, abstractmethod
import env
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

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
    
    @classmethod
    def get_framework(cls, name):
        model_class = cls.models[name]
        return model_class.framework
    

class BaseModel(ABC):
    framework = None

    @abstractmethod
    def get_hyperparameter_space(self, trial):
        """Return model-specific hyperparameters for Optuna trial"""
        pass


@ModelRegistry.register('FNN')
class FNN(BaseModel, torch.nn.Module):
    framework = 'pytorch'

    def __init__(self, **kwargs):
        super().__init__()
        self.input_size = kwargs.get('input_size', env.DEFAULT_FP_SIZE)
        self.task_type = kwargs.get('task_type', env.DEFAULT_FP_SIZE)
        self.hidden_sizes = kwargs.get('hidden_sizes', [128, 64])
        self.dropout_rate = kwargs.get('dropout_rate', 0.3)
        self.activation = kwargs.get('activation', 'relu')

        if self.task_type == 'regression':
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        
        layers = []
        prev_size = self.input_size
        
        for hidden_size in self.hidden_sizes:
            layers.extend([
                torch.nn.Linear(prev_size, hidden_size),
                get_activation(self.activation),
                torch.nn.Dropout(self.dropout_rate)
            ])
            prev_size = hidden_size
            
        layers.append(torch.nn.Linear(prev_size, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    @staticmethod
    def get_hyperparameter_space(trial):
        # commented out the big hyperparams for development
        # return {
        #     # Model hyperparameters
        #     'hidden_sizes': trial.suggest_categorical('hidden_sizes', [[128], [128, 64], [256, 128, 64]]),
        #     'dropout_rate': trial.suggest_categorical('dropout_rate', [0.0, 0.1, 0.25, 0.4, 0.5]),
        #     'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            
        #     # Training hyperparameters
        #     'epochs': trial.suggest_categorical('epochs', [100, 150, 200]),
        #     'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        #     'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
        # }
        return {
            # Model hyperparameters
            'hidden_sizes': trial.suggest_categorical('hidden_sizes', [[32], [32, 16], [32, 16, 8]]),
            'dropout_rate': trial.suggest_categorical('dropout_rate', [0.0, 0.1, 0.25, 0.4, 0.5]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            
            # Training hyperparameters
            'epochs': trial.suggest_categorical('epochs', [10, 15, 20]),
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
        }
    

@ModelRegistry.register('RF')
class RandomForestModel(BaseModel):
    framework = 'sklearn'

    def __init__(self, task_type='regression', **kwargs):
        if task_type == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                min_samples_split=kwargs.get('min_samples_split', 2),
                min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                min_samples_split=kwargs.get('min_samples_split', 2),
                min_samples_leaf=kwargs.get('min_samples_leaf', 1),
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
class XGBoostModel(BaseModel):
    framework = 'sklearn'

    def __init__(self, task_type='regression', **kwargs):
        if task_type == 'regression':
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 1.0),
                colsample_bytree=kwargs.get('colsample_bytree', 1.0),
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 1.0),
                colsample_bytree=kwargs.get('colsample_bytree', 1.0),
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
class SVMModel(BaseModel):
    framework = 'sklearn'

    def __init__(self, task_type='regression', **kwargs):
        if task_type == 'regression':
            self.model = SVR(
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale'),
                kernel=kwargs.get('kernel', 'rbf')
            )
        else:
            self.model = SVC(
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale'),
                kernel=kwargs.get('kernel', 'rbf'),
                probability=True  # Enable probability predictions
            )
    
    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto', 0.001, 0.01, 0.1, 1]),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
        }

@ModelRegistry.register('ElasticNet')
class ElasticNetModel(BaseModel):
    framework = 'sklearn'

    def __init__(self, task_type='regression', **kwargs):
        if task_type == 'regression':
            self.model = ElasticNet(
                alpha=kwargs.get('alpha', 1.0),
                l1_ratio=kwargs.get('l1_ratio', 0.5),
                random_state=42
            )
        else:
            self.model = LogisticRegression(
                C=1.0/kwargs.get('alpha', 1.0),  # C is inverse of alpha
                l1_ratio=kwargs.get('l1_ratio', 0.5),
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
class KNNModel(BaseModel):
    framework = 'sklearn'

    def __init__(self, task_type='regression', **kwargs):
        if task_type == 'regression':
            self.model = KNeighborsRegressor(
                n_neighbors=kwargs.get('n_neighbors', 5),
                weights=kwargs.get('weights', 'uniform'),
                metric=kwargs.get('metric', 'minkowski')
            )
        else:
            self.model = KNeighborsClassifier(
                n_neighbors=kwargs.get('n_neighbors', 5),
                weights=kwargs.get('weights', 'uniform'),
                metric=kwargs.get('metric', 'minkowski')
            )
    
    @staticmethod
    def get_hyperparameter_space(trial):
        return {
            'n_neighbors': trial.suggest_categorical('n_neighbors', [3, 5, 7, 9, 11]),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['minkowski', 'manhattan', 'chebyshev']),
        }