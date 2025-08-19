import torch
from abc import ABC, abstractmethod

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
    

class BaseModel(ABC):
    @abstractmethod
    def get_hyperparameter_space(self, trial):
        """Return model-specific hyperparameters for Optuna trial"""
        pass


@ModelRegistry.register('FNN')
class FNN(BaseModel, torch.nn.Module):
    def __init__(self, input_size, **kwargs):
        super().__init__()
        # Extract model-specific params with defaults
        self.hidden_sizes = kwargs.get('hidden_sizes', [128, 64])
        self.dropout_rate = kwargs.get('dropout_rate', 0.3)
        self.activation = kwargs.get('activation', 'relu')
        
        # Build network based on hyperparameters
        layers = []
        prev_size = input_size
        
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