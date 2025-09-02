import datasets, env, models, util
from database_manager import DatabaseManager
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch, optuna, os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict

class StudyManager:
    def __init__(self, studies_path: str = './studies/', predictions_path: str = 'studies/predictions.db'):
        self.studies_path = studies_path
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        self.db = DatabaseManager(predictions_path)
    
    def kfold_cv(self, X, Y, model_class, framework, task_type, hyperparams):   
        """Cross-validation on training data only"""
        kfold = KFold(env.N_FOLDS, shuffle=True, random_state=42)
        
        predictions = np.zeros_like(Y)                                                  
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)

            if framework == 'pytorch':
                X_train = torch.tensor(X_train, dtype=torch.float32)
                X_val = torch.tensor(X_val, dtype=torch.float32)
                Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
                Y_val = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)

                train_loader = DataLoader(
                    TensorDataset(X_train, Y_train), 
                    batch_size=hyperparams['batch_size'], 
                    shuffle=True
                )
                val_loader = DataLoader(
                    TensorDataset(X_val, Y_val), 
                    batch_size=hyperparams['batch_size'], 
                    shuffle=False
                )

                model = model_class(input_size=X_train.shape[1], task_type=task_type, **hyperparams).to(env.DEVICE)
                
                optimizer = torch.optim.Adam(
                    model.parameters(), 
                    lr=hyperparams['lr'],
                )
                
                model = util.train_with_val(model, optimizer, train_loader, val_loader, hyperparams['epochs'])
                X_val = X_val.to(env.DEVICE)
                predictions[val_idx] = model(X_val).cpu().detach().numpy().flatten()

            elif framework == 'sklearn':
                sklearn_model = model_class(task_type, **hyperparams)
                sklearn_model.model.fit(X_train, Y_train)
                predictions[val_idx] = sklearn_model.model.predict(X_val).flatten()

        return predictions
    
    def train_and_predict(self, X_train, Y_train, X_test, model_class, framework, task_type, hyperparams):
        """Train on full training set and predict on test set"""
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if framework == 'pytorch':
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            
            train_loader = DataLoader(
                TensorDataset(X_train_tensor, Y_train_tensor), 
                batch_size=hyperparams['batch_size'], 
                shuffle=True
            )
            
            model = model_class(input_size=X_train.shape[1], task_type=task_type, **hyperparams).to(env.DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])
            
            model = util.train_without_val(model, optimizer, train_loader, hyperparams['epochs'])
            
            model.eval()
            with torch.no_grad():
                X_test_tensor = X_test_tensor.to(env.DEVICE)
                test_predictions = model(X_test_tensor).cpu().detach().numpy().flatten()
                
        elif framework == 'sklearn':
            sklearn_model = model_class(task_type, **hyperparams)
            sklearn_model.model.fit(X_train_scaled, Y_train)
            test_predictions = sklearn_model.model.predict(X_test_scaled).flatten()
        
        return test_predictions
    
    def run_hyperparameter_optimization(self, fp_name: str, model_name: str, dataset_name: str) -> Dict:
        """Run hyperparameter optimization on entire dataset with cross-validation"""
        
        data = datasets.TDC_Dataset(dataset_name, fp_name)
        
        self.db.store_dataset_targets(dataset_name, data.Y)
        
        # Get components
        model_class = models.ModelRegistry.get_model(model_name)
        framework = models.ModelRegistry.get_framework(model_name)
        task_type = util.get_task_type(data.Y)
        
        study_id = f"{fp_name}_{model_name}_{dataset_name}"

        os.makedirs(self.studies_path, exist_ok=True)
        study = optuna.create_study(
            study_name=study_id,
            storage=f"sqlite:///{self.studies_path}/{study_id}.db",
            direction="minimize",
            load_if_exists=True
        )
        
        def objective(trial):
            # Get hyperparameters
            hyperparams = model_class.get_hyperparameter_space(trial)
            
            # Perform cross-validation on training data only
            cv_predictions = self.kfold_cv(data.X, data.Y, model_class, framework, task_type, hyperparams)
            return util.evaluate(data.Y, cv_predictions, task_type)
            
        study.optimize(objective, n_trials=env.N_TRIALS)
        
        best_params = study.best_params
        
        return best_params
    
    def run_multiple_train_test_evaluations(self, fp_name: str, model_name: str, dataset_name: str, best_hyperparams: Dict):
        """Run multiple train-test splits with different random seeds"""
        
        # Load data
        data = datasets.TDC_Dataset(dataset_name, fp_name)
        
        # Get components
        model_class = models.ModelRegistry.get_model(model_name)
        framework = models.ModelRegistry.get_framework(model_name)
        task_type = util.get_task_type(data.Y)
        
        for seed in range(env.N_TESTS):
            
            # Split data with different seed
            X_train, X_test, Y_train, Y_test, train_indices, test_indices = train_test_split(
                data.X, data.Y, np.arange(len(data.Y)), 
                test_size=env.TEST_SIZE, random_state=seed,
            )
            
            # Train model and get predictions
            test_predictions = self.train_and_predict(
                X_train, Y_train, X_test, model_class, framework, task_type, best_hyperparams
            )
            
            self.db.store_predictions(dataset_name, fp_name, model_name, test_predictions, test_indices, seed, 'random')
        
    
    def run_complete_study(self, fp_name: str, model_name: str, dataset_name: str) -> Dict:
        """Run complete study: nested cross-validation"""
        
        best_hyperparams = self.run_hyperparameter_optimization(fp_name, model_name, dataset_name)
        
        self.run_multiple_train_test_evaluations(fp_name, model_name, dataset_name, best_hyperparams)