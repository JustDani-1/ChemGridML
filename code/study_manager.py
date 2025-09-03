from time import sleep
import datasets, env, models, util
from database_manager import DatabaseManager
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import torch, optuna, os, sqlite3
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

class StudyManager:
    def __init__(self, studies_path: str = './studies/', predictions_path: str = 'studies/predictions.db'):
        self.studies_path = studies_path
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        self.db = DatabaseManager(predictions_path)
    
    def setup_optuna_storage(self, study_id):
        storage_path = f"{self.studies_path}/{study_id}.db"
        os.makedirs(self.studies_path, exist_ok=True)

        temp_storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path}")
        optuna.create_study(storage=temp_storage, study_name="__init__", direction="minimize")

        
        conn = sqlite3.connect(storage_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL") 
        conn.execute("PRAGMA cache_size=10000")
        conn.close()
        
        self.storage_url = f"sqlite:///{storage_path}?check_same_thread=false"

    def kfold_cv(self, X, Y, model_name, model_class, framework, task_type, hyperparams):
        kfold = KFold(env.N_FOLDS, shuffle=True, random_state=42)
        predictions = np.zeros_like(Y)

        if framework == 'sklearn':
            sklearn_model = model_class(task_type, **hyperparams)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            threshold = VarianceThreshold(threshold=0.0)
            threshold.fit(X_train)
            X_train, X_val = threshold.transform(X_train), threshold.transform(X_val)

            # only scale non-binary features
            #if X_train[0, 0] != 0 and X_train[0,0] != 1:

            if model_name == 'SVM':
                scaler = RobustScaler()
            else:
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
                optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])
                model = util.train_with_val(model, optimizer, train_loader, val_loader, hyperparams['epochs'])
                
                X_val = X_val.to(env.DEVICE)
                predictions[val_idx] = model(X_val).cpu().detach().numpy().flatten()
                
            elif framework == 'sklearn':
                sleep(0.001)
                sklearn_model.model.fit(X_train, Y_train)
                predictions[val_idx] = sklearn_model.model.predict(X_val).flatten()
        
        return predictions

    def train_and_predict(self, X_train, Y_train, X_test, model_name, model_class, framework, task_type, hyperparams):
        threshold = VarianceThreshold(threshold=0.0)
        threshold.fit(X_train)
        X_train, X_test = threshold.transform(X_train), threshold.transform(X_test)

        # only scale non-binary features
        #if X_train[0, 0] != 0 and X_train[0,0] != 1:
            #print("non binary")
        if model_name == 'SVM':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        scaler.fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        
        if framework == 'pytorch':
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            
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
            sklearn_model.model.fit(X_train, Y_train)
            test_predictions = sklearn_model.model.predict(X_test).flatten()
        
        return test_predictions

    def run_hyperparameter_optimization(self, X, Y, seed, fp_name: str, model_name: str, dataset_name: str) -> Dict:
        model_class = models.ModelRegistry.get_model(model_name)
        framework = models.ModelRegistry.get_framework(model_name)
        task_type = util.get_task_type(Y)
        
        study_id = f"{fp_name}_{model_name}_{dataset_name}"
        
        study = optuna.create_study(
            study_name=f"{study_id}_{seed}",
            storage=self.storage_url,
            direction="minimize",
            load_if_exists=True
        )
        
        def objective(trial):
            hyperparams = model_class.get_hyperparameter_space(trial)
            cv_predictions = self.kfold_cv(X, Y, model_name, model_class, framework, task_type, hyperparams)
            return util.evaluate(Y, cv_predictions, task_type)
        
        study.optimize(objective, n_trials=env.N_TRIALS)
        
        return study.best_params

    def run_single_experiment(self, seed: int, fp_name: str, model_name: str, dataset_name: str, data) -> Tuple[int, np.ndarray, np.ndarray]:
        
        X_train, X_test, Y_train, Y_test, train_indices, test_indices = train_test_split(
            data.X, data.Y, np.arange(len(data.Y)),
            test_size=env.TEST_SIZE, random_state=seed,
        )
        
        best_hyperparams = self.run_hyperparameter_optimization(
            X_train, Y_train, seed, fp_name, model_name, dataset_name
        )
        
        model_class = models.ModelRegistry.get_model(model_name)
        framework = models.ModelRegistry.get_framework(model_name)
        task_type = util.get_task_type(data.Y)
        
        test_predictions = self.train_and_predict(
            X_train, Y_train, X_test, model_name, model_class, framework, task_type, best_hyperparams
        )
        
        return seed, test_predictions, test_indices

    def run_nested_cv(self, fp_name, model_name, dataset_name):
        data = datasets.TDC_Dataset(dataset_name, fp_name)
        self.db.store_dataset_targets(dataset_name, data.Y)
        study_id = f"{fp_name}_{model_name}_{dataset_name}"
        self.setup_optuna_storage(study_id)
        
        predictions = [None for _ in range(env.N_TESTS)]
        indices = [None for _ in range(env.N_TESTS)]
        
        allocated_cores = int(os.environ.get('NSLOTS', multiprocessing.cpu_count()))
        max_workers = min(allocated_cores, env.N_TESTS)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_seed = {
                executor.submit(self.run_single_experiment, seed, fp_name, model_name, dataset_name, data): seed
                for seed in range(env.N_TESTS)
            }
            
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    seed_result, test_predictions, test_indices = future.result()
                    predictions[seed_result] = test_predictions
                    indices[seed_result] = test_indices
                except Exception as exc:
                    print(f"Seed {seed} failed: {exc}")
                    raise exc
        
        for seed in range(env.N_TESTS):
            if predictions[seed] is not None:
                self.db.store_predictions(
                    dataset_name, fp_name, model_name, 
                    predictions[seed], indices[seed], seed, 'random'
                )
