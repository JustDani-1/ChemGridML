# study_manager.py
from methods import Method
import datasets, env, models, util
from database_manager import DatabaseManager
from sklearn.model_selection import KFold, train_test_split
import optuna, os, sqlite3
import numpy as np
from typing import Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

class StudyManager:
    def __init__(self, studies_path: str = './studies/', predictions_path: str = 'studies/predictions.db'):
        self.studies_path = studies_path
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        self.db = DatabaseManager(predictions_path)
        self.optuna_init = False
    
    def setup_optuna_storage(self, study_id: str):
        storage_path = f"{self.studies_path}/{study_id}.db"
        os.makedirs(self.studies_path, exist_ok=True)

        temp_storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path}")
        optuna.create_study(storage=temp_storage, study_name="__init__", direction="minimize")

        # Add connection pooling parameters
        conn = sqlite3.connect(storage_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL") 
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.close()
        
        self.storage_url = f"sqlite:///{storage_path}?check_same_thread=false&pool_timeout=30"

    def kfold_cv(self, X, Y, method: Method, hyperparams: Dict):
        """Perform k-fold cross-validation using uniform model API"""
        kfold = KFold(env.N_FOLDS, shuffle=True, random_state=42)
        predictions = np.zeros_like(Y, dtype=np.float32)

        # Create model instance
        model_class = models.ModelRegistry.get_model(method.model)
        task_type = util.get_task_type(Y)
        model = model_class(task_type=task_type, **hyperparams)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            X_train, X_val, Y_train, Y_val = model.preprocess(X_train, X_val, Y_train, Y_val)

            model.fit(X_train, Y_train)
            
            fold_predictions = model.predict(X_val)
            predictions[val_idx] = fold_predictions
        
        return predictions

    def train_and_predict(self, X_train, Y_train, X_test, method: Method, hyperparams: Dict):
        """Train model and make predictions"""
        # Create model instance
        model_class = models.ModelRegistry.get_model(method.model)
        task_type = util.get_task_type(Y_train)
        model = model_class(task_type=task_type, **hyperparams)
        
        X_train, X_test, Y_train, _ = model.preprocess(X_train, X_test, Y_train, Y_train)

        # Train model
        model.fit(X_train, Y_train)
        
        return model.predict(X_test)

    def run_hyperparameter_optimization(self, X: np.ndarray, Y: np.ndarray, seed: int, method: Method, dataset_name: str) -> Dict:
        """Run hyperparameter optimization"""
        model_class = models.ModelRegistry.get_model(method.model)
        task_type = util.get_task_type(Y)
        
        study_id = f"{method.name}_{dataset_name}"
        
        study = optuna.create_study(
            study_name=f"{study_id}_{seed}",
            storage=self.storage_url,
            direction="minimize",
            load_if_exists=True
        )
        
        def objective(trial):
            hyperparams = model_class.get_hyperparameter_space(trial)
            cv_predictions = self.kfold_cv(X, Y, method, hyperparams)
            return util.evaluate(Y, cv_predictions, task_type)
        
        study.optimize(objective, n_trials=env.N_TRIALS)
        
        return study.best_params

    def run_single_experiment(self, seed: int, method: Method, dataset_name: str, data) -> Tuple[int, np.ndarray, np.ndarray]:
        """Run a single experiment (train-test split)"""
        
        X_train, X_test, Y_train, Y_test, train_indices, test_indices = train_test_split(
            data.X, data.Y, np.arange(len(data.Y)),
            test_size=env.TEST_SIZE, random_state=seed,
        )
        
        best_hyperparams = self.run_hyperparameter_optimization(
            X_train, Y_train, seed, method, dataset_name
        )
        
        test_predictions = self.train_and_predict(
            X_train, Y_train, X_test, method, best_hyperparams
        )
        
        return seed, test_predictions, test_indices

    def run_nested_cv(self, method: Method, dataset_name: str):
        """Run nested cross-validation experiment"""
        data = datasets.TDC_Dataset(dataset_name, method)
        self.db.store_dataset_targets(dataset_name, data.Y)
        
        study_id = f"{method.name}_{dataset_name}"
        self.setup_optuna_storage(study_id)
        
        predictions = [None for _ in range(env.N_TESTS)]
        indices = [None for _ in range(env.N_TESTS)]
        
        allocated_cores = int(os.environ.get('NSLOTS', multiprocessing.cpu_count()))
        max_workers = min(allocated_cores, env.N_TESTS)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_seed = {
                executor.submit(self.run_single_experiment, seed, method, dataset_name, data): seed
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
                    dataset_name, method.input_representation, method.model, 
                    predictions[seed], indices[seed], seed, 'random'
                )