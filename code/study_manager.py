import datasets, env, models, trainers
from sklearn.model_selection import KFold
import torch, optuna, os, joblib
from torch.utils.data import DataLoader, TensorDataset

class BenchmarkResults:
    def __init__(self):
        self.results = {}
    
    def add_result(self, fingerprint, model, dataset, best_score, best_params):
        if fingerprint not in self.results:
            self.results[fingerprint] = {}
        if model not in self.results[fingerprint]:
            self.results[fingerprint][model] = {}
        
        self.results[fingerprint][model][dataset] = {
            'score': best_score,
            'params': best_params
        }
    
    def get_summary_table(self):
        """Return pandas DataFrame with results summary"""
        # Implementation for creating comparison tables
        pass
    
    def plot_results(self):
        """Generate comparison plots"""
        pass

class StudyManager:
    def __init__(self, save_dir='studies'):
        self.save_dir = save_dir
        
    def kfold_cv(self, data, model_class, hyperparams, trainer):   
        # Load dataset
        X, Y = data.X_train, data.Y_train
        
        kfold = KFold(env.N_FOLDS, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
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

            model = model_class(input_size=X_train.shape[1], **hyperparams).to(env.DEVICE)
            
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=hyperparams['lr'],
            )
            
            trained_model = trainer.train(model, optimizer, train_loader, val_loader, hyperparams['epochs'])
            fold_score = trainer.evaluate(trained_model, val_loader)
            fold_scores.append(fold_score)

        return sum(fold_scores) / len(fold_scores)
    
    def run_study(self, fp_name, model_name, dataset_name):
        """Run hyperparameter optimization"""
        study_id = f"{fp_name}_{model_name}_{dataset_name}"

        os.makedirs(self.save_dir, exist_ok=True)
        study = optuna.create_study(
            study_name=study_id,
            storage="sqlite:///studies/fp_study_02.db",
            direction="minimize",
            load_if_exists=True
        )

        data = datasets.TDC_Dataset(dataset_name, fp_name)
        
        # Get components
        model_class = models.ModelRegistry.get_model(model_name)
        task_type = trainers.get_task_type(data.Y_test)
        trainer = trainers.get_trainer(task_type)
        
        def objective(trial):
            # Get all hyperparameters from model class
            hyperparams = model_class.get_hyperparameter_space(trial)
            
            # Run k-fold CV
            return self.kfold_cv(data, model_class, hyperparams, trainer)
        
        study.optimize(objective, n_trials=env.N_TRIALS)
        
        return study