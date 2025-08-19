import datasets, env, models, trainers
from sklearn.model_selection import KFold
import torch, optuna, os, time
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
    

class StudyManager:
    def __init__(self, save_dir='studies'):
        self.save_dir = save_dir
        
    def kfold_cv(self, X, Y, model_class, hyperparams, trainer: trainers.BaseTrainer):   
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
            
            model = trainer.train_with_val(model, optimizer, train_loader, val_loader, hyperparams['epochs'])
            fold_score = trainer.evaluate(model, val_loader)
            fold_scores.append(fold_score)

        return sum(fold_scores) / len(fold_scores)
    
    def final_model_score(self, data: datasets.TDC_Dataset, model_class, hyperparams, trainer: trainers.BaseTrainer):
        train_loader = DataLoader(
            TensorDataset(data.X_train, data.Y_train), 
            batch_size=hyperparams['batch_size'], 
            shuffle=True
        )

        test_loader = DataLoader(
            TensorDataset(data.X_test, data.Y_test), 
            batch_size=hyperparams['batch_size'], 
            shuffle=False
        )

        model = model_class(input_size=data.X_train.shape[1], **hyperparams).to(env.DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=hyperparams['lr'],
        )

        model = trainer.train_without_val(model, optimizer, train_loader, hyperparams['epochs'])
        score = trainer.metric(model, test_loader)
        
        return score
        
    
    def run_study(self, fp_name, model_name, dataset_name, timestamp):
        """Run hyperparameter optimization"""
        study_id = f"{fp_name}_{model_name}_{dataset_name}"

        os.makedirs(self.save_dir, exist_ok=True)
        study = optuna.create_study(
            study_name=study_id,
            storage=f"sqlite:///studies/fp_study_{timestamp}.db",
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
            return self.kfold_cv(data.X_train, data.Y_train, model_class, hyperparams, trainer)
        
        study.optimize(objective, n_trials=env.N_TRIALS)

        best_params = study.best_params

        score = self.final_model_score(data, model_class, best_params, trainer)
        
        return study, score