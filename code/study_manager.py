import datasets, env, models, trainers
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
import torch, optuna, os
from torch.utils.data import DataLoader, TensorDataset

class StudyManager:
    def __init__(self, save_dir='studies'):
        self.save_dir = save_dir
        
    def kfold_cv(self, X, Y, model_class, hyperparams, trainer: trainers.BaseTrainer):   
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
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
            TensorDataset(torch.tensor(data.X_train, dtype=torch.float32), torch.tensor(data.Y_train, dtype=torch.float32).unsqueeze(1)), 
            batch_size=hyperparams['batch_size'], 
            shuffle=True
        )

        test_loader = DataLoader(
            TensorDataset(torch.tensor(data.X_test, dtype=torch.float32), torch.tensor(data.Y_test, dtype=torch.float32).unsqueeze(1)), 
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
        framework = models.ModelRegistry.get_framework(model_name)
        task_type = trainers.get_task_type(data.Y_test)
        
        def objective(trial):
            # Get all hyperparameters from model class
            hyperparams = model_class.get_hyperparameter_space(trial)

            if framework == 'pytorch':
                trainer = trainers.get_trainer(task_type)
                return self.kfold_cv(data.X_train, data.Y_train, model_class, hyperparams, trainer)
            elif framework == 'sklearn':
                sklearn = model_class(task_type, **hyperparams)
                cv_scores = cross_val_score(
                    sklearn.model, data.X_train, data.Y_train, 
                    cv=env.N_FOLDS, 
                    scoring='neg_mean_squared_error'
                )
                return -cv_scores.mean()
            
            
        
        study.optimize(objective, n_trials=env.N_TRIALS)

        best_params = study.best_params

        if framework == 'pytorch':
            trainer = trainers.get_trainer(task_type)
            score = self.final_model_score(data, model_class, best_params, trainer)
        elif framework == 'sklearn':
            sklearn = model_class(task_type, **best_params)
            sklearn.model.fit(data.X_train, data.Y_train)
            predictions = sklearn.model.predict(data.X_test)
            if task_type == 'regression':
                score = {'RRMSE': trainers.rrmse(predictions, data.Y_test)}
            elif task_type == 'classification':
                score = {'AUROC': roc_auc_score(data.Y_test, predictions)}
        
        return study, score