import sqlite3
import threading
import numpy as np
import pandas as pd
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional
import json

class DatabaseManager:
    """Thread-safe singleton database manager for storing molecular property prediction results"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: str):
        if not self._initialized:
            self.db_path = db_path
            self.connection_lock = threading.Lock()
            self._create_tables()
            self._initialized = True
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table for storing dataset target values
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dataset_targets (
                    dataset_name TEXT,
                    data_index INTEGER,
                    target_value REAL,
                    smiles TEXT,
                    PRIMARY KEY (dataset_name, data_index)
                )
            ''')
            
            # Table for storing predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    dataset_name TEXT,
                    fingerprint TEXT,
                    model_name TEXT,
                    data_index INTEGER,
                    seed INTEGER,
                    split_type TEXT,  -- 'train' or 'test'
                    prediction REAL,
                    PRIMARY KEY (dataset_name, fingerprint, model_name, data_index, seed),
                    FOREIGN KEY (dataset_name, data_index) REFERENCES dataset_targets(dataset_name, data_index)
                )
            ''')
            
            # Table for storing hyperparameters and study results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS study_results (
                    dataset_name TEXT,
                    fingerprint TEXT,
                    model_name TEXT,
                    best_hyperparams TEXT,  -- JSON string
                    cv_score REAL,
                    cv_std REAL,
                    n_trials INTEGER,
                    study_completed BOOLEAN,
                    PRIMARY KEY (dataset_name, fingerprint, model_name)
                )
            ''')
            
            # Table for storing final test scores across seeds
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_scores (
                    dataset_name TEXT,
                    fingerprint TEXT,
                    model_name TEXT,
                    seed INTEGER,
                    metric_name TEXT,
                    metric_value REAL,
                    PRIMARY KEY (dataset_name, fingerprint, model_name, seed, metric_name)
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection context manager"""
        with self.connection_lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
            try:
                yield conn
            finally:
                conn.close()
    
    def store_dataset_targets(self, dataset_name: str, targets: np.ndarray, smiles: List[str] = None):
        """Store dataset target values"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Clear existing data for this dataset
            cursor.execute('DELETE FROM dataset_targets WHERE dataset_name = ?', (dataset_name,))
            
            # Insert new data
            for idx, target in enumerate(targets):
                smiles_str = smiles[idx] if smiles else None
                cursor.execute('''
                    INSERT INTO dataset_targets (dataset_name, data_index, target_value, smiles)
                    VALUES (?, ?, ?, ?)
                ''', (dataset_name, idx, float(target), smiles_str))
            
            conn.commit()
    
    def store_predictions(self, dataset_name: str, fingerprint: str, model_name: str,
                         predictions: np.ndarray, indices: np.ndarray, seed: int, split_type: str):
        """Store predictions for a specific seed and split"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for idx, pred in zip(indices, predictions):
                cursor.execute('''
                    INSERT INTO predictions (dataset_name, fingerprint, model_name, data_index, seed, split_type, prediction)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (dataset_name, fingerprint, model_name, int(idx), seed, split_type, float(pred)))
            
            conn.commit()
    
    def store_test_scores(self, dataset_name: str, fingerprint: str, model_name: str,
                         seed: int, metrics: Dict[str, float]):
        """Store test scores for a specific seed"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Clear existing scores for this combination and seed
            cursor.execute('''
                DELETE FROM test_scores 
                WHERE dataset_name = ? AND fingerprint = ? AND model_name = ? AND seed = ?
            ''', (dataset_name, fingerprint, model_name, seed))
            
            # Insert new scores
            for metric_name, metric_value in metrics.items():
                cursor.execute('''
                    INSERT INTO test_scores (dataset_name, fingerprint, model_name, seed, metric_name, metric_value)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (dataset_name, fingerprint, model_name, seed, metric_name, float(metric_value)))
            
            conn.commit()
    
    def get_study_results(self, dataset_name: str, fingerprint: str, model_name: str) -> Optional[Dict]:
        """Get stored hyperparameter optimization results"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT best_hyperparams, cv_score, cv_std, n_trials, study_completed
                FROM study_results
                WHERE dataset_name = ? AND fingerprint = ? AND model_name = ?
            ''', (dataset_name, fingerprint, model_name))
            
            result = cursor.fetchone()
            if result:
                return {
                    'best_hyperparams': json.loads(result[0]),
                    'cv_score': result[1],
                    'cv_std': result[2],
                    'n_trials': result[3],
                    'study_completed': bool(result[4])
                }
            return None
    
    def get_predictions_dataframe(self, dataset_name: str, fingerprint: str = None, 
                                 model_name: str = None) -> pd.DataFrame:
        """Get predictions as a pandas DataFrame"""
        with self._get_connection() as conn:
            query = '''
                SELECT p.*, dt.target_value, dt.smiles
                FROM predictions p
                JOIN dataset_targets dt ON p.dataset_name = dt.dataset_name AND p.data_index = dt.data_index
                WHERE p.dataset_name = ?
            '''
            params = [dataset_name]
            
            if fingerprint:
                query += ' AND p.fingerprint = ?'
                params.append(fingerprint)
            
            if model_name:
                query += ' AND p.model_name = ?'
                params.append(model_name)
            
            return pd.read_sql_query(query, conn, params=params)
    
    def get_test_scores_summary(self, dataset_name: str = None) -> pd.DataFrame:
        """Get summary of test scores across all seeds"""
        with self._get_connection() as conn:
            query = '''
                SELECT dataset_name, fingerprint, model_name, metric_name,
                       AVG(metric_value) as mean_score,
                       STDEV(metric_value) as std_score,
                       MIN(metric_value) as min_score,
                       MAX(metric_value) as max_score,
                       COUNT(*) as n_seeds
                FROM test_scores
            '''
            params = []
            
            if dataset_name:
                query += ' WHERE dataset_name = ?'
                params.append(dataset_name)
            
            query += ' GROUP BY dataset_name, fingerprint, model_name, metric_name'
            
            return pd.read_sql_query(query, conn, params=params)
    
    def check_combination_completed(self, dataset_name: str, fingerprint: str, model_name: str) -> bool:
        """Check if a combination has been completed (has study results)"""
        result = self.get_study_results(dataset_name, fingerprint, model_name)
        return result is not None and result.get('study_completed', False)
    
    def get_completed_combinations(self) -> List[Tuple[str, str, str]]:
        """Get list of completed combinations"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT dataset_name, fingerprint, model_name
                FROM study_results
                WHERE study_completed = 1
            ''')
            return cursor.fetchall()
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database"""
        with self._get_connection() as conn:
            backup = sqlite3.connect(backup_path)
            conn.backup(backup)
            backup.close()
    
    def vacuum_database(self):
        """Optimize database by running VACUUM"""
        with self._get_connection() as conn:
            conn.execute('VACUUM')
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count records in each table
            for table in ['dataset_targets', 'predictions', 'study_results', 'test_scores']:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Count unique combinations
            cursor.execute('''
                SELECT COUNT(DISTINCT dataset_name || '_' || fingerprint || '_' || model_name)
                FROM study_results
            ''')
            stats['unique_combinations'] = cursor.fetchone()[0]
            
            return stats