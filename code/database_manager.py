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
                    split_type TEXT,  -- 'random' or 'scaffold'
                    prediction REAL,
                    PRIMARY KEY (dataset_name, fingerprint, model_name, data_index, seed),
                    FOREIGN KEY (dataset_name, data_index) REFERENCES dataset_targets(dataset_name, data_index)
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
    
    def store_dataset_targets(self, dataset_name: str, targets: np.ndarray):
        """Store dataset target values (only if not already present)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if any entries exist for this dataset
            cursor.execute(
                'SELECT COUNT(*) FROM dataset_targets WHERE dataset_name = ?', 
                (dataset_name,)
            )
            existing_count = cursor.fetchone()[0]
            
            # If entries exist, assume they are complete and skip insertion
            if existing_count > 0:
                return
            
            # Insert new data only if no entries exist
            for idx, target in enumerate(targets):
                cursor.execute('''
                    INSERT INTO dataset_targets (dataset_name, data_index, target_value)
                    VALUES (?, ?, ?)
                ''', (dataset_name, idx, float(target)))
            
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
    
    def get_predictions_dataframe(self, dataset_name: str, fingerprint: str = None, 
                                 model_name: str = None) -> pd.DataFrame:
        """Get predictions as a pandas DataFrame"""
        with self._get_connection() as conn:
            query = '''
                SELECT p.*, dt.target_value
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