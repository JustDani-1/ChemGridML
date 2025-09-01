import sqlite3
import numpy as np
import pandas as pd
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional
import json
import time
import random
import os

class DatabaseManager:
    """Multi-process safe database manager for storing molecular property prediction results"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables_if_needed()
    
    def _create_tables_if_needed(self):
        """Create necessary tables if they don't exist (with retry logic for concurrent access)"""
        max_retries = 10
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
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
                    return  # Success, exit retry loop
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                    continue
                else:
                    raise  # Re-raise if not a lock error or max retries exceeded
    
    @contextmanager
    def _get_connection(self):
        """Multi-process safe database connection context manager with retry logic"""
        max_retries = 10
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(
                    self.db_path, 
                    timeout=30.0,
                    isolation_level='IMMEDIATE'  # Use IMMEDIATE for better multi-process handling
                )
                conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
                conn.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and performance
                conn.execute("PRAGMA cache_size=10000")  # Increase cache size
                conn.execute("PRAGMA temp_store=memory")  # Use memory for temporary storage
                
                try:
                    yield conn
                    return  # Success, exit retry loop
                finally:
                    conn.close()
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                    continue
                else:
                    raise  # Re-raise if not a lock error or max retries exceeded
    
    def store_dataset_targets(self, dataset_name: str, targets: np.ndarray):
        """Store dataset target values (only if not already present) with retry logic"""
        max_retries = 10
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
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
                    data_to_insert = [(dataset_name, idx, float(target)) for idx, target in enumerate(targets)]
                    cursor.executemany('''
                        INSERT OR IGNORE INTO dataset_targets (dataset_name, data_index, target_value)
                        VALUES (?, ?, ?)
                    ''', data_to_insert)
                    
                    conn.commit()
                    return  # Success, exit retry loop
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                    continue
                else:
                    raise
    
    def store_predictions(self, dataset_name: str, fingerprint: str, model_name: str,
                         predictions: np.ndarray, indices: np.ndarray, seed: int, split_type: str):
        """Store predictions for a specific seed and split with retry logic"""
        max_retries = 10
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Prepare data for batch insert
                    data_to_insert = [
                        (dataset_name, fingerprint, model_name, int(idx), seed, split_type, float(pred))
                        for idx, pred in zip(indices, predictions)
                    ]
                    
                    cursor.executemany('''
                        INSERT OR REPLACE INTO predictions 
                        (dataset_name, fingerprint, model_name, data_index, seed, split_type, prediction)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', data_to_insert)
                    
                    conn.commit()
                    return  # Success, exit retry loop
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                    continue
                else:
                    raise
    
    def get_predictions_dataframe(self, dataset_name: str, fingerprint: str = None, 
                                 model_name: str = None) -> pd.DataFrame:
        """Get predictions as a pandas DataFrame with retry logic"""
        max_retries = 10
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
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
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                    continue
                else:
                    raise