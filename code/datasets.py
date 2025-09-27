# datasets.py
from tdc.single_pred import ADME
from rdkit import Chem
import features
import numpy as np
import deepchem as dc
from experiments import Method
from pathlib import Path

class Dataset():
    def __init__(self, method: Method):
        """
        Initialize dataset with appropriate input representation based on method
        
        Args:
            method: Method object from MethodRegistry
        """
        # self._ensure_datasets()

        if method.dataset.startswith('Solubility_'):
            # Extract percentage from dataset name (e.g., 'Solubility_010' -> 10%)
            parts = method.dataset.split('_')
            if len(parts) > 1:
                percentage_str = parts[-1]
                percentage = int(percentage_str)
            else:
                percentage = 100
            
            data = ADME(name='Solubility_AqSolDB')
            df = data.get_data()
            
            # Sample the specified percentage of the dataset
            if percentage < 100:
                df = df.sample(frac=percentage/100, random_state=42)
        else:
            data = ADME(name=method.dataset)
            df = data.get_data()
            
        smiles = df['Drug']
        labels = df['Y']

        mols = [Chem.MolFromSmiles(x) for x in smiles]

        # Get features and valid indices
        self.X, valid_indices = features.getFeature(mols, method.feature)
        
        self.Y = np.array(labels)[valid_indices]