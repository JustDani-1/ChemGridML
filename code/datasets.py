# datasets.py
from tdc.single_pred import ADME
from rdkit import Chem
import features
import numpy as np
from experiments import Method
import pandas as pd

class Dataset():
    def __init__(self, method: Method):
        """
        Initialize dataset with appropriate input representation based on method
        
        Args:
            method: Method object from MethodRegistry
        """

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

            smiles = df['Drug']
            mols = [Chem.MolFromSmiles(x) for x in smiles]
            self.X = features.getFeature(mols, method.feature)
        else:
            try:
                data = ADME(name=method.dataset)
                df = data.get_data()
                smiles = df['Drug']
                mols = [Chem.MolFromSmiles(x) for x in smiles]
                self.X = features.getFeature(mols, method.feature)
            except:
                df = pd.read_csv(f"./data/{method.dataset}")
                feature_columns = [col for col in df.columns if col != 'Y']
                self.X = df[feature_columns].values.astype(np.float32)

            
        labels = df['Y']
        self.Y = np.array(labels)