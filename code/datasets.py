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
        # Get data
        data = ADME(name=method.dataset)
        df = data.get_data()
        smiles = df['Drug']
        labels = df['Y']

        mols = [Chem.MolFromSmiles(x) for x in smiles]

        # Labels
        self.Y = np.array(labels)

        self.X = features.getFeature(mols, method.feature)
        
        # Store original data for potential use
        # self.smiles = smiles
        # self.mols = mols
        # self.method = method

    def _ensure_datasets(self):
        if not Path("./data/HIV.csv").exists():
            print("loading HIV")
            _ = dc.deepchem.molnet.load_hiv(data_dir="./data", reload=False)


if __name__ == '__main__':
    
    print("done")