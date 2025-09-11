# datasets.py (Updated)
from tdc.single_pred import ADME
from rdkit import Chem
import fingerprints
import numpy as np
from methods import InputType

class TDC_Dataset():
    def __init__(self, dataset_name: str, method):
        """
        Initialize dataset with appropriate input representation based on method
        
        Args:
            dataset_name: Name of the TDC dataset
            method: Method object from MethodRegistry
        """
        # Get data
        data = ADME(name=dataset_name)
        df = data.get_data()
        smiles = df['Drug']
        labels = df['Y']

        mols = [Chem.MolFromSmiles(x) for x in smiles]

        # Labels
        self.Y = np.array(labels)

        # Features based on input type
        if method.input_type == InputType.FINGERPRINT:
            self.X = fingerprints.getFP(mols, method.input_representation)
        elif method.input_type == InputType.GRAPH:
            self.X = fingerprints.getFP(mols, method.input_representation)
        else:
            raise ValueError(f"Unsupported input type: {method.input_type}")
        
        # Store original data for potential use
        # self.smiles = smiles
        # self.mols = mols
        # self.method = method