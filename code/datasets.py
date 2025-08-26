from tdc.single_pred import ADME
from rdkit import Chem
import fingerprints
import numpy as np

class TDC_Dataset():
    def __init__(self, dataset_name: str, fingerprint: str):
        # Get data
        data = ADME(name = dataset_name)
        df = data.get_data()
        smiles = df['Drug']
        labels = df['Y']

        mols = [Chem.MolFromSmiles(x) for x in smiles]

        #Labels
        self.Y = np.array(labels)

        # Features
        self.X = fingerprints.getFP(mols, fingerprint)