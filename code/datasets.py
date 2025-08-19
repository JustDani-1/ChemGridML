from tdc.single_pred import ADME
from rdkit import Chem
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import fingerprints, env
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
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        # Features
        fps = fingerprints.getFP(mols, fingerprint)
        X_train, X_test, self.Y_train, self.Y_test = train_test_split(fps, labels, test_size=env.TEST_SIZE, random_state=42)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        self.X_train, self.X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)