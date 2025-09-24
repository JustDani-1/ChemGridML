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
            labels = df['Y']
        else:
            try:
                data = ADME(name=method.dataset)
                df = data.get_data()
                smiles = df['Drug']
                mols = [Chem.MolFromSmiles(x) for x in smiles]
                self.X = features.getFeature(mols, method.feature)
                labels = df['Y']
            except:
                df = pd.read_csv(f"./data/{method.dataset}")
                df = df[(df['SMILES_LIGANDS'] != '') & (df['SMILES_LIGANDS'].notna())]
                labels = df['Dock']
                smiles_CD = df['SMILES_CD']
                smiles_LIG = df['SMILES_LIGANDS']
                #smiles_LIG = [x for x in smiles_LIG if x != '']
                mols_CD = [Chem.MolFromSmiles(x) for x in smiles_CD]
                mols_LIG = [Chem.MolFromSmiles(x) for x in smiles_LIG]
                ECFP_CD = features.getFeature(mols_CD, 'ECFP')
                ECFP_LIG = features.getFeature(mols_LIG, 'ECFP')
                RDKit_CD = features.getFeature(mols_CD, 'RDKit')
                RDKit_LIG = features.getFeature(mols_LIG, 'RDKit')
                self.X = np.concatenate([ECFP_CD, ECFP_LIG, RDKit_CD, RDKit_LIG], axis=1)
                
        self.Y = np.array(labels)



if __name__ == '__main__':
    df = pd.read_csv(f"./data/FinalCSV.csv")
    smiles_CD = df['SMILES_CD']
    mols_CD = [Chem.MolFromSmiles(x) for x in smiles_CD]
    print("here")
    RdKit_CD = features.getFeature(mols_CD, 'RDKit')