# features.py
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys
import numpy as np
import env
import numpy as np
import deepchem as dc

def generate(mols, gen):
    fingerprints = []
    valid_indices = []  # Track which molecules were successfully processed
    failed_count = 0
    
    for i, mol in enumerate(mols):
        try:
            if mol is not None:
                fp = gen.GetFingerprint(mol)
                fingerprints.append(np.array(fp))
                valid_indices.append(i)
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
    
    print(f"Feature failed for {failed_count} molecules")
    
    if fingerprints:
        return np.stack(fingerprints), valid_indices
    else:
        return np.array([]), []

def AtomPair(mols, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetAtomPairGenerator(includeChirality=True, fpSize=fpSize)
    return generate(mols, gen)

def ECFP(mols, radius=2, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetMorganGenerator(includeChirality=True, radius=radius, fpSize=fpSize)
    return generate(mols, gen)

def MACCS(mols):
    fingerprints = []
    valid_indices = []
    failed_count = 0
    
    for i, mol in enumerate(mols):
        try:
            if mol is not None:
                fp = MACCSkeys.GenMACCSKeys(mol)
                fingerprints.append(np.array(fp))
                valid_indices.append(i)
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
    
    print(f"Feature failed for {failed_count} molecules")
    
    if fingerprints:
        return np.stack(fingerprints), valid_indices
    else:
        return np.array([]), []

def RDKitFP(mols, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpSize)
    return generate(mols, gen)

def TOPOTOR(mols, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(includeChirality=True, fpSize=fpSize)
    return generate(mols, gen)

def MOL2VEC(mols):
    fingerprints = []
    valid_indices = []
    failed_count = 0
    
    featurizer = dc.deepchem.feat.Mol2VecFingerprint()
    
    for i, mol in enumerate(mols):
        try:
            if mol is not None:
                fp = featurizer.featurize([mol])
                if fp is not None and len(fp) > 0 and fp[0] is not None:
                    fingerprints.append(fp[0])
                    valid_indices.append(i)
                else:
                    failed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
    
    print(f"Feature failed for {failed_count} molecules")
    
    if fingerprints:
        return np.stack(fingerprints), valid_indices
    else:
        return np.array([]), []

def GRAPH(mols):
    fingerprints = []
    valid_indices = []
    failed_count = 0
    
    featurizer = dc.deepchem.feat.MolGraphConvFeaturizer()
    
    for i, mol in enumerate(mols):
        try:
            if mol is not None:
                fp = featurizer.featurize([mol])
                if fp is not None and len(fp) > 0 and fp[0] is not None:
                    fingerprints.append(fp[0])
                    valid_indices.append(i)
                else:
                    failed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
    
    print(f"Feature failed for {failed_count} molecules")
    
    if fingerprints:
        return fingerprints, valid_indices  # Don't stack for graph features
    else:
        return [], []

def getFeature(mols, fingerprint: str):
    if fingerprint not in globals():
        raise ValueError(f"Unknown fingerprint type: {fingerprint}")
    
    func = globals()[fingerprint]
    if not callable(func):
        raise ValueError(f"{fingerprint} is not a callable function")
    
    return func(mols)