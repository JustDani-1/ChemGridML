from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys
import numpy as np
import env
import numpy as np
import deepchem as dc

def AtomPair(mols, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetAtomPairGenerator(includeChirality=True, fpSize=fpSize)
    return np.stack([np.array(gen.GetFingerprint(x)) for x in mols])

def ECFP(mols, radius=2, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetMorganGenerator(includeChirality=True, radius=radius, fpSize=fpSize)
    return np.stack([np.array(gen.GetFingerprint(x)) for x in mols])

def MACCS(mols):
    return np.stack([np.array(MACCSkeys.GenMACCSKeys(x)) for x in mols])

def RDKitFP(mols, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpSize)
    return np.stack([np.array(gen.GetFingerprint(x)) for x in mols])

def TOPOTOR(mols, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(includeChirality=True, fpSize=fpSize)
    return np.stack([np.array(gen.GetFingerprint(x)) for x in mols])

def MOL2VEC(mols):
    return dc.deepchem.feat.Mol2VecFingerprint().featurize(mols)

def GRAPH(mols):
    # TODO: check options of this featurizer
    return dc.deepchem.feat.MolGraphConvFeaturizer().featurize(mols)

def getFP(mols, fingerprint: str):
    if fingerprint not in globals():
        raise ValueError(f"Unknown fingerprint type: {fingerprint}")
    
    func = globals()[fingerprint]
    if not callable(func):
        raise ValueError(f"{fingerprint} is not a callable function")
    
    return func(mols)

