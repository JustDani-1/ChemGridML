# features.py
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys
import numpy as np
import env
import numpy as np
import deepchem as dc

def generate(mols, gen):
    fingerprints = []
    failed_count = 0
    
    for mol in mols:
        try:
            if mol is not None:
                fp = gen.GetFingerprint(mol)
                fingerprints.append(np.array(fp))
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
    
    print(f"Feature failed for {failed_count} molecules")
    
    return np.stack(fingerprints)

def AtomPair(mols, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetAtomPairGenerator(includeChirality=True, fpSize=fpSize)
    return generate(mols, gen)

def ECFP(mols, radius=2, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetMorganGenerator(includeChirality=True, radius=radius, fpSize=fpSize)
    return generate(mols, gen)

def MACCS(mols):
    return np.stack([np.array(MACCSkeys.GenMACCSKeys(x)) for x in mols])

def RDKitFP(mols, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpSize)
    return generate(mols, gen)

def TOPOTOR(mols, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(includeChirality=True, fpSize=fpSize)
    return generate(mols, gen)

def MOL2VEC(mols):
    return dc.deepchem.feat.Mol2VecFingerprint().featurize(mols)

def GRAPH(mols):
    # TODO: check options of this featurizer
    return dc.deepchem.feat.MolGraphConvFeaturizer().featurize(mols)

def getFeature(mols, fingerprint: str):
    if fingerprint not in globals():
        raise ValueError(f"Unknown fingerprint type: {fingerprint}")
    
    func = globals()[fingerprint]
    if not callable(func):
        raise ValueError(f"{fingerprint} is not a callable function")
    
    return func(mols)



