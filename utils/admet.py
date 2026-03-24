from rdkit import Chem
from rdkit.Chem import Descriptors

def admet_profile(smiles):

    mol = Chem.MolFromSmiles(smiles)

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    return {
        "MW": mw,
        "LogP": logp
    }
