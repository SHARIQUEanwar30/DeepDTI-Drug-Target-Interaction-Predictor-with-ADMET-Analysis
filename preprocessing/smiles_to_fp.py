from rdkit import Chem
from rdkit.Chem import AllChem
import torch

def smiles_to_fp(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

    return torch.tensor(list(fp), dtype=torch.float)
