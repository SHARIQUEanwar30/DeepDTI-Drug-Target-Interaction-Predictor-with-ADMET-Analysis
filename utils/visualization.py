from rdkit import Chem
from rdkit.Chem import Draw

def visualize_molecule(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Draw.MolToImage(mol, size=(300,300))


def visualize_multiple_molecules(smiles_list):

    mols = []

    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)

        if mol is not None:
            mols.append(mol)

    if len(mols) == 0:
        return None

    return Draw.MolsToGridImage(
        mols,
        molsPerRow=len(mols),
        subImgSize=(200,200)
    )
