from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

def find_similar_drugs(query_smiles, df, top_n=5):

    query_mol = Chem.MolFromSmiles(query_smiles)

    if query_mol is None:
        return []

    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=1024)

    similarities = []

    for sm in df['Drug']:

        mol = Chem.MolFromSmiles(sm)

        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

        sim = TanimotoSimilarity(query_fp, fp)

        similarities.append({
            "smiles": sm,
            "score": sim
        })

    # Sort by similarity (descending)
    similarities = sorted(similarities, key=lambda x: x["score"], reverse=True)

    return similarities[:top_n]
