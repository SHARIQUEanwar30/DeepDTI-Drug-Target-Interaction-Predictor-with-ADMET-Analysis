import torch

amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
aa_dict = {aa:i for i,aa in enumerate(amino_acids)}

max_len = 512

def encode_protein(seq):

    arr = [aa_dict.get(a,0) for a in seq[:max_len]]

    if len(arr) < max_len:
        arr += [0]*(max_len - len(arr))

    return torch.tensor(arr, dtype=torch.long)
