import numpy as np
from tqdm import tqdm


def _ratio(geneset, data, genes, gs_name=""):
    genes_in_ds = [gene in geneset for gene in genes]
    in_gs = data[genes_in_ds, :]
    if in_gs.shape[0] == 0:
        print("Incorrect geneset format:", gs_name)
        return np.zeros(in_gs.shape[1])
    gs_expression = np.count_nonzero(in_gs, axis=0) / in_gs.shape[0]
    return gs_expression


def BINA(genesets, data, genes):
    res = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        res[i] = _ratio(geneset_genes, data, genes, gs_name)

    # Calculate BINA: log((DR + 0.1) / (1 - DR + 0.1))
    bina_scores = np.log((res + 0.1) / (1 - res + 0.1))
    return bina_scores
