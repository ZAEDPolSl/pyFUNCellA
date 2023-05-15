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


def calculate_ratios(genesets, data, genes):
    res = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        res[i] = _ratio(geneset_genes, data, genes, gs_name)
    return res
