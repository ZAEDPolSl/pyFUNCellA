import numpy as np
from tqdm import tqdm
import pandas as pd

def _ratio(geneset, data, genes):
    genes_in_ds = [gene in geneset for gene in genes]
    in_gs = data[genes_in_ds, :]
    gs_expression = np.count_nonzero(in_gs, axis=0)/in_gs.shape[0]
    return gs_expression

def calculate_ratios(genesets, data, genes):
    res = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in tqdm(enumerate(genesets.items()), total=len(genesets)):
        res[i] = _ratio(geneset_genes, data, genes)
    return res
