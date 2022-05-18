import warnings
import numpy as np
from tqdm import tqdm

def create_random_gs(data, n, rng):
    if np.any(data.std(axis=0)==0):
        raise ValueError('Some of the analyzed samples has 0 variation - the method cannot be applied.')
    genes = rng.choice(range(data.shape[0]), n, replace=False)
    in_gs = data[genes, :]
    rand_gs_std = in_gs.std(axis=0)
    resample = 0
    while np.any(rand_gs_std==0):
        resample += 1
        if resample == 10:
            warnings.warn("Check if your data is correct - variance of gene expressions of some samples for a random geneset is 0.")
        genes = rng.choice(range(data.shape[0]), n, replace=False)
        in_gs = data[genes, :]
        rand_gs_std = in_gs.std(axis=0)
    rand_gs_mean = np.mean(in_gs, axis=0)
    return rand_gs_mean, rand_gs_std

def _vision(geneset, data, genes, seed=0):
    genes_in_ds = [gene in geneset for gene in genes]
    n = len(genes_in_ds) # number of genes in pathway - for creating random pathway for comparison
    in_gs = data[genes_in_ds, :]
    gs_expression = np.mean(in_gs, axis=0)
    rng = np.random.default_rng(seed)
    random_gs_mean, random_gs_std = create_random_gs(data, n, rng)
    gs_expression = (gs_expression - random_gs_mean)/random_gs_std
    return gs_expression

def VISION(genesets, data, genes, seed=0):
    res = np.empty((len(genesets), data.shape[1]))
    for i, (gs_name, geneset_genes) in tqdm(enumerate(genesets.items()), total=len(genesets)):
        res[i] = _vision(geneset_genes, data, genes, seed=seed)
    return res
