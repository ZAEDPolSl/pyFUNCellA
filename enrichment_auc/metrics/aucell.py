import numpy as np
from sklearn.metrics import auc
from tqdm import tqdm


def _aucell(
    geneset, data, genes, take_first_n, gs_name=""
):  # ordered gene names for each patient, geneset
    aucell = np.zeros(data.shape[1])
    genes_in_ds = [gene in geneset for gene in genes]
    N_gs = sum(genes_in_ds)  # number of genes in GS
    N_tot = len(genes)  # total number of genes
    if N_gs <= 1:
        print("Incorrect geneset format:", gs_name)
        return aucell

    cumulative_occurence = np.cumsum(np.array(genes_in_ds)[data], axis=0)
    filter_ranks = np.arange(take_first_n)
    total_ranks = np.arange(N_tot)
    for i in range(data.shape[1]):
        filter_auc = auc(filter_ranks, cumulative_occurence[:take_first_n, i])
        total_auc = auc(total_ranks, cumulative_occurence[:, i])
        aucell[i] = filter_auc / total_auc
    return aucell


def AUCELL(
    genesets, data, genes, thr=0.05
):  # ordered gene names for each patient, genesets
    aucell = np.zeros((len(genesets), data.shape[1]))
    data = data - data.min()  # should start at 0
    N_tot = len(genes)
    take_first_n = max(int(np.ceil(N_tot * thr)), 2)
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        aucell[i] = _aucell(geneset_genes, data, genes, take_first_n, gs_name)
    return aucell
