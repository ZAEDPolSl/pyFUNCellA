import numpy as np
from tqdm import tqdm

from pyfuncella.metrics.rank import rank_genes


def _aucell(
    geneset, data, genes, take_first_n, gs_name=""
):  # ordered gene names for each patient, geneset
    aucell = np.zeros(data.shape[1])
    genes_in_ds = [gene in geneset for gene in genes]
    N_gs = sum(genes_in_ds)  # number of genes in GS
    if N_gs <= 1:
        print("Incorrect geneset format:", gs_name)
        return aucell

    # max auc
    x_th = np.arange(1, min(N_gs + 1, take_first_n))
    y_th = np.arange(1, min(N_gs + 1, take_first_n))
    x_th = np.append(x_th, [take_first_n])
    x_th = np.diff(x_th)
    max_auc = np.sum(x_th * y_th)

    x_in = data[genes_in_ds, :]
    for i in range(x_in.shape[1]):
        x_in_ = x_in[:, i]
        x_in_ = np.sort(x_in_[x_in_ < take_first_n])
        y_in = np.arange(1, x_in_.shape[0] + 1)
        x_in_ = np.append(x_in_, [take_first_n])
        x_in_ = np.diff(x_in_)
        aucell[i] = np.sum(x_in_ * y_in)
    aucell /= max_auc
    return aucell


def AUCELL(genesets, data, genes, thr=0.05):
    data = rank_genes(data, ordinal=True)
    aucell = np.zeros((len(genesets), data.shape[1]))
    N_tot = len(genes)
    take_first_n = max(int(np.round(N_tot * thr)), 2)
    for i, (gs_name, geneset_genes) in tqdm(
        enumerate(genesets.items()), total=len(genesets)
    ):
        aucell[i] = _aucell(geneset_genes, data, genes, take_first_n, gs_name)
    return aucell
