import numpy as np


def rank_genes(data, descending=True):
    # for each patient return the gene ranking
    if descending:
        return (-data).argsort(axis=0).argsort(axis=0) + 1
    return np.argsort(data, axis=0) + 1
