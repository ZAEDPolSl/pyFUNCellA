import numpy as np
from scipy import stats

from enrichment_auc.metrics.gsea import rank_expressions


def get_ranks(data, genes=None):
    print("Transforming the geneset...\n")
    data = stats.norm.cdf(
        data.transpose(), loc=np.mean(data, axis=1), scale=np.std(data, axis=1, ddof=1)
    ).transpose()
    print("Getting the ranks...\n")
    ranks = rank_expressions(data)
    return ranks, genes
