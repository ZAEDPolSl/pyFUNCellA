import numpy as np

import enrichment_auc.metrics.rank as rank


def test_rank_genes():
    x = np.array([[0, 2], [3, 5]])
    res = rank.rank_genes(x)
    correct = np.array([[2, 2], [1, 1]])
    assert np.array_equal(correct, res)
