import numpy as np

import pyfuncella.metrics.rank as rank


def test_rank_genes():
    x = np.array([[0, 2], [3, 5]])
    res = rank.rank_genes(x)
    correct = np.array([[2, 2], [1, 1]])
    assert np.array_equal(correct, res)


def test_rank_ties():
    x = np.array([[2, 2, 1], [3, 5, 1], [1, 5, 1]])
    res = rank.rank_genes(x)
    correct = np.array([[2, 3, 2], [1, 1.5, 2], [3, 1.5, 2]])
    assert np.array_equal(correct, res)


def test_rank_ties_fisher():
    x = np.array([[2, 2, 1], [3, 5, 1], [1, 5, 1]])
    res = rank.rank_genes(x, ordinal=True)
    correct = np.array([[2, 3, 1], [1, 1, 2], [3, 2, 3]])
    assert np.array_equal(correct, res)
