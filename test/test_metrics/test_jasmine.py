import numpy as np

from enrichment_auc.metrics import jasmine


def test_ranks():
    data = np.array(
        [[0.1, 0.7, 0.9], [0.8, 0.6, 0.2], [0.0, 0.3, 0.5], [0.7, 0.0, 0.1]]
    )
    data = np.ma.masked_equal(data, 0)
    ranks = np.array([[3, 1, 1], [1, 2, 3], [0, 3, 2], [2, 0, 4]])
    ranks = np.ma.masked_equal(ranks, 0)
    ranks_jasmine = jasmine.rank_genes(data)
    assert np.ma.allequal(ranks_jasmine, ranks, fill_value=True)


def test_jasmine_is_array():
    genes = ["a", "b", "c", "d"]
    genesets = {"geneset": ["a", "b", "d"]}
    ranks = np.array([[3, 1, 1], [1, 2, 3], [4, 3, 2], [2, 4, 4]])
    ranks = np.ma.masked_equal(ranks, 0)
    jasmine_ = jasmine._jasmine(genesets["geneset"], ranks, genes)
    assert isinstance(jasmine_, np.ndarray)


def test_jasmine_on_ranks():
    genes = ["a", "b", "c", "d"]
    genesets = {"geneset": ["a", "b", "d"]}
    ranks = np.array([[3, 1, 1], [1, 2, 3], [0, 3, 2], [2, 0, 4]])
    ranks = np.ma.masked_equal(ranks, 0)
    res_expected = np.array([2.0 / 3, 0.5, 2.0 / 3])
    jasmine_ = jasmine._jasmine(genesets["geneset"], ranks, genes)
    assert np.array_equal(res_expected, jasmine_)


def test_full_jasmine():
    genes = ["a", "b", "c", "d"]
    genesets = {"geneset": ["a", "b", "d"], "geneset1": ["d"]}
    data = np.array(
        [[0.1, 0.7, 0.9], [0.8, 0.6, 0.2], [0.0, 0.3, 0.5], [0.7, 0.0, 0.1]]
    )
    res_expected = np.array([[1.0, 0, 1.0], [2 / 3, 0, 1.0]])
    jasmine_ = jasmine.JASMINE(genesets, data, genes)
    assert np.array_equal(res_expected, jasmine_)


def test_jasmine_is_0_robust():
    genes = ["a", "b", "c", "d"]
    genesets = {"geneset": ["a", "b", "d"], "geneset1": ["d"]}
    data = np.array(
        [[0.1, 0.7, 0.0], [0.8, 0.6, 0.0], [0.0, 0.3, 0.0], [0.7, 0.0, 0.0]]
    )
    res_expected = np.array([[1.0, 0.75, 0.0], [1.0, 0, 0.0]])
    jasmine_ = jasmine.JASMINE(genesets, data, genes)
    assert np.array_equal(res_expected, jasmine_)
