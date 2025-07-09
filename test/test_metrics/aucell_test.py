import numpy as np

import enrichment_auc.metrics.aucell as aucell


def test_aucell():
    genes = ["a", "b", "c", "d", "e", "g", "1", "2", "3", "4", "5", "6"]
    genesets = {"geneset": ["a", "b", "e", "f", "g"]}
    data = np.array(
        [
            [2, 4, 5, 4, 1, 6],
            [1, 5, 2, 5, 2, 5],
            [4, 2, 3, 2, 3, 4],
            [5, 1, 1, 6, 4, 3],
            [6, 3, 6, 3, 5, 2],
            [3, 6, 4, 1, 6, 1],
            [7, 8, 9, 10, 11, 12],
            [8, 9, 10, 11, 12, 7],
            [9, 10, 11, 12, 7, 8],
            [10, 11, 12, 7, 8, 9],
            [11, 12, 7, 8, 9, 10],
            [12, 7, 8, 9, 10, 11],
        ]
    )
    score = aucell._aucell(genesets["geneset"], data, genes, 3)
    score_expected = np.array([1, 0, 1.0 / 3, 2.0 / 3, 1, 1])
    assert np.isclose(score_expected, score, atol=10e-4).all()


def test_aucell_sets_ranks():
    genes = ["a", "b", "c", "d", "e", "g", "1", "2", "3", "4", "5", "6"]
    genesets = {"geneset": ["a", "b", "e", "f", "g"]}
    data = np.array(
        [
            [1.0, 0.8, 0.7, 0.8, 1.1, 0.6],
            [1.1, 0.7, 1.0, 0.7, 1.0, 0.7],
            [0.8, 1.0, 0.9, 1.0, 0.9, 0.8],
            [0.7, 1.1, 1.1, 0.6, 0.8, 0.9],
            [0.6, 0.9, 0.6, 0.9, 0.7, 1.0],
            [0.9, 0.6, 0.8, 1.1, 0.6, 1.1],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            [0.4, 0.3, 0.2, 0.1, 0.0, 0.5],
            [0.3, 0.2, 0.1, 0.0, 0.5, 0.4],
            [0.2, 0.1, 0.0, 0.5, 0.4, 0.3],
            [0.1, 0.0, 0.5, 0.4, 0.3, 0.2],
            [0.0, 0.5, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    score = aucell.AUCELL(genesets, data, genes, thr=0.25)
    score_expected = np.array([1, 0, 1.0 / 3, 2.0 / 3, 1, 1])
    assert np.isclose(score_expected, score, atol=10e-4).all()
