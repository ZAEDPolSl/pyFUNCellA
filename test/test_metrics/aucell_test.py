import numpy as np

import enrichment_auc.metrics.aucell as aucell


def test_aucell():
    genes = ["a", "b", "c", "d", "e", "g"]
    genesets = {"geneset": ["a", "b", "e", "f"]}
    data = np.array(
        [
            [2, 4, 5, 4, 1, 6],
            [1, 5, 2, 5, 2, 5],
            [4, 2, 3, 2, 3, 4],
            [5, 1, 1, 6, 4, 3],
            [6, 3, 6, 3, 5, 2],
            [3, 6, 4, 1, 6, 1],
        ]
    )
    data = data - 1
    score = aucell._aucell(genesets["geneset"], data, genes, 2)
    score_expected = np.array(
        [1.0 / 8, 1.0 / 21, 1.0 / 8, 1.0 / 17, 3.0 / 22, 1.0 / 13]
    )
    assert np.isclose(score_expected, score, atol=10e-4).all()


def test_aucell_sets_ranks():
    genes = ["a", "b", "c", "d", "e", "g"]
    genesets = {"geneset": ["a", "b", "e", "f"]}
    data = np.array(
        [
            [2, 4, 5, 4, 1, 6],
            [1, 5, 2, 5, 2, 5],
            [4, 2, 3, 2, 3, 4],
            [5, 1, 1, 6, 4, 3],
            [6, 3, 6, 3, 5, 2],
            [3, 6, 4, 1, 6, 1],
        ]
    )
    score = aucell.AUCELL(genesets, data, genes)
    score_expected = np.array(
        [[1.0 / 8, 1.0 / 21, 1.0 / 8, 1.0 / 17, 3.0 / 22, 1.0 / 13]]
    )
    assert np.isclose(score_expected, score, atol=10e-4).all()
