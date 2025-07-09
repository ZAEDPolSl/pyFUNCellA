import numpy as np

import enrichment_auc.metrics.bina as bina


def test_ratio():
    genes = ["a", "b", "c", "d", "e"]
    geneset = {"geneset": ["a", "b", "e", "f"]}
    # used genes - a, b, e
    expected = np.array([2 / 3, 2 / 3, 1 / 3, 1.0, 0.0])
    data = np.array(
        [
            [1, 2, 0, 4, 0],
            [6, 0, 6, 2, 0],
            [0, 7, 3, 6, 0],
            [0, 9, 9, 0, 0],
            [0, 5, 0, 5, 0],
        ]
    )
    res = bina._ratio(geneset["geneset"], data, genes)
    assert np.array_equal(expected, res)


def test_calculate_bina():
    genes = ["a", "b", "c", "d", "e"]
    genesets = {"geneset": ["a", "b", "e", "f"], "geneset1": ["a", "b", "c"]}

    # Raw ratios that would be calculated
    ratios = np.array(
        [[2 / 3, 2 / 3, 1 / 3, 1.0, 0.0], [2 / 3, 2 / 3, 2 / 3, 3 / 3, 0 / 3]]
    )
    # Transform to BINA scores: log((DR + 0.1) / (1 - DR + 0.1))
    expected = np.log((ratios + 0.1) / (1 - ratios + 0.1))

    data = np.array(
        [
            [1.0, 2, 0, 4, 0],
            [6, 0, 6, 2, 0],
            [0.0, 7, 3, 6, 0],
            [0, 9, 9, 0, 0],
            [0, 5, 0, 5, 0],
        ]
    )
    res = bina.BINA(genesets, data, genes)
    assert np.array_equal(expected, res)
