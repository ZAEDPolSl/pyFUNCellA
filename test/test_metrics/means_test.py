import numpy as np

import pyfuncella.metrics.mean as mean


def test_mean():
    genes = ["a", "b", "c", "d", "e"]
    geneset = {"geneset": ["a", "b", "e", "f"]}
    # used genes - a, b, e
    expected = np.array([7.0 / 3, 7.0 / 3, 6.0 / 3, 11.0 / 3, 0.0])
    data = np.array(
        [
            [1, 2, 0, 4, 0.0],
            [6, 0, 6, 2, 0],
            [0, 7, 3, 6, 0],
            [0, 9, 9, 0, 0],
            [0, 5, 0, 5, 0],
        ]
    )
    res = mean._mean(geneset["geneset"], data, genes)
    assert np.array_equal(expected, res)


def test_calculate_mean():
    genes = ["a", "b", "c", "d", "e"]
    genesets = {"geneset": ["a", "b", "e", "f"], "geneset1": ["a", "b", "c"]}

    expected = np.array(
        [
            [7.0 / 3, 7.0 / 3, 6.0 / 3, 11.0 / 3, 0.0],
            [7 / 3, 9 / 3, 9 / 3, 12 / 3, 0 / 3],
        ]
    )
    data = np.array(
        [
            [1.0, 2, 0, 4, 0],
            [6, 0, 6, 2, 0],
            [0.0, 7, 3, 6, 0],
            [0, 9, 9, 0, 0],
            [0, 5, 0, 5, 0],
        ]
    )
    res = mean.MEAN(genesets, data, genes)
    assert np.array_equal(expected, res)
