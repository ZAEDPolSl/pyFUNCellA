import pytest
import pandas as pd
import numpy as np
import source.metrics.ratio as ratio

def test_ratio():
    genes = ["a", "b", "c", "d", "e"]
    geneset = {"geneset": ["a", "b", "e", "f"]}
    # used genes - a, b, e
    expected = np.array([2/3, 2/3, 1/3, 1.0, 0.0])
    data = np.array([[1,2,0,4,0],
                                  [6,0,6,2,0],
                                  [0,7,3,6,0],
                                  [0,9,9,0,0],
                                  [0,5,0,5,0]])
    res = ratio._ratio(geneset["geneset"], data, genes)
    assert np.array_equal(expected, res)

def test_calculate_ratio():
    genes = ["a", "b", "c", "d", "e"]
    genesets = {"geneset": ['a', 'b', 'e', 'f'], "geneset1": ['a', 'b', 'c']}

    expected = np.array([[2/3, 2/3, 1/3, 1.0, 0.0], [2/3, 2/3, 2/3, 3/3, 0/3]])
    data = np.array([[1.0,2,0,4,0],
                     [6,0,6,2,0],
                     [0.0,7,3,6,0],
                     [0,9,9,0,0],
                     [0,5,0,5,0]])
    res = ratio.calculate_ratios(genesets, data, genes)
    assert np.array_equal(expected, res)
