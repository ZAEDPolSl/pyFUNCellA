import pytest
import pandas as pd
import numpy as np
import source.metrics.svd as svd

def test_svd():
    genes = ["a", "b", "c", "d", "e"]
    geneset = {"geneset": ["a", "b", "e", "f"]}
    # used genes - a, b, e
    expected = np.array([0, 0, 0, 0, 0])
    data = np.array([[1,1,1,1,1],
                     [0,0,0,0,0],
                     [1,3,4,6,8],
                     [3,8,5,-1,3],
                     [0.0,0,0,0,0]])
    res = svd._svd(geneset["geneset"], data, genes)
    assert np.array_equal(expected, res)
