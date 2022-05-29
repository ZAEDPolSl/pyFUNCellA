import pytest
import pandas as pd
import numpy as np
import source.metrics.gsva as gsva

def test_calculate_z_scores():
    x = np.array([[1, 2, 3], [4,5,6]])
    res = gsva._calculate_gene_zscores(x)
    correct = np.array([[0.2271351, 0.5, 0.7728649],[0.2271351, 0.5, 0.7728649]])
    assert np.allclose(correct, res)
    
def test_rank_expressions():
    transformed = np.array([0, 0.3, 0.5, 0.75, 0.8, 1])
    expected = np.array([3, 2, 1, 0, 1, 2])
    res = gsva._rank_expressions(transformed)
    assert np.array_equal(expected, res)
    
def test_get_miss_increment():
    genes = [i for i in range(100)]
    genes_in_ds = [i for i in range(20)]
    expected = -0.0125
    res = gsva._get_miss_increment(genes_in_ds, genes)
    assert res==expected
    
def test_gsva():
    ranks = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    genes = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    geneset = ["a", "e", "f", "g", "h", "z"]
    expected = -0.2
    res = gsva._gsva(geneset, ranks, genes)
    assert pytest.approx(res) == expected
