import pytest
import pandas as pd
import numpy as np
import source.metrics as metrics

def test_rank_genes():
    x = np.array([[0, 2],[3,5]])
    res = metrics.rank_genes(x)
    correct = np.array([[2,2],[1,1]])
    assert np.array_equal(correct, res)

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
    res = metrics._ratio(geneset["geneset"], data, genes)
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
    res = metrics.calculate_ratios(genesets, data, genes)
    assert np.array_equal(expected, res)
    
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
    res = metrics._svd(geneset["geneset"], data, genes)
    assert np.array_equal(expected, res)

def test_cerno_auc():
    genes = ["a", "b", "c", "d", "e", "g"]
    genesets = {"geneset": ['a', 'b', 'e', 'f']}
    data = np.array([[2, 4, 5, 4, 1, 6],
                     [1, 5, 2, 5, 2, 5],
                     [4, 2, 3, 2, 3, 4],
                     [5, 1, 1, 6, 4, 3],
                     [6, 3, 6, 3, 5, 2],
                     [3, 6, 4, 1, 6, 1]])
    cerno, auc, pval = metrics._cerno(genesets["geneset"], data, genes)
    auc_expected = np.array([2/3, 1/3, 2/9, 3/9, 7/9, 2/9])
    assert np.array_equal(auc_expected, auc)
    
def test_cerno():
    genes = ["a", "b", "c", "d", "e", "g"]
    genesets = {"geneset": ['a', 'b', 'e', 'f']}
    data = np.array([[2, 4, 5, 4, 1, 6],
                     [1, 5, 2, 5, 2, 5],
                     [4, 2, 3, 2, 3, 4],
                     [5, 1, 1, 6, 4, 3],
                     [6, 3, 6, 3, 5, 2],
                     [3, 6, 4, 1, 6, 1]])
    cerno, auc, pval = metrics._cerno(genesets["geneset"], data, genes)
    cerno_expected = np.array([5.7807, 2.5619, 2.5619, 2.5619, 6.1454, 2.5619])
    assert np.isclose(cerno_expected, cerno, atol=10e-4).all()  
    
    
def test_pval():
    # change into corrected_pvals
    genes = ["a", "b", "c", "d", "e", "g"]
    genesets = {"geneset": ['a', 'b', 'e', 'f']}
    data = np.array([[2, 4, 5, 4, 1, 6],
                     [1, 5, 2, 5, 2, 5],
                     [4, 2, 3, 2, 3, 4],
                     [5, 1, 1, 6, 4, 3],
                     [6, 3, 6, 3, 5, 2],
                     [3, 6, 4, 1, 6, 1]])
    cerno, auc, pval = metrics._cerno(genesets["geneset"], data, genes)
    cerno_expected = 0.4481942337398913
    assert 0 == pval[0]
    
def test_cerno_format():
    genes = ["a", "b", "c", "d", "e", "g"]
    genesets = {"geneset": ['a', 'b', 'c', 'd', 'e', 'g']}
    data = np.array([[2, 4, 5, 4, 1, 6],
                     [1, 5, 2, 5, 2, 5],
                     [4, 2, 3, 2, 3, 4],
                     [5, 1, 1, 6, 4, 3],
                     [6, 3, 6, 3, 5, 2],
                     [3, 6, 4, 1, 6, 1]])
    cerno, auc, pval = metrics._cerno(genesets["geneset"], data, genes)
    auc_expected = np.array([0, 0, 0, 0, 0, 0])
    assert np.array_equal(auc_expected, auc)
    assert np.array_equal(auc_expected, cerno)
    assert np.array_equal(np.array([1, 1, 1, 1, 1, 1]), pval)