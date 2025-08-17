import numpy as np
from statsmodels.stats.multitest import multipletests

import pyfuncella.metrics.cerno as cerno


def test_auc():
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
    auc = cerno._auc(genesets["geneset"], data, genes)
    auc_expected = np.array([2 / 3, 1 / 3, 2 / 9, 3 / 9, 7 / 9, 2 / 9])
    assert np.array_equal(auc_expected, auc)


def test_auc_ties():
    genes = ["a", "b", "c", "d", "e", "g"]
    genesets = {"geneset": ["a", "b", "e", "f"]}
    data = np.array(
        [
            [2.0, 3.5, 5.0, 4.0, 1.0, 6.0],
            [1.0, 5.0, 3.0, 5.0, 2.0, 5.0],
            [4.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            [5.0, 1.0, 1.0, 6.0, 4.0, 3.0],
            [6.0, 3.5, 6.0, 3.0, 5.0, 2.0],
            [3.0, 6.0, 3.0, 1.0, 6.0, 1.0],
        ]
    )
    auc = cerno._auc(genesets["geneset"], data, genes)
    auc_expected = np.array([2.0 / 3, 1.0 / 3, 1.0 / 9, 1.0 / 3, 7.0 / 9, 2.0 / 9])
    assert np.array_equal(auc_expected, auc)


def test_fisher():
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
    cerno_, _, _ = cerno._fisher(genesets["geneset"], data, genes)
    cerno_expected = np.array([5.7807, 2.5619, 2.5619, 2.5619, 6.1454, 2.5619])
    assert np.isclose(cerno_expected, cerno_, atol=10e-4).all()


def test_pval():
    # change into corrected_pvals
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
    _, pval, qval = cerno._fisher(genesets["geneset"], data, genes)
    pval_expected = [
        0.44819423,
        0.86148045,
        0.86148045,
        0.86148045,
        0.40710257,
        0.86148045,
    ]
    _, qval_expected, _, _ = multipletests(pval_expected, alpha=0.05, method="fdr_tsbh")
    assert np.isclose(qval, qval_expected, atol=10e-4).all()
    assert np.isclose(pval, pval_expected, atol=10e-4).all()


def test_fisher_format():
    genes = ["a", "b", "c", "d", "e", "g"]
    genesets = {"geneset": ["a", "b", "c", "d", "e", "g"]}
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
    cerno_, pval, qval = cerno._fisher(genesets["geneset"], data, genes)
    cerno_expected = np.array([0, 0, 0, 0, 0, 0])
    assert np.array_equal(cerno_expected, cerno_)
    assert np.array_equal(np.array([1, 1, 1, 1, 1, 1]), pval)
    assert np.array_equal(np.array([1, 1, 1, 1, 1, 1]), qval)


def test_auc_format():
    genes = ["a", "b", "c", "d", "e", "g"]
    genesets = {"geneset": ["a", "b", "c", "d", "e", "g"]}
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
    auc = cerno._auc(genesets["geneset"], data, genes)
    auc_expected = np.array([0, 0, 0, 0, 0, 0])
    assert np.array_equal(auc_expected, auc)
