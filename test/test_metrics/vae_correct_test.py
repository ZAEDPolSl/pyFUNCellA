import numpy as np

import enrichment_auc.metrics.vae as vae


def test_find_most_enriched():
    genes = ["a", "b", "c", "d", "e"]
    genesets = {"geneset": ["a", "b", "e", "f"], "geneset1": ["a", "b", "c"]}

    expected = np.array([3, 3])
    data = np.array(
        [
            [1.0, 2, 0, 4, 0],
            [6, 0, 6, 2, 0],
            [0.0, 7, 3, 6, 0],
            [0, 9, 9, 0, 0],
            [0, 5, 0, 5, 0],
        ]
    )
    res = vae.find_most_enriched(genesets, data, genes)
    assert np.array_equal(expected, res)
    
    
def test_leaves_correct_order():
    enriched_idx = 3
    expected = np.array([7.0 / 3, 7.0 / 3, 6.0 / 3, 11.0 / 3, 0.0])
    res = vae.correct_pas(expected, enriched_idx)
    assert np.array_equal(expected, res)
    

def test_reverts_order():
    enriched_idx = 4
    pas = np.array([7.0 / 3, 7.0 / 3, 6.0 / 3, 11.0 / 3, 0.0])
    expected = np.array([-7.0 / 3, -7.0 / 3, -6.0 / 3, -11.0 / 3, 0.0])
    res = vae.correct_pas(pas, enriched_idx)
    assert np.array_equal(expected, res)
    

def test_corrects_order():
    genes = ["a", "b", "c", "d", "e"]
    genesets = {"geneset": ["a", "b", "e", "f"], "geneset1": ["a", "b", "c"]}
    data = np.array(
        [
            [1.0, 2, 0, 4, 0],
            [6, 0, 6, 2, 0],
            [0.0, 7, 3, 6, 0],
            [0, 9, 9, 0, 0],
            [0, 5, 0, 5, 0],
        ]
    )
    pas = np.array([[7.0 / 3, 7.0 / 3, 6.0 / 3, 11.0 / 3, 0.0],
                    [7.0 / 3, 7.0 / 3, 9.0 / 3, -1.0 / 3, 11.0]])
    expected = np.array([[7.0 / 3, 7.0 / 3, 6.0 / 3, 11.0 / 3, 0.0],
                        [-7.0 / 3, -7.0 / 3, -9.0 / 3, 1.0 / 3, -11.0]])
    res = vae.correct_order(data, genesets, genes, pas)
    assert np.array_equal(expected, res)
