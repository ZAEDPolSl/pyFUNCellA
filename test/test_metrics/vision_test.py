import numpy as np
import pytest

import pyfuncella.metrics.vision as vision


def test_gs_generator_raises_exception():
    rng = np.random.default_rng(0)
    data = np.array(
        [
            [1, 2, 0, 4, 0],
            [6, 0, 6, 2, 0],
            [0, 7, 3, 6, 0],
            [0, 9, 9, 0, 0],
            [0, 5, 0, 5, 0],
        ]
    )
    with pytest.raises(Exception) as e_info:
        random_gs = vision.create_random_gs(data, 3, rng)


def test_reproducible_result():
    genes = ["a", "b", "c", "d", "e"]
    geneset = {"geneset": ["a", "b", "e", "f"]}
    # used genes - a, b, e
    data = np.array(
        [
            [1, 2, 0, 4, 1],
            [6, 0, 6, 2, 0],
            [0, 7, 3, 6, 0],
            [0, 9, 9, 0, 0],
            [0, 5, 0, 5, 0],
        ]
    )
    res1 = vision._vision(geneset["geneset"], data, genes, seed=123)
    res2 = vision._vision(geneset["geneset"], data, genes, seed=123)
    assert np.all(res1 == res2)
