import numpy as np
import enrichment_auc.distributions as dist


def test_find_dist_zero_var():
    scores = np.ones(100)
    dist_found = dist.find_distribution(scores, "")
    assert dist_found["weights"].shape[0] == 0
    assert dist_found["mu"].shape[0] == 0
    assert dist_found["sigma"].shape[0] == 0
    assert np.isnan(dist_found["TIC"])
    assert np.isnan(dist_found["l_lik"])


def test_categorize_by_thresholds_no_thr():
    # check if all data is non significant and in one group when no thr is found
    scores = np.random.normal(0, 1, 100)
    labels = np.zeros(100)
    thresholds = np.array([])
    groupings = dist.categorize_by_thresholds(scores, thresholds)
    np.testing.assert_array_equal(groupings, labels)


def remove_redundant_thresholds_start():
    scores = np.linspace(0, 1, 100)
    thresholds = np.array([0])
    thr_found = dist._remove_redundant_thresholds(thresholds, scores)
    assert thr_found.shape[0] == 0


def remove_redundant_thresholds_before():
    scores = np.linspace(0, 1, 100)
    thresholds = np.array([-0.01])
    thr_found = dist._remove_redundant_thresholds(thresholds, scores)
    assert thr_found.shape[0] == 0


def remove_redundant_thresholds_end():
    scores = np.linspace(0, 1, 100)
    thresholds = np.array([1])
    thr_found = dist._remove_redundant_thresholds(thresholds, scores)
    assert thr_found.shape[0] == 0


def remove_redundant_thresholds_after():
    scores = np.linspace(0, 1, 100)
    thresholds = np.array([1.01])
    thr_found = dist._remove_redundant_thresholds(thresholds, scores)
    assert thr_found.shape[0] == 0


def remove_redundant_thresholds_mixed_ends():
    scores = np.linspace(0, 1, 100)
    thresholds = np.array([0, 1.0])
    thr_found = dist._remove_redundant_thresholds(thresholds, scores)
    assert thr_found.shape[0] == 0


def remove_redundant_thresholds_mix():
    scores = np.linspace(0, 1, 100)
    thresholds = np.array([0, 0.5, 0.75, 1.0])
    thr_found = dist._remove_redundant_thresholds(thresholds, scores)
    thr_expected = np.array([0.5, 0.75])
    np.testing.assert_array_equal(thr_found, thr_expected)
