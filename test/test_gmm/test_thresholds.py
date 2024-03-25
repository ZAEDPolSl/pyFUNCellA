import numpy as np
import pandas as pd
import pytest

import enrichment_auc.gmm.thresholds as thr_tools


def test_categorize_by_thresholds_no_thr():
    # check if all data is non significant and in one group when no thr is found
    scores = np.random.normal(0, 1, 100)
    labels = np.zeros(100)
    thresholds = np.array([])
    groupings = thr_tools.categorize_by_thresholds(scores, thresholds)
    np.testing.assert_array_equal(groupings, labels)


def test_correct_via_kmeans_skips_smaller_than_2():
    distributions_0 = {
        "mu": np.array([]),
        "sigma": np.array([]),
        "weights": np.array([]),
    }
    distributions_1 = {
        "mu": np.array([0]),
        "sigma": np.array([1.0]),
        "weights": np.array([1.0]),
    }
    distributions_2 = {
        "mu": np.array([0, 1]),
        "sigma": np.array([1, 1]),
        "weights": np.array([0.5, 0.5]),
    }
    thresholds_0 = np.array([])
    thresholds_1 = np.array([])
    thresholds_2 = np.array([0.5])
    thr_found_0 = thr_tools.correct_via_kmeans(distributions_0, thresholds_0)
    thr_found_1 = thr_tools.correct_via_kmeans(distributions_1, thresholds_1)
    thr_found_2 = thr_tools.correct_via_kmeans(distributions_2, thresholds_2)
    np.testing.assert_array_equal(thr_found_0, thresholds_0)
    np.testing.assert_array_equal(thr_found_1, thresholds_1)
    np.testing.assert_array_equal(thr_found_2, thresholds_2)


def test_correct_via_kmeans_skips_missing_thrs():
    distributions = {
        "mu": np.array([0, 1, 2]),
        "sigma": np.array([1, 1, 1]),
        "weights": np.array([0.25, 0.25, 0.5]),
    }
    thresholds = np.array([0.5])
    thr_found = thr_tools.correct_via_kmeans(distributions, thresholds)
    np.testing.assert_array_equal(thr_found, thresholds)


def test_correct_via_kmeans_scales_correctly():
    distributions = {
        "mu": np.array(
            [0.25, 0.5, 0.75],
        ),
        "sigma": np.array([0.1, 0.1, 0.1]),
        "weights": np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
    }
    thresholds = np.array([0.4, 0.6])
    try:
        _ = thr_tools.correct_via_kmeans(distributions, thresholds)
    except ValueError:
        pytest.fail("unexpected ValueError")


def test_filter_thresholds_retains_thresholds():
    localizer = np.array([0, 1, 2])
    mu = np.array([0, 1, 2])
    thresholds = np.array([0.5, 1.5])
    thr_found = thr_tools._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, thresholds)


def test_filter_thresholds_retains_thresholds_mixed_up():
    localizer = np.array([0, 1, 0])
    mu = np.array([0, 1, 2])
    thresholds = np.array([0.5, 1.5])
    thr_found = thr_tools._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, thresholds)


def test_filter_thresholds_removes_single_thresholds():
    localizer = np.array([1, 1, 0])
    mu = np.array([0, 1, 2])
    thresholds = np.array([0.5, 1.5])
    thr_found = thr_tools._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, np.array([1.5]))


def test_filter_thresholds_removes_multiple_thresholds_one_label():
    localizer = np.array([0, 1, 1, 1])
    mu = np.array([0, 1, 2, 3])
    thresholds = np.array([0.5, 1.5, 2.5])
    thr_found = thr_tools._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, np.array([0.5]))


def test_filter_thresholds_removes_multiple_thresholds_diff_label():
    localizer = np.array([0, 0, 1, 1])
    mu = np.array([0, 1, 2, 3])
    thresholds = np.array([0.5, 1.5, 2.5])
    thr_found = thr_tools._filter_thresholds(localizer, mu, thresholds)
    np.testing.assert_array_equal(thr_found, np.array([1.5]))


def test_detect_noncrossing():
    f1 = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [0, 1, 2, 4, 2, 0]])
    f2 = np.array([[1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5], [0, 0.5, 1, 6, 7, 1]])
    expected = np.array([0, 1])
    np.testing.assert_array_equal(thr_tools._detect_noncrossing(f1, f2)[0], expected)


def test_restrict_thr_ranges():
    ranges = np.array([0.1, 0.9])
    x_temp = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    pdfs = np.array(
        [
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        ]
    )
    x_expected = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    pdfs_expected = np.array(
        [
            [0.3, 0.4, 0.5, 0.6, 0.7],
            [0.3, 0.4, 0.5, 0.6, 0.7],
            [0.3, 0.4, 0.5, 0.6, 0.7],
        ]
    )
    x_restricted, pdfs_restricted = thr_tools._restrict_thr_ranges(ranges, x_temp, pdfs)
    np.testing.assert_array_equal(x_restricted, x_expected)
    np.testing.assert_array_equal(pdfs_restricted, pdfs_expected)


def test_find_closest_location():
    x_temp = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    f1 = np.array(
        [
            [0.02, 0.05, 0.1, 0.2, 0.3, 0.35, 0.3, 0.2, 0.1, 0.05, 0.02],
            [0.02, 0.05, 0.17, 0.2, 0.3, 0.35, 0.3, 0.2, 0.1, 0.05, 0.02],
        ]
    )
    f2 = np.array(
        [
            [0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.001, 0.0001, 0.00001],
            [0.2, 0.07, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.001, 0.0001, 0.00001],
        ]
    )
    x_expected = np.array([0.2, 1])
    x_found = thr_tools._find_closest_location(f1, f2, x_temp)
    np.testing.assert_array_equal(x_found, x_expected)


def test_find_thr_by_dist_2():
    distributions = {
        "mu": np.array(
            [0.25, 0.75],
        ),
        "sigma": np.array([0.25, 0.25]),
        "weights": np.array([0.5, 0.5]),
    }
    x_temp = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    pdfs = thr_tools.find_pdfs(
        distributions["mu"], distributions["sigma"], distributions["weights"], x_temp
    )
    thr_expected = np.array([0.5])
    thr_found = thr_tools._find_thr_by_dist(distributions, x_temp, pdfs)
    np.testing.assert_array_almost_equal(thr_found, thr_expected)


def test_find_thr_by_dist_more():
    distributions = {
        "mu": np.array(
            [0.25, 0.5, 0.75],
        ),
        "sigma": np.array([0.1, 0.1, 0.1]),
        "weights": np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
    }
    x_temp = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    pdfs = thr_tools.find_pdfs(
        distributions["mu"], distributions["sigma"], distributions["weights"], x_temp
    )
    thr_expected = np.array([0.4, 0.6])
    thr_found = thr_tools._find_thr_by_dist(distributions, x_temp, pdfs)
    np.testing.assert_array_almost_equal(thr_found, thr_expected)


def test_find_thresholds():
    df = pd.read_csv("test/test_gmm/test_data.csv", index_col=0)
    scores = df.to_numpy().flatten()
    distributions = {
        "mu": np.array(
            [-20.965105, -14.611753, -5.064705, 1.029677],
        ),
        "sigma": np.array([2.267210, 2.414377, 4.526973, 2.267210]),
        "weights": np.array([0.03698616, 0.38945294, 0.46400648, 0.10955443]),
    }
    thr_expected = np.array([-19.9026174, -10.785832, 0.5642248])
    thr_found = thr_tools.find_thresholds(distributions, scores, "")
    np.testing.assert_array_almost_equal(thr_found, thr_expected, decimal=2)
