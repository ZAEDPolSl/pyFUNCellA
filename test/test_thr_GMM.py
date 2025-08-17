import numpy as np

from pyfuncella.thr_GMM import thr_GMM


def test_thr_GMM_basic():
    gmms = {
        "pathway1": {
            "thresholds": np.array([1.0, 2.0, np.nan, 3.0]),
            "model": {
                "mu": np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
                "sigma": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                "alpha": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            },
        },
        "pathway2": {
            "thresholds": np.array([np.nan]),
            "model": {
                "mu": np.array([1.0, 2.0]),
                "sigma": np.array([0.1, 0.2]),
                "alpha": np.array([0.5, 0.5]),
            },
        },
        "pathway3": {
            "thresholds": np.array([1.5]),
            "model": {
                "mu": np.array([1.0, 2.0]),
                "sigma": np.array([0.1, 0.2]),
                "alpha": np.array([0.5, 0.5]),
            },
        },
        "pathway4": {
            "thresholds": np.array([]),
            "model": {
                "mu": np.array([1.0]),
                "sigma": np.array([0.1]),
                "alpha": np.array([1.0]),
            },
        },
    }
    result = thr_GMM(gmms)
    # pathway1: thresholds after nan removal [1.0, 2.0, 3.0], Top1_thr=3.0, All_thr=[1.0,2.0,3.0]
    assert result["pathway1"]["Top1_thr"] == 3.0
    assert result["pathway1"]["All_thr"] == [1.0, 2.0, 3.0]
    assert isinstance(result["pathway1"]["Kmeans_thr"], float)
    # pathway2: thresholds after nan removal [], Top1_thr=nan, All_thr=[]
    assert np.isnan(result["pathway2"]["Top1_thr"])
    assert result["pathway2"]["All_thr"] == []
    assert result["pathway2"]["Kmeans_thr"] == float("-inf")
    # pathway3: thresholds [1.5], Top1_thr=1.5, All_thr=[1.5], Kmeans_thr=1.5 (2 components)
    assert result["pathway3"]["Top1_thr"] == 1.5
    assert result["pathway3"]["All_thr"] == [1.5]
    assert result["pathway3"]["Kmeans_thr"] == 1.5
    # pathway4: no thresholds, Top1_thr=nan, All_thr=[], Kmeans_thr=-inf
    assert np.isnan(result["pathway4"]["Top1_thr"])
    assert result["pathway4"]["All_thr"] == []
    assert result["pathway4"]["Kmeans_thr"] == float("-inf")


def test_thr_GMM_edge_cases():
    # thresholds off by one from components
    gmms = {
        "p": {
            "thresholds": np.array([1.0, 2.0]),
            "model": {
                "mu": np.array([1.0, 2.0, 3.0, 4.0]),
                "sigma": np.array([0.1, 0.2, 0.3, 0.4]),
                "alpha": np.array([0.25, 0.25, 0.25, 0.25]),
            },
        }
    }
    result = thr_GMM(gmms)
    # Should fallback to Top1_thr
    assert result["p"]["Kmeans_thr"] == result["p"]["Top1_thr"]
    assert result["p"]["All_thr"] == [1.0, 2.0]
    assert result["p"]["Top1_thr"] == 2.0
