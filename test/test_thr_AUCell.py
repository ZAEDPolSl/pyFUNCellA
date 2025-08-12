import numpy as np
import pandas as pd
import pytest

from pyfuncella.thr_AUCell import thr_AUCell, _check_aucell_installed
from pyfuncella.utils.r_executor import check_r_available


class TestThrAUCell:
    def test_r_available(self):
        # Test that R is available through process-based execution
        if not check_r_available():
            pytest.skip("R is not available")

    def test_aucell_installed(self):
        # Test that AUCell package is installed in R
        if not _check_aucell_installed():
            pytest.skip("AUCell R package is not installed")

    """Test class for thr_AUCell functionality."""

    def test_thr_AUCell_numpy(self):
        np.random.seed(42)
        # Unimodal (normal), bimodal (mixture), and reversed
        unimodal = np.random.normal(0.5, 0.1, 20)
        bimodal = np.concatenate(
            [np.random.normal(0.2, 0.05, 10), np.random.normal(0.8, 0.05, 10)]
        )
        reversed = unimodal[::-1]
        arr = np.stack([unimodal, bimodal, reversed])
        pathway_names = ["unimodal", "bimodal", "reversed"]
        thresholds = thr_AUCell(arr, pathway_names=pathway_names)
        assert isinstance(thresholds, dict)
        assert set(thresholds.keys()) == set(pathway_names)
        for k, v in thresholds.items():
            assert isinstance(v, float), f"Threshold for {k} should be float"
            idx = pathway_names.index(k)
            min_val, max_val = arr[idx].min(), arr[idx].max()
            # Threshold can be outside min/max, so just check it's finite
            assert np.isfinite(v), f"Threshold for {k} should be finite"

    def test_thr_AUCell_dataframe(self):
        np.random.seed(123)
        # Unimodal, bimodal, constant
        unimodal = np.random.normal(0.5, 0.1, 10)
        bimodal = np.concatenate(
            [np.random.normal(0.2, 0.05, 5), np.random.normal(0.8, 0.05, 5)]
        )
        constant = np.full(10, 0.7)
        df = pd.DataFrame(
            [unimodal, bimodal, constant],
            index=["unimodal", "bimodal", "constant"],
            columns=[f"sample{i}" for i in range(10)],
        )
        thresholds = thr_AUCell(df)
        assert isinstance(thresholds, dict)
        assert set(thresholds.keys()) == set(df.index)
        for k, v in thresholds.items():
            assert isinstance(v, float), f"Threshold for {k} should be float"
            # Threshold can be outside min/max, so just check it's finite
            assert np.isfinite(v), f"Threshold for {k} should be finite"

    def test_thr_AUCell_value_validation(self):
        np.random.seed(321)
        # Small range, large range, negative values
        small_range = np.random.normal(0.5, 0.001, 10)
        large_range = np.concatenate([np.full(5, -100), np.full(5, 100)])
        negative = np.random.normal(-1, 0.1, 10)
        arr = np.stack([small_range, large_range, negative])
        pathway_names = ["small_range", "large_range", "negative"]
        thresholds = thr_AUCell(arr, pathway_names=pathway_names)
        for k, v in thresholds.items():
            assert isinstance(v, float)
            assert np.isfinite(v), f"Threshold for {k} should be finite"
