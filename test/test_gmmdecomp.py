"""
Test module for GMMdecomp implementation.
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any

from enrichment_auc.GMMdecomp import (
    GMMdecomp,
    _check_r_available,
    _check_dpgmm_installed,
)


class TestGMMdecomp:
    """Test class for GMMdecomp functionality."""

    def test_r_available(self):
        """Test that R is available through rpy2."""
        assert _check_r_available(), "R is not available through rpy2"

    def test_dpgmm_installed(self):
        """Test that dpGMM package is installed in R."""
        assert _check_dpgmm_installed(), "dpGMM package is not installed"

    def test_gmmdecomp_basic_functionality(self):
        """Test basic GMMdecomp functionality with pandas DataFrame input."""
        # Create test data
        np.random.seed(42)

        # Create some bimodal data for testing
        pathway1 = np.concatenate(
            [
                np.random.normal(0.2, 0.1, 10),  # Low activity component
                np.random.normal(0.8, 0.1, 10),  # High activity component
            ]
        )

        pathway2 = np.random.normal(0.5, 0.2, 20)  # Unimodal data

        # Create DataFrame more concisely
        test_data = pd.DataFrame(
            [pathway1, pathway2],
            index=["pathway_bimodal", "pathway_unimodal"],
            columns=[f"sample_{i}" for i in range(20)],
        )

        # Run GMM decomposition
        results = GMMdecomp(test_data, K=3, verbose=False)

        # Check results structure
        assert isinstance(results, dict), "Results should be a dictionary"
        assert len(results) == 2, "Should have results for 2 pathways"
        assert "pathway_bimodal" in results, "Should have results for pathway_bimodal"
        assert "pathway_unimodal" in results, "Should have results for pathway_unimodal"

    def test_gmmdecomp_result_structure(self):
        """Test that GMMdecomp results have the correct structure."""
        np.random.seed(42)
        test_data = pd.DataFrame(
            np.random.randn(2, 20),
            index=["pathway_1", "pathway_2"],
            columns=[f"sample_{i}" for i in range(20)],
        )

        results = GMMdecomp(test_data, K=2, verbose=False)

        # Check each pathway result structure
        for pathway_name, result in results.items():
            assert isinstance(
                result, dict
            ), f"Result for {pathway_name} should be a dictionary"
            assert (
                "model" in result
            ), f"Result for {pathway_name} should have 'model' key"
            assert (
                "thresholds" in result
            ), f"Result for {pathway_name} should have 'thresholds' key"

            # Check model structure
            model = result["model"]
            assert isinstance(
                model, dict
            ), f"Model for {pathway_name} should be a dictionary"
            assert "alpha" in model, f"Model for {pathway_name} should have 'alpha' key"
            assert "mu" in model, f"Model for {pathway_name} should have 'mu' key"
            assert "sigma" in model, f"Model for {pathway_name} should have 'sigma' key"

            # Check model components are numpy arrays
            assert isinstance(
                model["alpha"], np.ndarray
            ), f"Alpha for {pathway_name} should be numpy array"
            assert isinstance(
                model["mu"], np.ndarray
            ), f"Mu for {pathway_name} should be numpy array"
            assert isinstance(
                model["sigma"], np.ndarray
            ), f"Sigma for {pathway_name} should be numpy array"

            # Check thresholds is a numpy array
            assert isinstance(
                result["thresholds"], np.ndarray
            ), f"Thresholds for {pathway_name} should be numpy array"

            # Check that all model components have the same length
            assert len(model["alpha"]) == len(
                model["mu"]
            ), f"Alpha and mu should have same length for {pathway_name}"
            assert len(model["alpha"]) == len(
                model["sigma"]
            ), f"Alpha and sigma should have same length for {pathway_name}"

            # Check that alpha values sum to approximately 1 (mixture weights)
            assert (
                np.abs(np.sum(model["alpha"]) - 1.0) < 1e-10
            ), f"Alpha values should sum to 1 for {pathway_name}"

    def test_gmmdecomp_value_validation(self):
        """Test that GMMdecomp produces reasonable values."""
        np.random.seed(42)

        # Create bimodal and unimodal data
        pathway1 = np.concatenate(
            [np.random.normal(0.2, 0.1, 10), np.random.normal(0.8, 0.1, 10)]
        )
        pathway2 = np.random.normal(0.5, 0.2, 20)

        test_data = pd.DataFrame(
            [pathway1, pathway2],
            index=["pathway_bimodal", "pathway_unimodal"],
            columns=[f"sample_{i}" for i in range(20)],
        )

        results = GMMdecomp(test_data, K=3, verbose=False)

        bimodal_result = results["pathway_bimodal"]
        unimodal_result = results["pathway_unimodal"]

        # Check threshold values are within reasonable range
        bimodal_data = test_data.loc["pathway_bimodal"]
        unimodal_data = test_data.loc["pathway_unimodal"]

        # For bimodal data, check thresholds if any exist
        bimodal_range = bimodal_data.max() - bimodal_data.min()
        bimodal_min_extended = bimodal_data.min() - 0.5 * bimodal_range
        bimodal_max_extended = bimodal_data.max() + 0.5 * bimodal_range

        bimodal_thresholds = bimodal_result["thresholds"]
        if len(bimodal_thresholds) > 0:
            for threshold in bimodal_thresholds:
                assert (
                    bimodal_min_extended <= threshold <= bimodal_max_extended
                ), f"Bimodal threshold {threshold} should be within extended range [{bimodal_min_extended}, {bimodal_max_extended}]"

        # For unimodal data, check thresholds if any exist
        unimodal_range = unimodal_data.max() - unimodal_data.min()
        unimodal_min_extended = unimodal_data.min() - 0.5 * unimodal_range
        unimodal_max_extended = unimodal_data.max() + 0.5 * unimodal_range

        unimodal_thresholds = unimodal_result["thresholds"]
        if len(unimodal_thresholds) > 0:
            for threshold in unimodal_thresholds:
                assert (
                    unimodal_min_extended <= threshold <= unimodal_max_extended
                ), f"Unimodal threshold {threshold} should be within extended range [{unimodal_min_extended}, {unimodal_max_extended}]"

        # Check that component means are reasonable
        for pathway_name, result in results.items():
            pathway_data = test_data.loc[pathway_name]
            model = result["model"]

            # All component means should be within expanded data range
            data_range = pathway_data.max() - pathway_data.min()
            data_min = pathway_data.min() - 0.5 * data_range
            data_max = pathway_data.max() + 0.5 * data_range

            for mu in model["mu"]:
                assert (
                    data_min <= mu <= data_max
                ), f"Component mean {mu} should be within reasonable range for {pathway_name}"

        # Check that all sigma values are positive (or zero for constant data)
        for pathway_name, result in results.items():
            model = result["model"]
            assert np.all(
                model["sigma"] >= 0
            ), f"All sigma values should be non-negative for {pathway_name}"

    def test_gmmdecomp_bimodal_detection(self):
        """Test that GMMdecomp can detect bimodal patterns."""
        np.random.seed(42)

        # Create clearly bimodal and unimodal data
        pathway1 = np.concatenate(
            [np.random.normal(0.2, 0.1, 10), np.random.normal(0.8, 0.1, 10)]
        )
        pathway2 = np.random.normal(0.5, 0.2, 20)

        test_data = pd.DataFrame(
            [pathway1, pathway2],
            index=["pathway_bimodal", "pathway_unimodal"],
            columns=[f"sample_{i}" for i in range(20)],
        )

        results = GMMdecomp(test_data, K=3, verbose=False)

        bimodal_result = results["pathway_bimodal"]
        unimodal_result = results["pathway_unimodal"]

        # Check that bimodal data likely has more components than unimodal
        bimodal_components = len(bimodal_result["model"]["alpha"])
        unimodal_components = len(unimodal_result["model"]["alpha"])

        # This is probabilistic, but with our designed data it should usually hold
        assert (
            bimodal_components >= unimodal_components
        ), "Bimodal data should generally have at least as many components as unimodal"

    def test_gmmdecomp_with_numpy_array(self):
        """Test GMMdecomp with numpy array input."""
        np.random.seed(42)

        # Create test data as numpy array
        test_data = np.random.randn(2, 10)

        # Run GMM decomposition
        results = GMMdecomp(test_data, K=2, verbose=False)

        # Check results structure
        assert isinstance(results, dict), "Results should be a dictionary"
        assert len(results) == 2, "Should have results for 2 pathways"
        assert "pathway_0" in results, "Should have results for pathway_0"
        assert "pathway_1" in results, "Should have results for pathway_1"

    def test_gmmdecomp_parameter_validation_k(self):
        """Test K parameter validation in GMMdecomp."""
        np.random.seed(42)
        test_data = pd.DataFrame(np.random.randn(2, 10))

        # Test invalid K
        with pytest.raises(ValueError, match="K must be a positive integer"):
            GMMdecomp(test_data, K=0, verbose=False)

        with pytest.raises(ValueError, match="K must be a positive integer"):
            GMMdecomp(test_data, K=-1, verbose=False)

    def test_gmmdecomp_parameter_validation_types(self):
        """Test type validation for boolean parameters in GMMdecomp."""
        np.random.seed(42)
        test_data = pd.DataFrame(np.random.randn(2, 10))

        # Test invalid multiply
        with pytest.raises(ValueError, match="multiply must be a boolean value"):
            GMMdecomp(test_data, multiply="true", verbose=False)  # type: ignore

        # Test invalid parallel
        with pytest.raises(ValueError, match="parallel must be a boolean value"):
            GMMdecomp(test_data, parallel="false", verbose=False)  # type: ignore

    def test_gmmdecomp_parameter_validation_ic(self):
        """Test IC parameter validation in GMMdecomp."""
        np.random.seed(42)
        test_data = pd.DataFrame(np.random.randn(2, 10))

        # Test invalid IC
        with pytest.raises(ValueError, match="IC must be one of"):
            GMMdecomp(test_data, IC="INVALID", verbose=False)

    def test_gmmdecomp_parameter_validation_input(self):
        """Test input data validation in GMMdecomp."""
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Input data X cannot be empty"):
            GMMdecomp(empty_df, verbose=False)

        # Test invalid input type
        with pytest.raises(
            ValueError, match="X must be a pandas DataFrame or numpy array"
        ):
            GMMdecomp([1, 2, 3], verbose=False)  # type: ignore

    def test_gmmdecomp_different_ic_criteria(self):
        """Test GMMdecomp with different IC criteria."""
        np.random.seed(42)
        test_data = pd.DataFrame(np.random.randn(1, 20), index=["pathway_test"])

        ic_criteria = ["AIC", "AICc", "BIC", "ICL-BIC", "LR"]

        for ic in ic_criteria:
            results = GMMdecomp(test_data, K=2, IC=ic, verbose=False)
            assert isinstance(
                results, dict
            ), f"Results should be a dictionary for IC={ic}"
            assert len(results) == 1, f"Should have results for 1 pathway for IC={ic}"
            assert (
                "pathway_test" in results
            ), f"Should have results for pathway_test for IC={ic}"

    def test_gmmdecomp_multiply_equivalence(self):
        """Test that multiply parameter produces equivalent results."""
        np.random.seed(42)
        test_data = pd.DataFrame(np.random.randn(1, 20), index=["pathway_test"])

        # Test with multiply=True
        results_multiply = GMMdecomp(test_data, K=2, multiply=True, verbose=False)

        # Test with multiply=False
        results_no_multiply = GMMdecomp(test_data, K=2, multiply=False, verbose=False)

        # Both should return valid results
        assert isinstance(
            results_multiply, dict
        ), "Results with multiply=True should be a dictionary"
        assert isinstance(
            results_no_multiply, dict
        ), "Results with multiply=False should be a dictionary"
        assert (
            len(results_multiply) == 1
        ), "Should have results for 1 pathway with multiply=True"
        assert (
            len(results_no_multiply) == 1
        ), "Should have results for 1 pathway with multiply=False"

        # Check that results are equivalent (multiply should not change final results)
        # Since we scale and then unscale, the final results should be very similar
        result_mult = results_multiply["pathway_test"]
        result_no_mult = results_no_multiply["pathway_test"]

        # Thresholds should be similar (allowing for small numerical differences)
        thresholds_mult = result_mult["thresholds"]
        thresholds_no_mult = result_no_mult["thresholds"]
        
        assert len(thresholds_mult) == len(thresholds_no_mult), "Number of thresholds should be equal"
        
        if len(thresholds_mult) > 0:
            np.testing.assert_allclose(
                thresholds_mult,
                thresholds_no_mult,
                rtol=1e-10,
                err_msg="Thresholds should be equivalent regardless of multiply parameter"
            )

        # Component means should be similar
        np.testing.assert_allclose(
            result_mult["model"]["mu"],
            result_no_mult["model"]["mu"],
            rtol=1e-10,
            atol=1e-10,
            err_msg="Component means should be equivalent regardless of multiply parameter",
        )

        # Component standard deviations should be similar
        np.testing.assert_allclose(
            result_mult["model"]["sigma"],
            result_no_mult["model"]["sigma"],
            rtol=1e-10,
            atol=1e-10,
            err_msg="Component sigmas should be equivalent regardless of multiply parameter",
        )

    def test_gmmdecomp_edge_case_constant_data(self):
        """Test GMMdecomp with constant data (all same values)."""
        np.random.seed(42)

        # Test with constant data (all same values)
        constant_data = pd.DataFrame(np.full((1, 20), 0.5), index=["pathway_constant"])
        results_constant = GMMdecomp(constant_data, K=2, verbose=False)

        # Should still work, likely with 1 component
        assert isinstance(results_constant, dict)
        assert len(results_constant) == 1
        assert "pathway_constant" in results_constant

        # For constant data, should have no thresholds (single component)
        constant_result = results_constant["pathway_constant"]
        assert len(constant_result["thresholds"]) == 0, "Constant data should have no thresholds (single component)"
        
        # The single component should be centered around the constant value
        assert len(constant_result["model"]["mu"]) == 1, "Should have exactly one component for constant data"
        assert abs(constant_result["model"]["mu"][0] - 0.5) < 0.1, "Component mean should be near the constant value"

    def test_gmmdecomp_edge_case_small_range(self):
        """Test GMMdecomp with very small data range."""
        np.random.seed(42)

        # Test with very small data range
        small_range_data = pd.DataFrame(
            np.random.normal(0.5, 0.001, (1, 20)), index=["pathway_small_range"]
        )
        results_small = GMMdecomp(small_range_data, K=2, verbose=False)

        assert isinstance(results_small, dict)
        assert len(results_small) == 1
        assert "pathway_small_range" in results_small

    def test_gmmdecomp_edge_case_extreme_values(self):
        """Test GMMdecomp with extreme values."""
        np.random.seed(42)

        # Test with extreme values
        extreme_data = pd.DataFrame(
            np.concatenate(
                [
                    np.full(10, -100),  # Very negative values
                    np.full(10, 100),  # Very positive values
                ]
            ).reshape(1, -1),
            index=["pathway_extreme"],
        )
        results_extreme = GMMdecomp(extreme_data, K=2, verbose=False)

        assert isinstance(results_extreme, dict)
        assert len(results_extreme) == 1
        assert "pathway_extreme" in results_extreme

        # Thresholds should be somewhere between the extremes if any exist
        extreme_result = results_extreme["pathway_extreme"]
        thresholds = extreme_result["thresholds"]
        if len(thresholds) > 0:
            for threshold in thresholds:
                assert -100 <= threshold <= 100, f"Threshold {threshold} should be between extremes"
