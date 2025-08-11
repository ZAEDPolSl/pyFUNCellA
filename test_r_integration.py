#!/usr/bin/env python3
"""
Test script to verify R and process-based R integration
Run this to check if your Docker environment has proper R setup
"""

import pandas as pd
import numpy as np


def test_basic_r_integration():
    """Test basic R availability"""
    try:
        print("Testing R process-based integration...")
        from enrichment_auc.utils.r_executor import check_r_available, execute_r_code

        if not check_r_available():
            print("✗ R is not available")
            return False

        print("✓ R is available")

        # Test basic R calculation
        result = execute_r_code("test_result <- 2 + 2")
        if result.get("success") and result.get("test_result") == 4:
            print("✓ Basic R calculation works")
        else:
            print("✗ Basic R calculation failed")
            return False

        return True

    except Exception as e:
        print(f"✗ R integration test failed: {e}")
        return False


def test_r_packages():
    """Test R package availability"""
    try:
        print("\nTesting R packages...")
        from enrichment_auc.utils.r_executor import execute_r_code

        r_code = """
        packages <- c('AUCell', 'GSVA', 'dpGMM')
        package_status <- list()
        
        for (pkg in packages) {
            status <- tryCatch({
                library(pkg, character.only=TRUE, quietly=TRUE)
                TRUE
            }, error = function(e) {
                FALSE
            })
            package_status[[pkg]] <- status
            cat('Package', pkg, ':', ifelse(status, 'available', 'not available'), '\\n')
        }
        
        test_result <- package_status
        """

        result = execute_r_code(r_code)

        if result.get("success"):
            packages = result.get("test_result", {})
            available_count = sum(packages.values()) if packages else 0
            total_count = len(packages) if packages else 0
            print(
                f"✓ Package test completed: {available_count}/{total_count} packages available"
            )
            return True
        else:
            print("✗ Package test failed:", result.get("stdout", "Unknown error"))
            return False

    except Exception as e:
        print(f"✗ Package test failed: {e}")
        return False


def test_data_transfer():
    """Test Python-R data transfer"""
    try:
        print("\nTesting data transfer...")
        from enrichment_auc.utils.r_executor import execute_r_code

        # Create test data
        test_data = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50]}
        )

        r_code = """
        if (exists('test_data')) {
            cat("Received data with dimensions:", dim(test_data), "\\n")
            cat("Column names:", colnames(test_data), "\\n")
            data_mean <- mean(as.matrix(test_data))
            cat("Data mean:", data_mean, "\\n")
            test_result <- list(
                dimensions = dim(test_data),
                mean_value = data_mean,
                success = TRUE
            )
        } else {
            cat("No test data received\\n")
            test_result <- list(success = FALSE)
        }
        """

        result = execute_r_code(r_code, {"test_data": test_data})

        if result.get("success") and result.get("test_result", {}).get("success"):
            print("✓ Data transfer successful")
            return True
        else:
            print("✗ Data transfer failed")
            return False

    except Exception as e:
        print(f"✗ Data transfer test failed: {e}")
        return False


def test_gmm_decomposition():
    """Test GMM decomposition functionality"""
    try:
        print("\nTesting GMM decomposition...")
        from enrichment_auc.GMMdecomp import _check_dpgmm_installed

        if not _check_dpgmm_installed():
            print("✗ dpGMM package not available")
            return False

        print("✓ dpGMM package available")

        # Create simple test data for GMM
        test_data = pd.DataFrame(np.random.randn(3, 10))

        from enrichment_auc.GMMdecomp import GMMdecomp

        results = GMMdecomp(test_data, K=2, verbose=True)

        if results and len(results) > 0:
            print("✓ GMM decomposition completed")
            return True
        else:
            print("✗ GMM decomposition failed")
            return False

    except Exception as e:
        print(f"✗ GMM decomposition test failed: {e}")
        return False


def test_aucell_thresholding():
    """Test AUCell thresholding functionality"""
    try:
        print("\nTesting AUCell thresholding...")
        from enrichment_auc.thr_AUCell import _check_aucell_installed

        if not _check_aucell_installed():
            print("✗ AUCell package not available")
            return False

        print("✓ AUCell package available")
        return True

    except Exception as e:
        print(f"✗ AUCell test failed: {e}")
        return False


def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("ENRICHMENT-AUC R INTEGRATION TEST SUITE")
    print("=" * 60)

    tests = [
        ("test_basic_r_integration", test_basic_r_integration),
        ("test_r_packages", test_r_packages),
        ("test_data_transfer", test_data_transfer),
        ("test_gmm_decomposition", test_gmm_decomposition),
        ("test_aucell_thresholding", test_aucell_thresholding),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for i, (test_name, success) in enumerate(results, 1):
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{i}. {test_name}: {status}")
        if success:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! R integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    return passed == len(results)


if __name__ == "__main__":
    main()
