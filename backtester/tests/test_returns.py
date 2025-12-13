"""
=============================================================================
PHASE 8: Validation & Edge Cases - Test Suite for Returns
=============================================================================

PURPOSE:
    Test the returns calculations to ensure correctness.
    This is CRITICAL - all other metrics depend on returns!

TESTING PHILOSOPHY:
-------------------
    1. Test known values (where you can calculate by hand)
    2. Test edge cases (flat prices, zeros, single element)
    3. Test mathematical properties (log returns are additive)
    4. Test defensive behavior (what happens with bad input?)

NUMPY TESTING CONCEPTS:
    - np.testing.assert_array_almost_equal() for floating point comparison
    - np.allclose() for approximate equality
    - Testing with known mathematical relationships

RUN TESTS:
    cd backtester/tests
    python -m pytest test_returns.py -v

=============================================================================
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from indicators import simple_returns, log_returns, cumulative_returns


class TestSimpleReturns:
    """Test suite for simple returns calculation."""
    
    def test_basic_calculation(self):
        """
        Test basic return calculation with known values.
        
        NUMPY LEARNING: Using known values to verify calculations.
        Hand-calculate expected results, then verify code matches.
        """
        # Known values: prices go from 100 to 110 (10% return)
        prices = np.array([100.0, 110.0, 105.0, 115.0])
        
        # Hand-calculated returns:
        # Day 0: no return (first day)
        # Day 1: (110-100)/100 = 0.10 (10%)
        # Day 2: (105-110)/110 = -0.0455 (-4.55%)
        # Day 3: (115-105)/105 = 0.0952 (9.52%)
        
        expected = np.array([0.0, 0.10, -0.04545455, 0.0952381])
        
        result = simple_returns(prices)
        
        # NUMPY CONCEPT: np.allclose for floating point comparison
        # Never use == for floats due to precision issues!
        assert np.allclose(result, expected, rtol=1e-5), \
            f"Expected {expected}, got {result}"
        
        print("✓ test_basic_calculation passed")
    
    def test_length_preservation(self):
        """
        Test that returns array has same length as prices.
        
        IMPORTANT: Many implementations return n-1 elements.
        Our implementation prepends 0 to maintain alignment.
        """
        prices = np.array([100.0, 105.0, 110.0, 108.0, 112.0])
        
        result = simple_returns(prices)
        
        assert len(result) == len(prices), \
            f"Length mismatch: prices={len(prices)}, returns={len(result)}"
        
        print("✓ test_length_preservation passed")
    
    def test_first_element_is_zero(self):
        """
        Test that first element is 0 (no return on first day).
        
        FINANCIAL MEANING: On day 0, there's no previous price to compare to.
        """
        prices = np.array([100.0, 105.0, 110.0])
        
        result = simple_returns(prices)
        
        assert result[0] == 0.0, f"First element should be 0, got {result[0]}"
        
        print("✓ test_first_element_is_zero passed")
    
    def test_flat_prices(self):
        """
        Test with constant prices (edge case: no volatility).
        
        EDGE CASE: If prices never change, all returns should be 0.
        """
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        result = simple_returns(prices)
        
        assert np.allclose(result, expected), \
            f"Flat prices should give zero returns. Got {result}"
        
        print("✓ test_flat_prices passed")
    
    def test_single_price(self):
        """
        Test with single price (minimum input).
        
        EDGE CASE: Only one price → only one return (0).
        """
        prices = np.array([100.0])
        
        result = simple_returns(prices)
        
        assert len(result) == 1, f"Expected length 1, got {len(result)}"
        assert result[0] == 0.0, f"Expected 0, got {result[0]}"
        
        print("✓ test_single_price passed")
    
    def test_large_return(self):
        """
        Test with large price movements (stress test).
        
        EDGE CASE: 100% gain and 50% loss.
        """
        # Price doubles (100% return), then halves (50% loss)
        prices = np.array([100.0, 200.0, 100.0])
        
        # Expected: 0, +1.0 (100%), -0.5 (-50%)
        expected = np.array([0.0, 1.0, -0.5])
        
        result = simple_returns(prices)
        
        assert np.allclose(result, expected), \
            f"Expected {expected}, got {result}"
        
        print("✓ test_large_return passed")


class TestLogReturns:
    """Test suite for log returns calculation."""
    
    def test_basic_calculation(self):
        """Test log return calculation with known values."""
        prices = np.array([100.0, 110.0, 105.0])
        
        # Log returns: ln(P[t]/P[t-1])
        # Day 0: 0
        # Day 1: ln(110/100) = ln(1.1) = 0.0953
        # Day 2: ln(105/110) = ln(0.9545) = -0.0465
        
        expected = np.array([0.0, np.log(1.1), np.log(105/110)])
        
        result = log_returns(prices)
        
        assert np.allclose(result, expected, rtol=1e-5), \
            f"Expected {expected}, got {result}"
        
        print("✓ test_basic_calculation (log) passed")
    
    def test_time_additivity(self):
        """
        Test that log returns are time-additive.
        
        MATHEMATICAL PROPERTY:
        log_return_total = sum(log_returns)
        
        This is WHY we use log returns for multi-period calculations!
        """
        prices = np.array([100.0, 110.0, 99.0, 120.0, 115.0])
        
        log_rets = log_returns(prices)
        
        # Sum of log returns should equal total log return
        sum_log_returns = np.sum(log_rets)
        total_log_return = np.log(prices[-1] / prices[0])
        
        assert np.isclose(sum_log_returns, total_log_return, rtol=1e-10), \
            f"Sum of log returns ({sum_log_returns}) != total ({total_log_return})"
        
        print("✓ test_time_additivity passed")
    
    def test_symmetry(self):
        """
        Test log return symmetry around zero.
        
        MATHEMATICAL PROPERTY:
        ln(1.10) ≈ -ln(1/1.10) = -ln(0.909)
        
        This means +10% and -10% have similar magnitudes in log space.
        """
        # +10% gain
        gain = np.log(1.10)
        
        # -10% loss (approximately)
        loss = np.log(0.90)
        
        # They should be approximately opposite in magnitude
        # (Not exactly due to asymmetry of returns, but close)
        assert np.abs(gain + loss) < 0.02, \
            f"Log returns not symmetric: +10% = {gain}, -10% = {loss}"
        
        print("✓ test_symmetry passed")


class TestCumulativeReturns:
    """Test suite for cumulative returns."""
    
    def test_final_cumulative_matches_total(self):
        """
        Test that final cumulative return matches total return.
        
        PROPERTY: cum_returns[-1] should equal total return.
        """
        prices = np.array([100.0, 110.0, 99.0, 120.0])
        
        simple_rets = simple_returns(prices)
        cum_rets = cumulative_returns(simple_rets, use_log=False)
        
        # Total return
        total_return = prices[-1] / prices[0] - 1
        
        assert np.isclose(cum_rets[-1], total_return, rtol=1e-10), \
            f"Final cumulative ({cum_rets[-1]}) != total return ({total_return})"
        
        print("✓ test_final_cumulative_matches_total passed")
    
    def test_cumulative_is_monotonic_with_positive_returns(self):
        """
        Test that cumulative returns increase with positive returns.
        
        If all returns are positive, cumulative should be monotonically increasing.
        """
        # All positive returns
        returns = np.array([0.0, 0.01, 0.02, 0.015, 0.01])
        
        cum_rets = cumulative_returns(returns, use_log=False)
        
        # Each element should be >= previous
        # NUMPY CONCEPT: np.diff should be all non-negative
        differences = np.diff(cum_rets)
        
        assert np.all(differences >= 0), \
            f"Cumulative returns should be monotonically increasing with positive returns"
        
        print("✓ test_cumulative_is_monotonic_with_positive_returns passed")


class TestEdgeCases:
    """Test edge cases and defensive programming."""
    
    def test_with_random_data(self):
        """
        Test with random data (sanity check).
        
        PRINCIPLE: Random test data should not crash the code.
        """
        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.02, 1000)))
        
        # Should not raise any exceptions
        simple_rets = simple_returns(prices)
        log_rets = log_returns(prices)
        cum_rets = cumulative_returns(simple_rets)
        
        # Basic sanity checks
        assert len(simple_rets) == len(prices)
        assert len(log_rets) == len(prices)
        assert len(cum_rets) == len(prices)
        assert not np.any(np.isnan(simple_rets))
        assert not np.any(np.isnan(log_rets))
        assert not np.any(np.isnan(cum_rets))
        
        print("✓ test_with_random_data passed")
    
    def test_very_small_returns(self):
        """
        Test numerical stability with very small price changes.
        
        EDGE CASE: Prices that change by tiny amounts.
        """
        prices = np.array([100.0, 100.0001, 100.0002, 100.0001])
        
        simple_rets = simple_returns(prices)
        log_rets = log_returns(prices)
        
        # Should be very close to each other for small returns
        # Because for small x: log(1+x) ≈ x
        assert np.allclose(simple_rets, log_rets, atol=1e-6), \
            "For small returns, simple and log returns should be nearly equal"
        
        print("✓ test_very_small_returns passed")
    
    def test_preservation_of_dtype(self):
        """
        Test that output dtype is appropriate.
        
        NUMPY CONCEPT: dtype should be float for returns.
        """
        prices = np.array([100.0, 105.0, 110.0], dtype=np.float64)
        
        result = simple_returns(prices)
        
        assert result.dtype == np.float64, \
            f"Expected float64, got {result.dtype}"
        
        print("✓ test_preservation_of_dtype passed")


def run_all_tests():
    """Run all tests and report results."""
    
    print("=" * 70)
    print("RUNNING RETURNS TEST SUITE")
    print("=" * 70)
    
    test_classes = [
        TestSimpleReturns(),
        TestLogReturns(),
        TestCumulativeReturns(),
        TestEdgeCases()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__class__.__name__}")
        print("-" * 40)
        
        # Get all test methods
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_class, method_name)
                method()
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {method_name} FAILED: {e}")
            except Exception as e:
                print(f"✗ {method_name} ERROR: {e}")
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 70)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
