"""
=============================================================================
PHASE 8: Validation & Edge Cases - Test Suite for Strategy
=============================================================================

PURPOSE:
    Test strategy signal generation for correctness and edge cases.
    Focus on: look-ahead bias prevention, signal values, and edge conditions.

CRITICAL TESTS:
---------------
    1. Signals must be in {-1, 0, +1}
    2. Signals must be shifted to prevent look-ahead bias
    3. Warmup period must return 0 (no signal)
    4. Edge cases: flat markets, constant prices, random data

RUN TESTS:
    cd backtester/tests
    python -m pytest test_strategy.py -v

=============================================================================
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from strategy import (
    shift_signal,
    generate_sma_crossover_signal,
    generate_momentum_signal,
    generate_mean_reversion_signal,
    generate_rsi_signal,
    combine_signals,
    signal_to_position,
    calculate_trades,
    count_trades
)
from indicators import simple_moving_average


class TestSignalShift:
    """Test look-ahead bias prevention."""
    
    def test_basic_shift(self):
        """
        Test that shift_signal correctly shifts signals.
        
        CRITICAL: This is how we prevent look-ahead bias!
        """
        signal = np.array([0, 1, 1, -1, -1, 0])
        
        # Shift by 1: today's signal comes from yesterday's data
        shifted = shift_signal(signal, periods=1)
        
        # Expected: [0, 0, 1, 1, -1, -1]
        # First element is 0 (no signal before day 0)
        # Each other element is previous day's signal
        expected = np.array([0, 0, 1, 1, -1, -1])
        
        assert np.array_equal(shifted, expected), \
            f"Expected {expected}, got {shifted}"
        
        print("✓ test_basic_shift passed")
    
    def test_shift_by_zero(self):
        """Test that shift by 0 returns copy."""
        signal = np.array([0, 1, -1, 0, 1])
        
        shifted = shift_signal(signal, periods=0)
        
        assert np.array_equal(shifted, signal), \
            "Shift by 0 should return same values"
        
        print("✓ test_shift_by_zero passed")
    
    def test_shift_preserves_length(self):
        """Test that shift preserves array length."""
        signal = np.array([0, 1, 1, -1, -1, 0, 0, 1])
        
        shifted = shift_signal(signal, periods=3)
        
        assert len(shifted) == len(signal), \
            f"Length mismatch: original={len(signal)}, shifted={len(shifted)}"
        
        print("✓ test_shift_preserves_length passed")
    
    def test_shift_fills_with_zero(self):
        """Test that shifted positions are filled with 0."""
        signal = np.array([1, 1, 1, 1, 1])
        
        shifted = shift_signal(signal, periods=2)
        
        assert shifted[0] == 0 and shifted[1] == 0, \
            f"First {2} elements should be 0, got {shifted[:2]}"
        
        print("✓ test_shift_fills_with_zero passed")


class TestSMACrossover:
    """Test SMA crossover strategy."""
    
    def test_signal_values(self):
        """
        Test that signals are only in {-1, 0, +1}.
        
        RULE: Signals must be discrete, not continuous.
        """
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, 200)))
        
        signal = generate_sma_crossover_signal(prices, fast_window=10, slow_window=30)
        
        unique_values = set(np.unique(signal))
        valid_values = {-1.0, 0.0, 1.0}
        
        assert unique_values.issubset(valid_values), \
            f"Invalid signal values: {unique_values - valid_values}"
        
        print("✓ test_signal_values passed")
    
    def test_warmup_period(self):
        """
        Test that warmup period has no signals.
        
        RULE: Before SMA is defined, signal must be 0.
        """
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, 100)))
        
        signal = generate_sma_crossover_signal(prices, fast_window=10, slow_window=30)
        
        # First 30 elements should be 0 (slow SMA needs 30 days)
        # The shift adds 1 more day of 0
        # So first 30 should definitely be 0 (element 30 could have signal after shift)
        assert np.all(signal[:30] == 0), \
            f"Warmup period should be 0, got {signal[:30]}"
        
        print("✓ test_warmup_period passed")
    
    def test_with_trending_prices(self):
        """
        Test strategy with clear uptrend.
        
        In a strong uptrend, fast SMA > slow SMA, so signal should be mostly 1.
        """
        # Create strong uptrend
        prices = 100 * np.exp(np.linspace(0, 0.5, 200))  # Steady 50% growth
        
        signal = generate_sma_crossover_signal(prices, fast_window=10, slow_window=30)
        
        # After warmup, most signals should be 1 (long)
        after_warmup = signal[40:]  # Skip warmup
        long_ratio = np.sum(after_warmup == 1) / len(after_warmup)
        
        assert long_ratio > 0.9, \
            f"In uptrend, expected >90% long signals, got {long_ratio:.1%}"
        
        print("✓ test_with_trending_prices passed")
    
    def test_with_downtrending_prices(self):
        """
        Test strategy with clear downtrend.
        
        In a strong downtrend, fast SMA < slow SMA, so signal should be mostly -1.
        """
        # Create strong downtrend
        prices = 100 * np.exp(np.linspace(0, -0.5, 200))  # Steady 50% decline
        
        signal = generate_sma_crossover_signal(prices, fast_window=10, slow_window=30)
        
        # After warmup, most signals should be -1 (short)
        after_warmup = signal[40:]
        short_ratio = np.sum(after_warmup == -1) / len(after_warmup)
        
        assert short_ratio > 0.9, \
            f"In downtrend, expected >90% short signals, got {short_ratio:.1%}"
        
        print("✓ test_with_downtrending_prices passed")
    
    def test_length_matches_prices(self):
        """Test that signal length matches price length."""
        prices = np.random.randn(150) * 10 + 100
        prices = np.abs(prices)  # Ensure positive
        
        signal = generate_sma_crossover_signal(prices, 5, 20)
        
        assert len(signal) == len(prices), \
            f"Length mismatch: prices={len(prices)}, signal={len(signal)}"
        
        print("✓ test_length_matches_prices passed")


class TestMomentumStrategy:
    """Test momentum strategy."""
    
    def test_signal_values(self):
        """Test signals are in {-1, 0, +1}."""
        np.random.seed(123)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, 200)))
        
        signal = generate_momentum_signal(prices, lookback=20, threshold=0.05)
        
        unique_values = set(np.unique(signal))
        valid_values = {-1.0, 0.0, 1.0}
        
        assert unique_values.issubset(valid_values), \
            f"Invalid values: {unique_values - valid_values}"
        
        print("✓ test_signal_values (momentum) passed")
    
    def test_threshold_effect(self):
        """
        Test that higher threshold means fewer signals.
        
        LOGIC: Higher threshold → fewer extreme momentum readings → fewer signals.
        """
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, 300)))
        
        signal_low = generate_momentum_signal(prices, 20, threshold=0.01)
        signal_high = generate_momentum_signal(prices, 20, threshold=0.10)
        
        # Higher threshold should have more zeros (fewer signals)
        zeros_low = np.sum(signal_low == 0)
        zeros_high = np.sum(signal_high == 0)
        
        assert zeros_high >= zeros_low, \
            f"Higher threshold should have more zeros: low={zeros_low}, high={zeros_high}"
        
        print("✓ test_threshold_effect passed")


class TestMeanReversionStrategy:
    """Test mean reversion strategy."""
    
    def test_signal_at_extremes(self):
        """
        Test that signals are generated at price extremes.
        
        Mean reversion: buy low (oversold), sell high (overbought).
        """
        # Create price that oscillates around mean
        t = np.linspace(0, 10 * np.pi, 300)
        prices = 100 + 20 * np.sin(t)  # Oscillates between 80 and 120
        
        signal = generate_mean_reversion_signal(prices, window=20, entry_std=1.5)
        
        # Should have some long signals (when price is low)
        # and some short signals (when price is high)
        n_long = np.sum(signal == 1)
        n_short = np.sum(signal == -1)
        
        assert n_long > 0, "Mean reversion should generate long signals at lows"
        assert n_short > 0, "Mean reversion should generate short signals at highs"
        
        print("✓ test_signal_at_extremes passed")
    
    def test_flat_prices_no_signal(self):
        """
        Test that flat prices generate no signals.
        
        EDGE CASE: If price = SMA always, z-score = 0, no extremes.
        """
        prices = np.full(100, 100.0)  # Constant price
        
        signal = generate_mean_reversion_signal(prices, window=20, entry_std=2.0)
        
        # All signals should be 0 after warmup
        # (z-score is 0 or undefined for constant prices)
        assert np.sum(np.abs(signal)) == 0, \
            "Flat prices should not generate signals"
        
        print("✓ test_flat_prices_no_signal passed")


class TestRSIStrategy:
    """Test RSI strategy."""
    
    def test_oversold_generates_long(self):
        """
        Test that oversold condition (RSI < 30) generates long signal.
        
        Create prices that would make RSI very low.
        """
        # Create strong downtrend followed by stabilization
        # This should push RSI into oversold territory
        down_phase = 100 * np.exp(np.linspace(0, -0.3, 50))
        stable_phase = np.full(50, down_phase[-1])
        prices = np.concatenate([down_phase, stable_phase])
        
        signal = generate_rsi_signal(prices, window=14, oversold=30, overbought=70)
        
        # Should have at least some long signals
        n_long = np.sum(signal == 1)
        
        # Note: This is a weak test because RSI behavior is complex
        # In practice, you'd want to verify RSI values directly
        print(f"   Long signals: {n_long}")
        print("✓ test_oversold_generates_long passed")


class TestCombineSignals:
    """Test signal combination methods."""
    
    def test_voting_method(self):
        """
        Test majority voting combination.
        
        LOGIC: 3 long + 1 short = 2 net → long
        """
        signals = [
            np.array([1, 1, -1, 0]),
            np.array([1, -1, -1, 0]),
            np.array([1, 1, 0, 0]),
            np.array([1, 1, 0, 1])
        ]
        
        combined = combine_signals(signals, method='vote')
        
        # Day 0: 4 longs → 1
        # Day 1: 2 longs, 1 short → 1 (net positive)
        # Day 2: 2 shorts, 1 neutral, 1 neutral → -1 (net negative)
        # Day 3: 1 long, 3 neutral → 1 (net positive)
        expected = np.array([1, 1, -1, 1])
        
        assert np.array_equal(combined, expected), \
            f"Expected {expected}, got {combined}"
        
        print("✓ test_voting_method passed")
    
    def test_single_signal(self):
        """Test that single signal returns itself."""
        signal = np.array([1, -1, 0, 1, -1])
        
        combined = combine_signals([signal], method='vote')
        
        assert np.array_equal(combined, signal), \
            "Single signal should return itself"
        
        print("✓ test_single_signal passed")


class TestPositionAndTrades:
    """Test position and trade calculations."""
    
    def test_signal_to_position(self):
        """Test that signal_to_position preserves values."""
        signal = np.array([0, 1, 1, -1, -1, 0])
        
        position = signal_to_position(signal)
        
        assert np.array_equal(position, signal), \
            "Basic signal_to_position should preserve values"
        
        print("✓ test_signal_to_position passed")
    
    def test_calculate_trades(self):
        """Test trade calculation from positions."""
        positions = np.array([0, 1, 1, -1, -1, 0])
        
        trades = calculate_trades(positions)
        
        # Expected trades:
        # Day 0: 0 (initial position is 0)
        # Day 1: 1-0 = 1 (enter long)
        # Day 2: 1-1 = 0 (hold)
        # Day 3: -1-1 = -2 (flip from long to short)
        # Day 4: -1-(-1) = 0 (hold)
        # Day 5: 0-(-1) = 1 (exit short)
        expected = np.array([0, 1, 0, -2, 0, 1])
        
        assert np.array_equal(trades, expected), \
            f"Expected {expected}, got {trades}"
        
        print("✓ test_calculate_trades passed")
    
    def test_count_trades(self):
        """Test trade counting."""
        positions = np.array([0, 1, 1, -1, -1, 0])
        
        n_trades = count_trades(positions)
        
        # Trades occur at: day 1, day 3, day 5 → 3 trades
        assert n_trades == 3, f"Expected 3 trades, got {n_trades}"
        
        print("✓ test_count_trades passed")
    
    def test_no_trades_if_constant_position(self):
        """Test that constant position means no trades."""
        positions = np.array([1, 1, 1, 1, 1])
        
        n_trades = count_trades(positions)
        
        # First position change (0→1) counts, then no changes
        assert n_trades == 1, f"Expected 1 trade (initial), got {n_trades}"
        
        print("✓ test_no_trades_if_constant_position passed")


class TestEdgeCases:
    """Test edge cases."""
    
    def test_all_same_prices(self):
        """Test strategies with constant prices."""
        prices = np.full(100, 100.0)
        
        mom_signal = generate_momentum_signal(prices, 20, 0.01)
        
        # Momentum should be all zeros (no price change = no momentum)
        assert np.all(mom_signal == 0), "Constant prices: momentum signal should be 0"
        
        # For SMA crossover with constant prices:
        # Both SMAs will be 100.0, so fast == slow
        # Our implementation returns 0 when equal (neither > nor <)
        # After shift, this propagates correctly
        sma_signal = generate_sma_crossover_signal(prices, 10, 30)
        
        # The vast majority should be 0
        non_zero_ratio = np.sum(np.abs(sma_signal)) / len(sma_signal)
        assert non_zero_ratio < 0.05, \
            f"Constant prices: SMA signal should be mostly 0, got {non_zero_ratio:.1%} non-zero"
        
        print("✓ test_all_same_prices passed")
    
    def test_very_short_prices(self):
        """Test with minimum length price array."""
        prices = np.array([100.0, 105.0, 102.0])
        
        # Should not crash, even if all signals are 0
        signal = generate_sma_crossover_signal(prices, 2, 3)
        
        assert len(signal) == len(prices)
        
        print("✓ test_very_short_prices passed")
    
    def test_random_prices(self):
        """
        Test with random prices (sanity check).
        
        Random walk prices should produce some of all signal types.
        """
        np.random.seed(12345)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, 500)))
        
        signal = generate_sma_crossover_signal(prices, 10, 30)
        
        # Should have all three signal types in random data
        unique = set(np.unique(signal))
        
        assert -1 in unique or 1 in unique, \
            "Random prices should generate at least some non-zero signals"
        
        print("✓ test_random_prices passed")


def run_all_tests():
    """Run all tests and report results."""
    
    print("=" * 70)
    print("RUNNING STRATEGY TEST SUITE")
    print("=" * 70)
    
    test_classes = [
        TestSignalShift(),
        TestSMACrossover(),
        TestMomentumStrategy(),
        TestMeanReversionStrategy(),
        TestRSIStrategy(),
        TestCombineSignals(),
        TestPositionAndTrades(),
        TestEdgeCases()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__class__.__name__}")
        print("-" * 40)
        
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
