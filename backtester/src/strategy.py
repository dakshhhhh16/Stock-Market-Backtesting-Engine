"""
=============================================================================
PHASE 5: Strategy Logic (Signals, Not Trades)
=============================================================================

PURPOSE:
    Generate trading signals from indicators.
    Strategies produce signals {-1, 0, +1}, NOT trades.

KEY MENTAL MODEL:
-----------------
    Signal = What you WANT to do
    Position = What you actually HOLD
    Trade = The ACTION to go from position to new position

    signal[t] = +1  → Want to be LONG
    signal[t] = -1  → Want to be SHORT
    signal[t] =  0  → Want to be FLAT (no position)

CRITICAL CONCEPT - LOOK-AHEAD BIAS:
------------------------------------
    The #1 bug in backtesting is using future information!
    
    WRONG: signal[t] = f(price[t])  → Trading on current info (impossible!)
    RIGHT: signal[t] = f(price[t-1]) → Trading on yesterday's info
    
    We achieve this with array shifting.

NUMPY CONCEPTS YOU'LL LEARN:
    - Boolean masks and np.where()
    - Array shifting for look-ahead prevention
    - Signal combination logic
    - Vectorized conditional logic

=============================================================================
"""

import numpy as np
from typing import Optional, Callable
from indicators import (
    simple_moving_average, 
    exponential_moving_average,
    relative_strength_index,
    bollinger_bands,
    simple_returns
)


def shift_signal(signal: np.ndarray, periods: int = 1) -> np.ndarray:
    """
    Shift signal array to prevent look-ahead bias.
    
    CRITICAL FUNCTION - UNDERSTANDING LOOK-AHEAD BIAS:
    ---------------------------------------------------
    If we generate a signal based on today's price, we can't trade
    on that signal until TOMORROW. So we must shift signals forward.
    
    signal[t] based on price[t] → can only be ACTED ON at t+1
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.roll() - Circular shift of array elements
       - roll([1,2,3,4], 1) = [4,1,2,3]
       - Problem: wraps around, not what we want!
    
    2. Manual shift with slicing
       - shifted[periods:] = original[:-periods]
       - Fill first 'periods' elements with 0 or NaN
    
    3. np.empty_like() - Create array with same shape/dtype
       - Faster than np.zeros() when we'll overwrite anyway
    
    Parameters:
    -----------
    signal : np.ndarray
        Original signal array
    periods : int
        Number of periods to shift (default 1 = use yesterday's signal today)
        
    Returns:
    --------
    np.ndarray: Shifted signal where signal[t] reflects decision from t-periods
    """
    
    # Create output array with same shape and dtype
    # NUMPY CONCEPT: empty_like creates uninitialized array (faster)
    shifted = np.empty_like(signal)
    
    if periods > 0:
        # Forward shift: move data to the right
        # First 'periods' elements become 0 (no signal initially)
        shifted[:periods] = 0
        shifted[periods:] = signal[:-periods]
    elif periods < 0:
        # Backward shift (rarely used, but included for completeness)
        shifted[periods:] = 0
        shifted[:periods] = signal[-periods:]
    else:
        # No shift
        shifted = signal.copy()
    
    return shifted


def generate_sma_crossover_signal(prices: np.ndarray,
                                   fast_window: int = 10,
                                   slow_window: int = 30) -> np.ndarray:
    """
    Generate signals based on SMA crossover strategy.
    
    STRATEGY LOGIC:
    ---------------
        signal = +1  when SMA_fast > SMA_slow  (uptrend, go long)
        signal = -1  when SMA_fast < SMA_slow  (downtrend, go short)
        signal =  0  when either SMA is undefined (warmup period)
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.where(condition, value_if_true, value_if_false)
       - Vectorized if-else statement
       - Can be nested for multiple conditions
    
    2. Boolean comparison on arrays
       - fast > slow returns boolean array
       - True where condition holds, False otherwise
    
    3. np.nan handling
       - Need to handle warmup period where SMAs are NaN
       - np.isnan() returns boolean array
    
    Parameters:
    -----------
    prices : np.ndarray
        Price array
    fast_window : int
        Fast SMA period
    slow_window : int
        Slow SMA period
        
    Returns:
    --------
    np.ndarray: Signal array with values in {-1, 0, +1}
    """
    
    # Calculate SMAs
    sma_fast = simple_moving_average(prices, fast_window)
    sma_slow = simple_moving_average(prices, slow_window)
    
    # Initialize signal array with zeros
    # NUMPY CONCEPT: np.zeros creates array of zeros
    signal = np.zeros(len(prices))
    
    # Generate signals using np.where
    # NUMPY CONCEPT: Nested np.where for multiple conditions
    # 
    # Logic:
    #   - If either SMA is NaN, signal = 0 (no trade during warmup)
    #   - If fast > slow, signal = 1 (bullish)
    #   - If fast < slow, signal = -1 (bearish)
    #   - If fast == slow, signal = 0 (neutral)
    
    # First, create a mask for valid (non-NaN) values
    valid_mask = ~np.isnan(sma_fast) & ~np.isnan(sma_slow)
    
    # Calculate the difference to avoid floating point comparison issues
    # NUMPY CONCEPT: Using difference threshold for robust comparison
    # A tiny threshold like 1e-10 handles floating point precision issues
    diff = sma_fast - sma_slow
    threshold = 1e-10  # Negligible difference treated as equal
    
    # Where valid, apply the strategy logic
    # NUMPY CONCEPT: np.where with three arrays
    signal = np.where(
        ~valid_mask,  # condition: if NOT valid
        0,            # return 0 (neutral during warmup)
        np.where(
            diff > threshold,   # if fast clearly > slow
            1,                  # long signal
            np.where(
                diff < -threshold,  # elif fast clearly < slow
                -1,                 # short signal
                0                   # else (approximately equal) neutral
            )
        )
    )
    
    # CRITICAL: Shift signal to prevent look-ahead bias
    # Today's signal is based on today's SMA, so we can only act tomorrow
    signal = shift_signal(signal, periods=1)
    
    return signal


def generate_momentum_signal(prices: np.ndarray,
                             lookback: int = 20,
                             threshold: float = 0.0) -> np.ndarray:
    """
    Generate signals based on momentum (past returns).
    
    STRATEGY LOGIC:
    ---------------
        signal = +1  when momentum > threshold  (positive momentum)
        signal = -1  when momentum < -threshold (negative momentum)
        signal =  0  otherwise
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. Calculating momentum with array slicing
       - momentum[t] = price[t] / price[t-lookback] - 1
       - Uses ratio of current to past price
    
    2. np.roll() vs manual slicing
       - roll wraps around, slicing doesn't
       - For financial data, prefer slicing (no wrap-around)
    
    Parameters:
    -----------
    prices : np.ndarray
        Price array
    lookback : int
        Number of periods to look back for momentum
    threshold : float
        Minimum momentum to trigger signal
        
    Returns:
    --------
    np.ndarray: Signal array with values in {-1, 0, +1}
    """
    
    n = len(prices)
    
    # Calculate momentum
    # momentum[t] = (price[t] - price[t-lookback]) / price[t-lookback]
    #             = price[t] / price[t-lookback] - 1
    
    # NUMPY CONCEPT: Array slicing for offset calculations
    # prices[lookback:] = current prices (from day 'lookback' onwards)
    # prices[:-lookback] = lagged prices (from day 0 to n-lookback-1)
    momentum = np.zeros(n)
    momentum[lookback:] = prices[lookback:] / prices[:-lookback] - 1
    
    # Generate signals
    # NUMPY CONCEPT: Compound conditions with & (AND) and | (OR)
    signal = np.where(
        momentum > threshold,
        1,
        np.where(
            momentum < -threshold,
            -1,
            0
        )
    )
    
    # Shift to prevent look-ahead bias
    signal = shift_signal(signal, periods=1)
    
    return signal


def generate_mean_reversion_signal(prices: np.ndarray,
                                    window: int = 20,
                                    entry_std: float = 2.0,
                                    exit_std: float = 0.5) -> np.ndarray:
    """
    Generate signals based on mean reversion (Bollinger Band style).
    
    STRATEGY LOGIC:
    ---------------
        signal = +1  when price < lower_band (oversold, expect bounce)
        signal = -1  when price > upper_band (overbought, expect drop)
        signal =  0  when price near middle (no extreme)
    
    MEAN REVERSION CONCEPT:
    -----------------------
    Assumes prices tend to return to their mean.
    - Buy when price is "too low" (below average)
    - Sell when price is "too high" (above average)
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. Z-score calculation
       - z = (price - mean) / std
       - Measures how many std deviations from mean
    
    2. Combining multiple signals
       - Can use different thresholds for entry/exit
    
    Parameters:
    -----------
    prices : np.ndarray
        Price array
    window : int
        Rolling window for mean and std
    entry_std : float
        Number of std devs for entry signal (e.g., 2.0)
    exit_std : float
        Number of std devs to exit back to neutral (e.g., 0.5)
        
    Returns:
    --------
    np.ndarray: Signal array with values in {-1, 0, +1}
    """
    
    # Calculate rolling mean and std
    sma = simple_moving_average(prices, window)
    
    # Calculate rolling std using sliding window
    from numpy.lib.stride_tricks import sliding_window_view
    
    n = len(prices)
    rolling_std = np.full(n, np.nan)
    
    windows = sliding_window_view(prices, window)
    rolling_std[window-1:] = np.std(windows, axis=1, ddof=1)
    
    # Calculate z-score
    # NUMPY CONCEPT: Division with NaN handling
    # Where std is 0 or NaN, z-score should be 0
    with np.errstate(divide='ignore', invalid='ignore'):
        z_score = np.where(
            (rolling_std == 0) | np.isnan(rolling_std) | np.isnan(sma),
            0,
            (prices - sma) / rolling_std
        )
    
    # Generate entry signals based on z-score extremes
    # NUMPY CONCEPT: Compound boolean operations
    signal = np.where(
        z_score < -entry_std,  # Price too low (oversold)
        1,                      # Buy signal
        np.where(
            z_score > entry_std,  # Price too high (overbought)
            -1,                    # Sell signal
            0                      # No signal
        )
    )
    
    # Shift to prevent look-ahead bias
    signal = shift_signal(signal, periods=1)
    
    return signal


def generate_rsi_signal(prices: np.ndarray,
                        window: int = 14,
                        oversold: float = 30.0,
                        overbought: float = 70.0) -> np.ndarray:
    """
    Generate signals based on RSI indicator.
    
    STRATEGY LOGIC:
    ---------------
        signal = +1  when RSI < oversold (30)  → Oversold, expect bounce
        signal = -1  when RSI > overbought (70) → Overbought, expect drop
        signal =  0  otherwise
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. Using pre-built indicators in strategy
    2. Simple threshold comparisons
    
    Parameters:
    -----------
    prices : np.ndarray
        Price array
    window : int
        RSI period
    oversold : float
        RSI level below which asset is considered oversold
    overbought : float
        RSI level above which asset is considered overbought
        
    Returns:
    --------
    np.ndarray: Signal array with values in {-1, 0, +1}
    """
    
    # Calculate RSI
    rsi = relative_strength_index(prices, window)
    
    # Generate signals
    signal = np.where(
        rsi < oversold,
        1,  # Buy when oversold
        np.where(
            rsi > overbought,
            -1,  # Sell when overbought
            0    # Neutral otherwise
        )
    )
    
    # Handle NaN values in RSI
    signal = np.where(np.isnan(rsi), 0, signal)
    
    # Shift to prevent look-ahead bias
    signal = shift_signal(signal, periods=1)
    
    return signal


def generate_breakout_signal(prices: np.ndarray,
                             window: int = 20) -> np.ndarray:
    """
    Generate signals based on price breakout (new highs/lows).
    
    STRATEGY LOGIC:
    ---------------
        signal = +1  when price makes new high over window
        signal = -1  when price makes new low over window
        signal =  0  otherwise
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. Rolling maximum and minimum
       - np.maximum.accumulate() for running max
       - Using sliding window for rolling max/min
    
    2. Comparing current value to historical range
    
    Parameters:
    -----------
    prices : np.ndarray
        Price array
    window : int
        Lookback window for high/low detection
        
    Returns:
    --------
    np.ndarray: Signal array with values in {-1, 0, +1}
    """
    
    from numpy.lib.stride_tricks import sliding_window_view
    
    n = len(prices)
    
    # Calculate rolling high and low
    rolling_high = np.full(n, np.nan)
    rolling_low = np.full(n, np.nan)
    
    windows = sliding_window_view(prices, window)
    
    # NUMPY CONCEPT: np.max and np.min with axis parameter
    # axis=1 means "compute across columns" (across each window)
    rolling_high[window-1:] = np.max(windows, axis=1)
    rolling_low[window-1:] = np.min(windows, axis=1)
    
    # Generate signals
    # New high: current price == rolling high
    # New low: current price == rolling low
    signal = np.where(
        prices == rolling_high,
        1,  # Breakout up
        np.where(
            prices == rolling_low,
            -1,  # Breakout down
            0    # No breakout
        )
    )
    
    # Handle NaN
    signal = np.where(np.isnan(rolling_high), 0, signal)
    
    # Shift to prevent look-ahead bias
    signal = shift_signal(signal, periods=1)
    
    return signal


def combine_signals(signals: list, 
                    weights: Optional[np.ndarray] = None,
                    method: str = 'vote') -> np.ndarray:
    """
    Combine multiple strategy signals into a single signal.
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.stack() - Combine arrays into new dimension
       - stack([arr1, arr2, arr3]) creates 2D array
       - Each original array becomes a row
    
    2. np.sum() with axis parameter
       - axis=0 sums across first dimension
    
    3. np.sign() - Get sign of numbers
       - sign(5) = 1, sign(-3) = -1, sign(0) = 0
    
    COMBINING METHODS:
    ------------------
    'vote':    Majority voting - final signal based on sum of signals
    'all':     All signals must agree (AND logic)
    'any':     Any signal triggers (OR logic)
    'weighted': Weighted sum of signals
    
    Parameters:
    -----------
    signals : list of np.ndarray
        List of signal arrays to combine
    weights : np.ndarray, optional
        Weights for each strategy (for 'weighted' method)
    method : str
        Combining method: 'vote', 'all', 'any', 'weighted'
        
    Returns:
    --------
    np.ndarray: Combined signal array
    """
    
    if len(signals) == 0:
        raise ValueError("Must provide at least one signal")
    
    if len(signals) == 1:
        return signals[0].copy()
    
    # Stack signals into 2D array
    # NUMPY CONCEPT: np.stack combines arrays along new axis
    # Shape: (n_strategies, n_timepoints)
    stacked = np.stack(signals, axis=0)
    
    if method == 'vote':
        # Sum signals and take sign
        # NUMPY CONCEPT: np.sum with axis, then np.sign
        signal_sum = np.sum(stacked, axis=0)
        combined = np.sign(signal_sum)
        
    elif method == 'all':
        # All must be positive for +1, all must be negative for -1
        # NUMPY CONCEPT: np.all with axis parameter
        all_positive = np.all(stacked > 0, axis=0)
        all_negative = np.all(stacked < 0, axis=0)
        combined = np.where(all_positive, 1, np.where(all_negative, -1, 0))
        
    elif method == 'any':
        # Any positive gives +1, any negative gives -1 (positive takes precedence)
        # NUMPY CONCEPT: np.any with axis parameter
        any_positive = np.any(stacked > 0, axis=0)
        any_negative = np.any(stacked < 0, axis=0)
        combined = np.where(any_positive, 1, np.where(any_negative, -1, 0))
        
    elif method == 'weighted':
        if weights is None:
            # Equal weights if not specified
            weights = np.ones(len(signals)) / len(signals)
        
        # Weighted sum
        # NUMPY CONCEPT: Broadcasting weights across signals
        # weights[:, np.newaxis] adds new axis for broadcasting
        weighted_sum = np.sum(stacked * weights[:, np.newaxis], axis=0)
        combined = np.sign(weighted_sum)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return combined.astype(np.float64)


def signal_to_position(signal: np.ndarray) -> np.ndarray:
    """
    Convert signals to positions (same in basic case).
    
    CONCEPT:
    --------
    For simple strategies:
        position[t] = signal[t]
    
    Signal is what you WANT, position is what you HAVE.
    In basic case, we assume immediate execution so they're the same.
    
    For more complex scenarios (partial fills, delays), they differ.
    
    Parameters:
    -----------
    signal : np.ndarray
        Signal array with values in {-1, 0, +1}
        
    Returns:
    --------
    np.ndarray: Position array with values in {-1, 0, +1}
    """
    
    # In simple case, position follows signal exactly
    return signal.copy()


def calculate_trades(positions: np.ndarray) -> np.ndarray:
    """
    Calculate trades from position changes.
    
    FORMULA:
        trade[t] = position[t] - position[t-1]
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.diff() for position changes
    2. Trade values can be {-2, -1, 0, +1, +2}
       - +2: short to long
       - +1: flat to long, or short to flat
       -  0: no change
       - -1: long to flat, or flat to short
       - -2: long to short
    
    Parameters:
    -----------
    positions : np.ndarray
        Position array with values in {-1, 0, +1}
        
    Returns:
    --------
    np.ndarray: Trade array showing position changes
    """
    
    # Calculate position changes
    # NUMPY CONCEPT: np.diff gives p[t] - p[t-1]
    position_changes = np.diff(positions)
    
    # Prepend 0 (no trade before first position)
    trades = np.concatenate([[positions[0]], position_changes])
    
    return trades


def count_trades(positions: np.ndarray) -> int:
    """
    Count the number of trades (position changes).
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.diff() for changes
    2. np.count_nonzero() for counting
    
    Parameters:
    -----------
    positions : np.ndarray
        Position array
        
    Returns:
    --------
    int: Number of trades
    """
    
    trades = calculate_trades(positions)
    
    # Count non-zero trades
    # NUMPY CONCEPT: np.count_nonzero is faster than sum(x != 0)
    return np.count_nonzero(trades)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 5: Strategy Logic - Signal Generation")
    print("=" * 70)
    
    # Generate sample prices
    np.random.seed(42)
    n_days = 252  # 1 year
    
    # Create trending + mean-reverting price series
    trend = np.linspace(0, 0.3, n_days)  # Upward trend
    noise = np.cumsum(np.random.normal(0, 0.02, n_days))  # Random walk
    prices = 100 * np.exp(trend + noise)
    
    print(f"\nGenerated {n_days} days of price data")
    print(f"Starting price: ${prices[0]:.2f}")
    print(f"Ending price: ${prices[-1]:.2f}")
    
    # ==========================================================================
    # Test different strategies
    # ==========================================================================
    
    print("\n" + "-" * 70)
    print("Testing SMA Crossover Strategy (10/30)")
    print("-" * 70)
    
    sma_signal = generate_sma_crossover_signal(prices, 10, 30)
    positions = signal_to_position(sma_signal)
    n_trades = count_trades(positions)
    
    print(f"Signal values: {np.unique(sma_signal)}")
    print(f"Long days: {np.sum(sma_signal == 1)}")
    print(f"Short days: {np.sum(sma_signal == -1)}")
    print(f"Flat days: {np.sum(sma_signal == 0)}")
    print(f"Number of trades: {n_trades}")
    
    print("\n" + "-" * 70)
    print("Testing Momentum Strategy (20-day)")
    print("-" * 70)
    
    mom_signal = generate_momentum_signal(prices, lookback=20, threshold=0.05)
    n_trades = count_trades(mom_signal)
    
    print(f"Long days: {np.sum(mom_signal == 1)}")
    print(f"Short days: {np.sum(mom_signal == -1)}")
    print(f"Flat days: {np.sum(mom_signal == 0)}")
    print(f"Number of trades: {n_trades}")
    
    print("\n" + "-" * 70)
    print("Testing Mean Reversion Strategy")
    print("-" * 70)
    
    mr_signal = generate_mean_reversion_signal(prices, window=20, entry_std=2.0)
    n_trades = count_trades(mr_signal)
    
    print(f"Long days: {np.sum(mr_signal == 1)}")
    print(f"Short days: {np.sum(mr_signal == -1)}")
    print(f"Flat days: {np.sum(mr_signal == 0)}")
    print(f"Number of trades: {n_trades}")
    
    print("\n" + "-" * 70)
    print("Testing RSI Strategy")
    print("-" * 70)
    
    rsi_signal = generate_rsi_signal(prices, window=14, oversold=30, overbought=70)
    n_trades = count_trades(rsi_signal)
    
    print(f"Long days: {np.sum(rsi_signal == 1)}")
    print(f"Short days: {np.sum(rsi_signal == -1)}")
    print(f"Flat days: {np.sum(rsi_signal == 0)}")
    print(f"Number of trades: {n_trades}")
    
    print("\n" + "-" * 70)
    print("Combining Multiple Strategies")
    print("-" * 70)
    
    # Combine strategies using voting
    combined = combine_signals(
        [sma_signal, mom_signal, mr_signal, rsi_signal],
        method='vote'
    )
    n_trades = count_trades(combined)
    
    print(f"Combined (vote) - Long: {np.sum(combined == 1)}, "
          f"Short: {np.sum(combined == -1)}, Flat: {np.sum(combined == 0)}")
    print(f"Number of trades: {n_trades}")
    
    # Demonstrate look-ahead bias prevention
    print("\n" + "-" * 70)
    print("Look-Ahead Bias Prevention Demo")
    print("-" * 70)
    
    sample_prices = np.array([100, 102, 99, 105, 103])
    sample_sma_fast = simple_moving_average(sample_prices, 2)
    sample_sma_slow = simple_moving_average(sample_prices, 3)
    
    print(f"Prices:    {sample_prices}")
    print(f"SMA(2):    {sample_sma_fast}")
    print(f"SMA(3):    {sample_sma_slow}")
    
    # Signal without shift (WRONG - look-ahead bias)
    raw_signal = np.where(
        ~np.isnan(sample_sma_fast) & ~np.isnan(sample_sma_slow),
        np.sign(sample_sma_fast - sample_sma_slow),
        0
    )
    print(f"\nRaw signal (has look-ahead bias): {raw_signal}")
    
    # Signal with shift (CORRECT)
    shifted = shift_signal(raw_signal, 1)
    print(f"Shifted signal (correct):          {shifted}")
    print("\n→ Shifted signal uses yesterday's info for today's trade!")
    
    print("\n" + "=" * 70)
    print("Key NumPy concepts demonstrated:")
    print("  - np.where() for vectorized conditionals")
    print("  - Array slicing for signal shifting")
    print("  - np.stack() and np.sum(axis=) for combining signals")
    print("  - np.sign() for converting to {-1, 0, +1}")
    print("  - Boolean masks for filtering")
    print("=" * 70)
