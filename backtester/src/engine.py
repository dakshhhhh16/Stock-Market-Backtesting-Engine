"""
=============================================================================
PHASE 6: Backtesting Engine (Heart of the Project)
=============================================================================

PURPOSE:
    Simulate capital over time by applying strategy signals to price data.
    This is where all previous phases come together!

THE BACKTESTING PIPELINE:
-------------------------
    1. Prices → Returns (from indicators.py)
    2. Prices → Signals (from strategy.py)
    3. Signals → Positions (convert signals to what we hold)
    4. Positions × Returns → Strategy Returns (the magic!)
    5. Apply transaction costs
    6. Compute equity curve

ALL VECTORIZED - No loops in the core logic!

KEY NUMPY CONCEPTS:
    - Element-wise multiplication (positions * returns)
    - Cumulative products for equity curve
    - Array shifting for proper alignment
    - Broadcasting for transaction costs

CRITICAL INSIGHT:
-----------------
    strategy_returns[t] = position[t-1] * asset_returns[t]
    
    Why t-1? Because:
    - position[t-1] is what you HELD at end of day t-1
    - asset_returns[t] is the return from close t-1 to close t
    - You earn the return on what you held BEFORE the return happened

=============================================================================
"""

import numpy as np
from typing import Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from indicators import simple_returns, log_returns, cumulative_returns
from strategy import signal_to_position, calculate_trades, shift_signal


@dataclass
class BacktestResult:
    """
    Container for backtest results.
    
    PYTHON CONCEPT: @dataclass automatically generates __init__, __repr__, etc.
    Clean way to bundle related data together.
    """
    
    # Core arrays
    prices: np.ndarray           # Original prices
    signals: np.ndarray          # Strategy signals
    positions: np.ndarray        # Actual positions held
    asset_returns: np.ndarray    # Buy-and-hold returns
    strategy_returns: np.ndarray # Strategy returns (after positions)
    equity_curve: np.ndarray     # Portfolio value over time
    
    # Scalars
    initial_capital: float
    final_capital: float
    total_return: float
    n_trades: int
    transaction_costs_total: float


def calculate_strategy_returns(positions: np.ndarray,
                                asset_returns: np.ndarray) -> np.ndarray:
    """
    Calculate strategy returns from positions and asset returns.
    
    FORMULA:
        strategy_return[t] = position[t-1] * asset_return[t]
    
    WHY position[t-1]?
    ------------------
    Think about it:
    - At end of day t-1, you decide your position for tomorrow
    - Day t: market opens, moves, closes
    - Your return on day t depends on what you HELD overnight
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. Element-wise multiplication
       - positions * returns multiplies each element
       - Arrays must be same length!
    
    2. Array alignment is CRITICAL
       - positions[:-1] aligns with returns[1:]
       - Or use np.roll() for circular shift
    
    3. This is the core of backtesting!
       - Simple multiplication IS the simulation
       - No need for loops or complex logic
    
    Parameters:
    -----------
    positions : np.ndarray
        Position array with values in {-1, 0, +1}
    asset_returns : np.ndarray
        Returns of the underlying asset
        
    Returns:
    --------
    np.ndarray: Strategy returns
    
    Example:
    --------
    >>> positions = np.array([0, 1, 1, -1, -1])  # flat, long, long, short, short
    >>> returns = np.array([0, 0.01, -0.02, 0.015, -0.01])
    >>> strat_ret = calculate_strategy_returns(positions, returns)
    
    Day 0: pos[t-1] doesn't exist → return = 0
    Day 1: pos[0]=0, we were flat → return = 0 * 0.01 = 0
    Day 2: pos[1]=1, we were long → return = 1 * (-0.02) = -0.02
    Day 3: pos[2]=1, we were long → return = 1 * 0.015 = 0.015
    Day 4: pos[3]=-1, we were short → return = -1 * (-0.01) = 0.01
    """
    
    # Shift positions forward by 1
    # position at t-1 affects return at t
    # NUMPY CONCEPT: np.roll for circular shift, then fix first element
    shifted_positions = np.roll(positions, 1)
    shifted_positions[0] = 0  # No position before day 0
    
    # Element-wise multiplication
    # NUMPY CONCEPT: This is the core operation!
    # Vector * Vector = Vector of element-wise products
    strategy_returns = shifted_positions * asset_returns
    
    return strategy_returns


def apply_transaction_costs(strategy_returns: np.ndarray,
                            positions: np.ndarray,
                            cost_per_trade: float = 0.001) -> Tuple[np.ndarray, float]:
    """
    Apply transaction costs to strategy returns.
    
    CONCEPT:
    --------
    Every time we trade, we pay a cost (commission, slippage, spread).
    This cost reduces our returns.
    
    cost_per_trade = 0.001 means 0.1% cost per trade (10 bps)
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.diff() for detecting trades
       - Trade occurs when position changes
    
    2. np.abs() for absolute value of trade size
       - Going from +1 to -1 is a trade of size 2
    
    3. Subtracting costs from returns
       - Simple element-wise subtraction
    
    Parameters:
    -----------
    strategy_returns : np.ndarray
        Raw strategy returns before costs
    positions : np.ndarray
        Position array
    cost_per_trade : float
        Cost per unit traded (e.g., 0.001 = 0.1%)
        
    Returns:
    --------
    Tuple of (adjusted_returns, total_costs)
    """
    
    # Calculate trade sizes (absolute value of position changes)
    # NUMPY CONCEPT: np.diff gives position[t] - position[t-1]
    position_changes = np.diff(positions)
    
    # Take absolute value - cost is always positive regardless of direction
    # NUMPY CONCEPT: np.abs() is vectorized
    trade_sizes = np.abs(position_changes)
    
    # Prepend first trade (going from 0 to initial position)
    trade_sizes = np.concatenate([[np.abs(positions[0])], trade_sizes])
    
    # Calculate costs at each time point
    # NUMPY CONCEPT: Scalar multiplication broadcasts to all elements
    costs = trade_sizes * cost_per_trade
    
    # Subtract costs from returns
    adjusted_returns = strategy_returns - costs
    
    # Total costs
    total_costs = np.sum(costs)
    
    return adjusted_returns, total_costs


def calculate_equity_curve(returns: np.ndarray,
                           initial_capital: float = 10000.0) -> np.ndarray:
    """
    Calculate equity curve (portfolio value over time).
    
    FORMULA:
        equity[t] = initial_capital * cumprod(1 + returns)
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.cumprod() - Cumulative product
       - (1+r1) * (1+r2) * ... * (1+rn)
       - This IS the equity curve (normalized)
    
    2. Scaling by initial capital
       - Simple multiplication broadcasts
    
    Parameters:
    -----------
    returns : np.ndarray
        Return series
    initial_capital : float
        Starting portfolio value
        
    Returns:
    --------
    np.ndarray: Portfolio value at each time point
    """
    
    # Calculate cumulative wealth multiplier
    # NUMPY CONCEPT: cumprod of (1 + returns)
    # [1+r0, (1+r0)(1+r1), (1+r0)(1+r1)(1+r2), ...]
    wealth_multiplier = np.cumprod(1 + returns)
    
    # Scale by initial capital
    equity = initial_capital * wealth_multiplier
    
    return equity


def run_backtest(prices: np.ndarray,
                 signals: np.ndarray,
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001,
                 use_log_returns: bool = False) -> BacktestResult:
    """
    Run a complete backtest simulation.
    
    THIS IS THE MAIN FUNCTION - THE HEART OF THE ENGINE!
    
    PIPELINE:
    ---------
    1. Calculate asset returns from prices
    2. Convert signals to positions
    3. Calculate strategy returns (positions × asset returns)
    4. Apply transaction costs
    5. Build equity curve
    6. Package results
    
    NUMPY LEARNING POINTS:
    -----------------------
    This function ties everything together!
    - Uses all previously built functions
    - Everything is vectorized
    - No loops in the entire pipeline
    
    Parameters:
    -----------
    prices : np.ndarray
        Price array
    signals : np.ndarray
        Signal array with values in {-1, 0, +1}
    initial_capital : float
        Starting portfolio value
    transaction_cost : float
        Cost per unit traded
    use_log_returns : bool
        Whether to use log returns (more stable for long periods)
        
    Returns:
    --------
    BacktestResult: Complete backtest results
    """
    
    # Validate inputs
    if len(prices) != len(signals):
        raise ValueError(f"Price length ({len(prices)}) != Signal length ({len(signals)})")
    
    # Step 1: Calculate asset returns
    if use_log_returns:
        asset_returns = log_returns(prices)
    else:
        asset_returns = simple_returns(prices)
    
    # Step 2: Convert signals to positions
    # In basic case, position follows signal exactly
    positions = signal_to_position(signals)
    
    # Step 3: Calculate strategy returns
    # THIS IS THE CORE OPERATION
    strategy_returns = calculate_strategy_returns(positions, asset_returns)
    
    # Step 4: Apply transaction costs
    adjusted_returns, total_costs = apply_transaction_costs(
        strategy_returns, positions, transaction_cost
    )
    
    # Step 5: Calculate equity curve
    equity = calculate_equity_curve(adjusted_returns, initial_capital)
    
    # Step 6: Package results
    final_capital = equity[-1]
    total_return = final_capital / initial_capital - 1
    n_trades = np.count_nonzero(np.diff(positions))
    
    return BacktestResult(
        prices=prices,
        signals=signals,
        positions=positions,
        asset_returns=asset_returns,
        strategy_returns=adjusted_returns,
        equity_curve=equity,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return=total_return,
        n_trades=n_trades,
        transaction_costs_total=total_costs
    )


def run_backtest_vectorized_multi(prices: np.ndarray,
                                   signal_generator: Callable,
                                   param_grid: Dict[str, list],
                                   initial_capital: float = 10000.0,
                                   transaction_cost: float = 0.001) -> Dict:
    """
    Run multiple backtests with different parameters (parameter sweep).
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. This demonstrates how to efficiently test many parameter combinations
    2. Store results in structured format for analysis
    
    ADVANCED CONCEPT:
    -----------------
    In production, you'd vectorize across parameters too,
    but here we iterate for clarity.
    
    Parameters:
    -----------
    prices : np.ndarray
        Price array
    signal_generator : Callable
        Function that takes (prices, **params) and returns signals
    param_grid : Dict
        Dictionary of parameter names to lists of values to test
    initial_capital : float
        Starting capital
    transaction_cost : float
        Transaction cost per trade
        
    Returns:
    --------
    Dict with results for each parameter combination
    """
    
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    results = []
    
    for combo in combinations:
        # Create parameter dict for this combination
        params = dict(zip(param_names, combo))
        
        # Generate signals
        signals = signal_generator(prices, **params)
        
        # Run backtest
        result = run_backtest(
            prices, signals, initial_capital, transaction_cost
        )
        
        # Store results with parameters
        results.append({
            'params': params,
            'total_return': result.total_return,
            'n_trades': result.n_trades,
            'final_capital': result.final_capital,
            'costs': result.transaction_costs_total
        })
    
    # Convert to structured format for easy analysis
    # NUMPY CONCEPT: Create structured array from results
    returns_array = np.array([r['total_return'] for r in results])
    
    return {
        'results': results,
        'returns': returns_array,
        'best_params': results[np.argmax(returns_array)]['params'],
        'worst_params': results[np.argmin(returns_array)]['params'],
        'mean_return': np.mean(returns_array),
        'std_return': np.std(returns_array)
    }


def compare_to_benchmark(strategy_equity: np.ndarray,
                         prices: np.ndarray,
                         initial_capital: float = 10000.0) -> Dict:
    """
    Compare strategy performance to buy-and-hold benchmark.
    
    CONCEPT:
    --------
    Buy-and-hold is the simplest strategy:
    - Buy at day 0
    - Hold forever
    - No trades, no costs
    
    Every strategy should beat this benchmark (adjusted for risk)!
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. Normalizing prices to equity curve
    2. Calculating outperformance
    
    Parameters:
    -----------
    strategy_equity : np.ndarray
        Strategy equity curve
    prices : np.ndarray
        Asset prices (benchmark)
    initial_capital : float
        Starting capital
        
    Returns:
    --------
    Dict with comparison metrics
    """
    
    # Benchmark equity curve
    # NUMPY CONCEPT: Normalize prices and scale by capital
    benchmark_equity = initial_capital * prices / prices[0]
    
    # Calculate returns
    strategy_return = strategy_equity[-1] / initial_capital - 1
    benchmark_return = benchmark_equity[-1] / initial_capital - 1
    
    # Outperformance
    outperformance = strategy_return - benchmark_return
    
    # Information ratio (simplified): excess return / tracking error
    strategy_daily_returns = np.diff(strategy_equity) / strategy_equity[:-1]
    benchmark_daily_returns = np.diff(benchmark_equity) / benchmark_equity[:-1]
    excess_returns = strategy_daily_returns - benchmark_daily_returns
    
    # NUMPY CONCEPT: Mean and std of excess returns
    tracking_error = np.std(excess_returns) * np.sqrt(252)  # Annualized
    info_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0
    
    return {
        'strategy_return': strategy_return,
        'benchmark_return': benchmark_return,
        'outperformance': outperformance,
        'information_ratio': info_ratio,
        'strategy_equity': strategy_equity,
        'benchmark_equity': benchmark_equity
    }


def calculate_drawdown_series(equity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate drawdown at each time point.
    
    FORMULA:
        drawdown[t] = (peak[t] - equity[t]) / peak[t]
        where peak[t] = max(equity[0:t+1])
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.maximum.accumulate() - Running maximum
       - Creates array of peak values at each point
       - accumulate([1,3,2,5,4]) = [1,3,3,5,5]
    
    2. Element-wise division for percentage
    
    This is key for risk management!
    
    Parameters:
    -----------
    equity : np.ndarray
        Equity curve
        
    Returns:
    --------
    Tuple of (drawdown_series, peak_series)
    """
    
    # Calculate running maximum (peak equity)
    # NUMPY CONCEPT: np.maximum.accumulate is vectorized running max
    # Much faster than a loop!
    peak = np.maximum.accumulate(equity)
    
    # Calculate drawdown at each point
    # drawdown = (peak - current) / peak
    drawdown = (peak - equity) / peak
    
    return drawdown, peak


# =============================================================================
# MULTI-ASSET EXTENSION (PHASE 9-10)
# =============================================================================

def run_backtest_multi_asset(prices_matrix: np.ndarray,
                              signals_matrix: np.ndarray,
                              weights: Optional[np.ndarray] = None,
                              initial_capital: float = 10000.0,
                              transaction_cost: float = 0.001) -> Dict:
    """
    Run backtest on multiple assets simultaneously.
    
    ADVANCED NUMPY CONCEPTS:
    ------------------------
    1. 2D arrays: prices_matrix shape is (n_assets, n_time)
    2. Matrix operations for portfolio calculations
    3. Broadcasting weights across time
    
    MENTAL MODEL:
    -------------
    - Each ROW is an asset
    - Each COLUMN is a time point
    - Portfolio return = weighted sum of asset returns
    
    Parameters:
    -----------
    prices_matrix : np.ndarray
        2D array of shape (n_assets, n_time)
    signals_matrix : np.ndarray
        2D array of signals, shape (n_assets, n_time)
    weights : np.ndarray, optional
        Asset weights, shape (n_assets,). If None, equal weights.
    initial_capital : float
        Starting capital
    transaction_cost : float
        Cost per trade
        
    Returns:
    --------
    Dict with portfolio-level results
    """
    
    n_assets, n_time = prices_matrix.shape
    
    # Default to equal weights
    if weights is None:
        weights = np.ones(n_assets) / n_assets
    
    # Calculate returns for each asset
    # NUMPY CONCEPT: np.diff with axis parameter
    # axis=1 means "compute across columns" (time)
    price_changes = np.diff(prices_matrix, axis=1)
    
    # Returns: (P[t] - P[t-1]) / P[t-1]
    asset_returns = price_changes / prices_matrix[:, :-1]
    
    # Prepend zeros for first day
    # NUMPY CONCEPT: np.hstack for horizontal stacking
    zeros_col = np.zeros((n_assets, 1))
    asset_returns = np.hstack([zeros_col, asset_returns])
    
    # Convert signals to positions
    positions = signals_matrix.copy()
    
    # Shift positions for proper timing
    # NUMPY CONCEPT: np.roll with axis
    shifted_positions = np.roll(positions, 1, axis=1)
    shifted_positions[:, 0] = 0  # No position on first day
    
    # Calculate strategy returns for each asset
    # NUMPY CONCEPT: Element-wise multiplication of 2D arrays
    strategy_returns = shifted_positions * asset_returns
    
    # Apply weights to get portfolio returns
    # NUMPY CONCEPT: Matrix multiplication / weighted sum
    # weights[:, np.newaxis] makes weights column vector for broadcasting
    # Then sum across assets (axis=0)
    portfolio_returns = np.sum(weights[:, np.newaxis] * strategy_returns, axis=0)
    
    # Calculate equity curve
    equity = calculate_equity_curve(portfolio_returns, initial_capital)
    
    return {
        'equity': equity,
        'portfolio_returns': portfolio_returns,
        'asset_returns': strategy_returns,
        'total_return': equity[-1] / initial_capital - 1,
        'weights': weights
    }


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 6: Backtesting Engine - Complete Simulation")
    print("=" * 70)
    
    # Import strategy for demo
    from strategy import generate_sma_crossover_signal, generate_momentum_signal
    
    # Generate sample price data
    np.random.seed(42)
    n_days = 504  # ~2 years
    
    # Create a price series with trend and noise
    trend = np.linspace(0, 0.4, n_days)  # 40% trend over period
    noise = np.cumsum(np.random.normal(0, 0.015, n_days))
    prices = 100 * np.exp(trend + noise)
    
    print(f"\nSimulation Setup:")
    print(f"  Days: {n_days}")
    print(f"  Starting price: ${prices[0]:.2f}")
    print(f"  Ending price: ${prices[-1]:.2f}")
    print(f"  Buy-and-hold return: {(prices[-1]/prices[0] - 1)*100:.1f}%")
    
    # ==========================================================================
    # Test SMA Crossover Strategy
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Running SMA Crossover Strategy (10/30)")
    print("-" * 70)
    
    # Generate signals
    sma_signals = generate_sma_crossover_signal(prices, fast_window=10, slow_window=30)
    
    # Run backtest
    result = run_backtest(
        prices=prices,
        signals=sma_signals,
        initial_capital=10000.0,
        transaction_cost=0.001  # 0.1% per trade
    )
    
    print(f"\nBacktest Results:")
    print(f"  Initial Capital: ${result.initial_capital:,.2f}")
    print(f"  Final Capital:   ${result.final_capital:,.2f}")
    print(f"  Total Return:    {result.total_return*100:.2f}%")
    print(f"  Number of Trades: {result.n_trades}")
    print(f"  Transaction Costs: ${result.transaction_costs_total*10000:.2f}")
    
    # Compare to benchmark
    comparison = compare_to_benchmark(result.equity_curve, prices, 10000.0)
    
    print(f"\nVs Buy-and-Hold:")
    print(f"  Strategy Return:  {comparison['strategy_return']*100:.2f}%")
    print(f"  Benchmark Return: {comparison['benchmark_return']*100:.2f}%")
    print(f"  Outperformance:   {comparison['outperformance']*100:.2f}%")
    print(f"  Information Ratio: {comparison['information_ratio']:.2f}")
    
    # ==========================================================================
    # Test Momentum Strategy
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Running Momentum Strategy (20-day)")
    print("-" * 70)
    
    mom_signals = generate_momentum_signal(prices, lookback=20, threshold=0.02)
    result_mom = run_backtest(prices, mom_signals, 10000.0, 0.001)
    
    print(f"  Total Return: {result_mom.total_return*100:.2f}%")
    print(f"  Number of Trades: {result_mom.n_trades}")
    
    # ==========================================================================
    # Demonstrate Drawdown Calculation
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Drawdown Analysis")
    print("-" * 70)
    
    drawdown, peak = calculate_drawdown_series(result.equity_curve)
    
    max_drawdown = np.max(drawdown)
    max_dd_idx = np.argmax(drawdown)
    
    print(f"  Maximum Drawdown: {max_drawdown*100:.2f}%")
    print(f"  Occurred at day: {max_dd_idx}")
    print(f"  Peak before drawdown: ${peak[max_dd_idx]:,.2f}")
    print(f"  Trough: ${result.equity_curve[max_dd_idx]:,.2f}")
    
    # ==========================================================================
    # Parameter Sweep Example
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Parameter Sweep (SMA Crossover)")
    print("-" * 70)
    
    def sma_generator(prices, fast, slow):
        return generate_sma_crossover_signal(prices, fast, slow)
    
    param_grid = {
        'fast': [5, 10, 15, 20],
        'slow': [20, 30, 40, 50]
    }
    
    sweep_results = run_backtest_vectorized_multi(
        prices, sma_generator, param_grid
    )
    
    print(f"  Tested {len(sweep_results['results'])} combinations")
    print(f"  Best params: {sweep_results['best_params']} → {sweep_results['results'][np.argmax(sweep_results['returns'])]['total_return']*100:.1f}%")
    print(f"  Worst params: {sweep_results['worst_params']} → {sweep_results['results'][np.argmin(sweep_results['returns'])]['total_return']*100:.1f}%")
    print(f"  Mean return: {sweep_results['mean_return']*100:.1f}%")
    print(f"  Std of returns: {sweep_results['std_return']*100:.1f}%")
    
    # ==========================================================================
    # Multi-Asset Example
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Multi-Asset Portfolio (3 assets)")
    print("-" * 70)
    
    # Generate 3 correlated assets
    n_assets = 3
    prices_multi = np.zeros((n_assets, n_days))
    
    for i in range(n_assets):
        np.random.seed(42 + i)
        t = np.linspace(0, 0.3 + i*0.1, n_days)
        n = np.cumsum(np.random.normal(0, 0.02, n_days))
        prices_multi[i] = 100 * np.exp(t + n)
    
    # Generate signals for each
    signals_multi = np.zeros((n_assets, n_days))
    for i in range(n_assets):
        signals_multi[i] = generate_sma_crossover_signal(prices_multi[i], 10, 30)
    
    # Run multi-asset backtest
    multi_result = run_backtest_multi_asset(
        prices_multi, signals_multi,
        weights=np.array([0.4, 0.3, 0.3]),  # 40%, 30%, 30%
        initial_capital=10000.0
    )
    
    print(f"  Portfolio Return: {multi_result['total_return']*100:.2f}%")
    print(f"  Weights: {multi_result['weights']}")
    
    print("\n" + "=" * 70)
    print("Key NumPy concepts demonstrated:")
    print("  - Element-wise multiplication (positions × returns)")
    print("  - np.cumprod() for equity curve")
    print("  - np.roll() for array shifting")
    print("  - np.maximum.accumulate() for drawdowns")
    print("  - 2D arrays for multi-asset backtesting")
    print("  - Broadcasting for weighted sums")
    print("=" * 70)
