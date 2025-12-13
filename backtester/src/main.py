"""
=============================================================================
STOCK MARKET BACKTESTING ENGINE - MAIN ENTRY POINT
=============================================================================

This is the main demonstration file that shows how all components work together.

PROJECT STRUCTURE:
------------------
    backtester/
    ├── src/
    │   ├── data.py        ← Data loading and generation
    │   ├── indicators.py  ← Returns and technical indicators
    │   ├── strategy.py    ← Signal generation
    │   ├── engine.py      ← Backtesting simulation
    │   ├── metrics.py     ← Performance statistics
    │   └── main.py        ← This file (demo runner)
    └── tests/
        ├── test_returns.py
        └── test_strategy.py

WHAT YOU'LL LEARN:
------------------
    1. Complete NumPy-based backtesting workflow
    2. How vectorized operations replace loops
    3. Real quantitative finance concepts
    4. Performance optimization techniques

HOW TO RUN:
-----------
    cd backtester/src
    python main.py

=============================================================================
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
from data import generate_sample_data, generate_ohlcv_data, validate_price_data
from indicators import (
    simple_returns, log_returns, cumulative_returns,
    simple_moving_average, exponential_moving_average,
    rolling_volatility, bollinger_bands, relative_strength_index, macd
)
from strategy import (
    generate_sma_crossover_signal, generate_momentum_signal,
    generate_mean_reversion_signal, generate_rsi_signal,
    combine_signals, count_trades
)
from engine import run_backtest, compare_to_benchmark, calculate_drawdown_series
from metrics import calculate_all_metrics, format_metrics_report


def print_header(title: str, width: int = 70):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_section(title: str, width: int = 70):
    """Print a section divider."""
    print("\n" + "-" * width)
    print(f" {title}")
    print("-" * width)


def demo_data_module():
    """Demonstrate the data module (Phase 2)."""
    print_header("PHASE 2: Market Data Handling")
    
    print("\nGenerating synthetic price data using Geometric Brownian Motion...")
    print("This simulates realistic stock price behavior.")
    
    # Generate 2 years of daily prices
    prices = generate_sample_data(
        n_days=504,           # 2 years × 252 trading days
        initial_price=100.0,
        volatility=0.02,      # 2% daily volatility (~32% annual)
        drift=0.0003,         # ~7.5% annual drift
        seed=42
    )
    
    print(f"\n📊 Generated {len(prices)} days of price data")
    print(f"   Starting price: ${prices[0]:.2f}")
    print(f"   Ending price:   ${prices[-1]:.2f}")
    print(f"   Min price:      ${np.min(prices):.2f}")
    print(f"   Max price:      ${np.max(prices):.2f}")
    
    # Validate the data
    is_valid, message = validate_price_data(prices)
    print(f"\n✓ Validation: {message}")
    
    # Show NumPy array properties
    print_section("NumPy Array Properties")
    print(f"   Shape: {prices.shape}")
    print(f"   Dtype: {prices.dtype}")
    print(f"   Memory: {prices.nbytes} bytes")
    print(f"   First 5 prices: {prices[:5]}")
    print(f"   Last 5 prices:  {prices[-5:]}")
    
    return prices


def demo_indicators_module(prices: np.ndarray):
    """Demonstrate the indicators module (Phase 3 & 4)."""
    print_header("PHASE 3 & 4: Returns & Technical Indicators")
    
    # ==========================================================================
    # Returns (Phase 3)
    # ==========================================================================
    print_section("Returns Calculation")
    
    simple_rets = simple_returns(prices)
    log_rets = log_returns(prices)
    cum_rets = cumulative_returns(simple_rets)
    
    print("\nSimple Returns:")
    print(f"   Mean daily:  {np.mean(simple_rets):.4%}")
    print(f"   Std daily:   {np.std(simple_rets):.4%}")
    print(f"   Annualized:  {np.mean(simple_rets) * 252:.2%}")
    
    print("\nLog Returns:")
    print(f"   Sum (total): {np.sum(log_rets):.4f}")
    print(f"   Verify:      {np.log(prices[-1]/prices[0]):.4f}")
    print("   → Log returns are time-additive!")
    
    print("\nCumulative Returns:")
    print(f"   Final:       {cum_rets[-1]:.2%}")
    
    # ==========================================================================
    # Technical Indicators (Phase 4)
    # ==========================================================================
    print_section("Technical Indicators (All Vectorized)")
    
    # Moving Averages
    sma_20 = simple_moving_average(prices, 20)
    ema_20 = exponential_moving_average(prices, 20)
    
    print("\nMoving Averages (20-day):")
    print(f"   Current SMA: ${sma_20[-1]:.2f}")
    print(f"   Current EMA: ${ema_20[-1]:.2f}")
    print(f"   Current Price: ${prices[-1]:.2f}")
    
    # Volatility
    vol = rolling_volatility(prices, 20)
    print(f"\nRolling Volatility (20-day, annualized): {vol[-1]:.2%}")
    
    # Bollinger Bands
    upper, middle, lower = bollinger_bands(prices, 20, 2.0)
    print(f"\nBollinger Bands:")
    print(f"   Upper:  ${upper[-1]:.2f}")
    print(f"   Middle: ${middle[-1]:.2f}")
    print(f"   Lower:  ${lower[-1]:.2f}")
    
    position = "Above middle" if prices[-1] > middle[-1] else "Below middle"
    print(f"   Price position: {position}")
    
    # RSI
    rsi = relative_strength_index(prices, 14)
    rsi_status = "Overbought" if rsi[-1] > 70 else "Oversold" if rsi[-1] < 30 else "Neutral"
    print(f"\nRSI (14-day): {rsi[-1]:.1f} ({rsi_status})")
    
    # MACD
    macd_line, signal_line, histogram = macd(prices)
    macd_signal = "Bullish" if histogram[-1] > 0 else "Bearish"
    print(f"\nMACD: {macd_line[-1]:.4f} (Signal: {macd_signal})")
    
    return simple_rets


def demo_strategy_module(prices: np.ndarray):
    """Demonstrate the strategy module (Phase 5)."""
    print_header("PHASE 5: Strategy Signal Generation")
    
    strategies = {}
    
    # SMA Crossover
    print_section("Strategy 1: SMA Crossover (10/30)")
    sma_signal = generate_sma_crossover_signal(prices, fast_window=10, slow_window=30)
    strategies['SMA Crossover'] = sma_signal
    
    long_days = np.sum(sma_signal == 1)
    short_days = np.sum(sma_signal == -1)
    flat_days = np.sum(sma_signal == 0)
    
    print(f"   Long days:  {long_days} ({long_days/len(prices)*100:.1f}%)")
    print(f"   Short days: {short_days} ({short_days/len(prices)*100:.1f}%)")
    print(f"   Flat days:  {flat_days} ({flat_days/len(prices)*100:.1f}%)")
    print(f"   Trades:     {count_trades(sma_signal)}")
    
    # Momentum
    print_section("Strategy 2: Momentum (20-day)")
    mom_signal = generate_momentum_signal(prices, lookback=20, threshold=0.03)
    strategies['Momentum'] = mom_signal
    
    print(f"   Long days:  {np.sum(mom_signal == 1)}")
    print(f"   Short days: {np.sum(mom_signal == -1)}")
    print(f"   Trades:     {count_trades(mom_signal)}")
    
    # Mean Reversion
    print_section("Strategy 3: Mean Reversion (Bollinger)")
    mr_signal = generate_mean_reversion_signal(prices, window=20, entry_std=2.0)
    strategies['Mean Reversion'] = mr_signal
    
    print(f"   Long days:  {np.sum(mr_signal == 1)}")
    print(f"   Short days: {np.sum(mr_signal == -1)}")
    print(f"   Trades:     {count_trades(mr_signal)}")
    
    # RSI
    print_section("Strategy 4: RSI Reversal")
    rsi_signal = generate_rsi_signal(prices, window=14, oversold=30, overbought=70)
    strategies['RSI'] = rsi_signal
    
    print(f"   Long days:  {np.sum(rsi_signal == 1)}")
    print(f"   Short days: {np.sum(rsi_signal == -1)}")
    print(f"   Trades:     {count_trades(rsi_signal)}")
    
    # Combined Strategy
    print_section("Combined Strategy (Voting)")
    combined = combine_signals(
        [sma_signal, mom_signal, mr_signal, rsi_signal],
        method='vote'
    )
    strategies['Combined'] = combined
    
    print(f"   Long days:  {np.sum(combined == 1)}")
    print(f"   Short days: {np.sum(combined == -1)}")
    print(f"   Trades:     {count_trades(combined)}")
    
    return strategies


def demo_backtest_engine(prices: np.ndarray, strategies: dict):
    """Demonstrate the backtesting engine (Phase 6)."""
    print_header("PHASE 6: Backtesting Engine")
    
    results = {}
    
    for name, signals in strategies.items():
        result = run_backtest(
            prices=prices,
            signals=signals,
            initial_capital=10000.0,
            transaction_cost=0.001  # 10 bps per trade
        )
        results[name] = result
    
    # Print comparison table
    print_section("Strategy Comparison")
    print(f"\n{'Strategy':<20} {'Return':>10} {'Trades':>8} {'Costs':>10}")
    print("-" * 50)
    
    for name, result in results.items():
        print(f"{name:<20} {result.total_return:>9.2%} {result.n_trades:>8} ${result.transaction_costs_total*10000:>8.2f}")
    
    # Benchmark comparison
    print_section("Best Strategy vs Buy-and-Hold")
    
    best_name = max(results, key=lambda x: results[x].total_return)
    best_result = results[best_name]
    
    comparison = compare_to_benchmark(best_result.equity_curve, prices, 10000.0)
    
    print(f"\n   Best Strategy: {best_name}")
    print(f"   Strategy Return:  {comparison['strategy_return']:.2%}")
    print(f"   Benchmark Return: {comparison['benchmark_return']:.2%}")
    print(f"   Outperformance:   {comparison['outperformance']:.2%}")
    print(f"   Information Ratio: {comparison['information_ratio']:.2f}")
    
    return results, best_result


def demo_metrics(result):
    """Demonstrate the metrics module (Phase 7)."""
    print_header("PHASE 7: Performance Metrics")
    
    # Calculate all metrics
    metrics = calculate_all_metrics(
        equity_curve=result.equity_curve,
        returns=result.strategy_returns,
        positions=result.positions,
        risk_free_rate=0.02  # 2% annual risk-free rate
    )
    
    # Print formatted report
    print(format_metrics_report(metrics))
    
    # Drawdown analysis
    print_section("Drawdown Analysis")
    
    drawdown, peak = calculate_drawdown_series(result.equity_curve)
    max_dd = np.max(drawdown)
    max_dd_idx = np.argmax(drawdown)
    
    print(f"\n   Maximum Drawdown: {max_dd:.2%}")
    print(f"   Occurred at day: {max_dd_idx}")
    print(f"   Peak equity: ${peak[max_dd_idx]:,.2f}")
    print(f"   Trough equity: ${result.equity_curve[max_dd_idx]:,.2f}")
    print(f"   Recovery needed: {(peak[max_dd_idx]/result.equity_curve[max_dd_idx] - 1):.2%}")
    
    # Underwater chart data
    underwater_days = np.sum(drawdown > 0)
    print(f"\n   Days underwater: {underwater_days} ({underwater_days/len(drawdown)*100:.1f}%)")
    
    return metrics


def demo_monte_carlo(prices: np.ndarray, n_simulations: int = 100):
    """Demonstrate Monte Carlo simulation (Phase 10 Extension)."""
    print_header("PHASE 10: Monte Carlo Simulation")
    
    print(f"\nRunning {n_simulations} random strategy simulations...")
    
    # Generate random signals
    np.random.seed(42)
    n_days = len(prices)
    
    final_returns = []
    
    for i in range(n_simulations):
        # Random signal: uniform distribution over {-1, 0, 1}
        random_signal = np.random.choice([-1, 0, 1], size=n_days, p=[0.3, 0.4, 0.3])
        
        # Run backtest
        result = run_backtest(prices, random_signal, 10000.0, 0.001)
        final_returns.append(result.total_return)
    
    final_returns = np.array(final_returns)
    
    print_section("Monte Carlo Results")
    print(f"\n   Mean return: {np.mean(final_returns):.2%}")
    print(f"   Std return:  {np.std(final_returns):.2%}")
    print(f"   Min return:  {np.min(final_returns):.2%}")
    print(f"   Max return:  {np.max(final_returns):.2%}")
    print(f"   Median:      {np.median(final_returns):.2%}")
    
    # Percentiles
    print(f"\n   5th percentile:  {np.percentile(final_returns, 5):.2%}")
    print(f"   25th percentile: {np.percentile(final_returns, 25):.2%}")
    print(f"   75th percentile: {np.percentile(final_returns, 75):.2%}")
    print(f"   95th percentile: {np.percentile(final_returns, 95):.2%}")
    
    # Probability of profit
    prob_profit = np.sum(final_returns > 0) / n_simulations
    print(f"\n   Probability of profit: {prob_profit:.1%}")
    
    return final_returns


def summarize_numpy_concepts():
    """Print summary of NumPy concepts learned."""
    print_header("NUMPY CONCEPTS SUMMARY")
    
    concepts = """
┌─────────────────────────────────────────────────────────────────────┐
│                     NUMPY CONCEPTS YOU'VE LEARNED                   │
├─────────────────────────────────────────────────────────────────────┤
│ ARRAY BASICS                                                        │
│   • np.array(), np.zeros(), np.ones(), np.empty()                  │
│   • Array attributes: shape, dtype, ndim, nbytes                    │
│   • Indexing and slicing: arr[0], arr[-1], arr[1:5], arr[::2]      │
│                                                                     │
│ MATHEMATICAL OPERATIONS (Vectorized!)                               │
│   • Element-wise: +, -, *, /, **, np.sqrt(), np.exp(), np.log()    │
│   • Aggregations: np.sum(), np.mean(), np.std(), np.min(), np.max()│
│   • Cumulative: np.cumsum(), np.cumprod()                          │
│   • Differences: np.diff()                                          │
│                                                                     │
│ CONDITIONAL OPERATIONS                                              │
│   • np.where(condition, true_val, false_val)                       │
│   • Boolean masking: arr[arr > 0]                                  │
│   • np.sign(), np.abs()                                            │
│                                                                     │
│ ROLLING WINDOWS                                                     │
│   • np.convolve() for moving averages                              │
│   • sliding_window_view for advanced rolling operations            │
│   • Cumsum trick for efficient rolling calculations                │
│                                                                     │
│ SPECIAL FUNCTIONS                                                   │
│   • np.maximum.accumulate() - running maximum (drawdowns!)         │
│   • np.roll() - circular shift                                     │
│   • np.percentile() - quantile calculations                        │
│   • np.concatenate(), np.stack() - combining arrays                │
│                                                                     │
│ RANDOM NUMBERS                                                      │
│   • np.random.default_rng() - modern RNG                           │
│   • rng.normal(), rng.uniform(), rng.choice()                      │
│                                                                     │
│ MULTI-DIMENSIONAL                                                   │
│   • 2D arrays for multi-asset portfolios                           │
│   • axis parameter: axis=0 (rows), axis=1 (columns)                │
│   • Broadcasting for weighted calculations                          │
└─────────────────────────────────────────────────────────────────────┘

KEY INSIGHT: 
In finance, almost EVERYTHING can be vectorized.
If you're writing a loop, there's probably a better way!
"""
    print(concepts)


def main():
    """Main entry point - runs complete demonstration."""
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "    NUMPY-BASED STOCK MARKET BACKTESTING ENGINE    ".center(68) + "█")
    print("█" + "    A Complete Learning Project                     ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Phase 2: Data
    prices = demo_data_module()
    
    # Phase 3 & 4: Returns and Indicators  
    returns = demo_indicators_module(prices)
    
    # Phase 5: Strategy
    strategies = demo_strategy_module(prices)
    
    # Phase 6: Backtesting Engine
    results, best_result = demo_backtest_engine(prices, strategies)
    
    # Phase 7: Metrics
    metrics = demo_metrics(best_result)
    
    # Phase 10: Monte Carlo (Extension)
    mc_results = demo_monte_carlo(prices, n_simulations=100)
    
    # Summary
    summarize_numpy_concepts()
    
    print_header("WHAT'S NEXT?")
    
    next_steps = """
Now that you've built the engine, try these exercises:

1. EXTEND THE STRATEGIES
   • Implement a trend-following strategy with ADX
   • Create a pairs trading strategy (requires 2 assets)
   • Build a volatility-based position sizing strategy

2. OPTIMIZE PERFORMANCE (Phase 9)
   • Profile the code with cProfile
   • Replace loops with vectorized alternatives
   • Test with 10,000+ days and 100+ assets
   • Experiment with float32 vs float64

3. ADD REAL DATA
   • Load CSV data from Yahoo Finance
   • Handle missing data and weekends
   • Implement data validation

4. ADVANCED FEATURES
   • Add portfolio rebalancing logic
   • Implement stop-loss and take-profit
   • Create a walk-forward optimization framework

5. VISUALIZATION (not covered)
   • Plot equity curves with matplotlib
   • Create drawdown charts
   • Build an interactive dashboard

RESUME LINE (Honest & Strong):
"Built a NumPy-based stock market backtesting engine implementing 
vectorized indicators, signal generation, portfolio simulation, 
and risk metrics without pandas or ML frameworks."

This directly transfers to:
   • ML preprocessing pipelines
   • Signal processing
   • Large-scale data analytics
   • Quantitative research

Happy coding! 🚀
"""
    print(next_steps)


if __name__ == "__main__":
    main()
