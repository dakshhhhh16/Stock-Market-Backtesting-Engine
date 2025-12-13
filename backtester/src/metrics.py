"""
=============================================================================
PHASE 7: Risk & Performance Metrics
=============================================================================

PURPOSE:
    Calculate all performance and risk metrics from backtest results.
    This is real quantitative finance math - implemented from scratch!

METRICS IMPLEMENTED:
--------------------
    Performance:
        - Total Return
        - CAGR (Compound Annual Growth Rate)
        - Annualized Return
        
    Risk:
        - Volatility (Standard Deviation)
        - Maximum Drawdown
        - Value at Risk (VaR)
        - Conditional VaR (CVaR / Expected Shortfall)
        
    Risk-Adjusted:
        - Sharpe Ratio
        - Sortino Ratio
        - Calmar Ratio
        
    Trade Analysis:
        - Win Rate
        - Win/Loss Ratio
        - Profit Factor

NUMPY CONCEPTS YOU'LL LEARN:
    - np.maximum.accumulate() for running max (drawdowns)
    - np.percentile() for VaR
    - np.mean() and np.std() with conditional masking
    - Numerical stability techniques

=============================================================================
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# Constant: Trading days per year (used for annualization)
TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    """
    Container for all performance metrics.
    Organized for easy access and reporting.
    """
    # Returns
    total_return: float
    cagr: float
    annualized_return: float
    
    # Risk
    volatility: float          # Annualized
    max_drawdown: float
    var_95: float              # 95% Value at Risk
    cvar_95: float             # 95% Conditional VaR
    
    # Risk-Adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade Analysis
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float


def calculate_total_return(equity_curve: np.ndarray) -> float:
    """
    Calculate total return from equity curve.
    
    FORMULA:
        total_return = (final_value / initial_value) - 1
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. Array indexing
       - equity_curve[0] = first element
       - equity_curve[-1] = last element
    
    2. Simple arithmetic on scalars extracted from array
    
    Parameters:
    -----------
    equity_curve : np.ndarray
        Portfolio value over time
        
    Returns:
    --------
    float: Total return as decimal (0.50 = 50%)
    """
    
    initial_value = equity_curve[0]
    final_value = equity_curve[-1]
    
    total_return = final_value / initial_value - 1
    
    return total_return


def calculate_cagr(equity_curve: np.ndarray, 
                   trading_days: int = None) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    FORMULA:
        CAGR = (final / initial) ^ (252 / n_days) - 1
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. Power operations with np.power() or **
       - Both work, ** is more readable for scalars
    
    2. Handling edge cases (zero or negative values)
    
    FINANCIAL MEANING:
    ------------------
    CAGR answers: "What constant annual return would give the same result?"
    - Smooths out volatility
    - Comparable across different time periods
    
    Parameters:
    -----------
    equity_curve : np.ndarray
        Portfolio value over time
    trading_days : int, optional
        Number of trading days. If None, uses length of equity curve.
        
    Returns:
    --------
    float: CAGR as decimal
    """
    
    if trading_days is None:
        trading_days = len(equity_curve)
    
    initial_value = equity_curve[0]
    final_value = equity_curve[-1]
    
    # Calculate years
    years = trading_days / TRADING_DAYS_PER_YEAR
    
    if years == 0:
        return 0.0
    
    # CAGR formula
    # NUMPY CONCEPT: Power operation
    cagr = (final_value / initial_value) ** (1 / years) - 1
    
    return cagr


def calculate_annualized_return(returns: np.ndarray) -> float:
    """
    Calculate annualized return from daily returns.
    
    FORMULA:
        annualized = mean_daily_return * 252
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.mean() - Average of array
    2. Simple annualization by scaling
    
    NOTE: This assumes returns are roughly normally distributed
    and ignores compounding effects. CAGR is more accurate.
    
    Parameters:
    -----------
    returns : np.ndarray
        Daily returns
        
    Returns:
    --------
    float: Annualized return
    """
    
    mean_daily = np.mean(returns)
    annualized = mean_daily * TRADING_DAYS_PER_YEAR
    
    return annualized


def calculate_volatility(returns: np.ndarray, 
                         annualize: bool = True) -> float:
    """
    Calculate volatility (standard deviation of returns).
    
    FORMULA:
        volatility = std(returns) * sqrt(252)  # if annualized
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.std() - Standard deviation
       - ddof=1 for sample std (N-1 denominator)
       - ddof=0 for population std (N denominator)
    
    2. np.sqrt() - Square root
    
    3. Annualization: multiply by sqrt(252)
       - Variance scales linearly with time
       - Std scales with sqrt(time)
    
    WHY sqrt(252)?
    --------------
    If daily variance is σ², then annual variance is 252 * σ²
    Therefore annual std = sqrt(252) * daily std
    
    Parameters:
    -----------
    returns : np.ndarray
        Daily returns
    annualize : bool
        Whether to annualize (multiply by sqrt(252))
        
    Returns:
    --------
    float: Volatility
    """
    
    # Calculate standard deviation
    # NUMPY CONCEPT: np.std with ddof parameter
    # ddof=1 gives sample std (divide by N-1)
    daily_vol = np.std(returns, ddof=1)
    
    if annualize:
        # Annualize by multiplying by sqrt(252)
        # NUMPY CONCEPT: np.sqrt is vectorized, but works on scalars too
        return daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        return daily_vol


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate Maximum Drawdown (MDD) - the worst peak-to-trough decline.
    
    FORMULA:
        drawdown[t] = (peak[t] - equity[t]) / peak[t]
        max_drawdown = max(drawdown)
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.maximum.accumulate() - Running maximum
       - This is THE key function for drawdown calculation
       - Creates array where each element is max of all previous elements
       - accumulate([1,3,2,5,4]) → [1,3,3,5,5]
    
    2. np.argmax() - Index of maximum value
    
    3. np.argmax on sliced arrays for peak/trough indices
    
    FINANCIAL MEANING:
    ------------------
    MDD answers: "What's the worst I could have experienced?"
    - Critical for risk assessment
    - Investors care about losses more than gains
    - Used in Calmar ratio
    
    Parameters:
    -----------
    equity_curve : np.ndarray
        Portfolio value over time
        
    Returns:
    --------
    Tuple of (max_drawdown, peak_idx, trough_idx)
    """
    
    # Calculate running peak (high-water mark)
    # NUMPY CONCEPT: np.maximum.accumulate()
    # This is a vectorized running maximum - very efficient!
    running_peak = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown at each point
    # drawdown = (peak - current) / peak
    drawdown = (running_peak - equity_curve) / running_peak
    
    # Find maximum drawdown
    max_dd = np.max(drawdown)
    trough_idx = np.argmax(drawdown)
    
    # Find peak before the trough
    # NUMPY CONCEPT: Slice array and find argmax within slice
    peak_idx = np.argmax(equity_curve[:trough_idx + 1])
    
    return max_dd, peak_idx, trough_idx


def calculate_sharpe_ratio(returns: np.ndarray, 
                           risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe Ratio - risk-adjusted return measure.
    
    FORMULA:
        Sharpe = (mean_return - risk_free) / std(return) * sqrt(252)
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.mean() for average return
    2. np.std() for standard deviation
    3. Handling edge case of zero volatility
    
    FINANCIAL MEANING:
    ------------------
    Sharpe ratio answers: "How much return per unit of risk?"
    - Higher is better
    - > 1.0 is considered good
    - > 2.0 is very good
    - > 3.0 is excellent (rare in practice)
    
    CAVEAT:
    -------
    Assumes returns are normally distributed.
    Penalizes upside volatility same as downside (see Sortino).
    
    Parameters:
    -----------
    returns : np.ndarray
        Daily returns
    risk_free_rate : float
        Daily risk-free rate (default 0)
        
    Returns:
    --------
    float: Annualized Sharpe ratio
    """
    
    # Calculate excess returns
    excess_returns = returns - risk_free_rate
    
    # Mean and std
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns, ddof=1)
    
    # Handle zero volatility (avoid division by zero)
    # NUMPY CONCEPT: Conditional check before division
    if std_returns == 0:
        return 0.0 if mean_excess == 0 else np.inf * np.sign(mean_excess)
    
    # Calculate Sharpe (annualized)
    # Note: We annualize at the end, not separately for mean and std
    sharpe = (mean_excess / std_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return sharpe


def calculate_sortino_ratio(returns: np.ndarray, 
                            risk_free_rate: float = 0.0,
                            target_return: float = 0.0) -> float:
    """
    Calculate Sortino Ratio - downside-risk adjusted return.
    
    FORMULA:
        Sortino = (mean_return - target) / downside_std * sqrt(252)
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. Boolean masking for filtering
       - returns[returns < target] selects only negative excess returns
    
    2. np.where() for conditional calculations
    
    3. Downside deviation calculation
    
    WHY SORTINO > SHARPE:
    ---------------------
    Sharpe penalizes ALL volatility equally.
    But upside volatility is GOOD! We want big gains!
    
    Sortino only penalizes DOWNSIDE volatility.
    More realistic for investors who fear losses.
    
    Parameters:
    -----------
    returns : np.ndarray
        Daily returns
    risk_free_rate : float
        Daily risk-free rate
    target_return : float
        Target return (usually 0 or risk-free rate)
        
    Returns:
    --------
    float: Annualized Sortino ratio
    """
    
    # Calculate excess returns
    excess_returns = returns - risk_free_rate
    
    # Calculate downside returns (only negative deviations from target)
    # NUMPY CONCEPT: np.minimum clips values at the target
    downside_returns = np.minimum(returns - target_return, 0)
    
    # Downside deviation: std of negative returns only
    # NUMPY CONCEPT: We use all values but only count negative ones
    # This is the proper "semi-deviation"
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    
    # Mean excess return
    mean_excess = np.mean(excess_returns)
    
    # Handle zero downside deviation
    if downside_std == 0:
        return 0.0 if mean_excess == 0 else np.inf * np.sign(mean_excess)
    
    # Calculate Sortino (annualized)
    sortino = (mean_excess / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return sortino


def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """
    Calculate Calmar Ratio - return relative to max drawdown.
    
    FORMULA:
        Calmar = CAGR / Max_Drawdown
    
    FINANCIAL MEANING:
    ------------------
    Calmar answers: "How much return do I get per unit of max pain?"
    - Focuses on the WORST case (drawdown)
    - Good for risk-averse investors
    - > 1.0 means your return exceeds your worst drawdown
    
    Parameters:
    -----------
    cagr : float
        Compound Annual Growth Rate
    max_drawdown : float
        Maximum Drawdown (as positive number)
        
    Returns:
    --------
    float: Calmar ratio
    """
    
    if max_drawdown == 0:
        return np.inf if cagr > 0 else 0.0
    
    return cagr / max_drawdown


def calculate_var(returns: np.ndarray, 
                  confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) - historical simulation method.
    
    FORMULA:
        VaR_α = percentile(returns, (1-α) * 100)
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.percentile() - Calculate percentile values
       - percentile(data, 5) gives value below which 5% of data falls
       - Used for VaR calculation
    
    2. Understanding percentiles vs quantiles
    
    FINANCIAL MEANING:
    ------------------
    VaR at 95% answers: "What's the loss I won't exceed 95% of days?"
    Or: "On a bad day (5% worst), how much could I lose?"
    
    Example: VaR_95 = -2% means:
    - 95% of days, daily loss is less than 2%
    - 5% of days, loss could exceed 2%
    
    Parameters:
    -----------
    returns : np.ndarray
        Daily returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
        
    Returns:
    --------
    float: VaR (as positive number, representing potential loss)
    """
    
    # Calculate percentile for losses
    # For 95% VaR, we want the 5th percentile
    # NUMPY CONCEPT: np.percentile()
    percentile = (1 - confidence_level) * 100
    var = -np.percentile(returns, percentile)  # Negative to make VaR positive
    
    return var


def calculate_cvar(returns: np.ndarray, 
                   confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional VaR (CVaR) / Expected Shortfall.
    
    FORMULA:
        CVaR_α = mean of returns below VaR threshold
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. Boolean masking for filtering
       - returns[returns < threshold] selects values below threshold
    
    2. np.mean() on filtered array
    
    WHY CVaR > VaR:
    ---------------
    VaR tells you the threshold, but not how bad it gets BEYOND.
    CVaR tells you the AVERAGE loss on bad days.
    
    Example:
    - VaR_95 = -2% (5% of days you lose more than 2%)
    - CVaR_95 = -4% (on those bad days, average loss is 4%)
    
    CVaR is more informative and mathematically nicer (coherent risk measure).
    
    Parameters:
    -----------
    returns : np.ndarray
        Daily returns
    confidence_level : float
        Confidence level
        
    Returns:
    --------
    float: CVaR (as positive number)
    """
    
    # Get VaR threshold
    percentile = (1 - confidence_level) * 100
    var_threshold = np.percentile(returns, percentile)
    
    # Calculate mean of returns below VaR
    # NUMPY CONCEPT: Boolean masking
    # returns[returns <= var_threshold] filters to only tail returns
    tail_returns = returns[returns <= var_threshold]
    
    if len(tail_returns) == 0:
        return 0.0
    
    cvar = -np.mean(tail_returns)  # Negative to make positive
    
    return cvar


def calculate_trade_statistics(returns: np.ndarray, 
                               positions: np.ndarray) -> Dict:
    """
    Calculate trade-level statistics.
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.diff() for detecting trade changes
    2. Boolean arrays for filtering wins vs losses
    3. np.sum() with boolean masks
    
    Parameters:
    -----------
    returns : np.ndarray
        Strategy returns
    positions : np.ndarray
        Position array
        
    Returns:
    --------
    Dict with trade statistics
    """
    
    # Identify trades (position changes)
    trades = np.diff(positions)
    trade_indices = np.where(trades != 0)[0]
    
    n_trades = len(trade_indices)
    
    if n_trades < 2:
        return {
            'n_trades': n_trades,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
    
    # Calculate returns between trades
    trade_returns = []
    for i in range(len(trade_indices) - 1):
        start = trade_indices[i]
        end = trade_indices[i + 1]
        # NUMPY CONCEPT: np.sum on slice
        trade_ret = np.sum(returns[start:end])
        trade_returns.append(trade_ret)
    
    trade_returns = np.array(trade_returns)
    
    # Separate wins and losses
    # NUMPY CONCEPT: Boolean masking
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    
    # Calculate statistics
    n_wins = len(wins)
    n_losses = len(losses)
    
    win_rate = n_wins / len(trade_returns) if len(trade_returns) > 0 else 0.0
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    
    # Profit factor: sum of wins / abs(sum of losses)
    total_wins = np.sum(wins) if len(wins) > 0 else 0.0
    total_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
    
    return {
        'n_trades': n_trades,
        'n_wins': n_wins,
        'n_losses': n_losses,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'trade_returns': trade_returns
    }


def calculate_all_metrics(equity_curve: np.ndarray,
                          returns: np.ndarray,
                          positions: np.ndarray,
                          risk_free_rate: float = 0.0) -> PerformanceMetrics:
    """
    Calculate all performance metrics in one call.
    
    This is the main function to use after running a backtest.
    
    Parameters:
    -----------
    equity_curve : np.ndarray
        Portfolio value over time
    returns : np.ndarray
        Strategy returns
    positions : np.ndarray
        Position array
    risk_free_rate : float
        Annual risk-free rate (will be converted to daily)
        
    Returns:
    --------
    PerformanceMetrics: All metrics in structured format
    """
    
    # Convert annual risk-free rate to daily
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    
    # Calculate all metrics
    total_return = calculate_total_return(equity_curve)
    cagr = calculate_cagr(equity_curve)
    ann_return = calculate_annualized_return(returns)
    
    volatility = calculate_volatility(returns)
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    var_95 = calculate_var(returns, 0.95)
    cvar_95 = calculate_cvar(returns, 0.95)
    
    sharpe = calculate_sharpe_ratio(returns, daily_rf)
    sortino = calculate_sortino_ratio(returns, daily_rf)
    calmar = calculate_calmar_ratio(cagr, max_dd)
    
    trade_stats = calculate_trade_statistics(returns, positions)
    
    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        annualized_return=ann_return,
        volatility=volatility,
        max_drawdown=max_dd,
        var_95=var_95,
        cvar_95=cvar_95,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        n_trades=trade_stats['n_trades'],
        win_rate=trade_stats['win_rate'],
        profit_factor=trade_stats['profit_factor'],
        avg_win=trade_stats['avg_win'],
        avg_loss=trade_stats['avg_loss']
    )


def format_metrics_report(metrics: PerformanceMetrics) -> str:
    """
    Format metrics into a readable report.
    
    Parameters:
    -----------
    metrics : PerformanceMetrics
        Calculated metrics
        
    Returns:
    --------
    str: Formatted report
    """
    
    report = """
╔══════════════════════════════════════════════════════════════════════╗
║                      PERFORMANCE REPORT                               ║
╠══════════════════════════════════════════════════════════════════════╣
║ RETURNS                                                               ║
║   Total Return:         {:>10.2%}                                     ║
║   CAGR:                 {:>10.2%}                                     ║
║   Annualized Return:    {:>10.2%}                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║ RISK                                                                  ║
║   Volatility (Ann.):    {:>10.2%}                                     ║
║   Max Drawdown:         {:>10.2%}                                     ║
║   VaR (95%):            {:>10.2%}                                     ║
║   CVaR (95%):           {:>10.2%}                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║ RISK-ADJUSTED                                                         ║
║   Sharpe Ratio:         {:>10.2f}                                     ║
║   Sortino Ratio:        {:>10.2f}                                     ║
║   Calmar Ratio:         {:>10.2f}                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║ TRADE ANALYSIS                                                        ║
║   Number of Trades:     {:>10d}                                       ║
║   Win Rate:             {:>10.2%}                                     ║
║   Profit Factor:        {:>10.2f}                                     ║
║   Avg Win:              {:>10.2%}                                     ║
║   Avg Loss:             {:>10.2%}                                     ║
╚══════════════════════════════════════════════════════════════════════╝
""".format(
        metrics.total_return,
        metrics.cagr,
        metrics.annualized_return,
        metrics.volatility,
        metrics.max_drawdown,
        metrics.var_95,
        metrics.cvar_95,
        metrics.sharpe_ratio,
        metrics.sortino_ratio,
        metrics.calmar_ratio,
        metrics.n_trades,
        metrics.win_rate,
        metrics.profit_factor,
        metrics.avg_win,
        metrics.avg_loss
    )
    
    return report


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 7: Risk & Performance Metrics")
    print("=" * 70)
    
    # Generate sample data
    np.random.seed(42)
    n_days = 504  # 2 years
    
    # Create realistic equity curve
    daily_returns = np.random.normal(0.0005, 0.015, n_days)  # ~12% annual, 24% vol
    daily_returns[0] = 0
    
    equity = 10000 * np.cumprod(1 + daily_returns)
    
    # Create sample positions
    positions = np.sign(np.random.randn(n_days))
    positions = np.where(np.random.rand(n_days) > 0.5, positions, 0)
    
    print(f"\nSample Data:")
    print(f"  Days: {n_days}")
    print(f"  Initial capital: ${equity[0]:,.2f}")
    print(f"  Final capital: ${equity[-1]:,.2f}")
    
    # Calculate individual metrics
    print("\n" + "-" * 70)
    print("Individual Metric Calculations")
    print("-" * 70)
    
    print(f"\nTotal Return: {calculate_total_return(equity):.2%}")
    print(f"CAGR: {calculate_cagr(equity):.2%}")
    print(f"Annualized Return: {calculate_annualized_return(daily_returns):.2%}")
    
    print(f"\nVolatility (Ann.): {calculate_volatility(daily_returns):.2%}")
    
    max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity)
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"  Peak at day {peak_idx}: ${equity[peak_idx]:,.2f}")
    print(f"  Trough at day {trough_idx}: ${equity[trough_idx]:,.2f}")
    
    print(f"\nVaR (95%): {calculate_var(daily_returns, 0.95):.2%}")
    print(f"CVaR (95%): {calculate_cvar(daily_returns, 0.95):.2%}")
    
    print(f"\nSharpe Ratio: {calculate_sharpe_ratio(daily_returns):.2f}")
    print(f"Sortino Ratio: {calculate_sortino_ratio(daily_returns):.2f}")
    print(f"Calmar Ratio: {calculate_calmar_ratio(calculate_cagr(equity), max_dd):.2f}")
    
    # Calculate all metrics at once
    print("\n" + "-" * 70)
    print("Complete Metrics Report")
    print("-" * 70)
    
    all_metrics = calculate_all_metrics(equity, daily_returns, positions)
    print(format_metrics_report(all_metrics))
    
    # Demonstrate NumPy concepts
    print("\n" + "=" * 70)
    print("Key NumPy concepts demonstrated:")
    print("  - np.maximum.accumulate() for running maximum (drawdown)")
    print("  - np.percentile() for VaR calculation")
    print("  - np.mean() and np.std() for return statistics")
    print("  - Boolean masking for filtering (wins vs losses)")
    print("  - np.cumprod() for equity curve")
    print("  - Numerical stability checks (zero division)")
    print("=" * 70)
