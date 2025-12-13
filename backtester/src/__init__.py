"""
Stock Market Backtesting Engine - Source Package

A NumPy-based backtesting engine for learning vectorized financial computations.
"""

from .data import generate_sample_data, generate_ohlcv_data, validate_price_data
from .indicators import (
    simple_returns, log_returns, cumulative_returns,
    simple_moving_average, exponential_moving_average,
    rolling_volatility, bollinger_bands, relative_strength_index, macd
)
from .strategy import (
    generate_sma_crossover_signal, generate_momentum_signal,
    generate_mean_reversion_signal, generate_rsi_signal,
    combine_signals
)
from .engine import run_backtest, BacktestResult
from .metrics import calculate_all_metrics, PerformanceMetrics

__all__ = [
    # Data
    'generate_sample_data',
    'generate_ohlcv_data', 
    'validate_price_data',
    
    # Indicators
    'simple_returns',
    'log_returns',
    'cumulative_returns',
    'simple_moving_average',
    'exponential_moving_average',
    'rolling_volatility',
    'bollinger_bands',
    'relative_strength_index',
    'macd',
    
    # Strategy
    'generate_sma_crossover_signal',
    'generate_momentum_signal',
    'generate_mean_reversion_signal',
    'generate_rsi_signal',
    'combine_signals',
    
    # Engine
    'run_backtest',
    'BacktestResult',
    
    # Metrics
    'calculate_all_metrics',
    'PerformanceMetrics',
]
