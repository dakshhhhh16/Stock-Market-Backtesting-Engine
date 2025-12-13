"""
=============================================================================
PHASE 2: Market Data Handling
=============================================================================

PURPOSE:
    Convert raw price data → clean NumPy arrays.
    This module is the foundation - all other modules depend on clean data.

KEY NUMPY CONCEPTS YOU'LL LEARN:
    1. dtype control - Specifying data types for memory efficiency
    2. Array slicing - Extracting specific columns/rows
    3. Missing value handling - np.isnan, np.nan
    4. Data validation - Ensuring array integrity

MENTAL MODEL:
    Price series = 1D NumPy array where:
    - Each element is a price at time t
    - Index represents time (t=0, t=1, t=2, ...)
    - Time alignment is CRITICAL (most bugs live here)

RULE: Convert to NumPy ONCE and stay there. No pandas in core logic.
=============================================================================
"""

import numpy as np
from typing import Tuple, Optional
import os


def load_csv_data(filepath: str, 
                  date_col: int = 0,
                  ohlcv_cols: Tuple[int, int, int, int, int] = (1, 2, 3, 4, 5),
                  skip_header: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load OHLCV data from CSV file using pure NumPy.
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.genfromtxt() - Reads CSV files directly into NumPy arrays
       - delimiter: column separator
       - skip_header: skip first row if it contains column names
       - usecols: select specific columns (avoid loading unnecessary data)
       - dtype: control memory usage (float64 vs float32)
    
    2. Why we use dtype=np.float64:
       - Financial data needs precision
       - float32 can introduce rounding errors in cumulative calculations
       - Trade-off: memory vs precision
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file with OHLCV data
    date_col : int
        Column index for dates (we'll skip this for NumPy processing)
    ohlcv_cols : tuple
        Column indices for (Open, High, Low, Close, Volume)
    skip_header : bool
        Whether to skip the first row
        
    Returns:
    --------
    Tuple of 5 NumPy arrays: (open_prices, high_prices, low_prices, close_prices, volume)
    """
    
    # Load data using genfromtxt
    # This is more flexible than loadtxt - handles missing values automatically
    data = np.genfromtxt(
        filepath,
        delimiter=',',
        skip_header=1 if skip_header else 0,
        usecols=ohlcv_cols,  # Only load OHLCV columns, not dates
        dtype=np.float64,    # Use float64 for financial precision
        missing_values=['', 'NA', 'NaN', 'null'],  # Common missing value representations
        filling_values=np.nan  # Replace missing with NaN for later handling
    )
    
    # Extract individual price arrays using column slicing
    # NUMPY CONCEPT: Array slicing with [:, index]
    # data[:, 0] means "all rows, column 0"
    open_prices = data[:, 0]
    high_prices = data[:, 1]
    low_prices = data[:, 2]
    close_prices = data[:, 3]
    volume = data[:, 4]
    
    return open_prices, high_prices, low_prices, close_prices, volume


def generate_sample_data(n_days: int = 252, 
                         initial_price: float = 100.0,
                         volatility: float = 0.02,
                         drift: float = 0.0001,
                         seed: Optional[int] = 42) -> np.ndarray:
    """
    Generate synthetic price data using Geometric Brownian Motion (GBM).
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.random.default_rng() - Modern random number generation
       - Reproducible with seed
       - Better statistical properties than legacy np.random
    
    2. np.exp() - Element-wise exponential
       - Vectorized operation (no loops needed!)
       - Applied to entire array at once
    
    3. np.cumsum() - Cumulative sum
       - Running total of all elements
       - Essential for converting returns to prices
    
    FINANCIAL CONCEPT - GBM:
    ------------------------
    Stock prices often modeled as:
        S(t) = S(0) * exp(cumsum(returns))
    
    Where returns follow:
        r(t) = drift + volatility * random_shock
    
    Parameters:
    -----------
    n_days : int
        Number of trading days to generate (252 = 1 year)
    initial_price : float
        Starting price
    volatility : float
        Daily volatility (0.02 = 2% daily std dev)
    drift : float
        Expected daily return (0.0001 = 0.01% daily)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray: Array of prices with shape (n_days,)
    """
    
    # Set random seed for reproducibility
    # NUMPY CONCEPT: Modern RNG with default_rng()
    rng = np.random.default_rng(seed)
    
    # Generate random daily returns
    # NUMPY CONCEPT: rng.normal() generates array of normally distributed values
    # Shape: (n_days,) - 1D array
    # Each element: drift + volatility * N(0,1)
    daily_returns = drift + volatility * rng.normal(size=n_days)
    
    # First day has no return (it's our starting point)
    # NUMPY CONCEPT: Direct array indexing and assignment
    daily_returns[0] = 0
    
    # Convert log returns to prices using GBM formula
    # NUMPY CONCEPT: 
    # - np.cumsum() creates running sum: [r1, r1+r2, r1+r2+r3, ...]
    # - np.exp() converts log returns to price multipliers
    # - Multiply by initial price to get actual prices
    # ALL VECTORIZED - no loops!
    prices = initial_price * np.exp(np.cumsum(daily_returns))
    
    return prices


def generate_ohlcv_data(n_days: int = 252,
                        initial_price: float = 100.0,
                        volatility: float = 0.02,
                        seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic OHLCV data for testing.
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.maximum() / np.minimum() - Element-wise max/min
       - Compare two arrays element by element
       - Returns array of same shape
    
    2. Array arithmetic - Broadcasting
       - close * 0.98 multiplies EVERY element by 0.98
       - No loops needed!
    
    3. np.abs() - Absolute value (element-wise)
    
    Returns:
    --------
    Tuple of (open, high, low, close, volume) arrays
    """
    rng = np.random.default_rng(seed)
    
    # Generate base close prices
    close = generate_sample_data(n_days, initial_price, volatility, seed=seed)
    
    # Generate intraday variations
    # NUMPY CONCEPT: Broadcasting - scalar operations apply to all elements
    intraday_vol = volatility * 0.5  # Intraday volatility is typically lower
    
    # Open price: close of previous day with overnight gap
    # NUMPY CONCEPT: np.roll() shifts array elements
    # roll(close, 1) shifts everything right by 1, wrapping around
    open_prices = np.roll(close, 1)
    open_prices[0] = initial_price  # First day starts at initial price
    
    # Add small random variation to open
    open_prices = open_prices * (1 + rng.normal(0, intraday_vol * 0.5, n_days))
    
    # High and Low based on close with random intraday range
    # NUMPY CONCEPT: np.maximum ensures high >= max(open, close)
    high_variation = np.abs(rng.normal(0, intraday_vol, n_days))
    low_variation = np.abs(rng.normal(0, intraday_vol, n_days))
    
    high = np.maximum(open_prices, close) * (1 + high_variation)
    low = np.minimum(open_prices, close) * (1 - low_variation)
    
    # Generate random volume
    # NUMPY CONCEPT: rng.integers for integer arrays
    base_volume = 1_000_000
    volume = rng.integers(base_volume // 2, base_volume * 2, size=n_days).astype(np.float64)
    
    return open_prices, high, low, close, volume


def clean_price_data(prices: np.ndarray) -> np.ndarray:
    """
    Clean price data by handling missing values and ensuring validity.
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.isnan() - Detect NaN values (returns boolean array)
       - NaN = Not a Number (represents missing data)
       - True where NaN exists, False otherwise
    
    2. np.any() - Check if ANY element is True
       - Useful for validation checks
    
    3. Boolean indexing - prices[~np.isnan(prices)]
       - ~ is the NOT operator
       - Select only non-NaN values
    
    4. np.interp() - Linear interpolation
       - Fill gaps in data smoothly
       - Better than forward-fill for financial data
    
    5. np.where() - Conditional element selection
       - Returns indices where condition is True
    
    Parameters:
    -----------
    prices : np.ndarray
        Raw price array, possibly with NaN values
        
    Returns:
    --------
    np.ndarray: Cleaned price array with no NaN values
    """
    
    # Create a copy to avoid modifying original data
    # NUMPY CONCEPT: .copy() creates independent array
    # Without copy, changes would affect original!
    cleaned = prices.copy()
    
    # Check for NaN values
    # NUMPY CONCEPT: np.isnan() returns boolean array same shape as input
    nan_mask = np.isnan(cleaned)
    
    if not np.any(nan_mask):
        # No NaN values, return as-is
        return cleaned
    
    # Get indices of valid (non-NaN) and invalid (NaN) values
    # NUMPY CONCEPT: np.where() returns tuple of index arrays
    valid_indices = np.where(~nan_mask)[0]  # [0] because where returns tuple
    invalid_indices = np.where(nan_mask)[0]
    
    # Interpolate missing values
    # NUMPY CONCEPT: np.interp(x, xp, fp)
    # - x: where to interpolate
    # - xp: known x coordinates (indices with valid data)
    # - fp: known y values (valid prices)
    cleaned[invalid_indices] = np.interp(
        invalid_indices,
        valid_indices,
        cleaned[valid_indices]
    )
    
    return cleaned


def validate_price_data(prices: np.ndarray) -> Tuple[bool, str]:
    """
    Validate price data for common issues.
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. prices.ndim - Number of dimensions
       - 1D array has ndim=1
       - 2D array (matrix) has ndim=2
    
    2. prices.shape - Tuple of dimensions
       - (100,) for 1D array with 100 elements
       - (100, 5) for 2D array with 100 rows, 5 columns
    
    3. Comparison operators on arrays
       - prices <= 0 returns boolean array
       - np.any(prices <= 0) checks if ANY are non-positive
    
    4. np.isinf() - Check for infinity values
    
    Returns:
    --------
    Tuple of (is_valid: bool, message: str)
    """
    
    # Check if input is numpy array
    if not isinstance(prices, np.ndarray):
        return False, "Input must be a NumPy array"
    
    # Check dimensionality
    # NUMPY CONCEPT: .ndim gives number of dimensions
    if prices.ndim != 1:
        return False, f"Expected 1D array, got {prices.ndim}D"
    
    # Check if empty
    # NUMPY CONCEPT: len() or .shape[0] for array length
    if len(prices) == 0:
        return False, "Price array is empty"
    
    # Check for NaN values
    if np.any(np.isnan(prices)):
        return False, "Price array contains NaN values"
    
    # Check for infinite values
    # NUMPY CONCEPT: np.isinf() detects +inf and -inf
    if np.any(np.isinf(prices)):
        return False, "Price array contains infinite values"
    
    # Check for non-positive prices (invalid in finance)
    # NUMPY CONCEPT: Comparison operators work element-wise
    if np.any(prices <= 0):
        return False, "Price array contains non-positive values"
    
    return True, "Price data is valid"


def ensure_sorted_chronologically(prices: np.ndarray, 
                                   timestamps: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Ensure prices are in chronological order (oldest to newest).
    
    NUMPY LEARNING POINTS:
    -----------------------
    1. np.argsort() - Returns INDICES that would sort the array
       - Not the sorted values themselves
       - Useful for sorting one array based on another
    
    2. Fancy indexing - prices[indices]
       - Use array of indices to reorder
       - Very powerful for data alignment
    
    3. np.diff() - Differences between consecutive elements
       - diff([1,3,6,10]) = [2,3,4]
       - Useful for detecting order issues
    
    Parameters:
    -----------
    prices : np.ndarray
        Price array
    timestamps : np.ndarray, optional
        Timestamp array (if available)
        
    Returns:
    --------
    np.ndarray: Properly ordered price array
    """
    
    if timestamps is not None:
        # Sort by timestamps
        # NUMPY CONCEPT: np.argsort() returns sorting indices
        sort_indices = np.argsort(timestamps)
        
        # Apply sorting using fancy indexing
        # NUMPY CONCEPT: array[index_array] reorders elements
        return prices[sort_indices]
    
    # If no timestamps, assume already chronological
    # Just return a copy
    return prices.copy()


def extract_close_prices(ohlcv_data: np.ndarray) -> np.ndarray:
    """
    Extract close prices from OHLCV array.
    
    NUMPY LEARNING POINTS:
    -----------------------
    This demonstrates proper slicing for 2D arrays:
    
    For a 2D array with shape (n_days, 5):
        - ohlcv[:, 0] = Open prices (all rows, column 0)
        - ohlcv[:, 1] = High prices
        - ohlcv[:, 2] = Low prices  
        - ohlcv[:, 3] = Close prices
        - ohlcv[:, 4] = Volume
    
    Parameters:
    -----------
    ohlcv_data : np.ndarray
        2D array with shape (n_days, 5) containing OHLCV data
        
    Returns:
    --------
    np.ndarray: 1D array of close prices
    """
    
    # Validate input shape
    if ohlcv_data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {ohlcv_data.ndim}D")
    
    if ohlcv_data.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns, got {ohlcv_data.shape[1]}")
    
    # Extract close prices (column index 3)
    # NUMPY CONCEPT: 2D slicing
    # [:, 3] means "all rows, column 3"
    close_prices = ohlcv_data[:, 3]
    
    return close_prices


# ============================================================================
# DEMONSTRATION: How to use this module
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2: Market Data Handling - Demonstration")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating synthetic price data...")
    prices = generate_sample_data(n_days=252, initial_price=100.0, volatility=0.02)
    
    print(f"   Shape: {prices.shape}")
    print(f"   Data type: {prices.dtype}")
    print(f"   First 5 prices: {prices[:5]}")
    print(f"   Last 5 prices: {prices[-5:]}")
    
    # Validate data
    print("\n2. Validating price data...")
    is_valid, message = validate_price_data(prices)
    print(f"   Valid: {is_valid}")
    print(f"   Message: {message}")
    
    # Generate OHLCV data
    print("\n3. Generating OHLCV data...")
    open_p, high_p, low_p, close_p, volume = generate_ohlcv_data(n_days=10)
    
    print("   Sample OHLCV data (first 5 days):")
    print("   Day  |  Open   |  High   |   Low   |  Close  |  Volume")
    print("   " + "-" * 55)
    for i in range(5):
        print(f"   {i+1:3d}  | {open_p[i]:7.2f} | {high_p[i]:7.2f} | {low_p[i]:7.2f} | {close_p[i]:7.2f} | {volume[i]:,.0f}")
    
    # Demonstrate NaN handling
    print("\n4. Demonstrating NaN handling...")
    prices_with_nan = prices.copy()
    prices_with_nan[5] = np.nan  # Introduce a NaN
    prices_with_nan[10] = np.nan
    
    print(f"   Prices with NaN (indices 5,10): {prices_with_nan[4:12]}")
    
    cleaned = clean_price_data(prices_with_nan)
    print(f"   After cleaning: {cleaned[4:12]}")
    
    print("\n" + "=" * 60)
    print("Data module ready! Key NumPy concepts demonstrated:")
    print("  - Array creation and dtype control")
    print("  - Random number generation (GBM simulation)")
    print("  - NaN handling and interpolation")
    print("  - Array validation and slicing")
    print("=" * 60)
