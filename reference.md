# Stock Market Backtesting Engine - Reference Guide

## What is This Project?

This is a **backtesting engine** - a tool that lets you test trading strategies on historical stock data to see how they would have performed in the past. Think of it like a "time machine" for trading strategies!

---

## How Does Stock Trading Work (Simplified)?

1. **Buy Low, Sell High**: The basic goal is to buy stocks when prices are low and sell when they're high
2. **Signals**: A trading strategy tells you WHEN to buy or sell
3. **Positions**: 
   - `+1` = You own the stock (Long position - you profit when price goes UP)
   - `-1` = You've "shorted" the stock (Short position - you profit when price goes DOWN)
   - `0` = You're not in the market (holding cash)

---

## Project Structure Explained

```
backtester/
├── src/
│   ├── data.py        → Creates fake stock price data
│   ├── indicators.py  → Calculates things like returns and moving averages
│   ├── strategy.py    → Decides when to buy/sell
│   ├── engine.py      → Runs the backtest simulation
│   ├── metrics.py     → Measures how good your strategy performed
│   └── main.py        → Demo that runs everything together
└── tests/             → Tests to make sure everything works
```

---

## Module-by-Module Explanation

### 1. `data.py` - The Data Generator

**What it does**: Creates realistic-looking fake stock prices

**Key Concept - Random Walk**:
```
Tomorrow's Price = Today's Price × (1 + small random change)
```

This is called **Geometric Brownian Motion (GBM)** - it's how real stock prices actually move!

**NumPy Learning**:
- `np.random.randn(n)` → Generates n random numbers from a bell curve
- `np.cumprod()` → Cumulative product (multiplies all previous values together)
- `np.isnan()` → Checks for missing values (NaN = Not a Number)

---

### 2. `indicators.py` - The Calculator

**What it does**: Calculates useful numbers from price data

#### Returns (How much you made/lost)

**Simple Returns**:
```
Return = (Today's Price - Yesterday's Price) / Yesterday's Price
```
Example: Stock goes from $100 to $105 → Return = 5/100 = 5%

**Log Returns**:
```
Return = log(Today's Price / Yesterday's Price)
```
Why use log? They add up nicely over time (mathematically convenient)

**NumPy Learning**:
- `np.diff(prices)` → Calculates difference between consecutive elements
- `np.log()` → Natural logarithm

#### Moving Averages (Smoothing out the noise)

**Simple Moving Average (SMA)**:
```
SMA = Average of last N days' prices
```
Example: 5-day SMA = (Day1 + Day2 + Day3 + Day4 + Day5) / 5

**Why useful?** Smooths out daily noise to see the trend

**NumPy Learning**:
- `np.convolve()` → Sliding window operation (like dragging a window across data)
- `np.ones(n)/n` → Creates weights for averaging

#### Other Indicators

- **Bollinger Bands**: Price channels showing if stock is "expensive" or "cheap"
- **RSI (Relative Strength Index)**: Measures if stock is "overbought" (too high) or "oversold" (too low)
- **MACD**: Compares fast and slow moving averages to spot trend changes

---

### 3. `strategy.py` - The Decision Maker

**What it does**: Looks at indicators and decides: BUY (+1), SELL (-1), or HOLD (0)

#### Strategy 1: SMA Crossover
```
If Fast SMA > Slow SMA → BUY (trend going up)
If Fast SMA < Slow SMA → SELL (trend going down)
```

**Analogy**: Fast SMA is like a speedboat, Slow SMA is like a cruise ship. When the speedboat overtakes the cruise ship, the market is speeding up!

#### Strategy 2: Momentum
```
If price went UP over last N days → BUY (momentum continuing)
If price went DOWN over last N days → SELL
```

#### Strategy 3: Mean Reversion
```
If price is BELOW its average → BUY (expecting it to bounce back)
If price is ABOVE its average → SELL (expecting it to fall back)
```

#### Strategy 4: RSI
```
If RSI < 30 → BUY (oversold, too cheap)
If RSI > 70 → SELL (overbought, too expensive)
```

#### Important: Look-Ahead Bias Prevention

**Problem**: In real life, you can't use today's closing price to make today's trade!

**Solution**: We "shift" signals by one day - today's signal is based on yesterday's data

**NumPy Learning**:
- `np.where(condition, value_if_true, value_if_false)` → Vectorized if-else
- `np.roll(array, 1)` → Shifts array elements (for signal shifting)

---

### 4. `engine.py` - The Simulator

**What it does**: Simulates what would have happened if you followed a strategy

#### Core Calculation:
```
Strategy Returns = Position × Market Returns
```

**Example**:
- If you're LONG (+1) and market goes UP 2% → You make +2%
- If you're LONG (+1) and market goes DOWN 2% → You lose -2%
- If you're SHORT (-1) and market goes DOWN 2% → You make +2%
- If you're OUT (0) and market moves → You make 0%

#### Transaction Costs
Every time you change position (buy/sell), you pay a small fee (like brokerage)

**NumPy Learning**:
- `np.diff(positions) != 0` → Finds where trades happened
- `np.cumprod(1 + returns)` → Builds the equity curve (your wealth over time)

---

### 5. `metrics.py` - The Report Card

**What it does**: Calculates how well your strategy performed

#### Key Metrics Explained:

**1. Total Return**
```
How much money you made overall (as a percentage)
```

**2. Sharpe Ratio** ⭐ (Most Important!)
```
Sharpe = (Average Return) / (Risk)
```
- Higher is better (more return per unit of risk)
- Above 1.0 = Good, Above 2.0 = Excellent

**3. Max Drawdown**
```
The biggest drop from a peak (worst losing streak)
```
Example: If your $100 grew to $150, then fell to $120, drawdown = 20%

**4. Win Rate**
```
What percentage of your trades were profitable
```

**5. Sortino Ratio**
```
Like Sharpe, but only considers downside risk
```

**6. VaR (Value at Risk)**
```
"On a bad day (worst 5%), how much might I lose?"
```

**NumPy Learning**:
- `np.maximum.accumulate()` → Running maximum (for drawdown calculation)
- `np.percentile()` → Finds value at a given percentile (for VaR)
- `np.std()` → Standard deviation (measures risk)

---

## How Everything Works Together

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  data.py    │ ──► │ indicators.py│ ──► │ strategy.py│
│ (Get Prices)│     │ (Calculate   │     │ (Generate  │
│             │     │  Indicators) │     │  Signals)  │
└─────────────┘     └──────────────┘     └────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  metrics.py │ ◄── │   engine.py  │ ◄───│  Signals   │
│ (Evaluate   │     │ (Simulate    │     │  [+1,+1,-1]│
│  Results)   │     │  Trading)    │     │            │
└─────────────┘     └──────────────┘     └────────────┘
```

---

## Key NumPy Concepts Used

| Function | What It Does | Where Used |
|----------|--------------|------------|
| `np.diff()` | Difference between consecutive elements | Returns calculation |
| `np.cumsum()` | Running total | Cumulative returns |
| `np.cumprod()` | Running product | Equity curve |
| `np.convolve()` | Sliding window operation | Moving averages |
| `np.where()` | Vectorized if-else | Signal generation |
| `np.roll()` | Shift array elements | Look-ahead bias prevention |
| `np.maximum.accumulate()` | Running maximum | Max drawdown |
| `np.percentile()` | Find percentile value | VaR calculation |
| `np.isnan()` | Check for NaN | Data validation |
| `np.random.randn()` | Random normal numbers | Price simulation |

---

## Running the Project

```bash
# Run the demo
cd backtester/src
python main.py

# Run tests
cd backtester/tests
python -m pytest test_returns.py -v
python -m pytest test_strategy.py -v
```

---

## What You'll Learn

1. **Vectorized Operations**: Doing math on entire arrays at once (no loops!)
2. **Financial Math**: Returns, moving averages, risk metrics
3. **Trading Concepts**: Positions, signals, backtesting, transaction costs
4. **Testing**: Writing unit tests to verify your code works

---

## Next Steps to Explore

1. **Add Real Data**: Use Yahoo Finance API to get real stock prices
2. **Visualize**: Add matplotlib charts to see the equity curve
3. **New Strategies**: Create your own trading rules
4. **Optimize**: Find the best parameters for each strategy

---

## Glossary

| Term | Meaning |
|------|---------|
| **Backtest** | Testing a strategy on historical data |
| **Long** | Owning a stock (profit when price goes up) |
| **Short** | Betting against a stock (profit when price goes down) |
| **Drawdown** | Drop from a peak value |
| **Sharpe Ratio** | Risk-adjusted return measure |
| **SMA** | Simple Moving Average |
| **RSI** | Relative Strength Index |
| **Look-ahead Bias** | Accidentally using future data to make decisions |
| **Vectorization** | Operating on entire arrays instead of looping |

---

*Happy Learning! 🚀📈*
