# NumPy-Based Stock Market Backtesting Engine

A complete stock market backtesting engine built entirely with NumPy. This project is designed as a **learning resource** to master NumPy through practical financial applications.

## Purpose

This project teaches you to:
- **Think in vectorized pipelines** instead of loops
- **Understand time alignment** deeply (critical in finance)
- **Learn real numerical finance** concepts
- **Use NumPy like an engineer**, not a student

## 📁 Project Structure

```
backtester/
├── src/
│   ├── data.py        # Phase 2: Data loading & generation
│   ├── indicators.py  # Phase 3-4: Returns & technical indicators
│   ├── strategy.py    # Phase 5: Signal generation
│   ├── engine.py      # Phase 6: Backtesting simulation
│   ├── metrics.py     # Phase 7: Performance statistics
│   └── main.py        # Complete demo runner
│
├── tests/
│   ├── test_returns.py   # Phase 8: Returns validation
│   └── test_strategy.py  # Phase 8: Strategy validation
│
└── README.md
```

## Quick Start

```bash
# Navigate to the project
cd backtester/src

# Run the complete demo
python main.py

# Run tests
cd ../tests
python test_returns.py
python test_strategy.py
```


## NumPy Concepts Cheat Sheet

| Concept | Function | Use Case |
|---------|----------|----------|
| Running max | `np.maximum.accumulate()` | Drawdowns |
| Rolling sum | `np.convolve()` | SMA |
| Differences | `np.diff()` | Returns |
| Cumulative | `np.cumsum()`, `np.cumprod()` | Equity curve |
| Conditional | `np.where()` | Signal generation |
| Percentile | `np.percentile()` | VaR |
| Shifting | `np.roll()` | Position alignment |

## 📊 Sample Output

```
╔══════════════════════════════════════════════════════════════════════╗
║                      PERFORMANCE REPORT                               ║
╠══════════════════════════════════════════════════════════════════════╣
║ RETURNS                                                               ║
║   Total Return:              45.23%                                   ║
║   CAGR:                      20.15%                                   ║
║   Annualized Return:         18.75%                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║ RISK                                                                  ║
║   Volatility (Ann.):         24.32%                                   ║
║   Max Drawdown:              15.67%                                   ║
║   VaR (95%):                  2.15%                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║ RISK-ADJUSTED                                                         ║
║   Sharpe Ratio:               0.82                                    ║
║   Sortino Ratio:              1.15                                    ║
║   Calmar Ratio:               1.29                                    ║
╚══════════════════════════════════════════════════════════════════════╝
```


## 🔗 Skills Transfer

This knowledge directly applies to:
- **ML preprocessing** - Feature engineering, time series
- **Signal processing** - Filtering, convolutions
- **Large-scale analytics** - Vectorized data processing
- **Quantitative research** - Financial modeling

## 📝 Requirements

- Python 3.8+
- NumPy 1.20+

```bash
pip install numpy
```

## 📖 Further Reading

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Quantitative Finance with Python](https://www.quantstart.com/)
- [Investopedia - Technical Indicators](https://www.investopedia.com/terms/t/technicalindicator.asp)

---

