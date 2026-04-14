# Data Flow Architecture

## Overview
This automated trading algorithm analyzes three stocks (AAPL, TSLA, NVDA) using three independent strategies, aggregates signals, sizes positions using Kelly Criterion, ensures portfolio diversification (max 80% per stock), and calculates risk metrics—all in **paper trading mode only** (no real trades executed).

## Pipeline

### 1. **Data Fetching** (`trading/strategies.py`)
- Retrieves 500 days of OHLCV data for AAPL, TSLA, NVDA
- Calculates daily percent changes and returns

### 2. **Strategy Analysis** (3 Independent Strategies per Stock)

#### a) **Simple Moving Average (SMA-50)**
- Compares closing price vs 50-day moving average
- **Signal**: Buy if price > SMA, Sell if price < SMA

#### b) **Markov Chain**
- Models price movement states: Up (>0.5%), Steady (0-0.5%), Down (<0%)
- Builds transition probability matrices from historical data
- **Signal**: Buy if predicted Up, Sell if predicted Down, Hold if Steady

#### c) **LSTM Neural Network**
- Trains 1-layer LSTM on normalized returns (75/25 train-test split)
- Predicts tomorrow's return from last 60 days
- **Signal**: Buy if predicted return > today's return, else Sell

### 3. **Signal Aggregation** (`trading/strategies.py`)
- Per stock: counts buy/sell votes from 3 strategies
- **Final Decision**: Buy (≥2 buy votes), Sell (≥2 sell votes), Hold (otherwise)
- **Output**: `signals[ticker]` dict with all strategy votes + final signal

### 4. **Position Sizing & Portfolio Balancing** (`trading/execution.py`)
- **Kelly Criterion**: Calculates optimal position size based on Markov transition matrix
  - Formula: kelly_fraction = (Expected Return / Expected Return²) × 0.25 (capped at 100%)
- **Portfolio Check**: Simulates post-trade weights for all three stocks
- **80% Limit Enforcement**: If any stock would exceed 80% of portfolio, reduces shares to maintain limit
- **Paper Trading**: Logs all trades as "Would BUY/SELL" without actual execution

### 5. **Risk Metrics Calculation** (`trading/position_risk.py`)
- **Sharpe Ratio**: (Mean Return − Risk-Free Rate) / Std Dev
  - ✅ Green if > 0.5 | ⚠️ Orange if moderate | ❌ Red if < 0
- **VaR (95% Confidence)**: −1.645 × Std Dev (assumes normal distribution)
  - ✅ Green if < 5% | ⚠️ Alert if > 5%

### 6. **Reporting & Visualization** (`trading/reporting.py`)
- Generates 6-panel dashboard:
  1. **Final Trading Signals** (by stock)
  2. **Sharpe Ratios** (risk-adjusted returns)
  3. **VaR 95%** (downside risk)
  4. **Individual Strategy Votes** (SMA, Markov, LSTM breakdown)
  5. **Mean Daily Returns** (historical average)
  6. **Volatility** (historical standard deviation)
- **Exports**: PDF report named `algo_results_YYYY-MM-DD.pdf`

## Execution

### Run Everything:
```bash
python -m main.run_algorithm
```

### What Happens:
1. Strategies analyze all 3 stocks independently
2. Execution simulates trades with portfolio constraints
3. Reporting generates PDF with analytics
4. Output: `algo_results_YYYY-MM-DD.pdf` + console logs

## Key Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **3 Strategies** | Diversified signals reduce false positives; majority vote (2/3) for robustness |
| **Kelly Criterion** | Optimal position sizing based on win probability from Markov states |
| **80% Limit** | Prevents over-concentration; industry standard for risk management |
| **Sharpe + VaR** | Sharpe captures risk-adjusted returns; VaR quantifies downside exposure |
| **Paper Trading** | All trades simulated; no real capital at risk; PDF shows what *would* happen |
| **Daily Execution** | Run `run_algorithm.py` daily; generates timestamped report each run |

## Constraints & Assumptions

- **Normal Distribution**: VaR assumes normal returns (may underestimate tail risk in crisis)
- **Markov Independence**: Assumes tomorrow's state depends only on today's state
- **LSTM Training**: Uses only AAPL/TSLA/NVDA historical data; may not generalize to new stocks
- **Paper Trading**: Portfolio weights & cash flows are simulated; actual broker balances may differ

## Paper Trading Guarantee

⚠️ **CRITICAL**: No real trades are executed at any stage.
- `execution.py` outputs: `"PAPER TRADE: Would BUY/SELL X shares"` (logged, not executed)
- API calls use `paper=True` (Alpaca paper trading endpoint)
- PDF report shows simulated portfolio state only
