# Tradebot — A Simple Guide

## What Is This?

Tradebot is a tool that tries to predict Bitcoin price turning points (called **fractals**) using machine learning, then tests whether trading on those predictions would actually make money.

Think of it as: **spot the reversal, take the trade, measure the results.**

---

## The Big Idea

Bitcoin's price makes swing highs and swing lows. These are called **fractals** — a pattern where a candle's high (or low) sticks out above (or below) the surrounding candles.

Tradebot does three things:

1. **Detects** these fractals in historical price data
2. **Predicts** where the next one will happen using ML
3. **Backtests** whether trading those predictions is profitable

---

## How It Works (The Pipeline)

```
Price Data → Indicators → Features → ML Model → Predictions → Backtest
```

**Step by step:**

| Step | What happens | Why it matters |
|------|-------------|----------------|
| **Import data** | Load BTCUSDT hourly candles into the database | You need price history to work with |
| **Run indicators** | Detect fractals, compute support/resistance levels | These are the "signals" the ML learns from |
| **Compute features** | Calculate RSI, MACD, ATR, level proximity, candle patterns | Turn raw data into numbers the ML can digest |
| **Train a model** | Feed features to LightGBM (or Random Forest, XGBoost) | The model learns which patterns precede fractals |
| **Generate predictions** | Model scores each candle: no fractal / bullish / bearish | Now we know what the model thinks will happen |
| **Run backtest** | Simulate trading those predictions with real rules | Find out if the model's predictions make money |

---

## Key Concepts

**Fractals** — A 5-candle pattern. A swing high (bearish fractal) has a candle whose high is above both neighbors. A swing low (bullish fractal) is the opposite. These mark potential turning points.

**Support & Resistance Levels** — Price zones where buying or selling pressure has historically appeared. Levels come from higher-timeframe structure, Fibonacci retracements, and Volume Profile analysis.

**Features** — Measurable characteristics of each candle that the ML model uses to make predictions. Examples: how far is price from the nearest support? What's the RSI? How volatile is the market (ATR)?

**Confidence** — How sure the model is about its prediction (0–100%). Trades only trigger when confidence is above 55%.

---

## The Trading Strategy

When the model predicts a fractal with enough confidence:

- **LONG** (buy): Model predicts a swing low near a support level
- **SHORT** (sell): Model predicts a swing high near a resistance level
- **Stop-loss**: 1.5x ATR from entry (limits your loss)
- **Take-profit**: 3.0x ATR from entry (2:1 reward-to-risk ratio)
- **Position size**: Risk 2% of portfolio per trade

---

## The Web Interface

Open `http://localhost:5000` after starting the app. Six pages in the sidebar:

| Page | What you'll find |
|------|-----------------|
| **Dashboard** | Overview stats — how much data you have, recent pipeline runs |
| **Data** | Data coverage, level breakdown, import status |
| **Charts** | Interactive candlestick chart with fractal markers |
| **Features** | Feature engineering progress and coverage |
| **Models** | Trained ML models with accuracy/F1/ROC metrics |
| **Backtest** | Run backtests, view results with equity curves and trade logs |

---

## How To Use

### 1. Setup

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the web app
python run.py
```

### 2. Get Data In

```bash
# Import from CSV files in the datasets/ folder
python scripts/import_csv.py
```

Or use the web UI: go to the Dashboard and click **Fetch Data** (requires Binance API keys in `.env`).

### 3. Run the Pipeline

From the web UI, click these buttons in order (or use **Run Pipeline** to do all at once):

1. **Run Indicators** — detects fractals and computes levels
2. **Compute Features** — builds the ML feature set
3. **Train Model** — trains a classifier (try LightGBM)
4. **Predict** — generates predictions on unseen data

### 4. Backtest

**From the web UI:**
Go to `/backtest/`, pick a model, set your parameters, click **Run Backtest**.

**From the command line:**
```bash
python scripts/run_backtest.py --model-id 1 --cash 100000 --risk 0.02
```

### 5. Read the Results

The backtest page shows:
- **Total Return** — did you make or lose money?
- **Sharpe Ratio** — return adjusted for risk (higher is better, >1 is decent)
- **Max Drawdown** — worst peak-to-trough drop (lower is better)
- **Win Rate** — percentage of profitable trades
- **Profit Factor** — gross wins / gross losses (>1 means profitable)
- **Equity Curve** — visual chart of your portfolio over time
- **Trade Log** — every trade with entry, exit, and P&L

---

## Project Structure (Simplified)

```
app/
  services/          ← Business logic (indicators, features, ML, backtesting)
  models/            ← Database tables (candles, levels, predictions, etc.)
  views/             ← Web pages and API endpoints
  templates/         ← HTML pages
scripts/             ← Command-line tools (import data, run backtest)
tests/               ← Automated tests (78 tests)
instance/            ← SQLite database (created automatically)
```

---

## Configuration

Edit `app/config.py` or set environment variables in `.env`:

| Setting | Default | What it does |
|---------|---------|-------------|
| Initial cash | $100,000 | Starting portfolio for backtests |
| Commission | 0.1% | Trading fee per transaction |
| Risk per trade | 2% | Max portfolio risk on a single trade |
| Confidence threshold | 55% | Minimum model confidence to enter a trade |
| SL multiplier | 1.5x ATR | How far the stop-loss is from entry |
| TP multiplier | 3.0x ATR | How far the take-profit is from entry |
