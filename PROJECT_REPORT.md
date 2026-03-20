# BTCUSDT Fractal Prediction System — Project Report

**Authors:** Aldo + Claude Code (AI pair programming)
**Date:** March 2026
**Repo:** github.com/Mazelabcl/cctrade
**Stack:** Python, Flask, SQLite, SQLAlchemy, scikit-learn, TradingView Lightweight Charts

---

## 1. What Is This Project?

A system that detects key price levels on Bitcoin (BTCUSDT) across multiple timeframes, evaluates their historical performance via backtesting, and scores trade setups when price touches those levels — mimicking how professional traders (Daniel & Igor from Chart Champions) analyze the market.

The goal: **automate the "checklist" a coach uses before entering a trade**, and back it with data.

---

## 2. What Was Built (Pipeline Overview)

```
Binance API -> Candle Data (1min to 1M, 2017-2025)
    |
Fractal Detection (swing highs/lows on all TFs)
    |
Level Generation:
  - HTF Levels (Daily/Weekly/Monthly fractals)
  - Fibonacci (CC golden pocket + Igor 0.25/0.50/0.75)
  - Volume Profile (POC/VAH/VAL from 1-minute data)
  - Previous Session Levels (high/low/EQ/25%/75%/VP/VWAP for D/W/M)
    |
Backtest Engine (per-level, wick-based SL, multi-RR)
    |
Scoring Engine (rates setups using backtest win rates)
    |
ML Models (Random Forest, target = "was N-1 a fractal?")
    |
Flask Web App (dashboard, charts, settings, live sync)
```

---

## 3. Data Foundation

| Timeframe | Candles | Range | Fractals |
|-----------|---------|-------|----------|
| 1 min | 4,396,089 | Aug 2017 - Dec 2025 | N/A (used for VP only) |
| 1 hour | 73,261 | Aug 2017 - Dec 2025 | 10,386 bull / 10,389 bear |
| 4 hour | 18,332 | Aug 2017 - Dec 2025 | 2,614 bull / 2,672 bear |
| 6 hour | 12,227 | Aug 2017 - Dec 2025 | 1,746 bull / 1,766 bear |
| 8 hour | 9,172 | Aug 2017 - Dec 2025 | 1,284 bull / 1,322 bear |
| 12 hour | 6,116 | Aug 2017 - Dec 2025 | 848 bull / 849 bear |
| 1 day | 3,059 | Aug 2017 - Dec 2025 | 420 bull / 409 bear |
| 1 week | 437 | Aug 2017 - Dec 2025 | 54 bull / 55 bear |
| 1 month | 100 | Sep 2017 - Dec 2025 | 13 bull / 10 bear |

All data fetched via Binance public API. No API keys required for historical data.

---

## 4. Level Detection

Levels are generated from **Daily, Weekly, Monthly** timeframes (HTF = Higher Time Frame):

| Level Type | Description | Count (D/W/M) |
|-----------|-------------|----------------|
| Fractal_support | Confirmed swing lows (5-candle pattern) | ~650 |
| Fractal_resistance | Confirmed swing highs | ~650 |
| HTF_level | Multi-timeframe structural levels | ~1,300 |
| Fib_CC | Daniel's golden pocket (CC method) | ~2,100 |
| Fib_0.25 / 0.50 / 0.75 | Igor's quarter fibs | ~6,300 |
| VP_POC / VP_VAH / VP_VAL | Volume Profile (from 1-min data) | ~2,000 |
| PrevSession_* | Previous session high/low/EQ/25%/75%/VP/VWAP | ~28,000 |

**Key discovery:** Hourly levels (~67,000) were found to be pure noise. Removing them and keeping only D/W/M levels dramatically improved signal quality.

**Naked levels:** Only levels that have never been touched by price are considered valid. The system tracks `first_touched_at` and filters dynamically -- for any given candle, only levels that were naked at that point in time are used.

---

## 5. Backtest Results -- Which Levels Work?

Backtest methodology:
- **Entry:** Wick touches/pierces level + close confirms direction
- **SL:** Entry candle's wick extreme + 0.1% buffer
- **TP:** Fixed RR ratios (1:1, 2:1, 3:1)
- **Naked only:** Level must not have been previously touched

### Top 15 Levels by Win Rate (4h execution, RR 1:1)

| # | Level Type | Source TF | Win Rate | Profit Factor | Trades |
|---|-----------|-----------|----------|--------------|--------|
| 1 | Fractal_support | weekly | **80.6%** | 15.39 | 36 |
| 2 | Fractal_resistance | weekly | **79.1%** | 3.77 | 43 |
| 3 | Fractal_resistance | daily | **75.9%** | 3.25 | 245 |
| 4 | Fractal_support | daily | **72.2%** | 4.21 | 227 |
| 5 | Fib_CC | monthly | 58.3% | 1.40 | 12 |
| 6 | Fib_CC | weekly | 55.6% | 1.14 | 81 |
| 7 | PrevSession_VWAP | monthly | 55.1% | 1.07 | 89 |
| 8 | HTF_level | monthly | 54.3% | 2.27 | 35 |
| 9 | Fib_0.75 | monthly | 53.8% | 1.61 | 13 |
| 10 | VP_VAL | monthly | 51.0% | 1.18 | 98 |
| 11 | PrevSession_VWAP | weekly | 48.4% | 1.16 | 349 |
| 12 | PrevSession_High | weekly | 47.8% | 1.42 | 268 |
| 13 | PrevSession_75 | monthly | 47.3% | 0.97 | 91 |
| 14 | PrevSession_VP_VAH | weekly | 47.2% | 1.28 | 337 |
| 15 | PrevSession_25 | weekly | 46.6% | 1.29 | 296 |

### Worst Performers

| Level Type | Source TF | Win Rate | Trades |
|-----------|-----------|----------|--------|
| VP_POC | daily | 17.6% | 3,165 |
| VP_POC | monthly | 16.7% | 144 |
| Fib_0.25 | monthly | 22.1% | 231 |

### Key Takeaway

**Fractal levels are king.** Weekly fractals have 80% win rate with profit factor >15. Daily fractals at 72-76%. Everything else is secondary. VP_POC is consistently terrible (<18% WR) -- avoid.

---

## 6. ML Experiments

### 6.1 Feature Engineering

17 features computed for each candle:

| Category | Features | Source |
|----------|---------|--------|
| Candle shape | upper_wick_ratio, lower_wick_ratio, body_total_ratio, body_position_ratio | N-1 candle |
| Volume | volume_short_ratio (vs 6-bar MA), volume_long_ratio (vs 168-bar MA) | N-1 candle |
| Timing | utc_block, candles_since_last_up, candles_since_last_down | N-2 based |
| Level distance | support_distance_pct, resistance_distance_pct | N-2 close |
| Level quality | support/resistance_confluence_score (WR-weighted), support/resistance_liquidity_consumed | N-2 close |
| Volatility | atr_14, momentum_12 | up to N-1 |

**Target:** "Was N-1 a fractal?" (binary: 0=no, 1=yes)

**Leakage fix:** Timing counters (candles_since_last_up/down) were found to leak future info when based on candle N. Fixed by lagging to N-2 (N-1 is the target, so can't use it either). This dropped accuracy from 100% (leaked) to 77-83% (real).

**Removed "retail" indicators:** RSI, MACD, Bollinger Bands were removed. ATR and momentum kept.

### 6.2 Multi-Timeframe Comparison

| Exec TF | Candles | Acc (bull) | F1 (bull) | Level Weight |
|---------|---------|-----------|----------|-------------|
| 1h | 73k | 77.5% | 68.1% | 7.6% |
| **4h** | **18k** | **82.4%** | **71.3%** | **12.1%** |
| 6h | 12k | varies | varies | 8.5% |
| 12h | 6k | varies | varies | ~14% |
| 1d | 3k | varies | varies | 16.1% |

**Finding:** 4h is the sweet spot -- good accuracy with enough data. Level features become more important at higher TFs (7.6% at 1h to 16.1% at 1d).

### 6.3 Feature Importance (Random Forest, 4h)

**Bullish fractal prediction:**
1. candles_since_last_up -- 33% (cycle timing dominates)
2. lower_wick_ratio -- 18% (wick shape matters)
3. upper_wick_ratio -- 10%
4. volume_short_ratio -- 8%
5. Level features combined -- 12%

### 6.4 Honest Assessment

The ML model achieves 82% accuracy but has a hidden problem:
- **Precision for fractals: ~50%** -- when it says "fractal", it's right half the time
- **The model predicts WHEN** (timing) but not WHERE (which level will hold)
- **Cycle timing dominates** -- the model mostly learned "a fractal is due based on time since last one"

This led us to pivot from pure ML to a **scoring engine** approach.

---

## 7. Scoring Engine -- The Key Innovation

Instead of predicting fractals on every candle, the scoring engine only evaluates moments when **price touches a level** -- like a coach does.

### How It Works

Each "touch event" gets a score (0-12+ scale):

| Factor | Weight | Source |
|--------|--------|--------|
| **Level type quality** | 0-8 pts | Backtest win rate x 10 (e.g., Fractal weekly 80% = 8.0 pts) |
| Wick rejection | 0-3 pts | How much the entry candle's wick rejected the level |
| Volume | 0-2 pts | Relative volume vs 20-bar MA |
| Confluence | 0-3 pts | Other levels in the same zone |
| Touch precision | 0-2 pts | How close the wick got to the exact level |

**All weights are calibrated from actual backtest data** -- no arbitrary numbers.

### Scoring Engine Backtest Results (4h, RR 1:1)

| Min Score | Trades | Win Rate | Improvement vs Base |
|-----------|--------|----------|---------------------|
| 0 (no filter) | 30,165 | 42.6% | baseline |
| 6 | 17,630 | 44.2% | +1.5% |
| 7 | 7,864 | 45.6% | +3.0% |
| **8** | **1,894** | **52.6%** | **+10.0%** |
| **9** | **515** | **68.0%** | **+25.3%** |
| **10** | **118** | **80.5%** | **+37.9%** |

### Practical Trading Rule

**"Only enter trades with score >= 8"**
- 1,894 trades over 8 years = ~237 trades/year = ~1 trade every 1.5 days
- 52.6% win rate at RR 1:1 = profitable
- Score >= 9 for high-conviction: 68% WR but only ~64 trades/year

---

## 8. Level-to-Level TP (Daniel's Method)

Instead of fixed RR, TP is set at the next level in the opposite direction:

| Filter | Trades | WR | PF | Avg PnL |
|--------|--------|-----|-----|---------|
| All L2L | 4,499 | 32.8% | 0.77 | -0.10% |
| Score >= 8 | 290 | 28.3% | 1.04 | +0.15% |
| **Score >= 9** | **36** | **47.2%** | **3.19** | **+0.65%** |
| RR 1-3 + Score >= 7 | 468 | 34.4% | 1.01 | -0.02% |

L2L is conceptually closer to how coaches trade but the fixed RR 1:1 + scoring >= 8 remains the strongest quantified system so far.

---

## 9. Key Discoveries

1. **Fractal levels are king** -- 72-81% WR. Nothing else comes close. This validates the Chart Champions methodology.

2. **VP_POC is consistently terrible** -- <18% WR across all timeframes. Avoid trading the POC.

3. **4h is the optimal execution timeframe** -- balances signal quality with data quantity. 1h is too noisy, daily has too few data points.

4. **Level type matters more than candle shape** -- the type of level explains ~70% of trade outcome. Wick rejection, volume, and body shape add ~5-8% each.

5. **Naked levels only** -- levels that have been previously touched lose their predictive power. Fresh, untouched levels are key.

6. **The scoring engine works** -- filtering by score >= 8 turns a 42.6% WR system into 52.6% WR. Score >= 9 achieves 68% WR.

7. **ML adds value for timing, not level selection** -- the Random Forest model's main contribution is cycle timing (candles since last fractal). Level-specific features need the scoring engine approach.

8. **Previous Session VWAP is the best "new" level** -- monthly PrevSession VWAP ranks #7 overall at 55.1% WR.

9. **Hourly levels are noise** -- removing 67,000 hourly levels and keeping only D/W/M improved everything.

10. **Candle shape effects are modest** -- upper wick >50% improves SHORT WR by ~8% max. Volume >3x adds ~5%. These are real but secondary to level quality.

---

## 10. Technical Architecture

### Web Application
- Flask with Blueprints (dashboard, data, charts, features, models, backtest, settings)
- Bootstrap 5 + TradingView Lightweight Charts
- APScheduler for background tasks (live sync from Binance)
- SQLite database with Alembic migrations

### Services Layer (Pure Python)
- `data_fetcher.py` -- Binance API integration
- `indicators.py` -- Fractal detection, HTF levels, Fibonacci, Volume Profile, PrevSession, VWAP
- `feature_engine.py` -- 17 ML features with naked level filtering
- `ml_trainer.py` -- Random Forest / XGBoost / LightGBM training
- `level_trade_backtest_db.py` -- Per-level backtest with multi-RR
- `scoring_engine.py` -- Trade setup scoring from backtest win rates

### Database
- ~4.5M candles across 9 timeframes
- ~120k levels (D/W/M + PrevSession + VWAP)
- ~110k backtest trades
- ~120k computed features

---

## 11. What's Next

1. **Refine the scoring engine** -- test different score thresholds, add partial TP logic
2. **Live signal generation** -- real-time alerts when score >= 8 on current candles
3. **DataRobot comparison** -- export the feature dataset and compare with AutoML
4. **Local range detection** -- identify consolidation zones like the coaches do visually
5. **Order flow integration** -- add depth/footprint data for entry confirmation
6. **Paper trading** -- forward-test the scoring system on live data

---

## 12. Repository Structure

```
app/
  __init__.py          # Flask app factory
  config.py            # Configuration
  extensions.py        # SQLAlchemy, APScheduler
  models/              # ORM models (Candle, Level, Feature, MLModel, etc.)
  services/            # Business logic
    indicators.py      # Level detection (fractals, fibs, VP, PrevSession, VWAP)
    feature_engine.py  # ML feature computation
    ml_trainer.py      # Model training
    scoring_engine.py  # Trade setup scoring
    level_trade_backtest_db.py  # Per-level backtesting
  views/               # Flask routes (Blueprints)
  templates/           # Jinja2 + Bootstrap 5
  tasks/               # Background jobs

scripts/
  import_csv.py               # CSV data import
  fetch_1min.py               # 1-minute data for VP
  tf_experiment.py            # Multi-TF ML comparison
  proximity_experiment.py     # Level proximity analysis
  bounce_or_break.py          # Bounce/break ML experiment
  run_prevsession_backtest.py # PrevSession level backtest
  run_scoring_pipeline.py     # Full scoring pipeline

instance/              # SQLite database (gitignored)
legacy/                # Original code (reference)
```

---

*Generated with Claude Code -- March 2026*
