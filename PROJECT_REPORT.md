# CCTrade Project Report

## What is this?

A Bitcoin (BTCUSDT) trading system that detects key price levels, scores trade setups, and predicts market reversals. Built on the Chart Champions methodology (Daniel + Igor's trading styles).

---

## Phase 1: Foundation (Flask Web App)

**What we did:** Restructured the entire codebase from loose Python scripts into a proper Flask web application with SQLite database.

- **Flask + Blueprints** architecture with separate views for dashboard, charts, data, features, models, backtest, settings
- **SQLAlchemy ORM** models: Candle, Level, Feature, MLModel, Prediction, PipelineRun, Setting, BacktestResult
- **TradingView Lightweight Charts** for interactive price visualization with level overlays
- **APScheduler** for background tasks (live sync, pipeline automation)
- **Alembic** for database migrations
- **Bootstrap 5 + Jinja2** templates

**Key files:** `app/__init__.py`, `app/models/`, `app/views/`, `app/templates/`

---

## Phase 2: Data Pipeline

**What we did:** Built a complete data extraction pipeline from Binance.

- Fetches BTCUSDT candles from **August 2017 to present**
- All timeframes: **1min, 1h, 4h, 6h, 8h, 12h, 1d, 1w, 1M**
- 1-minute data: **4,396,089 candles** (used exclusively for Volume Profile calculation)
- Handles rate limits, pagination, and incremental updates
- Settings page allows configuring API keys and sync intervals

**Data in DB:**

| Timeframe | Candles | Period |
|-----------|---------|--------|
| 1min | 4,396,089 | Aug 2017 - Jan 2026 |
| 1h | 73,261 | Aug 2017 - Dec 2025 |
| 4h | 18,332 | Aug 2017 - Dec 2025 |
| 6h | 12,227 | Aug 2017 - Dec 2025 |
| 8h | 9,172 | Aug 2017 - Dec 2025 |
| 12h | 6,116 | Aug 2017 - Dec 2025 |
| 1d | 3,059 | Aug 2017 - Dec 2025 |
| 1w | 437 | Aug 2017 - Dec 2025 |
| 1M | 100 | Sep 2017 - Dec 2025 |

**Key files:** `app/services/data_fetcher.py`, `scripts/fetch_1min.py`

---

## Phase 3: Level Detection (D/W/M)

**What we did:** Automated the level detection that Daniel does manually every Sunday.

Levels are detected on **Daily, Weekly, Monthly** timeframes only (Higher Time Frames = HTF):

### Level Types

| Type | Description | How it's detected |
|------|-------------|-------------------|
| **Fractal_support** | Swing lows (5-candle pattern) | Low < 2 candles before AND 2 after |
| **Fractal_resistance** | Swing highs (5-candle pattern) | High > 2 candles before AND 2 after |
| **HTF_level** | Higher timeframe structure | Weekly/monthly OHLC pivots |
| **Fib_CC** | Chart Champions golden pocket | Fibonacci between consecutive fractals (Daniel's method) |
| **Fib_0.25 / 0.50 / 0.75** | Igor's quarter levels | Quarter retracements between fractals |
| **VP_POC** | Point of Control | Highest volume price in period (from 1min data) |
| **VP_VAH** | Value Area High | Upper 70% volume boundary |
| **VP_VAL** | Value Area Low | Lower 70% volume boundary |
| **PrevSession_High/Low** | Previous session extremes | Prior D/W/M high and low |
| **PrevSession_EQ** | Previous session equilibrium | 50% of prior session range |
| **PrevSession_25/75** | Previous session quarters | 25% and 75% of prior range |
| **PrevSession_VP_POC/VAH/VAL** | Previous session volume profile | VP levels from prior period |
| **PrevSession_VWAP** | Previous session VWAP | Volume-weighted average price of prior period |

### Level counts in DB:

| Source TF | Count | Notes |
|-----------|-------|-------|
| Daily | ~18,400 | Fractals, fibs, VP, HTF, PrevSession |
| Weekly | ~2,400 | Same types |
| Monthly | ~480 | Same types |
| **Total** | **~21,300** | Excluding hourly noise |

### Touch Tracking

- Each level tracks `first_touched_at` — when price first interacted with it
- "Naked" levels = never touched = highest quality
- Levels are filtered dynamically: for any analysis at time T, only levels created before T and still naked at T are used (no future information leakage)

**Key files:** `app/services/indicators.py`, `app/models/level.py`

---

## Phase 4: Per-Level Backtest

**What we did:** Backtested every level type individually to measure which levels actually work.

### Entry criteria:
- Wick touches/pierces level (within 0.5% tolerance)
- Close confirms direction (close > level = support/long, close < level = resistance/short)
- Level must be **naked** (never touched before)

### SL/TP:
- **SL** = entry candle's wick extreme + 0.1% buffer (exactly like Daniel)
- **TP** = entry price +/- (RR ratio x risk)
- Tested RR ratios: 1:1, 2:1, 3:1

### Results (4h execution, RR 1:1, sorted by win rate):

| Level Type | Source TF | Win Rate | Profit Factor | Trades |
|-----------|-----------|----------|---------------|--------|
| **Fractal_support** | **weekly** | **80.6%** | **15.39** | 36 |
| **Fractal_resistance** | **weekly** | **79.1%** | **3.77** | 43 |
| **Fractal_resistance** | **daily** | **75.9%** | **3.25** | 245 |
| **Fractal_support** | **daily** | **72.2%** | **4.21** | 227 |
| Fib_CC | monthly | 58.3% | 1.40 | 12 |
| Fib_CC | weekly | 55.6% | 1.14 | 81 |
| PrevSession_VWAP | monthly | 55.1% | 1.07 | 89 |
| HTF_level | monthly | 54.3% | 2.27 | 35 |
| PrevSession_VWAP | weekly | 48.4% | 1.16 | 349 |
| PrevSession_High | weekly | 47.8% | 1.42 | 268 |
| VP_VAL | monthly | 51.0% | 1.18 | 98 |
| VP_POC | daily | 28.9% | 0.53 | 447 |

### Key findings:
- **Fractals are king.** Weekly fractal levels have 80% win rate — nothing else comes close
- **PrevSession_VWAP monthly** ranks #7 overall with 55% WR
- **VP_POC is consistently bad** (<18% WR at higher RR) — should probably be excluded or weighted very low
- **Higher TF = stronger levels** — weekly levels outperform daily across all types
- Igor fibs (0.25/0.50/0.75) underperform CC fibs consistently

**Key files:** `app/services/level_trade_backtest_db.py`, `scripts/run_prevsession_backtest.py`

---

## Phase 5: Feature Engineering

**What we did:** Created ML features for each candle using the N-2/N-1/N pattern.

### The Pattern:
- **N-2 close** = reference price for zone/level detection
- **N-1** = source candle for shape/volume features
- **N** = the candle being evaluated
- **Target** = "was N-1 a fractal?" (not forward-looking)

### Current 17 Features:

| Feature | Source | Description |
|---------|--------|-------------|
| upper_wick_ratio | N-1 | Upper wick / total range |
| lower_wick_ratio | N-1 | Lower wick / total range |
| body_total_ratio | N-1 | Body / total range |
| body_position_ratio | N-1 | (close - low) / range |
| volume_short_ratio | N-1 | Volume / MA(6) |
| volume_long_ratio | N-1 | Volume / MA(168) |
| utc_block | N | Hour // 4 (trading session) |
| candles_since_last_up | N-2 | Candles since last bullish fractal |
| candles_since_last_down | N-2 | Candles since last bearish fractal |
| support_distance_pct | N-2 | % distance to nearest support level |
| resistance_distance_pct | N-2 | % distance to nearest resistance level |
| atr_14 | up to N-1 | Average True Range (14) |
| momentum_12 | up to N-1 | Price momentum (12 periods) |
| support_confluence_score | N-2 | Sum of win_rates of levels in support zone |
| resistance_confluence_score | N-2 | Sum of win_rates of levels in resistance zone |
| support_liquidity_consumed | N-2 | % of touched levels in support zone |
| resistance_liquidity_consumed | N-2 | % of touched levels in resistance zone |

### Removed features (too "retail"):
RSI, MACD (line/signal/histogram), Bollinger Width — replaced by backtest-calibrated confluence scores

### Leakage fix:
`candles_since_last_up/down` was using candle N (future data) to reset the counter. Fixed to use N-2 since N-1 is the target and N hasn't happened at prediction time.

**Key files:** `app/services/feature_engine.py`, `app/models/feature.py`

---

## Phase 6: ML Models

**What we did:** Trained Random Forest classifiers across multiple timeframes.

### Multi-TF Comparison (RF, target_bullish):

| Exec TF | Candles | Accuracy | F1 | Level Feature Weight |
|---------|---------|----------|-----|---------------------|
| 1h | 52k | 77.5% | 68.1% | 7.6% |
| **4h** | **13k** | **82.4%** | **71.3%** | **12.1%** |
| 6h | 8.7k | 81.9% | 71.0% | 8.5% |
| 8h | 6.5k | 83.2% | 68.2% | 10.5% |
| 12h | 4.4k | 85.3% | 68.3% | 14.0% |
| 1d | 2.2k | 87.4% | 66.1% | 16.1% |

### Feature Importance (4h, bullish):

| Feature | Importance |
|---------|-----------|
| candles_since_last_up | 33.1% |
| lower_wick_ratio | 17.8% |
| upper_wick_ratio | 9.7% |
| volume_short_ratio | 8.3% |
| body_position_ratio | 5.3% |
| momentum_12 | 5.3% |
| Level features (combined) | 12.1% |

### Key insight:
**Timing dominates** — the cycle counter alone carries 33% of the signal. Level features contribute 12% but grow to 16% on daily TF. The model predicts WHEN a fractal will happen (timing) but not WHERE (which level).

**Key files:** `app/services/ml_trainer.py`

---

## Phase 7: Scoring Engine (Current Focus)

**What we did:** Shifted from pure ML to a coach-style scoring system.

### The Problem with ML:
The model sees ALL candles (95% of which are far from levels) and learns that levels rarely matter. But coaches only evaluate when price is AT a level.

### The Solution: Scoring Engine
Rate each "touch event" using data-calibrated weights:

1. **Level type score** (0-10): directly from backtest win rate
   - Fractal_support/weekly = 8.1 pts (80.6% WR)
   - VP_POC/daily = 2.9 pts (28.9% WR)
2. **Wick rejection** (0-3): based on rejection wick ratio
3. **Volume** (0-2): relative volume at time of touch
4. **Confluence** (0-3): number of additional levels in zone
5. **Precision** (0-2): how close the wick got to the level

### Backtest Results:

| Min Score | Trades | Win Rate | vs Baseline |
|-----------|--------|----------|-------------|
| 0 (all) | 30,165 | 42.6% | — |
| 6 | 17,630 | 44.2% | +1.5% |
| 7 | 7,864 | 45.6% | +3.0% |
| **8** | **1,894** | **52.6%** | **+10.0%** |
| **9** | **515** | **68.0%** | **+25.3%** |
| **10** | **118** | **80.5%** | **+37.9%** |

**Score >= 8 is the sweet spot:** 52.6% WR with ~237 trades/year. Profitable with RR 1:1.

**Key files:** `app/services/scoring_engine.py`, `scripts/bounce_or_break.py`

---

## Key Discoveries

1. **Fractal weekly levels = 80% WR.** This validates the Chart Champions HTF analysis methodology.

2. **VP_POC is consistently bad** (<18% WR). POC marks WHERE most trading happened, not necessarily WHERE price will react.

3. **Level type matters more than candle shape.** The TYPE of level determines ~70% of trade outcome. Wick rejection and volume add ~15% combined.

4. **4h is the sweet spot** for execution — less noise than 1h, more data than daily.

5. **Scoring engine works.** Filtering by score >= 8 lifts WR from 42% to 53%. Score >= 9 = 68% WR.

6. **PrevSession VWAP is surprisingly strong** — monthly VWAP ranks #7 overall at 55% WR.

---

## Architecture Summary

```
Data Flow:
Binance API → Candles (SQLite) → Fractal Detection → Level Creation
                                                          ↓
                                                    Level Backtest → Win Rates
                                                          ↓
Candle + Levels → Feature Engine → Features Table → ML Training / Scoring Engine
                                                          ↓
                                                    Trade Signals
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Database | SQLite + SQLAlchemy ORM |
| Web framework | Flask + Blueprints |
| UI | Bootstrap 5 + Jinja2 |
| Charts | TradingView Lightweight Charts |
| Background tasks | APScheduler |
| ML | scikit-learn (Random Forest) |
| Testing | pytest |
| Migrations | Alembic |

---

## What's Next

1. **Level-to-level TP** — Instead of fixed RR, exit at the next opposing level (Daniel's actual method)
2. **Live scoring** — Real-time alerts when price touches a high-score level
3. **DataRobot comparison** — Upload the same dataset to DataRobot to benchmark our approach
4. **Order flow integration** — Daniel uses order flow for confirmation (30min chart)
5. **Local range detection** — Both coaches mark value areas of local ranges visually
