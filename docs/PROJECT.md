# BTCUSDT Fractal Predictor — Project Overview

## Core Objective

Predict **swing highs and swing lows (fractals)** on BTCUSDT using multi-timeframe technical levels as features.

This is a **classification problem**:
- `0` = no fractal on this candle
- `1` = bullish fractal (swing low — price bottom, potential long entry)
- `2` = bearish fractal (swing high — price top, potential short entry)

The hypothesis: **fractals don't form randomly — they form at meaningful price levels** (HTF structure, Fibonacci retracements, volume profile nodes). If we can measure how price is interacting with these levels at candle N, we can predict whether candle N+1/N+2 will be a fractal.

---

## Ground Truth (Target Variable)

A **fractal** is a 5-candle pattern:
- **Bullish fractal**: candle N has a lower low than candles N-2, N-1, N+1, N+2
- **Bearish fractal**: candle N has a higher high than the same neighbors

These are detected by the indicator pipeline and stored in `Candle.bullish_fractal` / `Candle.bearish_fractal`.

---

## Data Pipeline

```
Binance API
    ↓
fetch_candles()          → Candle table (1h, 4h, 1d, 1w, 1M)
    ↓
run_indicators()
  ├── Fractal detection  → Candle.bullish_fractal / bearish_fractal
  ├── HTF levels         → Level table (type=HTF_level)
  ├── Fibonacci levels   → Level table (type=Fib_CC, Fib_0.25, Fib_0.50, Fib_0.75)
  ├── Volume Profile     → Level table (type=VP_POC, VP_VAH, VP_VAL)
  └── Touch tracking     → Level.support_touches, resistance_touches, first_touched_at
    ↓
compute_features()       → Feature table (1 row per 1d candle)
    ↓
train_model()            → MLModel table
    ↓
predict()                → Prediction table
```

---

## Technology Stack

| Concern | Choice |
|---------|--------|
| Language | Python 3.11 |
| Database | SQLite + SQLAlchemy ORM |
| Web UI | Flask + Bootstrap 5 dark theme |
| Charts | TradingView Lightweight Charts v4.1.3 |
| Background jobs | APScheduler |
| ML | scikit-learn (current), extensible |
| Testing | pytest (112 tests) |
| Migrations | Alembic |

---

## Trading Context (Chart Champions Methodology)

The level system follows the **Chart Champions** framework:
- Price moves from level to level
- Fractals form **at** levels (that's the edge we're modeling)
- **Naked levels** (never touched) are the most important
- **Confluence** (multiple level types at same price) = stronger signal
