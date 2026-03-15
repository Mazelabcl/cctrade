# ML Pipeline — How the Model Works

## The Question We're Answering

> "On this daily candle, is price about to form a swing high or swing low?"

This is a **multiclass classification**:
- Class 0: no fractal (most candles)
- Class 1: bullish fractal (swing low)
- Class 2: bearish fractal (swing high)

---

## Why This Works (The Hypothesis)

Fractals don't form randomly. They form when price:
1. Reaches a significant level (HTF, Fibonacci, VP)
2. Gets rejected from that level
3. Leaves a 5-candle pattern as evidence

So if we measure **how price is interacting with levels** at candle N, we should be able to predict the fractal at N+1/N+2.

---

## Train / Test Split

| Period | Role |
|--------|------|
| 2020-01-01 → 2024-06-01 | **Training** (model learns patterns) |
| 2024-06-01 → 2025-12-31 | **Test / Unseen** (evaluate real performance) |

The cutoff is strict — the model never sees test data during training.

---

## Features (Current — v1)

Features are computed at candle N using information available before candle N closes.

| Feature | Description |
|---------|-------------|
| `dist_nearest_htf` | % distance from close to nearest HTF level |
| `dist_nearest_sfp` | % distance from close to nearest fractal level |
| `dist_nearest_fib` | % distance from close to nearest Fibonacci level |
| `dist_nearest_vp_poc` | % distance from close to nearest VP POC |
| `at_naked_level` | 1 if price is within 0.5% of a naked level |
| `level_confluence_score` | count of level types within 1% of close |
| `htf_bias` | weekly trend direction (1=bullish, -1=bearish) |
| `candle_body_pct` | body size as % of range (measures indecision) |
| `volume_ratio` | volume vs 20-period average |

---

## Planned Features (v2)

| Feature | Reason |
|---------|--------|
| `dist_vwap` | VWAP is a key institutional reference |
| `anchored_vwap_slope` | Trend direction from last major fractal |
| `fib_confluence` | How many Fib levels cluster near price |
| `level_age_days` | Fresh levels react stronger |
| `touch_count_at_level` | Level strength (naked=strongest) |
| `htf_fractal_proximity` | Is price near a weekly/monthly fractal? |
| `vp_position` | Is price above POC, in value area, or below? |
| `bb_squeeze` | Bollinger Band width (breakout potential) |
| `rsi_divergence` | Price makes new high but RSI doesn't |

---

## Backtest — Current State vs What We Need

### Current backtest (individual_level_backtest.py)
Tests: "when price touches level X → enter long/short → does SL or TP hit?"

This is a **classic bounce trading backtest**. Useful for validating level quality, but **not directly aligned** with fractal prediction.

### What we actually need
**Level Quality Score backtest**: For each level type, when price touches it AND a fractal forms there, how often does the expected move follow?

This gives us:
- Which level types produce the most reliable fractals?
- Which TFs have the best fractal-at-level hit rate?
- This directly feeds into which levels should be weighted more as features

### Backtest → Feature pipeline
```
Level quality backtest
    ↓
hit_rate per level_type (e.g., HTF=62%, VP_POC=71%, Fib_CC=58%)
    ↓
Use hit_rate as feature weight in ML model
    ↓ (or)
Use as standalone signal: "is this a high-quality level?"
```

---

## Model Selection

Current: Random Forest (fast to train, handles class imbalance, interpretable feature importance)

Planned comparison:
- XGBoost (often better on tabular data)
- LightGBM (faster)
- Simple threshold rules (baseline)

Class imbalance: fractals are ~5% of candles → use SMOTE or class weights.

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Precision (fractal) | >60% (avoid false signals) |
| Recall (fractal) | >50% (catch real fractals) |
| F1 score | >55% |
| Baseline (predict all 0s) | ~95% accuracy — meaningless |

**Important**: accuracy is useless here due to class imbalance. Use precision/recall/F1 on classes 1 and 2.
