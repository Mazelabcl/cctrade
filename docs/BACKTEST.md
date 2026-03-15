# Backtest System

## Current State

### Individual Level Backtest (`individual_level_backtest.py`)

Tests: "when price touches level X → enter long/short → does SL or TP hit first?"

**Flow:**
1. Price approaches a level within `tolerance_pct` (default 2%)
2. Detects direction from level context (support = long, resistance = short)
3. Enters trade on next candle open
4. Monitors SL/TP intra-bar using high/low
5. Exits on: TP hit, SL hit, or timeout (N candles)

**Strategies:**
- `FixedPercentStrategy`: SL/TP as fixed % of entry (e.g., 2% SL / 4% TP → 1:2 RR)
- `ATRBasedStrategy`: SL/TP as multiples of ATR (adapts to volatility)

**What it measures:** Win rate, avg profit/loss, Sharpe ratio per level type.

---

## Alignment with Project Goal

The current backtest answers: **"Is this a good bounce trading level?"**

The project goal is: **"Will a fractal form here?"**

These are related but not the same. A level can be a good bounce trade without producing a fractal pattern, and vice versa.

### Gap Analysis

| Question | Current Backtest | Needed for ML |
|----------|-----------------|---------------|
| Does price bounce at this level? | ✅ Yes | Partial |
| Does a fractal form at this level? | ❌ Not tracked | ✅ Critical |
| Which level types predict fractals? | ❌ No | ✅ Critical |
| How often does the post-fractal move follow? | ❌ No | ✅ Important |

---

## What We Actually Need: Level Quality Score

For each level type + timeframe combination, measure:

**Hit Rate** = (levels where a fractal formed within N candles) / (total levels touched)

```
Example:
- HTF_level daily: 68% of touches produce a fractal within 3 candles
- Fib_CC daily:    54% of touches produce a fractal within 3 candles
- VP_POC daily:    71% of touches produce a fractal within 3 candles
```

This gives us:
1. A **reliability ranking** of level types → informs feature weighting
2. A **baseline signal**: "price at VP_POC = high probability fractal forming"
3. Direct input to ML: level quality score as a feature

---

## Planned: Level Quality Backtest

**Algorithm:**
```
For each level L:
    For each candle C that touches L (after L.created_at):
        Look ahead N candles (no lookahead in features — only for evaluation)
        Did a fractal form within N candles at this level?
        → hit = 1 / miss = 0

Per level_type:
    hit_rate = sum(hits) / total_touches
    avg_move_after_fractal = mean(|close[fractal+5] - entry| / entry)
```

**Output per level type:**
- Hit rate (fractal formation probability)
- Average move size after fractal
- Average time to fractal (candles)
- Best timeframe for each level type

---

## How Backtest Feeds Into Features

```
Level Quality Backtest
    ↓
htf_hit_rate = 0.68
vp_poc_hit_rate = 0.71
fib_cc_hit_rate = 0.54
    ↓
Feature: is_at_high_quality_level = 1 if nearest level has hit_rate > 0.65
Feature: weighted_level_score = sum(proximity_weight * hit_rate) for all nearby levels
    ↓
ML Model uses these features to predict fractal class
```

---

## Existing Backtest — Keep or Rewrite?

**Keep** — it's useful for a different question: "if I trade every level bounce, what's my P&L?" This is valuable for:
- Validating that levels are real (not random noise)
- Understanding SL/TP sizing per level type
- Future paper trading strategy evaluation

**Add** the Level Quality Score backtest as a separate module. Both are complementary.
