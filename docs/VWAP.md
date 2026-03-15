# VWAP — Volume Weighted Average Price

## What Is VWAP?

VWAP = average price weighted by volume. It answers: "at what price has most volume traded?"

```
VWAP = Σ(price × volume) / Σ(volume)
```

Unlike a simple moving average, VWAP gives more weight to candles with high volume — so it tracks where institutions are actually active.

---

## Types We'll Implement

### 1. Session VWAP (intraday)
Resets each day. Standard reference for 1h/4h charts.
- Price above session VWAP = bullish intraday bias
- Price below session VWAP = bearish intraday bias

### 2. Weekly VWAP
Rolling 5-day VWAP. Good reference for daily charts.

### 3. Anchored VWAP (most important for Chart Champions)
VWAP calculated from a specific starting point — usually a major fractal or market event.

```
Anchored from: last major bullish fractal → tracks "fair value" since that swing low
Anchored from: last major bearish fractal → tracks "fair value" since that swing high
```

**Why Chart Champions uses it:** anchored VWAP from a key fractal shows the trend's "center of gravity." When price returns to anchored VWAP = mean reversion zone = potential fractal forming.

---

## As a Level Type

Store as `Level` with type = `VWAP_session`, `VWAP_weekly`, `VWAP_anchored`.

```
price_level = current VWAP value
timeframe = source timeframe (1h, 4h, 1d)
created_at = anchor point (for anchored VWAP)
```

Note: VWAP is **dynamic** (changes every candle) unlike static HTF/Fib levels. Implementation options:
- Store snapshot at each candle close (one row per candle per VWAP type) — queryable as history
- Compute on-the-fly in features pipeline — simpler, no storage needed

**Recommendation:** compute on-the-fly in features, show on chart via API endpoint.

---

## As Features for ML

| Feature | Description |
|---------|-------------|
| `dist_session_vwap` | % distance from close to session VWAP |
| `dist_weekly_vwap` | % distance from close to weekly VWAP |
| `dist_anchored_vwap_bull` | % distance from anchored VWAP (from last bullish fractal) |
| `dist_anchored_vwap_bear` | % distance from anchored VWAP (from last bearish fractal) |
| `above_vwap` | 1 if close > session VWAP, -1 if below |
| `vwap_slope` | VWAP direction over last 5 candles (trend proxy) |
| `vwap_bandwidth` | distance between anchored bull/bear VWAPs (squeeze indicator) |

---

## Implementation Plan

```python
# In app/services/indicators.py
def calculate_vwap(candles_df, anchor_time=None):
    """
    Calculate VWAP from anchor_time (or session start if None).
    Returns series of VWAP values aligned to candle index.
    """
    if anchor_time:
        df = candles_df[candles_df['open_time'] >= anchor_time].copy()
    else:
        df = candles_df.copy()

    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']
    df['vwap'] = df['tp_vol'].cumsum() / df['volume'].cumsum()
    return df['vwap']
```

Then in `compute_features()`:
- Calculate session VWAP per day using 1h candles
- Calculate weekly VWAP using 1d candles
- Calculate anchored VWAP from last 3 major fractals (daily)
- Add distances as feature columns
