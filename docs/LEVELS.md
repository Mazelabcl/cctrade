# Level Types Reference

All levels are stored in the `levels` table with a `level_type` field.

---

## HTF Levels (`HTF_level`)

**What**: Swing highs/lows on higher timeframes (Daily, Weekly, Monthly).
**Logic**: Detect the high/low of each completed period (day/week/month) and store as a level.
**Color**: Cyan (daily), Yellow (weekly), Purple (monthly).
**Why it matters**: Institutions and smart money trade from these zones. Price often reacts strongly.

---

## SFP / Fractals (`Fractal_support`, `Fractal_resistance`)

**What**: 5-candle swing highs and lows on Daily/Weekly/Monthly.
**Logic**: Candle N is a fractal if it has the highest high (or lowest low) compared to N-2, N-1, N+1, N+2.
**Color**: White/light gray.
**Why it matters**: These ARE the target variable. A fractal at a level = confirmation. A failed fractal (SFP = Swing Failure Pattern) = reversal signal.

---

## Fibonacci Levels

Calculated between consecutive fractal pairs (anchor → target).
`range = |target - anchor|`

| Type | Ratio | Formula (bullish) | Color | Origin |
|------|-------|-------------------|-------|--------|
| `Fib_CC` | 0.618–0.636 | `target - 0.628 * range` | Yellow | Daniel CC golden pocket |
| `Fib_0.25` | 0.25 | `target - 0.25 * range` | Red | Igor quarters |
| `Fib_0.50` | 0.50 | `target - 0.50 * range` | Yellow | Igor quarters (midpoint) |
| `Fib_0.75` | 0.75 | `target - 0.75 * range` | Red | Igor quarters |

**Example**: Anchor=40k, Target=50k, range=10k
- CC: 50k - 6,280 = **43,720**
- 0.25: 50k - 2,500 = **47,500**
- 0.50: 50k - 5,000 = **45,000**
- 0.75: 50k - 7,500 = **42,500**

**Direction**: Bearish range uses `anchor + ratio * range` (retracements from top).

---

## Volume Profile (`VP_POC`, `VP_VAH`, `VP_VAL`)

**What**: Volume-weighted price distribution over each completed period (1d/1w/1M).
**Requires**: 1-minute candle data to compute intra-period volume distribution.

| Type | Meaning | Color |
|------|---------|-------|
| `VP_POC` | Point of Control — price with most volume traded | Red (thick) |
| `VP_VAH` | Value Area High — top of 70% volume zone | Blue |
| `VP_VAL` | Value Area Low — bottom of 70% volume zone | Blue |

**Why it matters**: POC = "fair value" magnet. Price tends to return to POC. VAH/VAL = zone boundaries.

---

## VWAP (planned)

**What**: Volume-Weighted Average Price — average price weighted by volume.

| Type | Description |
|------|-------------|
| Session VWAP | Intraday reference (resets each day) |
| Anchored VWAP | From a specific fractal/event — shows trend from that point |
| Weekly VWAP | Rolling 5-day VWAP |

**Why it matters**: Chart Champions uses anchored VWAP as dynamic support/resistance. Institutional reference level.

---

## Touch Tracking

Every time price passes through a level's price zone (`candle.low <= price <= candle.high`), it's counted as a touch.

- `support_touches`: times price came from below
- `resistance_touches`: times price came from above
- `first_touched_at`: timestamp of first touch (used to end touched level lines on chart)
- **Naked level**: 0 total touches — the most important for trading

**Temporal integrity**: only candles AFTER `level.created_at` are counted (no lookahead bias).

---

## Level Confluence

When multiple level types cluster at the same price zone (within ~0.5%), that zone has **confluence** and is considered a stronger reference. This is a planned feature input for the ML model.
