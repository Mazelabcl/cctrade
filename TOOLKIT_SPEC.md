# TradeKit SDK — Specification for New Agent

## What is this?

This doc tells a NEW agent exactly what to extract from the `cctrade` repo
to build a clean trading toolkit. Read this FIRST, then read the source files.

## Source repo

Path: `C:\Users\aldot\.gemini\antigravity\scratch\claudecode\tradebot`
Branch: `claude/auto-research` (merged to main)
DB: `instance/tradebot.db` (1GB SQLite with all candle + level data)

## What the system does (for a non-trader)

Bitcoin (BTCUSDT) trading system that:
1. Downloads price data (candles) from Binance at multiple timeframes
2. Calculates "levels" — price points where the market is likely to react
3. Detects when price touches a level with "confluence" (multiple levels stacked)
4. Enters a trade at the touch, with a stop loss below the wick
5. Exits via trailing stop or timeout

## Core concepts a new agent MUST understand

### Candles
OHLCV data (Open, High, Low, Close, Volume) at different timeframes:
- 1m, 15m, 1h, 4h — for trading and analysis
- 1d (daily), 1w (weekly), 1M (monthly) — for calculating levels
- ALL timeframes must ALWAYS be synced together. Never partially.

### Levels (the most important concept)
A "level" is a horizontal price line where we expect a reaction.
Types (from Chart Champions methodology):
- **Fractal** — 5-candle swing high/low pattern. The strongest signal.
- **HTF_level** — Where candle direction changes (green→red or red→green)
- **Fib_CC** — Fibonacci 0.639 retracement between fractals (golden pocket)
- **Fib_0.25/0.50/0.75** — Igor's quarter system
- **PrevSession_High/Low/EQ/25/75** — Previous session's key prices
- **PrevSession_VWAP** — Volume-weighted average price of previous session
- **PrevSession_VP_POC/VAH/VAL** — Volume Profile of previous session

### Level lifecycle (CRITICAL — get this wrong and everything breaks)

**Structural levels** (Fractal, HTF, Fib):
- Born: when the pattern completes on a candle
- Source timeframes: Currently tested with daily, weekly, monthly. Other TFs
  (4h, 1h, etc.) could work but haven't been validated yet — this is an OPEN
  RESEARCH QUESTION. The toolkit should make it easy to experiment with any TF.
- Die: when price touches them (`first_touched_at` gets set)
- "Naked" = never touched = still valid for trading
- IMPORTANT: `first_touched_at` depends on which CANDLE TIMEFRAME you're analyzing.
  A daily HTF level at $68,000 might be first touched at 17:00 on 1h candles,
  but at 17:15 on 15m candles, or 16:45 on 4h candles. The touch time should be
  recalculated based on the granularity of the analysis timeframe (use the finest
  available — typically 15m — for accuracy). The level itself is the same across
  all analysis timeframes, only the touch detection granularity changes.

**Mobile levels** (PrevSession, VP):
- Born: at the START of the next session (e.g., Monday's PrevSession_High is
  calculated from Monday's data but becomes available on Tuesday 00:00 UTC)
- So for any given candle, "PrevSession" = the PREVIOUS completed session's data
- Die: when the NEXT level of same type+timeframe is created (`superseded_at`)
  (e.g., Monday's PrevSession_High dies when Tuesday's PrevSession_High is created on Wednesday)
- `first_touched_at` is NOT used for mobile levels — only `superseded_at`
- Only the most recent of each (type, timeframe) is valid at any point in time

### Confluence
When 2+ different level TYPES exist within a configurable zone around the same price.
Example: Fractal_support + Fib_CC + PrevSession_VWAP all near $67,000 = confluence zone.
NOTE: The zone width (tested at 1% but could be 0.5%, 2%, etc.), minimum number of types,
and which types count — ALL of these should be PARAMETERS that the researcher can experiment
with. Nothing is fixed. The toolkit should make it trivial to test different combinations.

### Touch detection
A candle "touches" a level when:
- `candle.low <= level_price * 1.003` AND `candle.close > level_price` → LONG
- `candle.high >= level_price * 0.997` AND `candle.close < level_price` → SHORT

#### Exit strategies (what we've tried so far — NOT exhaustive)
These are the exit strategies implemented so far. New ones can and should be invented:
- **breakeven_trail**: SL stays at original until price moves +be_rr×R, then trails with swing lows
- **atr_trail**: SL trails at price - atr_multiplier × ATR
- **fixed_rr**: Close at fixed risk/reward ratio
- **swing_trail**: SL trails with swing lows/highs immediately

The toolkit should make it EASY to add new exit strategies (just a function that
takes candles + entry + SL and returns exit price/time). Ideas not yet tried:
- Time-based partial closes
- Multi-target exits (close 50% at 2R, let rest run)
- Volatility-adjusted trailing
- Higher-TF confirmation-based exits (hold longer if 4h confirms)

## Proven code to extract (READ THESE FILES)

### Data fetching
- `app/services/data_fetcher.py` → `fetch_candles()` — downloads from Binance API
- `app/tasks/data_sync.py` → `sync_candle_data()` — orchestrates fetching all TFs

### Level calculation
- `app/services/indicators.py` — ALL level calculation functions:
  - `detect_fractals_df()` — fractal detection (5-candle pattern)
  - `calculate_htf_levels()` — HTF direction changes
  - `calculate_fibonacci_levels()` — Fib retracements from fractal pairs
  - `calculate_previous_session_levels()` — session High/Low/EQ/25/75
  - `calculate_vwap_levels()` — VWAP from 1m data
  - `calculate_volume_profile_levels()` — VP POC/VAH/VAL from 1m data

### Touch tracking
- `app/services/level_touch_tracker.py` → `update_touched_levels()` — marks levels as touched
  MUST run after every candle sync. Without it, touched levels appear naked.

### Level loading with proper filtering
- `app/services/level_trade_backtest_db.py` → `load_levels_db(source_timeframes=...)`
  ALWAYS use `source_timeframes=['daily', 'weekly', 'monthly']`
  This function also computes `superseded_at` for mobile levels.

### Confluence detection + trade simulation
- `scripts/autoresearch/confluence_scalper.py` — the main backtest engine:
  - `load_and_cache_data()` — loads candles + levels into numpy arrays
  - `find_touch_entries()` — vectorized touch detection (chunked 2D broadcasting)
  - `score_and_deduplicate()` — confluence scoring + cooldown
  - `_simulate_exit()` — exit simulation (all 4 strategies)

### Alerts
- `scripts/alert_bot.py` — Telegram bot with smart alerts
- `app/services/confluence_signal.py` → `get_market_status()` — nearest levels, distances

## Known issues / lessons learned (DO NOT REPEAT)

1. **Hourly levels contaminated early experiments.** When we calculated levels from 1h candles
   AND from daily/weekly/monthly candles, the 101K hourly levels drowned out the signal from
   the ~7K D-W-M levels. This made HTF and Fib appear useless when they weren't.
   LESSON: When experimenting, be intentional about which SOURCE TIMEFRAMES generate levels.
   Don't mix without purpose. Start with D-W-M (proven), then experiment with adding 4h, 1h
   etc. as a SEPARATE research question. The toolkit should make this a parameter.

2. **`first_touched_at` must be recalculated** every time new candles are added.
   Otherwise levels appear "naked" forever even after price crossed them.

3. **PrevSession levels use `superseded_at`, not `first_touched_at`.**
   They're replaced by the next session's level, not "touched" by price.

4. **The 1m sniper backtest was invalid** — it used future information (looked inside
   a closed 15m candle to find the best 1m entry point). Don't repeat this.

5. **Commission analysis is critical.** 1m scalping looks great gross but dies after
   commissions. Always calculate: position_size = risk / sl_pct, commission = position × 0.06%.

## Validated results (what actually works)

### Best configs (backtest 2017-2025 with corrected data)
| Config | PF | WR | $/mo ($10 risk) | Note |
|--------|-----|------|-----------------|------|
| Fractals only | 784 | 94% | $205 | Very few trades, amazing quality |
| Fractals + Fib_CC | 21.5 | 79% | $198 | More trades, still great |
| Old 4 (Fractal+VWAP+VP_POC) | 2.13 | 35% | $151 | Most trades |

### Forward test 2026 (corrected data)
Fractals alone: only 1-2 trades (all touched in current range).
PrevSession intraday with ATR trail 0.5: PF 2.35, WR 61%, works in chop.

### Universal truths
- Fractals are the strongest signal but rare
- Naked levels >> touched levels
- US+EU session has 3x more edge than Asia
- Trail exits >> fixed RR
- Volume >= 3x at touch = better trades
- Retail indicators (RSI, MACD) add nothing

## Database schema (SQLite)

### candles table
symbol, timeframe, open_time, open, high, low, close, volume,
quote_volume, num_trades, bullish_fractal, bearish_fractal

### levels table
id, price_level, level_type, timeframe, source, created_at,
invalidated_at, first_touched_at, support_touches, resistance_touches

## API keys
Binance API keys are stored in the `settings` table in the DB.
Telegram bot token: also in settings table.

## What the toolkit should provide

```python
# Fetch & cache data
from tradekit import data
data.sync_all()  # downloads all TFs from Binance, updates DB
candles = data.load('15m', start='2026-01-01')

# Calculate levels
from tradekit import levels
all_levels = levels.calculate_all(candles_1d, candles_1w, candles_1M, candles_1m)
# Returns DataFrame with: price, type, timeframe, created_at, first_touched_at, superseded_at

# Find signals
from tradekit import signals
touches = signals.find_confluence(
    candles_15m, all_levels,
    level_types=['Fractal_support', 'Fib_CC'],
    zone_width=0.01, min_types=2, naked_only=True,
    session='us_eu', volume_min=2.0,
)

# Backtest
from tradekit import backtest
result = backtest.simulate(touches, exit='atr_trail', atr_mult=0.5, timeout=20)
print(result.summary())  # PF, WR, $/mo, trade log

# Compare strategies
results = backtest.compare([
    {'name': 'ATR tight', 'exit': 'atr_trail', 'atr_mult': 0.5, 'timeout': 20},
    {'name': 'BE trail', 'exit': 'breakeven_trail', 'be_rr': 2.75, 'timeout': 45},
], touches=touches)

# Alerts
from tradekit import alerts
alerts.setup_telegram(token='...', chat_id='...')
alerts.check_and_notify()  # sends status/approaching/signal/update
```
