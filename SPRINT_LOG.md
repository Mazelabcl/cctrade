# Sprint Log — BTCUSDT Fractal Prediction System

## Sprint 1: Foundation & ML (completed)
**Branch:** `main`
**Period:** Initial setup

- Project restructure: Flask + SQLAlchemy + Blueprints
- Data pipeline: Binance fetcher, CSV import, multi-TF candles
- Indicator pipeline: Fractals, HTF levels, Fibonacci (CC + Igor), Volume Profile
- Feature engineering: N-2/N-1/N pattern, distance to levels, confluence scores
- ML: Random Forest for fractal prediction (4h: 84% acc, 77% F1)
- UI: Dashboard, Charts (TradingView Lightweight Charts), Settings, Live Sync

---

## Sprint 2: Backtesting & Scoring (completed)
**Branch:** `claude/bounce-or-break`
**Commits:** `de34264` → `b709b5b`

### What was built
- **Individual Level Backtests** — test each (level_type, timeframe) combo independently
  - Wick-based SL: SL at candle wick + 0.1% buffer
  - Configurable RR: 1:1, 2:1, 3:1
  - 167k+ trades across all combos
- **PrevSession levels** — High/Low/EQ/25%/75% for D/W/M periods
- **VWAP** — Volume-weighted average price from 1-minute data
- **PrevSession VP** — POC/VAH/VAL from 1-minute candles per session
- **Scoring Engine** — rates trade setups like a coach's checklist
  - Level type score (WR from backtests × 10)
  - Wick rejection (0-3), Body position (0-2)
  - Confluence (0-5), Precision (0-2)
  - Total range: 0-20 points
- **Trade Explorer** — visual trade browser with:
  - TradingView chart with entry/exit markers
  - Per-type level toggles (checkbox per level type)
  - Entry/exit condition annotations
  - Manual exit mode (click chart to simulate custom TP)
  - Directional analysis zone visualization
  - Score breakdown sidebar
- **Analytics Dashboard** — Plotly-based with 4 tabs:
  - Backtest Results: WR heatmap by (level_type × timeframe)
  - Feature Distribution: histograms, scatter plots, stats cards
  - Level Density: active levels over time (structural vs mobile)
  - Scoring Analysis: score vs PnL, violin plots, equity curves

### Key discoveries
1. **91k levels with 0.002% gaps** — hourly levels are noise. Filtered to D/W/M only.
2. **ML precision for fractals is ~50%** — coin flip. Pivoted to scoring engine.
3. **VP_POC is terrible** — <30% WR across all timeframes. Avoid.
4. **Fractals are gold** — 77-80% WR (daily), but few trades (~400-500 each)
5. **PrevSession/VP ~50% WR** — coin flip noise, but many trades

### Bugs found & fixed
- **Feature leakage**: `candles_since_last_up/down` used candle N (future). Fixed to N-2.
- **Naked level bug (CRITICAL)**: PrevSession/VP levels NEVER had `first_touched_at` set → ALL 23k+ PrevSession and 9k+ VP levels were treated as "naked" forever. This inflated confluence scores and made level proximity meaningless.
  - **Fix**: Mobile supersession — only the most recent level per (type, timeframe) is valid. Older ones are automatically superseded when a new session creates the replacement.
  - **Impact on ML**: Minimal (~0.2% accuracy change) — RF depends on candle ratios, not level features.
  - **Impact on scoring**: Scores were inflated (avg 14.2/20). Filtering by min_score barely reduced trade count.
  - **Impact on density**: Before fix: 99.4% of candles within 0.5% of a level. After: 82.3%.

### Recomputed results (post-fix)
| Component | Metric | Value |
|-----------|--------|-------|
| Features 4h | Count | 18,330 |
| Features 1h | Count | 73,259 |
| RF 4h Bullish | Accuracy | 84.4% |
| RF 4h Bearish | Accuracy | 82.9% |
| RF 1h Bullish | Accuracy | 78.9% |
| RF 1h Bearish | Accuracy | 78.0% |
| Scoring (4h, >=7) | WR / Trades | 51.1% / 18,123 |
| Fractals only | WR | 77-80% |
| PrevSession only | WR | 48-53% |

### Open questions
- Scoring threshold doesn't filter well (avg score 14.2, min 7 = barely filters)
- Need to recalibrate score ranges or change scoring formula
- 82.3% of candles still within 0.5% of a level — is this still too dense?

---

## Sprint 3: Exit Strategy Optimization (next)
**Branch:** `claude/exit-optimization` (to be created)

### Goals
Model the coach's exit criteria. Currently we model ENTRY well (Fractals 80% WR), but EXIT is naive (fixed RR). Trades that could go 20:1 are closed at 1:1.

### Planned work

#### Phase 1: MFE Analysis (Maximum Favorable Excursion)
- For each existing trade: remove TP, let run until SL hit or dataset end
- Measure: max_rr_achieved, candles_to_max, max_price
- Answer: "what % of trades reach 5:1? 10:1? 20:1?"
- This determines if there's actual edge in holding longer

#### Phase 2: Multi-TP Manual Annotation Tool
- Extend Trade Explorer: click chart to add TP1, TP2, TP3
- Save annotations to DB with features at each TP point
- After 50-100 annotations: reverse-engineer common patterns
  - Was TP1 at a fractal? Fib extension? VP level?
  - Was exit triggered by volume shift? Structure break?

#### Phase 3: Exit Strategy Backtest Framework
- Break-even stop: move SL to entry after reaching 1:1
- Trail stop by swings: SL follows swing lows (LONG)
- Trail stop ATR: SL = price - 1.5×ATR
- Partial TPs: close 50% at TP1, trail the rest
- Time stop: close after N candles without progress

#### Phase 4: AutoResearch Loop (Karpathy-inspired)
- Agent modifies exit rules → runs 5-min backtest → evaluates → keeps or discards
- Automatic overnight feature discovery and parameter optimization
- evaluate.py + experiment.py + loop

#### Phase 5: Compound / Position Sizing
- When to add to a winning position
- Pyramiding rules: add on pullbacks within trend
- Dynamic position sizing based on setup score

### Coach's wisdom to model
- "Even 30% WR can be profitable with good SL/TP management"
- "VAL to VAH rotation" — price tends to rotate between value area extremes
- Partial TPs at structure levels, trail the rest
- Compound on confirmed trend continuation

### MFE Key Discovery (Sprint 3.1)
**48,083 trades analyzed (4h, RR 1:1)**

Trades reaching each RR threshold:
- >= 1:1 => 42.8%, >= 5:1 => 16.2%, >= 10:1 => 9.8%, >= 20:1 => 5.5%, >= 50:1 => 3.1%

By level type (median max RR):
- Fractal_support: **4.0:1** median, 43% reach >=5:1, avg 892 candles to max
- Fractal_resistance: **4.6:1** median, 48% reach >=5:1, avg 144 candles to max
- PrevSession/VP: **0.3-0.8:1** median — confirms they're noise for swing trades

**Conclusion**: Fractals are swing trades (hold days-weeks), PrevSession are scalps.
Two completely different systems should be built, not one mixed scoring engine.

---

## Ideas Backlog ("Volas")

Ideas for future exploration, not prioritized yet:

1. **Sniper Timeframe Optimization**: If 4h Fractal gives 4:1 median RR with 2% SL, detecting the same signal on 1h (0.5% SL) gives 16:1 RR for the same price move. Descending through timeframes (4h -> 1h -> 30m -> 15m) for precision entry while using HTF for direction.

2. **Fractal-to-Fractal System**: Use bullish fractal as entry, bearish fractal as exit. Pure structure-based trading without fixed RR.

3. **Volume Divergence as Exit Signal**: Track if volume at TP point diverges from entry volume. Possible pattern: "exit when sell volume reaches 30% of entry buy volume."

4. **AutoResearch Loop**: Karpathy-inspired overnight experiment runner. Agent modifies exit rules, runs backtest, evaluates, keeps or discards. 100 experiments/night.

5. **Two-System Architecture**: System A (Fractal Swings) — few trades, high RR, trail stop. System B (Session Rotations) — many trades, low RR, fixed TP. Never mix them.
