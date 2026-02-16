# Individual Level Backtest System

**Branch**: `feature/individual-level-backtest`
**Objetivo**: Backtest individual de cada tipo de nivel para determinar cuáles son más efectivos y generar datasets para ML.

---

## 🎯 Objetivo del Sistema

Probar cada tipo de nivel por separado para:
1. **Determinar efectividad**: ¿Qué niveles tienen mejor win rate / profit factor?
2. **Generar data para ML**: Crear datasets con características de niveles ganadores vs perdedores
3. **Optimizar parámetros**: Encontrar mejores SL/TP/estrategias por tipo de nivel
4. **Validar visualmente**: Dashboard con replay para verificar que los niveles se detectan correctamente

---

## 📊 Tipos de Niveles a Probar

### Por Timeframe de Origen:
- **HTF**: `HTF_hourly`, `HTF_daily`, `HTF_weekly`, `HTF_monthly`
- **Fractal High**: `Fractal_High_hourly`, `Fractal_High_daily`, `Fractal_High_weekly`
- **Fractal Low**: `Fractal_Low_hourly`, `Fractal_Low_daily`, `Fractal_Low_weekly`
- **Fibonacci 0.50**: `Fib_0.50_daily`, `Fib_0.50_weekly`, `Fib_0.50_monthly`
- **Fibonacci 0.618**: `Fib_0.618_daily`, `Fib_0.618_weekly`, `Fib_0.618_monthly`
- **Fibonacci 0.786**: `Fib_0.786_daily`, `Fib_0.786_weekly`, `Fib_0.786_monthly` *(nuevo)*
- **Volume Profile POC**: `VP_POC_daily`, `VP_POC_weekly`, `VP_POC_monthly`
- **Volume Profile VAH**: `VP_VAH_daily`, `VP_VAH_weekly`, `VP_VAH_monthly`
- **Volume Profile VAL**: `VP_VAL_daily`, `VP_VAL_weekly`, `VP_VAL_monthly`

**Total**: ~45-50 combinaciones a probar

---

## 🎲 Estrategias de Entrada/Salida

Cada backtest puede usar una de estas estrategias:

### **1. fixed_percent** (Baseline)
- **SL**: X% del entry price (ej: 1%, 2%, 3%)
- **TP**: Y% del entry price (ej: 2%, 4%, 6%) o R:R ratio (1:1, 2:1, 3:1)
- **Timeout**: N candles (ej: 50, 100, 200)

### **2. atr_based** (Volatility-adjusted)
- **SL**: entry ± (ATR × multiplier) — ej: ATR × 1.5
- **TP**: entry ± (ATR × multiplier × RR) — ej: ATR × 3
- **Timeout**: N candles

### **3. level_based** (Price structure)
- **SL**: Próximo nivel opuesto (ej: si LONG en support, SL = support anterior)
- **TP**: Próximo nivel a favor (ej: si LONG, TP = resistance arriba)
- **Timeout**: N candles o hasta hit de nivel

### **4. fractal_based** (Swing structure)
- **SL**: Fractal opuesto más cercano
- **TP**: R:R basado en distancia del SL (ej: 2:1)
- **Timeout**: N candles

**Prioridad inicial**: Implementar estrategias 1 y 2. Luego agregar 3 y 4.

---

## 🚪 Regla de Entrada Unificada

**Para TODOS los niveles** (HTF, Fractal, Fib, VP):

### **LONG Signal** (Bounce from Support)
```python
# Condiciones:
1. Vela toca el nivel: candle.low <= level + tolerance
2. Vela cierra por encima: candle.close > level
→ Nivel actuó como soporte ✅
```

### **SHORT Signal** (Rejection from Resistance)
```python
# Condiciones:
1. Vela toca el nivel: candle.high >= level - tolerance
2. Vela cierra por debajo: candle.close < level
→ Nivel actuó como resistencia ✅
```

### **Parámetros**:
- `entry_tolerance_pct`: 0.5% - 3% (default: 2%)
- Solo se considera el primer toque de cada nivel (naked levels prioritarios)

---

## 🗄️ Database Schema

### **Table: `individual_level_backtest`**
```sql
id                          INTEGER PRIMARY KEY
level_type                  VARCHAR(50)    -- "HTF", "Fractal_High", "Fib_0.618", etc.
level_source_timeframe      VARCHAR(10)    -- "1h", "4h", "1d", "1w" (timeframe del nivel)
trade_execution_timeframe   VARCHAR(10)    -- "1h", "4h", "1d" (timeframe donde se ejecutan trades)
strategy_name               VARCHAR(50)    -- "fixed_percent", "atr_based", etc.
parameters                  JSON           -- {sl_pct: 2, tp_pct: 4, timeout: 50, ...}

start_date                  DATETIME
end_date                    DATETIME
status                      VARCHAR(20)    -- "running", "completed", "failed"

-- Métricas
total_trades                INTEGER
winning_trades              INTEGER
losing_trades               INTEGER
win_rate                    FLOAT          -- %
profit_factor               FLOAT          -- total_wins / total_losses
sharpe_ratio                FLOAT
max_drawdown                FLOAT          -- %
total_pnl                   FLOAT          -- $
avg_win                     FLOAT          -- $
avg_loss                    FLOAT          -- $
avg_trade_duration          FLOAT          -- candles

created_at                  DATETIME
finished_at                 DATETIME
error_message               TEXT
```

### **Table: `individual_level_trade`**
```sql
id                  INTEGER PRIMARY KEY
backtest_id         INTEGER FK → individual_level_backtest.id
level_id            INTEGER FK → levels.id  -- El nivel específico que generó la señal

entry_time          DATETIME
entry_price         FLOAT
direction           VARCHAR(10)    -- "LONG" or "SHORT"
stop_loss           FLOAT
take_profit         FLOAT

exit_time           DATETIME
exit_price          FLOAT
exit_reason         VARCHAR(20)    -- "TP_HIT", "SL_HIT", "TIMEOUT"

pnl                 FLOAT          -- $
pnl_pct             FLOAT          -- %
candles_held        INTEGER

-- Features para ML
entry_volatility    FLOAT          -- ATR at entry
volume_ratio        FLOAT          -- Volume vs MA
distance_to_level   FLOAT          -- % distance from exact level
zone_confluence     INTEGER        -- # of levels in zone

metadata_json       JSON           -- Additional context
```

---

## 🏗️ Implementation Plan

### ✅ Phase 0: Pre-requisitos
- [ ] Agregar Fibonacci 0.786 a `app/services/indicators.py`
- [ ] Re-run indicators pipeline para generar niveles Fib_0.786

### ✅ Phase 1: Database & Models (30-45 min)
- [ ] Crear modelo `IndividualLevelBacktest` en `app/models/backtest.py`
- [ ] Crear modelo `IndividualLevelTrade` en `app/models/backtest.py`
- [ ] Alembic migration: `alembic revision --autogenerate -m "Add individual level backtest tables"`
- [ ] `alembic upgrade head`
- [ ] Verificar tablas creadas en DB

### ✅ Phase 2: Service Layer - Core Logic (2-3 hrs)
**File**: `app/services/individual_level_backtest.py`

- [ ] **Entry signal detection**:
  - [ ] `detect_long_signal(candle, level, tolerance_pct)` → bool
  - [ ] `detect_short_signal(candle, level, tolerance_pct)` → bool

- [ ] **Strategy implementations**:
  - [ ] `FixedPercentStrategy`: SL/TP fijos en %
  - [ ] `ATRBasedStrategy`: SL/TP basados en ATR
  - [ ] Base class `TradingStrategy` para extensibilidad

- [ ] **Position management**:
  - [ ] `open_position(entry_time, entry_price, direction, sl, tp)` → Position
  - [ ] `check_exit(position, current_candle)` → (exit_reason, exit_price) or None
  - [ ] `calculate_pnl(entry, exit, direction)` → (pnl, pnl_pct)

- [ ] **Main backtest runner**:
  - [ ] `run_individual_level_backtest(db, level_type, source_tf, exec_tf, strategy, params, start_date, end_date)`
  - [ ] Loop through candles
  - [ ] Detect signals on relevant levels
  - [ ] Simulate trades
  - [ ] Calculate metrics
  - [ ] Persist results to DB

- [ ] **Metrics calculation**:
  - [ ] Win rate, profit factor, avg win/loss
  - [ ] Sharpe ratio, max drawdown
  - [ ] Equity curve generation

### ✅ Phase 3: API Endpoints (1 hr)
**File**: `app/views/backtest.py` (extend existing blueprint)

- [ ] `POST /api/backtest/individual-level/run`
  - Body: `{level_type, source_tf, exec_tf, strategy_name, params, start_date, end_date}`
  - Response: `{backtest_id, status: "running"}`

- [ ] `GET /api/backtest/individual-level/results`
  - Query params: `?level_type=HTF&source_tf=1d&strategy=fixed_percent`
  - Response: `[{backtest_id, metrics, ...}]`

- [ ] `GET /api/backtest/individual-level/<id>/trades`
  - Query params: `?filter=winning&sort=pnl_desc&limit=100`
  - Response: `[{trade_id, entry_time, pnl, ...}]`

- [ ] `GET /api/backtest/individual-level/<id>/metrics`
  - Response: `{win_rate, profit_factor, equity_curve, ...}`

- [ ] `GET /api/backtest/individual-level/<id>/replay-data`
  - Response: `{candles, levels, trades, timeline_events}`

### ✅ Phase 4: Dashboard UI - Results Table (2 hrs)
**File**: `app/templates/backtest/individual_levels.html`

- [ ] **Run Backtest Form**:
  - [ ] Level type selector (dropdown)
  - [ ] Source timeframe selector
  - [ ] Execution timeframe selector
  - [ ] Strategy selector
  - [ ] Parameter inputs (SL %, TP %, timeout, etc.)
  - [ ] Date range picker
  - [ ] "Run Backtest" button → POST to API

- [ ] **Results Comparison Table**:
  - [ ] Fetch all backtests via API
  - [ ] Display in sortable table (DataTables.js)
  - [ ] Columns: Level Type, TF, Strategy, Trades, Win Rate, PF, Total PNL, Sharpe, Actions
  - [ ] Actions: "View Trades" 👁️, "View Chart" 📊, "Delete" 🗑️

- [ ] **Filters & Sorting**:
  - [ ] Filter by level type, timeframe, strategy
  - [ ] Sort by any metric (win rate, PF, PNL, etc.)

### ✅ Phase 5: Dashboard UI - Trade Details (1 hr)
**File**: `app/templates/backtest/trade_details_modal.html`

- [ ] Modal popup for trade details
- [ ] Table of all trades for selected backtest
- [ ] Columns: Entry Time, Exit Time, Direction, Entry $, Exit $, PNL, PNL%, Exit Reason, Duration
- [ ] Filters: Winning/Losing, Exit Reason
- [ ] Export to CSV button

### ✅ Phase 6: Dashboard UI - Visual Charts (1-2 hrs)
**File**: `app/templates/backtest/individual_levels.html` (charts section)

- [ ] **Bar Chart**: Win Rate by Level Type (Chart.js)
- [ ] **Line Chart**: Cumulative PNL by Level Type (multi-line, timeline)
- [ ] **Scatter Plot**: Risk/Reward distribution
- [ ] **Heatmap**: Win Rate by (Level Type × Timeframe)

### ⭐ Phase 7: Replay Feature (2-3 hrs) - OPTIONAL BUT VALUABLE
**File**: `app/templates/backtest/replay.html`

- [ ] TradingView Lightweight Charts integration
- [ ] Timeline slider component
- [ ] Play/Pause/Speed controls (1x, 2x, 5x, 10x)
- [ ] Event markers:
  - [ ] Level creation (when level forms)
  - [ ] Entry signals (green/red markers)
  - [ ] SL/TP lines
  - [ ] Exit points with reason label
- [ ] PnL tracker (running total in corner)
- [ ] Current candle index indicator

### ✅ Phase 8: Testing & Validation (1 hr)
- [ ] Run test backtest on small date range (1 month)
- [ ] Verify trades make sense
- [ ] Check metrics calculations
- [ ] Compare results with manual calculation
- [ ] Fix any bugs

### ✅ Phase 9: Production Run (30 min)
- [ ] Run backtests for all level types on full dataset (2021-2024)
- [ ] Generate comparison report
- [ ] Export results to CSV for analysis
- [ ] Document findings in `BACKTEST_RESULTS.md`

---

## 📈 Success Metrics

After completion, we should be able to answer:
1. **Which level types have highest win rate?**
2. **Which timeframes are most reliable?**
3. **Which strategy works best for each level type?**
4. **What are optimal SL/TP parameters per level?**
5. **Which levels should be prioritized in the final ML model?**

---

## 🔄 Next Steps After This System

1. **Feature importance analysis**: Identify what makes a level more predictive
2. **Combine best levels**: Multi-level model using only top performers
3. **ML model training**: Train on best-performing level types with optimized weights
4. **Live trading integration**: Use backtest insights to configure live strategy

---

## 📝 Notes & Decisions

### Entry Logic Clarification
- **OLD (incorrect)**: Enter when fractal forms (vela 5)
- **NEW (correct)**: Fractal is a LEVEL. Enter when price TOUCHES that level and closes in expected direction.

### Strategy Evolution
- Start with 2 simple strategies (fixed_percent, atr_based)
- Add more complex ones (level_based, fractal_based) after validating the system works

### Fibonacci Update
- Add 0.786 ratio to `calculate_fibonacci_levels()`
- Remove 0.25 ratio (not a true Fibonacci level)
- Keep 0.50, 0.618, 0.75 (or replace 0.75 with 0.786)

---

## 🚀 Current Status

**Last Updated**: 2026-02-16

### Completed:
- [ ] Plan created

### In Progress:
- [ ] Phase 1: Database & Models

### Blocked:
- None

### Next Up:
- Phase 1: Create database models and migrations
