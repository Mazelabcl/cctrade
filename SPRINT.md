# Sprint: Exit Optimization & Data Foundation

**Branch**: `claude/exit-optimization`
**Started**: 2026-03-23
**Status**: Active

---

## Completado

### 1. Fix critico: Naked Levels (PrevSession/VP)
- **Bug**: PrevSession y VP levels nunca tenian `first_touched_at` seteado. TODOS se consideraban "naked" para siempre.
- **Fix**: Implementamos "mobile supersession": solo el nivel mas reciente por (tipo, timeframe) es valido. Los anteriores se ignoran.
- **Impacto**: Redujo niveles activos de ~260 a ~30 dentro del 1% del precio.

### 2. Analytics Dashboard (`/analytics/`)
6 tabs con Plotly, cada uno con TLDR explicativo:
1. **Backtest Results** - Heatmap WR por level_type x timeframe
2. **Feature Distribution** - Histogramas de distancia y confluencia
3. **Level Density** - Niveles activos a lo largo del tiempo
4. **Scoring Analysis** - Score vs PnL, equity curves por threshold
5. **MFE Analysis** - Maximum Favorable Excursion por tipo de nivel
6. **Level Breakdown** - WR y PF desglosado por source TF (daily/weekly/monthly)

### 3. Trade Explorer Mejorado
- Toggle de niveles por tipo (checkboxes individuales, no all/none)
- Entry/exit condition annotations en el chart
- Analysis zone visualization (toggle Z)
- Reference level resuelto desde DB (ya no muestra NULL)
- Mobile PrevSession query (solo el mas reciente)

### 4. Paridad de Datos (15m/30m/1h/4h)
Script `ensure_tf_complete.py` que garantiza paridad:
- Candles: 15m (293k), 30m (147k), 1h (73k), 4h (18k)
- Features: calculados para todos
- Backtests: 62 combos unicos por TF
- ML Models: RF bull+bear por TF
- MFE: analizado para todos

### 5. Trail Stop Backtest
Testeamos 6 estrategias de salida en todos los TFs:

**Resultados Fractals (breakeven trail, 4h):**
| Tipo | Trades | WR% | Avg R | PF |
|------|--------|-----|-------|-----|
| Fractal_support (weekly) | 36 | 80.6% | ~2R | **15.39** |
| Fractal_resistance (daily) | 279 | ~52% | ~1.9R | **8.76** |

**Hallazgo clave**: Trail stop multiplica el profit 4-5x vs fixed RR 1:1.

### 6. Candles sinteticas desde 1-min
Script `build_candles_from_1min.py` que construye cualquier TF a partir de 4.4M candles de 1 minuto.

---

## Hallazgos Clave

### Los datos dicen claramente:
1. **Fractal_support weekly es el santo grial**: PF 15.39, WR 80.6% en 4h
2. **VP_POC y VP_VAH son toxicos**: pierden dinero en TODA estrategia y TF. Excluirlos.
3. **PrevSession_VP_VAH funciona** (el de sesion pasada), pero VP_VAH actual no.
4. **Trail stop > Fixed RR**: Swing trail genera +1,601R vs +316R en fixed 1:1 (5x mas)
5. **4h es el mejor TF para trading**: mas limpio, menos ruido, mejor PF
6. **15m puede funcionar como sniper**: Fractal_support mantiene 1.37R en 15m

### Lo que "R" significa:
- R = unidad de riesgo. Si arriesgas $100 (distancia entry a SL):
  - 1R = ganaste $100, 5R = ganaste $500, -1R = perdiste $100
  - PF 3.0 = por cada $1 que pierdes, ganas $3

---

## Roadmap Actualizado

### Siguiente paso: Multi-TP Annotation Tool
**Prioridad: ALTA** - Extender Trade Explorer para:
- Click multiple TPs en el chart (TP1, TP2, TP3)
- Sistema guarda precio, tiempo, features del momento
- Despues de anotar 50-100 trades, reverse-engineering de patrones
- "TP1 siempre fue en un swing high fractal"

### Despues: Exit Rule AutoResearch
- Adaptar concepto de Karpathy autoresearch
- evaluate.py corre backtest, mide Sharpe/PF
- Agent modifica exit rules automaticamente
- Loop de N experimentos overnight

### Ideas en exploracion ("volas"):
- **Sniper TF**: usar 4h para senal + 15m/30m para entry preciso (reduce SL)
- **VAH multi-touch**: investigar si el 2do touch del VAH es mas rentable
- **SFP trigger en 30m**: el coach usa close de 30m debajo del SFP como trigger
- **Compound/pyramiding**: agregar a posiciones ganadoras en pullbacks
- **Dos sistemas separados**: Fractal Swings (hold largo, trail) vs Session Rotations (fixed RR, scalp)
