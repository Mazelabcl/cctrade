# AutoResearch: Hallazgos Consolidados

> Documento de referencia para el diseno de futuros experimentos.
> Ultima actualizacion: 2026-03-29

---

## Indice

1. [Mode A: Exit Optimization (4h fractals)](#mode-a-exit-optimization)
2. [Mode B: Feature Discovery (fractal prediction)](#mode-b-feature-discovery)
3. [Mode C: Confluence Scalper Discovery](#mode-c-confluence-scalper-discovery)
   - [v1: Resultados por timeframe](#v1-resultados-por-timeframe)
   - [v2: Weighted scoring + SFP entry](#v2-weighted-scoring--sfp-entry)
4. [Hallazgos transversales](#hallazgos-transversales)
5. [Preguntas abiertas para futura investigacion](#preguntas-abiertas)

---

## Mode A: Exit Optimization

**Objetivo:** Optimizar la salida del sistema de fractales en 4h.

**Resultado principal:** El sistema de fractales en 4h logra PF 10+, lo cual confirma que la estrategia base (entrar en fractal, salir con trail) es extremadamente rentable en timeframes altos.

### Hallazgos clave

| Parametro | Resultado |
|-----------|-----------|
| Profit Factor | 10+ |
| Mejor exit | `swing_trail` con lookback ajustado por timeframe |
| Trail stop vs fixed RR | Trail genera **5x mas profit** que fixed Risk-Reward |

### Conclusion

El swing trail es el exit definitivo para fractales en 4h. No vale la pena usar fixed RR; el trail captura los movimientos extendidos que los fractales anticipan.

---

## Mode B: Feature Discovery

**Objetivo:** Descubrir si ML puede predecir fractales usando features de una sola vela (single-candle features).

**Escala:** 200 experimentos, 21 features, modelo Random Forest.

### Resultado principal

**ML NO puede predecir fractales.** El mejor F1 fue 0.10, practicamente aleatorio.

### Features evaluados

| Categoria | Features | Resultado |
|-----------|----------|-----------|
| Distancia a extremos | `dist_from_high_20`, `dist_from_low_20` | Mejor grupo (41% importance combinado) |
| Volatilidad | `atr_14` | Util pero debil |
| Divergencia | `rsi_divergence` | Marginalmente util |
| RSI, momentum, volumen | Multiples variantes | **NO ayudaron en absoluto** |

Solo 4 de 21 features sobrevivieron: `dist_from_high_20`, `dist_from_low_20`, `atr_14`, `rsi_divergence`.

### Conclusion

Predecir fractales desde features de una sola vela es casi imposible (F1=0.10). La distancia al high/low reciente es lo mas predictivo, pero no es suficiente. **ML no puede reemplazar el ojo del trader para deteccion de fractales.** El sistema reactivo (esperar el fractal, entrar en touch) sigue siendo el approach mas fuerte.

> **Implicacion para futuros experimentos:** No invertir mas tiempo en predecir fractales con single-candle features. Si se quiere mejorar la prediccion, habria que explorar features multi-candle o patrones de secuencia (LSTM, etc.), pero el ROI esperado es bajo.

---

## Mode C: Confluence Scalper Discovery

**Objetivo:** Construir un scalper basado en confluencia de niveles tecnicos (fractales, Fibonacci, Volume Profile, session levels).

### v1: Resultados por timeframe

#### 1m (4.4M velas, 200 experimentos)

| Metrica | Baseline | Mejor |
|---------|----------|-------|
| Profit Factor | 1.15 | 2.14 |
| Win Rate | 32% | 54.4% |
| Trades | 26K | 43K |
| Frecuencia | - | 14 trades/dia |
| Fitness | 185 | 448 |

- **Mejor exit:** `atr_trail` con multiplier 0.5, timeout 5 velas
- 16 de 18 level types contribuyeron
- **PROBLEMA CRITICO:** Las comisiones destruyen TODO el profit en 1m (-$115K neto en Futures)
- El SL es tan pequeno que el position size es enorme, y las comisiones se comen todo

**Conclusion:** 1m es demasiado granular para scalping. Profitable en gross, negativo en net. **Descartado.**

---

#### 15m (293K velas, 200 experimentos)

| Metrica | Baseline | Mejor |
|---------|----------|-------|
| Profit Factor | 1.00 | 2.39 |
| Win Rate | 26.8% | 32.3% |
| Trades | 4758 | 2962 |
| Frecuencia | - | ~1 trade/dia |
| Fitness | 69 | 130 |

- **Mejor exit:** `breakeven_trail`, `swing_lookback=9`, `timeout=35`
- Solo 4 level types sobrevivieron: `Fractal_support`, `Fractal_resistance`, `PrevSession_VWAP`, `PrevSession_VP_POC`
- Touch tolerance: 0.1% (muy tight)
- **NET PROFITABLE:** $150-164/mes con $10 de riesgo despues de comisiones

**Conclusion:** 15m es el sweet spot para confluence scalping. Frecuencia razonable (~1/dia), profitable despues de comisiones, y pocas level types necesarias.

---

#### 30m (146K velas, 200 experimentos)

| Metrica | Baseline | Mejor |
|---------|----------|-------|
| Profit Factor | 0.99 (perdiendo) | 1.38 |
| Win Rate | 24.9% | 19.9% |
| Trades | 2587 | 2393 |

- 8 level types necesarios pero resultados mediocres
- **33 max consecutive losses** (terrible para la psicologia)

**Conclusion:** 30m es demasiado lento, pierde el edge. **Descartado.**

---

### v2: Weighted scoring + SFP entry

**15m con weighted scoring (200 experimentos)**

Se usaron los win rates reales del backtest como pesos de cada level type (ej: `Fractal_support` weekly = 87% WR, `VP_POC` daily = 31%).

| Metrica | Resultado |
|---------|-----------|
| Profit Factor | 1084 (!!) |
| Win Rate | 68.8% |
| Trades | 587 (~1 cada 5 dias) |
| Avg R | 2.77 |
| Max consecutive losses | Solo 4 |
| Net | $157-164/mes con $10 riesgo |

- Solo `Fractal_support` + `Fractal_resistance` sobrevivieron. **Todos los demas levels fueron eliminados.**
- El PF de 1084 refleja un sistema con muy pocos trades pero altisimo win rate

**KEY INSIGHT:** El weighted scoring hizo que el sistema convergiera a fractals-only porque sus win rates son mucho mas altos que todo lo demas. Esto confirma lo que Mode A ya demostro: los fractales son el edge real.

---

## Hallazgos transversales

Estos patrones se repitieron consistentemente a traves de todos los experimentos:

### 1. `naked_only=True` siempre gana

Los niveles que no han sido tocados (untouched/naked) son significativamente mas poderosos que los que ya fueron tocados. Esto tiene sentido intuitivo: un nivel "fresco" tiene mas liquidez acumulada.

### 2. Los fractales dominan todo

Cada optimizacion converge hacia configuraciones heavy en fractales. No importa el timeframe ni el exit: los fractales siempre son los niveles con mayor edge.

### 3. Jerarquia de exits (15m+)

```
breakeven_trail > swing_trail > atr_trail
```

Para 15m y timeframes superiores, `breakeven_trail` es superior. Para 4h, `swing_trail` gana (ver Mode A).

### 4. Score threshold optimo = 3

Se probaron 2, 2.5, 3.5, 4.0. Nunca mejoro sobre 3. Este parece ser el punto de equilibrio entre frecuencia y calidad de senales.

### 5. Zone width optimo = 1%

Se probo desde 0.6% hasta 1.4%. El 1% nunca fue superado. Zonas mas estrechas pierden trades; zonas mas amplias introducen ruido.

### 6. `PrevSession_Low` siempre es eliminado

Los highs de sesion son respetados consistentemente; los lows no. Esto puede reflejar el bias alcista de BTC en el periodo de datos (2017-2026).

### 7. Volume Profile levels son debiles

| Level Type | Win Rate tipico |
|------------|----------------|
| VP_POC | 31% |
| VP_VAH | ~35% |
| VP_VAL | ~43% |

Estos win rates son demasiado bajos para contribuir significativamente al sistema de confluencia. En v2 con weighted scoring, fueron eliminados automaticamente.

### 8. Analisis de comisiones es CRITICO

El 1m fue profitable en gross pero negativo en net. Sin el analisis de comisiones, habriamos desplegado un sistema perdedor. **Siempre calcular net profit despues de comisiones antes de considerar un sistema viable.**

---

## Preguntas abiertas

Estas son las preguntas que quedan sin responder y que deberian guiar futuros experimentos:

### 1. Que pasa SIN fractales?

Los otros level types (Fibonacci, VP, session levels) tienen edge por si solos? O solo funcionan como confirmacion de fractales? Esto determinaria si vale la pena un segundo sistema no-fractal.

### 2. Puede SFP entry mejorar resultados?

El Stop Hunt Pattern (SFP) se probo pero no se optimizo. Un SFP entry en lugar de touch entry podria mejorar el timing y reducir el drawdown.

### 3. Filtros de hora del dia

US session vs Asia vs Europe. Es posible que el edge se concentre en ciertas sesiones. Filtrar por hora podria mejorar el win rate sin reducir mucho la frecuencia.

### 4. Out-of-sample validation

Todos los resultados son in-sample sobre 2017-2026. Hay riesgo de overfitting. Se necesita:
- Walk-forward validation
- Train en 2017-2023, test en 2024-2026
- O al menos un holdout set

### 5. Sistema combinado: fractal (4h) + scalper (15m)

El sistema de fractales en 4h es raro pero muy rentable (PF 10+). El scalper en 15m es frecuente pero menos rentable (PF ~2). Combinar ambos daria income consistente (scalper) + home runs (fractales 4h). Falta definir como manejar la coexistencia de posiciones.

---

## Resumen ejecutivo

| Sistema | Timeframe | PF | Frecuencia | Status |
|---------|-----------|-----|------------|--------|
| Fractal swing | 4h | 10+ | Raro | VIABLE - el edge mas fuerte |
| Confluence scalper | 1m | 2.14 gross | 14/dia | DESCARTADO - comisiones |
| Confluence scalper | 15m | 2.39 | 1/dia | VIABLE - sweet spot |
| Confluence scalper | 30m | 1.38 | ~1/dia | DESCARTADO - sin edge |
| Weighted scalper | 15m | 1084 | 1/5 dias | VIABLE - pero converge a fractals-only |

**El edge real esta en los fractales.** Toda la evidencia apunta a que los fractales son los niveles tecnicos con mayor poder predictivo. El sistema optimo es reactivo: esperar el fractal, entrar en touch del nivel, salir con trail stop.
