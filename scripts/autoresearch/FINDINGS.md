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

## Mode C Addendum: Experimentos sin fractales

**Objetivo:** Determinar si los otros niveles (Fib, VP, Session, HTF) tienen edge sin fractales.

### unique_types scoring (200 experimentos, 15m)

| Metrica | Baseline | Mejor |
|---------|----------|-------|
| Profit Factor | 0.78 (perdiendo) | 1.39 |
| Win Rate | 18.5% | 48.1% |
| Trades | 6,738 | 21,260 |
| **Net despues de comisiones** | — | **-$40/mes** (perdedor) |

### tf_weighted scoring (200 experimentos, 15m)

Pesos por temporalidad: monthly=3, weekly=2, daily=1, hourly=0.25.

| Metrica | Resultado |
|---------|-----------|
| Profit Factor | 1.12 |
| Trades | 23,169 |
| **Net despues de comisiones** | **-$187/mes** (peor) |

### Conclusion

**Sin fractales no hay edge viable.** Los otros niveles solo logran PF ~1.1-1.4 en gross, insuficiente para cubrir comisiones con alta frecuencia. El tf_weighted scoring (pesar por temporalidad) no ayudo — de hecho empeoro. Sin fractales, todos los niveles tienen WR similar (~30-45%) y ninguno domina.

---

## Mode D: Fractal Prediction con Features de Niveles

**Objetivo:** Predecir si la vela actual sera fractal usando features de niveles (confluencia, distancia, TF weights) y price action (forma de vela).

**Diferencia clave vs Mode B:** Mode B uso indicadores retail (RSI, momentum, volumen) → F1=0.10. Mode D usa los niveles del coach (HTF, Fib, VP, Session) como features → F1=0.40.

### Resultados por timeframe (levels + PA only, 200 experimentos cada uno)

| Metrica | 1h | 4h |
|---------|-----|-----|
| F1 macro | **0.403** | **0.395** |
| Raw precision (fractal exacto) | 32.3% | 32.8% |
| **Adj precision (fractal ±2 bars)** | **86.3%** | **89.0%** |
| **Adj precision (reaccion wick)** | **79.5%** | **79.2%** |
| Near-miss: fractal en ±2 velas | 80% de FP | 84% de FP |
| Near-miss: wick de rechazo | 70% de FP | 69% de FP |
| Near-miss: precio movio +0.5% | 31% de FP | 59% de FP |
| Modelo | RF 450 trees, depth 13 | RF 50 trees, depth 10 |
| Zone width | 1.5% | 0.75% |

### Near-miss analysis (hallazgo clave)

La precision raw de 32% es enganosa. Cuando el modelo dice "fractal aqui" y no es exactamente fractal:
- **84% de las veces** hay un fractal en ±2 velas (timing casi perfecto)
- **70% de las veces** hay una vela de reaccion con wick >50% del rango (trade viable)
- **En 4h, 59%** de los FP igual movieron el precio 0.5%+ a favor

**Precision ajustada real: ~89%.** El modelo detecta ZONAS DE REACCION, no el pip exacto.

### Top features (consistentes en ambos timeframes)

| Feature | Importancia | Tipo |
|---------|-------------|------|
| `candles_since_bearish` | **17-19%** | Ritmo fractal |
| `candles_since_bullish` | **17%** | Ritmo fractal |
| `dist_from_high_20` | 7-8% | Price action |
| `upper_wick` / `lower_wick` | 6% cada uno | Price action |
| `body_ratio` | 6% | Price action |
| `nearest_support/resistance_dist` | presente | **Level context** |
| `nearest_support/resistance_tf` | presente | **Level context** |
| `naked_support/resistance_total` | presente | **Level context** |
| `has_htf_support/resistance` | presente | **Level context** |

### Validacion: retail vs levels-only

| Modo | F1 macro | Adj precision |
|------|----------|---------------|
| Con retail (RSI, momentum, vol) | 0.41 | 88% |
| **Solo levels + PA** | **0.40** | **86-89%** |

Quitar indicadores retail NO empeoro el modelo. Los niveles del coach + price action son todo lo necesario.

### Config optimo

```python
# 1h
{'model': 'rf', 'n_trees': 450, 'max_depth': 13, 'zone_width': 0.015}
# Features: 19 (5 PA + 14 level context)

# 4h
{'model': 'rf', 'n_trees': 50, 'max_depth': 10, 'zone_width': 0.0075}
# Features: 21 (5 PA + 16 level context)
```

### Out-of-sample validation (train 2017-2023, test 2024-2026)

**Resultado: NO hay overfitting.** El modelo entrenado sin ver 2024-2026 predice igual de bien.

| Metrica | 1h in-sample | 1h OOS | 4h in-sample | 4h OOS |
|---------|-------------|--------|-------------|--------|
| F1 macro | 0.403 | **0.406** | 0.395 | **0.399** |
| Raw precision | 32.3% | 32.0% | 32.8% | 32.7% |
| Adj precision strict | 86.3% | **86.7%** | 89.0% | **87.0%** |
| Adj precision practical | 79.5% | **79.3%** | 79.2% | **78.0%** |

Los patrones de interaccion precio-niveles son **estables en el tiempo**. Confirmado con 400 experimentos OOS adicionales (200 por timeframe).

---

## Hallazgos transversales (actualizado)

### NUEVO: Los niveles del coach funcionan como features de ML

Los niveles de Chart Champions (HTF, Fibonacci CC/quarters, Volume Profile, Session levels) son features predictivos reales. Combinados con price action (forma de vela), logran F1=0.40 para predecir fractales — 4x mejor que indicadores retail.

### NUEVO: Near-miss precision es la metrica real

La precision raw de ML para eventos raros (fractales ~5% de velas) siempre sera baja. Pero la precision ajustada (contando reacciones de precio y fractales en ±2 velas) es ~89%. El modelo detecta zonas de reaccion, no el pip exacto.

### NUEVO: Sin fractales, no hay edge

600 experimentos (3 modos de scoring) confirmaron que sin Fractal_support/resistance, los otros niveles solo logran PF 1.1-1.4 gross, insuficiente para cubrir comisiones.

*(Hallazgos anteriores 1-8 siguen vigentes)*

---

## Preguntas abiertas

### 1. ~~Que pasa SIN fractales?~~ RESPONDIDA

**Sin fractales no hay edge viable.** Confirmado con 600 experimentos. Los otros niveles son complementarios, no sustitutos.

### 2. Puede SFP entry mejorar resultados?

El Stop Hunt Pattern (SFP) se probo pero no se optimizo. Un SFP entry en lugar de touch entry podria mejorar el timing y reducir el drawdown.

### 3. Filtros de hora del dia

US session vs Asia vs Europe. Es posible que el edge se concentre en ciertas sesiones.

### 4. ~~Out-of-sample validation~~ RESPONDIDA

**No hay overfitting.** Train 2017-2023, test 2024-2026: F1 y adj precision se mantienen identicos. Los patrones son estables.

### 5. Predictor como filtro del scalper

Usar el fractal predictor (89% adj precision) como filtro: solo tomar trades del confluence scalper cuando el predictor dice "zona caliente". Podria mejorar PF y reducir trades perdedores.

### 7. ~~Backtest del predictor~~ RESPONDIDA

**El predictor es rentable como sistema de trading OOS.** Mejores resultados:
- 4h swing_trail: PF 1.59, +$21/mes ($10 risk), 161 trades en 2 anos
- 1h breakeven_trail: PF 1.30, +$13/mes ($10 risk), 618 trades en 2 anos
- Fixed RR 2:1 no funciona; trail es necesario
- WR bajo (21-26%) pero avg win 3-4R vs avg loss 0.6-0.8R

### 6. Multi-timeframe predictor

Combinar features de niveles de TODOS los timeframes en un solo modelo. Un nivel weekly cerca + un nivel daily cerca + fractal rhythm → prediccion mas robusta.

---

## Resumen ejecutivo

| Sistema | Timeframe | Resultado | Status |
|---------|-----------|-----------|--------|
| Fractal swing (Mode A) | 4h | PF 10+ | VIABLE - el edge mas fuerte |
| Feature discovery (Mode B) | 4h | F1=0.10 | DESCARTADO - retail features no predicen |
| Confluence scalper (Mode C) | 1m | PF 2.14 gross | DESCARTADO - comisiones |
| Confluence scalper (Mode C) | 15m | PF 2.39, +$164/mes | VIABLE - sweet spot |
| Confluence scalper (Mode C) | 30m | PF 1.38 | DESCARTADO - sin edge |
| Sin fractales (Mode C) | 15m | PF 1.39, -$40/mes | DESCARTADO - comisiones |
| tf_weighted sin fractales | 15m | PF 1.12, -$187/mes | DESCARTADO - peor |
| **Fractal predictor (Mode D)** | **1h** | **F1=0.40, adj prec 86%** | **PROMETEDOR** |
| **Fractal predictor (Mode D)** | **4h** | **F1=0.40, adj prec 89%** | **PROMETEDOR** |

**Los niveles del coach son el edge real.** ML confirma que los niveles de Chart Champions (HTF, Fib, VP, Session) combinados con price action predicen zonas de reaccion con 89% de precision ajustada. Los indicadores retail (RSI, momentum) no aportan nada adicional.
