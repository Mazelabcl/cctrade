# New ML Features Implementation Plan

## Project Overview
This project is a cryptocurrency trading bot that uses machine learning to predict potential swing points (fractals) in Bitcoin price action. The bot analyzes multiple timeframes (1-hour, daily, weekly, monthly) and various technical levels to make predictions. The main goal is to identify high-probability turning points in the market by combining multiple types of technical analysis with machine learning.

## Key Concepts

### Technical Levels
The bot tracks several types of significant price levels:

1. **HTF (Higher Timeframe) Levels**
   - Created when a candle changes direction in higher timeframes
   - Example: If a daily candle is bullish (green) and the next is bearish (red), the high of the bearish candle becomes an HTF level
   - These mark important market structure points

2. **Volume Profile Levels**
   - POC (Point of Control): Price level with highest trading volume
   - VAH (Value Area High): Upper boundary of 70% volume concentration
   - VAL (Value Area Low): Lower boundary of 70% volume concentration
   - Calculated for daily, weekly, and monthly timeframes

3. **Fibonacci Levels**
   - Created between significant swing points
   - Uses three key ratios:
     - 0.5 (50% retracement)
     - 0.618 (Golden Pocket)
     - 0.75 (75% retracement)

4. **Fractal Levels**
   - Swing highs and lows from higher timeframes
   - Used to identify key reversal points

## Level Validity Rules
- Each level starts as "naked" (untouched)
- Support touches and resistance touches are counted separately
- A level becomes invalid after:
  - 4 touches as support, OR
  - 4 touches as resistance
- Invalid levels are removed from analysis
- Only levels that existed before the current candle's timestamp are considered
  - This prevents look-ahead bias
  - Levels dataset is filtered by created_at <= current_candle_timestamp

## New Feature Engineering Approach

### 1. Finding Nearest Naked Levels
For each 1-hour candle (n), we look at candle n-2's close price and:

1. **Search Strategy**:
   - Look above and below for nearest naked level (pristine, 0 touches as resistance and support).

2. **Level Requirements**:
   - Must be untouched (touch_count = 0)
   - Must be valid (not expired)
   - Support levels must be below price
   - Resistance levels must be above price

### 2. Zone Analysis
Once nearest naked support and resistance are found:

1. **Zone Definition**:
   - Support Zone: [Nearest Support Level, Nearest Support Level - 1.5%]
   - Resistance Zone: [Nearest Resistance Level, Nearest Resistance Level + 1.5%]

2. **Zone Features** (calculated for both support and resistance zones):
   - `zone_X_total_levels`: Total number of levels in zone
   - `zone_X_naked_levels`: Number of untouched levels
   - `zone_X_monthly_levels`: Number of monthly levels (including HTF, VP, Fractals, Fibs from monthly timeframe)
   - `zone_X_weekly_levels`: Number of weekly levels (including HTF, VP, Fractals, Fibs from weekly timeframe)
   - `zone_X_daily_levels`: Number of daily levels (including HTF, VP, Fractals, Fibs from daily timeframe)
   - `zone_X_naked_fibs_golden_pocket`: Number of untouched 0.618 fib levels across all timeframes (daily, weekly, monthly)

### 3. Zone Interaction Features
Counts the number of levels present in a zone and which levels were touched by the candle's high/low:

```python
zone_levels = {
    # Daily levels
    'daily_htf': 0,      # Naked HTF levels
    'daily_poc': 0,      # Volume Profile POC
    'daily_vah': 0,      # Volume Profile VAH
    'daily_val': 0,      # Volume Profile VAL
    'daily_fib_618': 0,  # Fibonacci 0.618
    'daily_fib_75': 0,   # Fibonacci 0.75
    'daily_fib_50': 0,   # Fibonacci 0.50
    'daily_fractal_up': 0,   # Fractal Up
    'daily_fractal_down': 0, # Fractal Down
    # ...

    # Weekly levels (same structure)
    'weekly_htf': 0,
    'weekly_poc': 0,
    # ...
    
    # Monthly levels (same structure)
    'monthly_htf': 0,
    'monthly_poc': 0,
    # ...
}

touched_levels = {
    # Same structure as zone_levels
    #this should be the vector template

    'daily_htf': 0,
    'daily_poc': 0,
    # ...
}
```

### Important Notes:
1. Support/resistance touches are tracked separately
   - A level can have 4+ touches as support but still be valid as resistance
   - Touch counts reset when price breaks through

2. Zone Detection Logic:
   - Support zone: From highest level to (highest level * (1 - margin))
   - Resistance zone: From lowest level to (lowest level * (1 + margin))
   - Default margin: 1.5%

3. Level Touch Rules:
   - Only count levels with < 4 touches in their current role
   - A level is "touched" if price reaches within the zone margin
   - Multiple touches of the same level type are counted

### Support and Resistance Interaction Features

#### Zone Levels
Tracks which support and resistance levels are currently active in the trading zone.
- A level is considered "valid" if it has less than 4 touches
- Returns 1 for each valid level, 0 otherwise
- Feature format: `{timeframe}_{level_type}` (e.g., `monthly_htf`, `weekly_poc`, `daily_fib_618`)

#### Touched Levels
Tracks which support and resistance levels have been touched by the current price action.

##### Support Level Touch Logic
- A support level is considered touched if price goes down to or below the level
- Mathematically: `candle_low <= support_level`
- Example: If support is at 39500 and candle low reaches 39500 or goes below it, the level is touched

##### Resistance Level Touch Logic
- A resistance level is considered touched if price goes up to or above the level
- Mathematically: `candle_high >= resistance_level`
- Example: If resistance is at 40500 and candle high reaches 40500 or goes above it, the level is touched

### Feature Names
All features follow the format: `{timeframe}_{level_type}`
- Timeframes: monthly, weekly, daily
- Level types: htf, poc, fib_618

### 4. Swing Pattern Features
Detects potential swing points by comparing candle n-1 with previous candles:

```python
swing_patterns = {
    'potential_swing_high': 0,  # 1 if n-1 high > n-2 high AND n-1 high > n-3 high
    'potential_swing_low': 0,   # 1 if n-1 low < n-2 low AND n-1 low < n-3 low
}
```

### Important Notes:
1. Swing detection uses 3 candles (n-1, n-2, n-3)
2. A candle can be both a swing high and swing low
3. This is a "potential" swing as it may change when new candles form

### 5. Candle Structure Analysis
For candle n-1:

1. **Candle Ratios**:
   - `upper_wick_ratio = upper_wick_length / total_candle_length`
     - High ratio + resistance touch = potential reversal down
   - `lower_wick_ratio = lower_wick_length / total_candle_length`
     - High ratio + support touch = potential reversal up
   - `body_total_ratio = body_length / total_candle_length`
     - Small ratio (< 0.2) = indecision/potential reversal
     - Large ratio (> 0.8) = strong momentum
   - `body_position_ratio = (close - low) / (high - low)`
     - Close to 1.0 = Very bullish (closed near high)
     - Close to 0.0 = Very bearish (closed near low)
     - Around 0.5 = Indecision

### 6. Volume Analysis
For candle n-1:

1. **Volume Ratios**:
   - `volume_short_ratio = current_volume / MA(volume, 6)`
     - Detects immediate volume spikes
     - Ratio > 1 indicates higher than recent average
   - `volume_long_ratio = current_volume / MA(volume, 168)`

### 7. Time Block Feature
Divide 24 hours into 6 blocks of 4 hours each:
- Block 0: 00:00-03:59 UTC
- Block 1: 04:00-07:59 UTC
- Block 2: 08:00-11:59 UTC
- Block 3: 12:00-15:59 UTC
- Block 4: 16:00-19:59 UTC
- Block 5: 20:00-23:59 UTC

This captures trading session information (Asia, London, New York).

### 8. Fractal Timing Features
For each candle:
- `candles_since_last_up`: Number of candles since last swing low
- `candles_since_last_down`: Number of candles since last swing high
- `last_up_time`: Timestamp of last swing low
- `last_down_time`: Timestamp of last swing high
- `fractal_type`: Type of fractal (0=None, 1=Up/SwingLow, 2=Down/SwingHigh)

#### Fractal Detection Logic
- Uses 5 candles [N-5, N-4, N-3, N-2, N-1]
- Evaluates N-3 as potential fractal
- Fractal Down (Swing High): N-3 high > all other highs
- Fractal Up (Swing Low): N-3 low < all other lows
- Counts reset to 0 when new fractal is detected
- Maintains independent counts for up/down fractals

## Implementation Status (create_ml_features.py):

### 1. Finding Nearest Naked Levels 
- [x] Implemented expanding range search
- [x] Added pristine naked level filtering (no touches in either direction)
- [x] Proper support/resistance separation (strictly below/above price)
- [x] Tested with synthetic data

### 2. Zone Analysis 
- [x] 1.5% zone calculation
- [x] Level counting by type and timeframe
- [x] Naked level detection in zones
- [x] Golden pocket fib detection
- [x] Tested with synthetic data

### 3. Zone Interaction Features 
- [x] Level presence vector
- [x] Level touch vector
- [x] Support/resistance touch counting
- [x] Example scenarios

### 4. Swing Pattern Features 
- [x] Potential swing high/low detection
- [x] Swing detection logic

### 5. Candle Structure Analysis [DCP: CHECK, OK. ORDER 3]
- [x] Ratio calculations
- [x] Fractal pre-check logic

### 6. Volume Analysis  [DCP: CHECK, corregido. ORDER 2]
- [x] Volume ratio calculations
- [x] Time block features

### 7. Time Block Feature 
- [x] Implemented time block feature
- Note: match contra periodos apertura bolsas internacionales

### 8. Fractal Timing Features [DCP: CHECK, OK. ORDER 1]
- [ ] Implemented fractal timing features
- [ ] Added fractal type detection
- [ ] Tested with synthetic data

## Implementation Steps

1. **Level Management**
   - Implement level touch tracking system
   - Add logic to invalidate levels after 4 touches
   - Create efficient level search function with expanding range

2. **Zone Analysis**
   - Create zone boundary calculation
   - Implement level counting by type within zones
   - Add naked level filtering

3. **Zone Interaction Features**
   - Implement level presence vector
   - Implement level touch vector
   - Add support/resistance touch counting

4. **Swing Pattern Features**
   - Implement potential swing high/low detection
   - Implement swing detection logic

5. **Feature Generation Pipeline**
   - Create main feature generation function
   - Implement proper data alignment (n-2, n-1, n)
   - Add progress tracking and logging

## Output Format
Final dataset will have these columns for each 1-hour candle:

1. **Base Data**
   - timestamp
   - open, high, low, close, volume

2. **Zone Features** (both support and resistance)
   - All zone-related counts and ratios

3. **Zone Interaction Features**
   - Level presence vector
   - Level touch vector

4. **Swing Pattern Features**
   - Potential swing high/low detection

5. **Technical Features**
   - Candle structure ratios
   - Volume ratios
   - Time block

This dataset will be used by DataRobot to predict potential fractal formation in future candles.

## Candle Naming Convention
- N-1: Current candle (just closed)
- N-2: Previous candle
- N-3: Two candles back

## Example
```python
# Input Data
candle_n1 = {'low': 39500, 'high': 40100}  # Current candle
support_level = 39600  # Support level above current low

# Feature Detection
if candle_n1['low'] <= support_level:
    # Support touched because price went down below the level
    touched_levels['monthly_htf'] = 1
```

## DCP Revision

### Exec scripts

main -> data_fetching -> dataset_generation -> create_ml_features.

1. data_fetching.py - Gets the raw candle data from the exchange (trae de binance, no se ejecuta ahora porque tiene dataset local)
2. dataset_generation.py - Processes raw data and detects naked levels
3. create_ml_features.py - Creates features including zone detection and interactions
4. visualize_zones.py - Visualizes how zones evolve and interact

### Notes

main
dataset_generation (no hizo algo)
create_ml_features

### Dentro de create_ml_features.py

1. load data candles y levels (csv's creados en dataset gen).
2. calculate fractal timing para todas las velas. 
    - Llama a update_fractal_timing() contenido en fractal_timing.py.
    - update_fractal_timing() itera por todo el dataset de candles_df y toma grupos de a 5 velas para revisar si hay fractal.
    - analiza los grupos de 5 velas con detect_fractal() para chequear y guardar si hay fractal (swing low o swing high).

3. rellenar columnas vacias y vectores en 0.

4. iterar por cada vela.
    1. calcular volume ratios (llama calculate_volume_ratios() del archivo volume_ratios.py).
    2. calcular ratios de la vela (llama analyze_candle_ratios() del archivo candle_ratios.py).
    3. primer uso de levels_df. Asociar niveles al n-2 de la vela actual.
    4. encontrar support y resistance zones, llamando find_nearest_naked_levels() de test_nearest_levels.py. Le pasa el precio de cierre del n-2 y niveles validos para n-2 que calculamos en el paso anterior (selecciona ok las zonas).
    5. asigna fractales a la vela actual [i] segun el calculo de fractales que se hizo previamente (para todo el dataset). "update fractal timing" ml_features.loc[i, 'fractal_timing_high'] y [i, 'fractal_timing_low'].
    6. interaction_features: analiza interaccion vela n-1 contra zona/niveles n-2.
      - llama analyze_candle_interaction() de test_candle_interaction.py.
   7. 
