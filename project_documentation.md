# Trading Bot Project Documentation

## Project Overview
This trading bot project is designed for cryptocurrency trading, specifically focusing on Bitcoin. It uses advanced technical analysis features including fractal detection, volume analysis, and multi-timeframe support/resistance levels.

## Core Components

### 1. Fractal Detection System
#### Location: `fractal_timing.py`

##### Input Data Structure:
```python
candle = {
    'high': float,  # High price of the candle
    'low': float,   # Low price of the candle
    'open_time': datetime  # Candle opening time
}
candle_history = List[Dict]  # List of previous candles
```

##### Key Features:
1. **Potential Fractal Detection**:
   - Analyzes n-3, n-2, n-1 candles for potential fractal formation
   - For potential bullish fractal (swing low):
     ```
     if low(n-2) < low(n-3) and low(n-2) < low(n-1):
         potential_bullish = True
     ```
   - For potential bearish fractal (swing high):
     ```
     if high(n-2) > high(n-3) and high(n-2) > high(n-1):
         potential_bearish = True
     ```

2. **Fractal Confirmation**:
   - Requires 2 additional candles (n+1, n+2)
   - Bullish confirmation: `low(n-2) < low(n+1) and low(n-2) < low(n+2)`
   - Bearish confirmation: `high(n-2) > high(n+1) and high(n-2) > high(n+2)`

##### Output:
```python
{
    'fractal_type': int,  # 1=bullish, 2=bearish, 0=none
    'potential_fractal': bool,
    'confirmation_candles_needed': int,  # How many more candles needed for confirmation
    'fractal_price': float  # Price level of the potential/confirmed fractal
}
```

### 2. Candle Pattern Analysis
#### Location: `candle_ratios.py`

##### Input Data:
```python
candle = {
    'open': float,
    'high': float,
    'low': float,
    'close': float
}
```

##### Calculations:
1. **Body Ratio**:
   ```python
   body_size = abs(close - open)
   total_size = high - low
   body_ratio = body_size / total_size if total_size > 0 else 0
   ```

2. **Wick Ratios**:
   ```python
   upper_wick = high - max(open, close)
   lower_wick = min(open, close) - low
   upper_ratio = upper_wick / total_size if total_size > 0 else 0
   lower_ratio = lower_wick / total_size if total_size > 0 else 0
   ```

3. **Pattern Recognition**:
   - Hammer: `lower_ratio > 0.6 and upper_ratio < 0.1`
   - Shooting Star: `upper_ratio > 0.6 and lower_ratio < 0.1`
   - Doji: `body_ratio < 0.1`

##### Output:
```python
{
    'body_ratio': float,
    'upper_wick_ratio': float,
    'lower_wick_ratio': float,
    'pattern_type': str,  # 'hammer', 'shooting_star', 'doji', 'none'
    'strength': float  # Pattern strength score (0-1)
}
```

### 3. Volume Analysis
#### Location: `volume_ratios.py`

##### Input:
```python
candle = {
    'volume': float,
    'close': float
}
volume_history = pd.Series  # Historical volume data
```

##### Calculations:
1. **Volume Ratios**:
   ```python
   avg_10 = volume_history.rolling(10).mean()
   avg_20 = volume_history.rolling(20).mean()
   current_ratio_10 = current_volume / avg_10
   current_ratio_20 = current_volume / avg_20
   ```

2. **Volume Spikes**:
   - Major spike: `current_ratio_20 > 2.5`
   - Minor spike: `current_ratio_10 > 1.5`

##### Output:
```python
{
    'volume_ratio_10': float,
    'volume_ratio_20': float,
    'is_spike': bool,
    'spike_strength': float,  # 0-1 score
    'volume_trend': str  # 'increasing', 'decreasing', 'stable'
}
```

### 4. Level Touch Tracking
#### Location: `level_touch_tracker.py`

##### Input:
```python
level = {
    'price_level': float,
    'level_type': str,  # 'Fractal_Low', 'VP_poc', 'Fib_0.50', etc.
    'timeframe': str,
    'created_at': datetime
}
candle = {
    'high': float,
    'low': float,
    'close': float
}
```

##### Key Features:
1. **Touch Detection**:
   - Support touch: Price approaches from above
   - Resistance touch: Price approaches from below
   - Touch zone: ±0.1% of level price

2. **Level Strength**:
   - Based on number of touches
   - Time since creation
   - Success rate of bounces

##### Output:
```python
{
    'support_touches': int,
    'resistance_touches': int,
    'last_touch_time': datetime,
    'touch_type': str,  # 'support', 'resistance', 'none'
    'level_strength': float  # 0-1 score
}
```

### 5. Time Block Analysis
#### Location: `time_blocks.py`

##### Input:
```python
candle = {
    'open_time': datetime
}
```

##### Features:
1. **Session Detection**:
   - Asian: 00:00-08:00 UTC
   - European: 08:00-16:00 UTC
   - American: 16:00-24:00 UTC

2. **Volume Profile Periods**:
   - Daily blocks
   - Weekly aggregation
   - Monthly trends

##### Output:
```python
{
    'trading_session': str,
    'is_high_volume_period': bool,
    'historical_volatility': float,
    'session_progress': float  # 0-1 through current session
}
```

## Data Sources

### 1. Historical Data (ml_dataset.csv)
- 3 months of 1-hour candles
- OHLCV data with basic indicators
- Fractal detection results

### 2. Level Data (levels_dataset.csv)
- Support/Resistance levels
- Multiple level types:
  - Fractal levels (swing highs/lows)
  - Volume Profile (VAH, VAL, POC)
  - Fibonacci retracements
  - Higher Timeframe levels

## Feature Integration

### New Feature Generation Script
#### Location: `create_ml_features.py`

This script combines all the above components to create a comprehensive feature set for machine learning:

1. **Input Processing**:
   - Reads historical OHLCV data
   - Processes level information
   - Maintains calculation state

2. **Feature Generation**:
   - Calculates all technical indicators
   - Tracks level interactions
   - Generates time-based features

3. **Output Dataset**:
   - Combined features for ML
   - Normalized values
   - Ready for DataRobot input

## Next Steps

### Integration Tasks
1. **API Integration**:
   - Connect to exchange API
   - Implement real-time data fetching
   - Handle API rate limits

2. **Real-time Processing**:
   - Adapt feature calculations for streaming
   - Implement efficient updates
   - Maintain calculation accuracy

3. **Testing Requirements**:
   - Validate against historical results
   - Ensure calculation consistency
   - Monitor performance metrics

### Important Notes
- All calculations must maintain precision
- Real-time updates should be efficient
- Error handling is critical
- Data consistency must be maintained

## Project Location
Root Directory: `d:/GRUPO MAZELAB/9.ia/Visual Studio/trading_botWS`
