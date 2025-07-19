import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_price(start_price, volatility, trend, n_hours):
    """Generate synthetic price data with trends and volatility"""
    prices = [start_price]
    for _ in range(n_hours - 1):
        # Random walk with trend
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change/100)
        prices.append(new_price)
    return prices

def create_synthetic_candles(start_date, n_hours=336):  # 336 hours = 2 weeks
    """Create synthetic OHLCV data with realistic patterns"""
    
    # Initialize base parameters
    start_price = 40000.0
    base_volume = 1000.0
    
    # Create time series
    dates = [start_date + timedelta(hours=i) for i in range(n_hours)]
    
    # Generate different price series for different market phases
    prices = []
    volumes = []
    fractals = {'bearish': [], 'bullish': []}
    
    # Create different market phases
    phases = [
        {'hours': 48, 'trend': 0.02, 'volatility': 0.3},   # Slight uptrend
        {'hours': 72, 'trend': -0.05, 'volatility': 0.5},  # Strong downtrend
        {'hours': 96, 'trend': 0.01, 'volatility': 0.2},   # Consolidation
        {'hours': 72, 'trend': 0.08, 'volatility': 0.4},   # Strong uptrend
        {'hours': 48, 'trend': -0.02, 'volatility': 0.3},  # Slight downtrend
    ]
    
    current_price = start_price
    for phase in phases:
        phase_prices = generate_synthetic_price(current_price, phase['volatility'], phase['trend'], phase['hours'])
        prices.extend(phase_prices)
        current_price = phase_prices[-1]
        
        # Generate volumes (higher in trending phases)
        phase_volumes = np.random.normal(base_volume, base_volume * phase['volatility'], phase['hours'])
        volumes.extend(phase_volumes)
    
    # Create OHLCV data
    data = []
    for i in range(len(dates)):
        close = prices[i]
        # Generate OHLC around close price
        high = close * (1 + abs(np.random.normal(0, 0.002)))
        low = close * (1 - abs(np.random.normal(0, 0.002)))
        open_price = close * (1 + np.random.normal(0, 0.001))
        
        # Detect fractals (simplified for synthetic data)
        is_bearish = False
        is_bullish = False
        if i >= 2 and i < len(prices) - 2:
            # Bearish fractal
            if prices[i] > prices[i-2] and prices[i] > prices[i-1] and \
               prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                is_bearish = True
            # Bullish fractal
            if prices[i] < prices[i-2] and prices[i] < prices[i-1] and \
               prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                is_bullish = True
        
        data.append({
            'open_time': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volumes[i],
            'close_time': dates[i] + timedelta(hours=1) - timedelta(milliseconds=1),
            'quote_asset_volume': volumes[i] * close,
            'number_of_trades': int(volumes[i] * 10),
            'taker_buy_base_asset_volume': volumes[i] * 0.4,
            'taker_buy_quote_asset_volume': volumes[i] * 0.4 * close,
            'ignore': 0,
            'bearish_fractal': is_bearish,
            'bullish_fractal': is_bullish
        })
    
    return pd.DataFrame(data)

def create_synthetic_levels(candles_df):
    """Create synthetic levels based on price action"""
    levels = []
    
    # Track significant price points
    for i in range(len(candles_df)):
        row = candles_df.iloc[i]
        
        # Add fractal levels
        if row['bearish_fractal']:
            levels.append({
                'price_level': row['high'],
                'level_type': 'Fractal_High',
                'timeframe': 'daily',
                'created_at': row['open_time'],
                'source': 'fractal'
            })
        elif row['bullish_fractal']:
            levels.append({
                'price_level': row['low'],
                'level_type': 'Fractal_Low',
                'timeframe': 'daily',
                'created_at': row['open_time'],
                'source': 'fractal'
            })
        
        # Add Volume Profile levels every 24 hours
        if i % 24 == 0:
            # VP POC near VWAP
            vwap = (row['high'] + row['low'] + row['close']) / 3
            levels.append({
                'price_level': vwap,
                'level_type': 'VP_poc',
                'timeframe': 'daily',
                'created_at': row['open_time'],
                'source': 'volume_profile'
            })
            # VP VAH and VAL
            levels.append({
                'price_level': vwap * 1.005,
                'level_type': 'VP_vah',
                'timeframe': 'daily',
                'created_at': row['open_time'],
                'source': 'volume_profile'
            })
            levels.append({
                'price_level': vwap * 0.995,
                'level_type': 'VP_val',
                'timeframe': 'daily',
                'created_at': row['open_time'],
                'source': 'volume_profile'
            })
        
        # Add HTF levels at significant swings (every 48 hours)
        if i % 48 == 0:
            levels.append({
                'price_level': row['high'],
                'level_type': 'HTF_Level_daily',
                'timeframe': 'daily',
                'created_at': row['open_time'],
                'source': 'htf'
            })
    
    return pd.DataFrame(levels)

if __name__ == "__main__":
    # Generate synthetic data
    start_date = datetime(2024, 1, 1)
    
    # Create candles
    candles_df = create_synthetic_candles(start_date)
    candles_df.to_csv("test_data/ml_dataset_test_v3.csv", index=False)
    
    # Create levels
    levels_df = create_synthetic_levels(candles_df)
    levels_df.to_csv("test_data/levels_dataset_test_v3.csv", index=False)
    
    print(f"Created {len(candles_df)} candles and {len(levels_df)} levels in test_data/")
