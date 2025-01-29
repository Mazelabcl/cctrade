# indicators.py

import pandas as pd
import logging
import numpy as np
from tqdm import tqdm

def detect_fractals(df, timeframe):
    """
    Detect swing highs and lows (fractals) in the data.
    A bearish fractal (swing high) forms when there are 2 lower highs on each side of a high point.
    A bullish fractal (swing low) forms when there are 2 higher lows on each side of a low point.
    """
    try:
        df = df.copy()
        # Initialize fractal columns
        df['bearish_fractal'] = False
        df['bullish_fractal'] = False
        
        # Need at least 5 candles to form a fractal
        if len(df) < 5:
            return df
            
        # Detect bearish fractals (swing highs)
        for i in range(2, len(df) - 2):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                df.iloc[i, df.columns.get_loc('bearish_fractal')] = True
        
        # Detect bullish fractals (swing lows)
        for i in range(2, len(df) - 2):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                df.iloc[i, df.columns.get_loc('bullish_fractal')] = True
        
        return df
        
    except Exception as e:
        logging.error(f"Error detecting fractals: {e}")
        return df

def calculate_htf_levels(df, timeframe):
    """
    Calculate HTF levels based on candle direction changes.
    
    A level is created when:
    - Candle_i is down and candle_i+1 is up
    - Candle_i is up and candle_i+1 is down
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: The timeframe being analyzed (daily, weekly, monthly)
        
    Returns:
        DataFrame with HTF levels containing:
        - open_time: When the level was created
        - price_level: The open price where level was created
        - level_type: HTF_Level_{timeframe}
    """
    try:
        if df.empty:
            return pd.DataFrame()
            
        # Reset index if needed
        df = df.copy()
        if 'open_time' not in df.columns:
            df = df.reset_index()
            
        htf_levels = []
        
        # We need at least 2 candles to compare
        for i in range(len(df)-1):
            # Get candle directions
            candle_i_direction = 'up' if df.iloc[i]['close'] > df.iloc[i]['open'] else 'down'
            candle_i1_direction = 'up' if df.iloc[i+1]['close'] > df.iloc[i+1]['open'] else 'down'
            
            # Check for direction change
            if candle_i_direction != candle_i1_direction:
                htf_levels.append({
                    'open_time': df.iloc[i+1]['open_time'],
                    'price_level': df.iloc[i+1]['open'],
                    'level_type': f'HTF_Level_{timeframe}'
                })
        
        # Convert to DataFrame
        htf_df = pd.DataFrame(htf_levels)
        return htf_df
        
    except Exception as e:
        logging.error(f"Error calculating HTF levels for {timeframe}: {e}")
        return pd.DataFrame()

def calculate_fibonacci_levels(df, timeframe):
    """
    Calculate Fibonacci levels using quarters (0.25, 0.5, 0.75) and golden pocket (0.639).
    Each fractal becomes an anchor point that pulls to future opposite fractals if they expand the range.
    A pull remains valid until price pierces through its anchor point.
    
    For high to low pulls: levels are calculated by adding to the low
    For low to high pulls: levels are calculated by subtracting from the high
    """
    try:
        if df.empty:
            return pd.DataFrame()
            
        # Get fractals first
        df = detect_fractals(df, timeframe)
        
        # Initialize results
        fib_levels = []
        
        # Get high and low fractals
        high_fractals = df[df['bearish_fractal'] == True].copy()
        low_fractals = df[df['bullish_fractal'] == True].copy()
        
        if high_fractals.empty or low_fractals.empty:
            return pd.DataFrame()
            
        # Process each low fractal as an anchor (pulling up)
        for anchor_idx, anchor_row in low_fractals.iterrows():
            anchor_low = anchor_row['low']
            local_high = None  # Each anchor tracks its own local high
            
            # Look for valid high fractals to pull to
            future_highs = high_fractals[high_fractals.index > anchor_idx]
            for high_idx, high_row in future_highs.iterrows():
                # Check if price pierced through anchor before this target
                price_between = df[(df.index > anchor_idx) & (df.index < high_idx)]
                if any(price_between['low'] < anchor_low):
                    break  # Anchor invalidated, stop pulling
                    
                # Only pull if this high expands the range
                if local_high is None or high_row['high'] > local_high:
                    local_high = high_row['high']  # Update local high
                    
                    # Calculate range
                    range_high = high_row['high']
                    range_low = anchor_low
                    range_diff = range_high - range_low
                    
                    # Calculate quarter levels and golden pocket
                    # For low to high, we SUBTRACT from the high
                    levels = [
                        {'price': range_high - 0.25 * range_diff, 'ratio': '0.25'},
                        {'price': range_high - 0.5 * range_diff, 'ratio': '0.50'},
                        {'price': range_high - 0.639 * range_diff, 'ratio': '0.639'},
                        {'price': range_high - 0.75 * range_diff, 'ratio': '0.75'}
                    ]
                    
                    # Add levels with metadata
                    for level in levels:
                        fib_levels.append({
                            'open_time': high_idx,
                            'price_level': level['price'],
                            'level_type': f'Fib_{level["ratio"]}_{timeframe}',
                            'anchor_time': anchor_idx,
                            'completion_time': high_idx,
                            'direction': 'low_to_high'
                        })
        
        # Process each high fractal as an anchor (pulling down)
        for anchor_idx, anchor_row in high_fractals.iterrows():
            anchor_high = anchor_row['high']
            local_low = None  # Each anchor tracks its own local low
            
            # Look for valid low fractals to pull to
            future_lows = low_fractals[low_fractals.index > anchor_idx]
            for low_idx, low_row in future_lows.iterrows():
                # Check if price pierced through anchor before this target
                price_between = df[(df.index > anchor_idx) & (df.index < low_idx)]
                if any(price_between['high'] > anchor_high):
                    break  # Anchor invalidated, stop pulling
                    
                # Only pull if this low expands the range
                if local_low is None or low_row['low'] < local_low:
                    local_low = low_row['low']  # Update local low
                    
                    # Calculate range
                    range_high = anchor_high
                    range_low = low_row['low']
                    range_diff = range_high - range_low
                    
                    # Calculate quarter levels and golden pocket
                    # For high to low, we ADD from the low
                    levels = [
                        {'price': range_low + 0.25 * range_diff, 'ratio': '0.25'},
                        {'price': range_low + 0.5 * range_diff, 'ratio': '0.50'},
                        {'price': range_low + 0.639 * range_diff, 'ratio': '0.639'},
                        {'price': range_low + 0.75 * range_diff, 'ratio': '0.75'}
                    ]
                    
                    # Add levels with metadata
                    for level in levels:
                        fib_levels.append({
                            'open_time': low_idx,
                            'price_level': level['price'],
                            'level_type': f'Fib_{level["ratio"]}_{timeframe}',
                            'anchor_time': anchor_idx,
                            'completion_time': low_idx,
                            'direction': 'high_to_low'
                        })
        
        # Convert to DataFrame
        fib_df = pd.DataFrame(fib_levels)
        return fib_df
        
    except Exception as e:
        logging.error(f"Error calculating Fibonacci levels for {timeframe}: {e}")
        return pd.DataFrame()

def calculate_volume_profile(df_1min):
    """
    Calculate Volume Profile using 1-minute data.
    For each candle:
    1. Create range between low and high (rounded to integers)
    2. Divide candle volume by range size
    3. Assign volume to each price level
    """
    try:
        if df_1min.empty:
            return {}
            
        # Initialize volume by price dictionary
        volume_by_price = {}
        
        # Process each 1-minute candle
        for _, candle in df_1min.iterrows():
            # Round prices to integers
            low = round(candle['low'])
            high = round(candle['high'])
            volume = candle['volume']
            
            # Calculate price range and volume per level
            price_range = range(low, high + 1)  # +1 to include the high price
            range_size = len(price_range)
            
            if range_size > 0:  # Avoid division by zero
                volume_per_level = volume / range_size
                
                # Distribute volume to each price level
                for price in price_range:
                    if price not in volume_by_price:
                        volume_by_price[price] = 0
                    volume_by_price[price] += volume_per_level
        
        if not volume_by_price:
            return {}
            
        # Find POC (Point of Control) - price with highest volume
        poc = max(volume_by_price.items(), key=lambda x: x[1])[0]
        
        # Calculate Value Area (70% of total volume)
        total_volume = sum(volume_by_price.values())
        target_volume = total_volume * 0.7
        current_volume = volume_by_price[poc]
        
        # Sort prices for expanding from POC
        prices = sorted(volume_by_price.keys())
        poc_idx = prices.index(poc)
        
        # Initialize indices for expanding from POC
        above_idx = poc_idx - 1
        below_idx = poc_idx + 1
        
        # Expand value area until we reach 70% of volume
        while current_volume < target_volume and (above_idx >= 0 or below_idx < len(prices)):
            above_vol = volume_by_price[prices[above_idx]] if above_idx >= 0 else 0
            below_vol = volume_by_price[prices[below_idx]] if below_idx < len(prices) else 0
            
            # Add the larger volume to value area
            if above_vol >= below_vol and above_idx >= 0:
                current_volume += above_vol
                above_idx -= 1
            elif below_idx < len(prices):
                current_volume += below_vol
                below_idx += 1
            else:
                break
        
        # Get VAH and VAL
        vah = prices[above_idx + 1]
        val = prices[below_idx - 1]
        
        return {
            'poc': poc,
            'vah': vah,
            'val': val
        }
        
    except Exception as e:
        logging.error(f"Error calculating Volume Profile: {e}")
        return {}

def calculate_volume_profile_levels(data_1min, timeframe, price_precision=1, value_area_pct=0.70):
    """
    Calculate Volume Profile levels for a specific timeframe using 1-minute data.
    
    Args:
        data_1min: DataFrame with 1-minute data
        timeframe: 'daily', 'weekly', or 'monthly'
        price_precision: Round prices to this number of decimals
        value_area_pct: Percentage of volume to include in Value Area (default 70%)
    """
    try:
        logging.info(f"Calculating Volume Profile levels for {timeframe} timeframe")
        
        # Convert timestamp to datetime if it's not already
        data_1min = data_1min.copy()
        if not isinstance(data_1min.index, pd.DatetimeIndex):
            data_1min['open_time'] = pd.to_datetime(data_1min['open_time'])
            data_1min.set_index('open_time', inplace=True)
        
        # Get timeframe start and end
        if timeframe == 'daily':
            freq = 'D'
        elif timeframe == 'weekly':
            freq = 'W'
        elif timeframe == 'monthly':
            freq = 'M'
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Resample to get period boundaries
        period_boundaries = data_1min.resample(freq).agg({'close': 'last'}).index
        
        # Initialize results dictionary
        results = {}
        
        # Calculate Volume Profile for each period
        for start_time in period_boundaries[:-1]:
            end_time = start_time + pd.Timedelta(days=1 if freq == 'D' else 7 if freq == 'W' else 30)
            period_data = data_1min.loc[start_time:end_time]
            
            if not period_data.empty:
                # Calculate Volume Profile for this period
                vp = calculate_volume_profile(period_data)
                if vp:
                    period_key = start_time.strftime('%Y-%m-%d')
                    results[period_key] = vp
        
        return results
        
    except Exception as e:
        logging.error(f"Error calculating Volume Profile levels for {timeframe}: {e}")
        return {}
