from enum import IntEnum
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

class FractalType(IntEnum):
    NONE = 0
    UP = 1    # Swing Low
    DOWN = 2  # Swing High

def detect_fractal(candles: List[dict]) -> Optional[FractalType]:
    """
    Detect if candle N-3 forms a fractal pattern using N-5 to N-1.
    A swing high (fractal down) is formed when the middle candle's high is higher than 2 candles before and after.
    A swing low (fractal up) is formed when the middle candle's low is lower than 2 candles before and after.
    
    Args:
        candles: List of 5 candles [N-5, N-4, N-3, N-2, N-1]
        
    Returns:
        FractalType or None if not enough candles
    """
    if len(candles) != 5:
        return None
        
    n3 = candles[2]  # Middle candle to evaluate
    
    # Check Fractal Down (Swing High)
    # Middle candle must have highest high among 2 candles before and after
    is_down = (
        n3['high'] > candles[0]['high'] and  # Higher than N-5
        n3['high'] > candles[1]['high'] and  # Higher than N-4
        n3['high'] > candles[3]['high'] and  # Higher than N-2
        n3['high'] > candles[4]['high']      # Higher than N-1
    )
    
    if is_down:
        return FractalType.DOWN
        
    # Check Fractal Up (Swing Low)
    # Middle candle must have lowest low among 2 candles before and after
    is_up = (
        n3['low'] < candles[0]['low'] and  # Lower than N-5
        n3['low'] < candles[1]['low'] and  # Lower than N-4
        n3['low'] < candles[3]['low'] and  # Lower than N-2
        n3['low'] < candles[4]['low']      # Lower than N-1
    )
    
    if is_up:
        return FractalType.UP
        
    return FractalType.NONE

def update_fractal_timing(candles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Update fractal information for each candle.
    Adds/updates columns:
    - fractal_type: Type of fractal (0=None, 1=Up/SwingLow, 2=Down/SwingHigh)
    - candles_since_last_up: Candles since last swing low
    - candles_since_last_down: Candles since last swing high
    - last_up_time: Timestamp of last swing low
    - last_down_time: Timestamp of last swing high
    
    Args:
        candles_df: DataFrame with candle data
        
    Returns:
        DataFrame with updated fractal information
    """
    df = candles_df.copy()
    
    # Initialize new columns if they don't exist
    if 'fractal_type' not in df.columns:
        df['fractal_type'] = FractalType.NONE
    if 'candles_since_last_up' not in df.columns:
        df['candles_since_last_up'] = 0
    if 'candles_since_last_down' not in df.columns:
        df['candles_since_last_down'] = 0
    if 'last_up_time' not in df.columns:
        df['last_up_time'] = None
    if 'last_down_time' not in df.columns:
        df['last_down_time'] = None
    
    # We need at least 5 candles to detect fractals
    if len(df) < 5:
        return df
    
    # Process each candle from index 4 onwards (so we have enough history)
    last_up_count = 0
    last_down_count = 0
    last_up_time = None
    last_down_time = None
    
    for i in range(4, len(df)):
        # Get 5 candles for fractal detection
        candles = df.iloc[i-4:i+1].to_dict('records')
        
        # Detect fractal
        fractal = detect_fractal(candles)
        df.at[i-2, 'fractal_type'] = fractal  # Update N-3
        
        # Update timing information
        if fractal == FractalType.UP:
            last_up_count = 0
            last_up_time = df.iloc[i-2]['open_time']
        elif fractal == FractalType.DOWN:
            last_down_count = 0
            last_down_time = df.iloc[i-2]['open_time']
            
        # Update counts for current candle
        df.at[i, 'candles_since_last_up'] = last_up_count
        df.at[i, 'candles_since_last_down'] = last_down_count
        df.at[i, 'last_up_time'] = last_up_time
        df.at[i, 'last_down_time'] = last_down_time
        
        # Increment counts
        last_up_count += 1
        last_down_count += 1
    
    return df
