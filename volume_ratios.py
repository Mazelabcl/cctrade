import pandas as pd
import numpy as np
from typing import Dict

def calculate_volume_ratios(candle_n1: dict, volume_history: pd.Series) -> Dict:
    """
    Calculate volume ratios for candle N-1 comparing its volume against short and long MAs.
    
    Volume ratios:
    1. Short-term (6 periods) - For detecting immediate volume spikes
    2. Long-term (168 periods) - For detecting significant volume events
    
    Args:
        candle_n1: Previous candle (N-1, just closed)
        volume_history: Pandas Series of historical volume data
        
    Returns:
        dict: Dictionary with volume ratios:
            - volume_short_ratio: Current volume / 6-period MA
            - volume_long_ratio: Current volume / 168-period MA
    """
    try:
        # Initialize results
        results = {
            'volume_short_ratio': 0,
            'volume_long_ratio': 0
        }
        
        # Calculate short-term MA if we have enough data
        if len(volume_history) >= 6:
            # Get last 6 volumes excluding current candle
            recent_volumes = volume_history.iloc[-6:]
            ma_short = recent_volumes.mean()
            
            if ma_short > 0:
                results['volume_short_ratio'] = candle_n1['volume'] / ma_short
        
        # Calculate long-term MA if we have enough data
        if len(volume_history) >= 168:
            recent_volumes = volume_history.iloc[-168:]
            ma_long = recent_volumes.mean()
            
            if ma_long > 0:
                results['volume_long_ratio'] = candle_n1['volume'] / ma_long
        
        return results
        
    except Exception as e:
        print(f"Error calculating volume ratios: {e}")
        return {
            'volume_short_ratio': 0,
            'volume_long_ratio': 0
        }
