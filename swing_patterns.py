import pandas as pd
from typing import Dict

def analyze_swing_patterns(candle_n1: dict, candle_n2: dict, candle_n3: dict) -> Dict:
    """
    Analyze if candle N-1 forms a potential swing pattern by comparing with N-2 and N-3.
    This is the first condition for swing pattern formation.
    
    A Swing High (Fractal Down) forms when:
    - N-1's high is higher than both N-2 and N-3's highs
    - This indicates potential downward reversal
    
    A Swing Low (Fractal Up) forms when:
    - N-1's low is lower than both N-2 and N-3's lows
    - This indicates potential upward reversal
    
    Args:
        candle_n1: Previous candle (N-1, just closed)
        candle_n2: Two candles back (N-2)
        candle_n3: Three candles back (N-3)
    
    Returns:
        dict: Dictionary with swing pattern features:
            - potential_swing_high: 1 if N-1 forms swing high (fractal down)
            - potential_swing_low: 1 if N-1 forms swing low (fractal up)
    """
    results = {
        'potential_swing_high': 0,  # Fractal Down
        'potential_swing_low': 0    # Fractal Up
    }
    
    # Check for potential swing high (fractal down)
    if (candle_n1['high'] > candle_n2['high'] and 
        candle_n1['high'] > candle_n3['high']):
        results['potential_swing_high'] = 1
    
    # Check for potential swing low (fractal up)
    if (candle_n1['low'] < candle_n2['low'] and 
        candle_n1['low'] < candle_n3['low']):
        results['potential_swing_low'] = 1
    
    return results
