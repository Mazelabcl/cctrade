import pandas as pd
from typing import Dict

def analyze_candle_ratios(candle_n1: dict) -> Dict:
    """
    Analyze various ratios of candle N-1 to identify potential reversal signals.
    
    Key ratios analyzed:
    1. Upper Wick Ratio = Upper wick length / Total candle length
       - High upper wick with resistance touch could indicate reversal down
    
    2. Lower Wick Ratio = Lower wick length / Total candle length
       - High lower wick with support touch could indicate reversal up
    
    3. Body to Total Ratio = Body length / Total candle length
       - Small body with large wicks suggests indecision
       - Large body suggests strong momentum
    
    4. Body Position Ratio = (Close - Low) / (High - Low)
       - Shows where body closes within the candle's range
       - Close to 1: Strong bullish
       - Close to 0: Strong bearish
       - Around 0.5: Indecision
    
    Args:
        candle_n1: Previous candle (N-1, just closed)
    
    Returns:
        dict: Dictionary with candle ratios:
            - upper_wick_ratio: Length of upper wick relative to total length
            - lower_wick_ratio: Length of lower wick relative to total length
            - body_total_ratio: Length of body relative to total length
            - body_position_ratio: Where body closes within candle range
    """
    # Get candle measurements
    high = candle_n1['high']
    low = candle_n1['low']
    open_price = candle_n1['open']
    close = candle_n1['close']
    
    # Calculate total candle length
    total_length = high - low
    if total_length == 0:  # Avoid division by zero
        return {
            'upper_wick_ratio': 0,
            'lower_wick_ratio': 0,
            'body_total_ratio': 1,
            'body_position_ratio': 0.5
        }
    
    # Calculate body high/low
    body_high = max(open_price, close)
    body_low = min(open_price, close)
    body_length = body_high - body_low
    
    # Calculate wick lengths
    upper_wick = high - body_high
    lower_wick = body_low - low
    
    # Calculate ratios
    results = {
        'upper_wick_ratio': upper_wick / total_length,
        'lower_wick_ratio': lower_wick / total_length,
        'body_total_ratio': body_length / total_length,
        'body_position_ratio': (close - low) / total_length
    }
    
    return results
