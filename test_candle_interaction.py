import pandas as pd
from typing import Dict, List
from test_nearest_levels import get_level_vector_template

def analyze_candle_interaction(candle_n1: dict, support_zone: dict, resistance_zone: dict) -> Dict[str, List[int]]:
    """
    Analyze how candle N-1 interacts with support and resistance zones.
    A zone is considered touched as soon as price reaches the level that initiated the zone.
    Only levels actually touched by price are counted and added to the touched vector.
    
    Args:
        candle_n1: Previous candle (N-1, just closed) with keys: open, high, low, close
        support_zone: Support zone dict with keys: zone_start, zone_end, level_vector, level_prices
        resistance_zone: Resistance zone dict with keys: zone_start, zone_end, level_vector, level_prices
    
    Returns:
        dict: Dictionary containing:
            - support_touched_vector: List[int] showing which levels were touched in support zone
            - resistance_touched_vector: List[int] showing which levels were touched in resistance zone
            - total_support_touches: Number of levels touched in support zone
            - total_resistance_touches: Number of levels touched in resistance zone
    """
    # Initialize results with empty vectors
    result = {
        'support_touched_vector': get_level_vector_template(),
        'resistance_touched_vector': get_level_vector_template(),
        'total_support_touches': 0,
        'total_resistance_touches': 0
    }
    
    # Check support zone touches
    if support_zone and candle_n1['low'] <= support_zone['zone_end']:
        # Get the touched levels and their indices
        touched_levels = [
            (idx, price) for idx, price in enumerate(support_zone['level_prices'])
            if price >= candle_n1['low']  # Level was touched if price went below or equal to it
        ]
        
        # Create vector counting only touched levels
        touched_vector = get_level_vector_template()
        for idx, _ in touched_levels:
            touched_vector[idx] = 1  # Count each touched level once
            
        result['support_touched_vector'] = touched_vector
        result['total_support_touches'] = len(touched_levels)
    
    # Check resistance zone touches
    if resistance_zone and candle_n1['high'] >= resistance_zone['zone_start']:
        # Get the touched levels and their indices
        touched_levels = [
            (idx, price) for idx, price in enumerate(resistance_zone['level_prices'])
            if price <= candle_n1['high']  # Level was touched if price went above or equal to it
        ]
        
        # Create vector counting only touched levels
        touched_vector = get_level_vector_template()
        for idx, _ in touched_levels:
            touched_vector[idx] = 1  # Count each touched level once
            
        result['resistance_touched_vector'] = touched_vector
        result['total_resistance_touches'] = len(touched_levels)
    
    return result

if __name__ == "__main__":
    pass
