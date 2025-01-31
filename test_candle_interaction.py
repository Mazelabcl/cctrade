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
    
    # Check if candle_n1 is None
    if candle_n1 is None:
        return result
    
    # Process support zone
    if support_zone and 'low' in candle_n1 and 'zone_end' in support_zone and 'level_vector' in support_zone:
        if candle_n1['low'] <= support_zone['zone_end']:
            # Create touched vector based on the original level_vector
            touched_vector = get_level_vector_template()
            original_vector = support_zone['level_vector']
            
            # Only mark as touched the levels that exist in the original vector
            for i in range(len(touched_vector)):
                if i < len(original_vector) and original_vector[i] > 0:
                    touched_vector[i] = 1
            
            touched_count = sum(1 for i in range(len(touched_vector)) 
                              if i < len(original_vector) and original_vector[i] > 0)
            
            result['support_touched_vector'] = touched_vector
            result['total_support_touches'] = touched_count
    
    # Process resistance zone
    if resistance_zone and 'high' in candle_n1 and 'zone_start' in resistance_zone and 'level_vector' in resistance_zone:
        if candle_n1['high'] >= resistance_zone['zone_start']:
            # Create touched vector based on the original level_vector
            touched_vector = get_level_vector_template()
            original_vector = resistance_zone['level_vector']
            
            # Only mark as touched the levels that exist in the original vector
            for i in range(len(touched_vector)):
                if i < len(original_vector) and original_vector[i] > 0:
                    touched_vector[i] = 1
            
            touched_count = sum(1 for i in range(len(touched_vector)) 
                              if i < len(original_vector) and original_vector[i] > 0)
            
            result['resistance_touched_vector'] = touched_vector
            result['total_resistance_touches'] = touched_count
    
    return result

if __name__ == "__main__":
    pass
