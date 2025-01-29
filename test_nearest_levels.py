import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

def get_level_vector_template() -> List[int]:
    """Get template for level vector with all possible level types."""
    # Return a list of 30 zeros (one for each level type)
    return [0] * 30

def get_level_type_index(level_type: str, timeframe: str) -> int:
    """Convert level type and timeframe to vector index."""
    # Order: daily, weekly, monthly for each type
    # Types: htf, vah, val, poc, fib25, fib50, fib618, fib75, fractal_up, fractal_down
    timeframe_offset = {'daily': 0, 'weekly': 10, 'monthly': 20}
    type_offset = {
        'htf': 0, 'vah': 1, 'val': 2, 'poc': 3,
        'fib25': 4, 'fib50': 5, 'fib618': 6, 'fib75': 7,
        'fractal_up': 8, 'fractal_down': 9
    }
    
    base_type = level_type.lower()
    if 'fib' in base_type or 'fibonacci' in base_type:
        if '25' in base_type or '0.25' in base_type:
            base_type = 'fib25'
        elif '50' in base_type or '0.5' in base_type:
            base_type = 'fib50'
        elif '618' in base_type or '0.618' in base_type:
            base_type = 'fib618'
        elif '75' in base_type or '0.75' in base_type:
            base_type = 'fib75'
        else:
            base_type = 'htf'  # Still keep HTF as fallback for unknown fibs
    elif 'fractal' in base_type:
        if 'low' in base_type:
            base_type = 'fractal_up'  # Fractal low = swing low = potential up move
        else:
            base_type = 'fractal_down'  # Fractal high = swing high = potential down move
    elif base_type.startswith('vp_'):
        if 'vah' in base_type:
            base_type = 'vah'
        elif 'val' in base_type:
            base_type = 'val'
        elif 'poc' in base_type:
            base_type = 'poc'
    elif base_type.startswith('htf'):
        base_type = 'htf'
    else:
        base_type = 'htf'  # Default to HTF for unknown types
    
    return timeframe_offset[timeframe.lower()] + type_offset[base_type]

def find_nearest_naked_levels(price: float, levels_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Find nearest pristine naked levels and analyze their zones.
    Only pristine levels can start a zone, but touched levels can be included
    if they have less than 4 touches in their respective role.
    
    Args:
        price: Price from candle N-2 to search from
        levels_df: DataFrame containing all levels
    
    Returns:
        Tuple containing support and resistance zone dictionaries, each with:
            - zone_start: Start price of zone
            - zone_end: End price of zone
            - distance_pct: Distance from price to nearest naked level (%)
            - level_vector: Vector showing count of each level type in zone (List[int])
            - level_prices: List of level prices in zone
            - daily_count: Number of daily levels in zone
            - weekly_count: Number of weekly levels in zone
            - monthly_count: Number of monthly levels in zone
            - fib618_count: Number of fib618 levels in zone
            - naked_count: Number of naked levels in zone
    """
    if len(levels_df) == 0:
        return None, None
    
    # Initialize vectors for counting level types
    support_vector = get_level_vector_template()
    resistance_vector = get_level_vector_template()
    
    # Find nearest naked support (below price)
    support_levels = levels_df[levels_df['price_level'] <= price].copy()
    support_naked = support_levels[
        (support_levels['support_touches'] == 0) & 
        (support_levels['resistance_touches'] == 0)  # Must be completely naked
    ]
    
    # Find nearest naked resistance (above price)
    resistance_levels = levels_df[levels_df['price_level'] >= price].copy()
    resistance_naked = resistance_levels[
        (resistance_levels['resistance_touches'] == 0) & 
        (resistance_levels['support_touches'] == 0)  # Must be completely naked
    ]
    
    support_zone = {}
    if len(support_naked) > 0:
        # Get nearest naked support (highest below price)
        nearest_support = support_naked.iloc[support_naked['price_level'].argmax()]
        support_start = nearest_support['price_level']
        
        # Calculate distance percentage
        distance_pct = ((price - support_start) / price) * 100
        
        # Define zone width (1.5% of price)
        zone_width = price * 0.015
        support_end = support_start - zone_width  # Expand downward
        
        # Get all levels in zone
        zone_levels = support_levels[
            (support_levels['price_level'] >= support_end) & 
            (support_levels['price_level'] <= support_start)
        ]
        
        # Count level types in zone
        for _, level in zone_levels.iterrows():
            idx = get_level_type_index(level['level_type'], level['timeframe'])
            support_vector[idx] += 1
        
        # Count level categories
        daily_count = len(zone_levels[zone_levels['timeframe'] == 'daily'])
        weekly_count = len(zone_levels[zone_levels['timeframe'] == 'weekly'])
        monthly_count = len(zone_levels[zone_levels['timeframe'] == 'monthly'])
        fib618_count = len(zone_levels[zone_levels['level_type'].str.contains('618', na=False)])
        naked_count = len(zone_levels[
            (zone_levels['support_touches'] == 0) & 
            (zone_levels['resistance_touches'] == 0)
        ])
        
        # Get list of level prices in zone
        level_prices = sorted(zone_levels['price_level'].tolist())
        
        support_zone = {
            'zone_start': support_end,
            'zone_end': support_start,
            'distance_pct': distance_pct,
            'level_vector': support_vector,
            'level_prices': level_prices,
            'daily_count': daily_count,
            'weekly_count': weekly_count,
            'monthly_count': monthly_count,
            'fib618_count': fib618_count,
            'naked_count': naked_count
        }
    
    resistance_zone = {}
    if len(resistance_naked) > 0:
        # Get nearest naked resistance (lowest above price)
        nearest_resistance = resistance_naked.iloc[resistance_naked['price_level'].argmin()]
        resistance_end = nearest_resistance['price_level']
        
        # Calculate distance percentage
        distance_pct = ((resistance_end - price) / price) * 100
        
        # Define zone width (1.5% of price)
        zone_width = price * 0.015
        resistance_start = resistance_end + zone_width  # Expand upward
        
        # Get all levels in zone
        zone_levels = resistance_levels[
            (resistance_levels['price_level'] >= resistance_end) & 
            (resistance_levels['price_level'] <= resistance_start)
        ]
        
        # Count level types in zone
        for _, level in zone_levels.iterrows():
            idx = get_level_type_index(level['level_type'], level['timeframe'])
            resistance_vector[idx] += 1
        
        # Count level categories
        daily_count = len(zone_levels[zone_levels['timeframe'] == 'daily'])
        weekly_count = len(zone_levels[zone_levels['timeframe'] == 'weekly'])
        monthly_count = len(zone_levels[zone_levels['timeframe'] == 'monthly'])
        fib618_count = len(zone_levels[zone_levels['level_type'].str.contains('618', na=False)])
        naked_count = len(zone_levels[
            (zone_levels['support_touches'] == 0) & 
            (zone_levels['resistance_touches'] == 0)
        ])
        
        # Get list of level prices in zone
        level_prices = sorted(zone_levels['price_level'].tolist())
        
        resistance_zone = {
            'zone_start': resistance_end,
            'zone_end': resistance_start,
            'distance_pct': distance_pct,
            'level_vector': resistance_vector,
            'level_prices': level_prices,
            'daily_count': daily_count,
            'weekly_count': weekly_count,
            'monthly_count': monthly_count,
            'fib618_count': fib618_count,
            'naked_count': naked_count
        }
    
    return support_zone, resistance_zone
