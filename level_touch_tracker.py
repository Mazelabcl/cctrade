import pandas as pd
from typing import Dict

def update_level_touches(levels_df: pd.DataFrame, candle_n1: dict) -> pd.DataFrame:
    """
    Update level touches based on candle N-1 interaction.
    Each level maintains separate counts for support and resistance touches.
    
    Args:
        levels_df: DataFrame containing all levels
        candle_n1: Previous candle (N-1) that just closed
        
    Returns:
        DataFrame with updated touch counts
    """
    updated_levels = levels_df.copy()
    
    for idx, level in updated_levels.iterrows():
        # Check support touch (price went down to/through level)
        if candle_n1['low'] <= level['price_level']:
            updated_levels.at[idx, 'support_touches'] += 1
            
        # Check resistance touch (price went up to/through level)
        if candle_n1['high'] >= level['price_level']:
            updated_levels.at[idx, 'resistance_touches'] += 1
    
    return updated_levels
