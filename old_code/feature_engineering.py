# feature_engineering.py

import pandas as pd
import numpy as np
import logging
from config import ROUNDING_PRECISION, CONFLUENCE_THRESHOLD, PRICE_RANGE

def assign_levels_to_ranges(levels_df, price_range):
    """Assign levels to price buckets."""
    if levels_df.empty:
        return levels_df
    
    if 'price_level' not in levels_df.columns:
        print("Warning: 'price_level' column not found in levels dataset")
        return levels_df
        
    levels_df['price_bucket'] = (levels_df['price_level'] // price_range) * price_range
    return levels_df

def identify_confluence_zones(levels_df, price_range, confluence_threshold):
    """Identify zones where multiple levels converge."""
    if levels_df.empty:
        return pd.DataFrame()
        
    if 'price_level' not in levels_df.columns:
        print("Warning: 'price_level' column not found in levels dataset")
        return pd.DataFrame()
        
    # Assign levels to price buckets
    levels_df = assign_levels_to_ranges(levels_df, price_range)
    
    # Count levels in each bucket
    confluence_zones = levels_df.groupby('price_bucket').agg({
        'price_level': 'count',
        'level_type': lambda x: list(x),
        'timeframe': lambda x: list(x)
    }).reset_index()
    
    # Rename columns
    confluence_zones.columns = ['price_bucket', 'level_count', 'level_types', 'timeframes']
    
    # Filter by confluence threshold
    confluence_zones = confluence_zones[confluence_zones['level_count'] >= confluence_threshold]
    
    return confluence_zones

def merge_confluence_with_price_data(price_data, confluence_zones, price_range):
    """Merge confluence zones information with price data."""
    if price_data.empty or confluence_zones.empty:
        return price_data
        
    # Create price buckets for the price data
    price_data['price_bucket'] = (price_data['close'] // price_range) * price_range
    
    # Merge with confluence zones
    price_data = pd.merge(price_data, confluence_zones, on='price_bucket', how='left')
    
    # Drop the price bucket column
    price_data.drop('price_bucket', axis=1, inplace=True)
    
    return price_data
