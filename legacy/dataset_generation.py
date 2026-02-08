# dataset_generation.py

import logging
import pandas as pd
import numpy as np
from indicators import detect_fractals
from feature_engineering import identify_confluence_zones, merge_confluence_with_price_data
from config import ROUNDING_PRECISION, CONFLUENCE_THRESHOLD, PRICE_RANGE, HIT_COUNT_THRESHOLD

def generate_datasets(prepared_data):
    """
    Generate two datasets:
    1. ML Dataset: 1-hour timeframe with fractal labels
    2. Levels Dataset: All technical levels with timeframe information
    """
    try:
        # 1. Generate ML Datasets for both timeframes
        timeframes = {'hour': '1h', '12hour': '12h'}
        
        for tf_key, tf_label in timeframes.items():
            data = prepared_data['data_timeframes'].get(tf_key)
            if data is not None and not data.empty:
                # Add fractal labels
                ml_dataset = detect_fractals(data.copy(), tf_key)
                
                # Save ML dataset with all OHLCV data and fractal labels
                filename = f'ml_dataset_{tf_label}.csv'
                ml_dataset.to_csv(filename)
                print(f"ML dataset for {tf_label} timeframe saved successfully as {filename}")
        
        # 2. Generate Levels Dataset
        levels = []
        
        # Add Fractals as levels for HTF timeframes
        for timeframe in ['daily', 'weekly', 'monthly']:
            df = prepared_data['data_timeframes'][timeframe]
            if not df.empty:
                # Detect fractals
                df_with_fractals = detect_fractals(df.copy(), timeframe)
                
                # Add bullish fractals (swing lows)
                bullish_fractals = df_with_fractals[df_with_fractals['bullish_fractal'] == True]
                for idx, row in bullish_fractals.iterrows():
                    levels.append({
                        'price_level': row['low'],
                        'level_type': f'Fractal_Low',
                        'timeframe': timeframe,
                        'created_at': idx,
                        'source': 'fractal'
                    })
                
                # Add bearish fractals (swing highs)
                bearish_fractals = df_with_fractals[df_with_fractals['bearish_fractal'] == True]
                for idx, row in bearish_fractals.iterrows():
                    levels.append({
                        'price_level': row['high'],
                        'level_type': f'Fractal_High',
                        'timeframe': timeframe,
                        'created_at': idx,
                        'source': 'fractal'
                    })
                
                print(f"Added {len(bullish_fractals)} bullish and {len(bearish_fractals)} bearish fractals for {timeframe}")
        
        # Add HTF levels
        for timeframe in ['daily', 'weekly', 'monthly']:
            htf_df = prepared_data['htf_levels'].get(timeframe)
            if isinstance(htf_df, pd.DataFrame) and not htf_df.empty:
                for _, row in htf_df.iterrows():
                    levels.append({
                        'price_level': row['price_level'],
                        'level_type': row['level_type'],
                        'timeframe': timeframe,
                        'created_at': row['open_time'],
                        'source': 'htf'
                    })
        
        # Add Fibonacci levels
        for timeframe in ['daily', 'weekly', 'monthly']:
            fib_df = prepared_data['fib_levels'].get(timeframe)
            if isinstance(fib_df, pd.DataFrame) and not fib_df.empty:
                for _, row in fib_df.iterrows():
                    levels.append({
                        'price_level': row['price_level'],
                        'level_type': row['level_type'],
                        'timeframe': timeframe,
                        'created_at': row['open_time'],
                        'anchor_time': row['anchor_time'],
                        'completion_time': row['completion_time'],
                        'direction': row['direction'],
                        'source': 'fibonacci'
                    })
        
        # Add Volume Profile levels
        for timeframe in ['daily', 'weekly', 'monthly']:
            vp_data = prepared_data['volume_profile'].get(timeframe)
            if isinstance(vp_data, dict):
                for period, level_info in vp_data.items():
                    if isinstance(level_info, dict):
                        for level_type, value in level_info.items():
                            levels.append({
                                'price_level': value,
                                'level_type': f'VP_{level_type}',
                                'timeframe': timeframe,
                                'period': period,
                                'source': 'volume_profile'
                            })
        
        # Create and save levels dataset
        if levels:
            levels_df = pd.DataFrame(levels)
            # Sort by timeframe and price level for better organization
            levels_df.sort_values(['timeframe', 'price_level'], inplace=True)
            levels_df.to_csv('levels_dataset.csv', index=False)
            print("Levels dataset saved successfully")
        else:
            print("No levels found to save")
            
    except Exception as e:
        print(f"Error generating datasets: {e}")
        logging.error(f"Error generating datasets: {e}")

def create_levels_dataset(htf_levels, fib_levels):
    """Create a separate dataset for levels with hit counts and validity."""
    levels = []
    # HTF Levels
    for tf, df in htf_levels.items():
        for idx, row in df.iterrows():
            levels.append({
                'level_id': f"HTF_{tf}_{idx}",
                'price_level': row['level'],
                'time_created': row['time'],
                'level_type': 'HTF',
                'timeframe': tf
            })
    # Fibonacci Levels
    for tf, df in fib_levels.items():
        for idx, row in df.iterrows():
            levels.append({
                'level_id': f"Fib_{tf}_{idx}",
                'price_level': row['fib_level'],
                'time_created': row['start_time'],
                'level_type': 'Fibonacci',
                'fractal_type': row['type'],  # support or resistance
                'hit_count': 0,
                'validity': 1  # 1 for valid, 0 for invalid
            })
    levels_df = pd.DataFrame(levels)
    return levels_df
