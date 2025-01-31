import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

from volume_ratios import calculate_volume_ratios
from candle_ratios import analyze_candle_ratios
from test_nearest_levels import find_nearest_naked_levels, get_level_vector_template
from test_candle_interaction import analyze_candle_interaction
from fractal_timing import update_fractal_timing

def create_feature_dataset(candles_file: str, levels_file: str, output_file: str) -> pd.DataFrame:
    """Create ML feature dataset from candles and levels data."""
    print("Loading datasets...")
    candles_df = pd.read_csv(candles_file)
    levels_df = pd.read_csv(levels_file)
    
    # Create a backup of the levels dataframe before saving changes
    levels_backup_file = f"levels_dataset_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Creating backup: {levels_backup_file}")
    levels_df.to_csv(levels_backup_file, index=False)
    
    # Create a backup of the candles file before overwriting it
    candles_backup_file = f"candles_dataset_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Creating backup: {candles_backup_file}")
    candles_df.to_csv(candles_backup_file, index=False)
    
    # Convert time columns
    print("Convert time columns...")
    candles_df['open_time'] = pd.to_datetime(candles_df['open_time'])
    levels_df['created_at'] = pd.to_datetime(levels_df['created_at'])
    
    # Initialize level tracking columns
    print("Initialize level tracking columns...")
    levels_df['support_touches'] = 0
    levels_df['resistance_touches'] = 0
    
    # Calculate fractal timing for all candles
    print("Calculating fractal timing...")
    fractal_timing_df = update_fractal_timing(candles_df)
    
    # Create empty DataFrame for ML features
    print("Creating empty DataFrame for ML features...")
    ml_features = pd.DataFrame()
    
    # Add basic candle data
    print("Add basic candle data...")
    ml_features['timestamp'] = candles_df['open_time']
    ml_features['open'] = candles_df['open']
    ml_features['high'] = candles_df['high']
    ml_features['low'] = candles_df['low']
    ml_features['close'] = candles_df['close']
    ml_features['volume'] = candles_df['volume']
    
    # Initialize level features
    print("Initialize level features...")
    ml_features['support_distance_pct'] = None
    ml_features['resistance_distance_pct'] = None
    ml_features['support_zone_start'] = None
    ml_features['support_zone_end'] = None
    ml_features['support_level_prices'] = None
    ml_features['resistance_zone_start'] = None
    ml_features['resistance_zone_end'] = None
    ml_features['resistance_level_prices'] = None
    ml_features['support_daily_count'] = 0
    ml_features['support_weekly_count'] = 0
    ml_features['support_monthly_count'] = 0
    ml_features['support_fib618_count'] = 0
    ml_features['support_naked_count'] = 0
    ml_features['resistance_daily_count'] = 0
    ml_features['resistance_weekly_count'] = 0
    ml_features['resistance_monthly_count'] = 0
    ml_features['resistance_fib618_count'] = 0
    ml_features['resistance_naked_count'] = 0
    
    # Initialize fractal timing features
    print("Initialize fractal timing features...")
    ml_features['fractal_timing_high'] = None
    ml_features['fractal_timing_low'] = None
    
    # Initialize vectors as lists of zeros
    print("Initialize vectors as lists of zeros...")
    ml_features['support_level_vector'] = [get_level_vector_template() for _ in range(len(candles_df))]
    ml_features['resistance_level_vector'] = [get_level_vector_template() for _ in range(len(candles_df))]
    
    # Initialize interaction features
    print("Initialize interaction features...")
    ml_features['support_touched_vector'] = [get_level_vector_template() for _ in range(len(candles_df))]
    ml_features['resistance_touched_vector'] = [get_level_vector_template() for _ in range(len(candles_df))]
    ml_features['total_support_touches'] = 0
    ml_features['total_resistance_touches'] = 0
    
    print("Processing features...")
    # Process each candle
    for i in range(2, len(candles_df)):
        if i % 100 == 0:
            print(f"Processing candle {i} of {len(candles_df)}")
            
        row = candles_df.iloc[i]
        row_n1 = candles_df.iloc[i-1] if i > 0 else None  # Previous candle
        row_n2 = candles_df.iloc[i-2] if i > 1 else None  # Candle before previous
        
        # Calculate volume ratios for this candle
        volume_features = calculate_volume_ratios(
            candle_n1={'volume': row_n1['volume']} if row_n1 is not None else None,
            volume_history=candles_df['volume'].iloc[:i+1]
        )
        for feat_name, feat_value in volume_features.items():
            ml_features.loc[i, feat_name] = feat_value
        
        # Calculate candle ratios
        candle_features = analyze_candle_ratios({
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        })
        for feat_name, feat_value in candle_features.items():
            ml_features.loc[i, feat_name] = feat_value
        
        # Get valid levels up to N-2 candle
        valid_levels = levels_df[levels_df['created_at'] <= (row_n2['open_time'] if row_n2 is not None else row['open_time'])].copy()
        
        # Get nearest levels and analyze zones from N-2 perspective
        support_zone, resistance_zone = find_nearest_naked_levels(
            price=(row_n2['close'] if row_n2 is not None else row['close']),
            levels_df=valid_levels
        )
        
        # Update level features
        if support_zone:
            ml_features.loc[i, 'support_distance_pct'] = support_zone['distance_pct']
            ml_features.loc[i, 'support_zone_start'] = support_zone['zone_start']
            ml_features.loc[i, 'support_zone_end'] = support_zone['zone_end']
            ml_features.loc[i, 'support_level_prices'] = str(support_zone['level_prices'])
            ml_features.loc[i, 'support_daily_count'] = support_zone['daily_count']
            ml_features.loc[i, 'support_weekly_count'] = support_zone['weekly_count']
            ml_features.loc[i, 'support_monthly_count'] = support_zone['monthly_count']
            ml_features.loc[i, 'support_fib618_count'] = support_zone['fib618_count']
            ml_features.loc[i, 'support_naked_count'] = support_zone['naked_count']
            ml_features.loc[i, 'support_level_vector'] = str(support_zone['level_vector'])
        
        if resistance_zone:
            ml_features.loc[i, 'resistance_distance_pct'] = resistance_zone['distance_pct']
            ml_features.loc[i, 'resistance_zone_start'] = resistance_zone['zone_start']
            ml_features.loc[i, 'resistance_zone_end'] = resistance_zone['zone_end']
            ml_features.loc[i, 'resistance_level_prices'] = str(resistance_zone['level_prices'])
            ml_features.loc[i, 'resistance_daily_count'] = resistance_zone['daily_count']
            ml_features.loc[i, 'resistance_weekly_count'] = resistance_zone['weekly_count']
            ml_features.loc[i, 'resistance_monthly_count'] = resistance_zone['monthly_count']
            ml_features.loc[i, 'resistance_fib618_count'] = resistance_zone['fib618_count']
            ml_features.loc[i, 'resistance_naked_count'] = resistance_zone['naked_count']
            ml_features.loc[i, 'resistance_level_vector'] = str(resistance_zone['level_vector'])
        
        # Update fractal timing
        ml_features.loc[i, 'fractal_timing_high'] = fractal_timing_df.loc[i, 'last_down_time']  # Swing high = fractal down
        ml_features.loc[i, 'fractal_timing_low'] = fractal_timing_df.loc[i, 'last_up_time']  # Swing low = fractal up
        
        # Check how N-1 candle interacted with zones from N-2
        if row_n1 is not None:
            interaction_features = analyze_candle_interaction(
                candle_n1={
                    'open': row_n1['open'],
                    'high': row_n1['high'],
                    'low': row_n1['low'],
                    'close': row_n1['close']
                },
                support_zone=support_zone,
                resistance_zone=resistance_zone
            )
        
            # Update interaction features
            ml_features.loc[i, 'support_touched_vector'] = str(interaction_features['support_touched_vector'])
            ml_features.loc[i, 'resistance_touched_vector'] = str(interaction_features['resistance_touched_vector'])
            ml_features.loc[i, 'total_support_touches'] = interaction_features['total_support_touches']
            ml_features.loc[i, 'total_resistance_touches'] = interaction_features['total_resistance_touches']
        else:
            # Manejar el caso cuando no hay vela anterior
            ml_features.loc[i, 'support_touched_vector'] = str(get_level_vector_template())
            ml_features.loc[i, 'resistance_touched_vector'] = str(get_level_vector_template())
            ml_features.loc[i, 'total_support_touches'] = 0
            ml_features.loc[i, 'total_resistance_touches'] = 0
        
        # ==== TOUCHED LEVELS ====
        # Update level touches in levels_df when zones are touched
        if support_zone and interaction_features['total_support_touches'] > 0:
            # Get all levels in the support zone
            zone_levels = levels_df[
                (levels_df['price_level'] >= support_zone['zone_start']) &
                (levels_df['price_level'] <= support_zone['zone_end'])
            ]
            # Only mark levels as touched if price actually reached them
            touched_levels = zone_levels[
                zone_levels['price_level'] >= (row_n1['low'] if row_n1 is not None else row['low'])
            ].index
            # Increment support touches only for actually touched levels
            levels_df.loc[touched_levels, 'support_touches'] += 1
            
        if resistance_zone and interaction_features['total_resistance_touches'] > 0:
            # Get all levels in the resistance zone
            zone_levels = levels_df[
                (levels_df['price_level'] >= resistance_zone['zone_start']) &
                (levels_df['price_level'] <= resistance_zone['zone_end'])
            ]
            # Only mark levels as touched if price actually reached them
            touched_levels = zone_levels[
                zone_levels['price_level'] <= (row_n1['high'] if row_n1 is not None else row['high'])
            ].index
            # Increment resistance touches only for actually touched levels
            levels_df.loc[touched_levels, 'resistance_touches'] += 1


    # Save the updated levels to the original file
    print(f"Updating levels file: {levels_file}")
    levels_df.to_csv(levels_file, index=False)
    
    # Save features
    print("Saving results...")
    ml_features.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
    
    return ml_features

if __name__ == "__main__":
    # Input files
    candles_file = "ml_dataset.csv"  # Main candles dataset
    levels_file = "levels_dataset.csv"  # Levels dataset
    output_file = "ml_features_dataset.csv"  # Output features
    
    # Create feature dataset
    features_df = create_feature_dataset(candles_file, levels_file, output_file)
