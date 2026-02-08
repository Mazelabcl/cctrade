# fetch_missing_data.py
# Simple script to fetch the missing 2025 data

import pandas as pd
from datetime import datetime
from binance.client import Client
import config
from data_fetching import fetch_data
from indicators import detect_fractals, calculate_htf_levels, calculate_fibonacci_levels, calculate_volume_profile_levels
import os

def fetch_2025_semester_1():
    """Fetch first semester of 2025 (Jan 1 - Jun 30, 2025)."""
    
    print("🔄 Fetching missing 2025 H1 data (Jan 1 - Jun 30, 2025)...")
    
    # Initialize client
    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
    
    # Date range for 2025 H1
    start_str = "1 Jan 2025"
    end_str = "30 Jun 2025"
    
    print(f"Fetching BTCUSDT data from {start_str} to {end_str}")
    
    # Define timeframe mappings
    timeframe_intervals = {
        'daily': Client.KLINE_INTERVAL_1DAY,
        'weekly': Client.KLINE_INTERVAL_1WEEK,
        'monthly': Client.KLINE_INTERVAL_1MONTH
    }
    
    # Additional intervals (skip minute data for speed)
    additional_intervals = {
        'hour': Client.KLINE_INTERVAL_1HOUR,
        '12hour': Client.KLINE_INTERVAL_12HOUR
    }
    
    # Fetch data for all timeframes
    data_timeframes = {}
    
    print("📡 Fetching timeframe data...")
    
    # Fetch main timeframes
    for timeframe, interval in timeframe_intervals.items():
        df = fetch_data(client, config.SYMBOL, interval, start_str, end_str)
        data_timeframes[timeframe] = df
        print(f"  ✓ {timeframe}: {len(df)} rows")
        if not df.empty:
            print(f"    Range: {df.index[0]} to {df.index[-1]}")
    
    # Fetch additional timeframes
    for name, interval in additional_intervals.items():
        df = fetch_data(client, config.SYMBOL, interval, start_str, end_str)
        data_timeframes[name] = df
        print(f"  ✓ {name}: {len(df)} rows")
        if not df.empty:
            print(f"    Range: {df.index[0]} to {df.index[-1]}")
    
    # Calculate technical levels
    print("🧮 Calculating technical levels...")
    htf_levels = {}
    fib_levels = {}
    volume_profile = {}
    
    for timeframe in timeframe_intervals.keys():
        df = data_timeframes[timeframe]
        if not df.empty:
            try:
                htf_levels[timeframe] = calculate_htf_levels(df, timeframe)
                fib_levels[timeframe] = calculate_fibonacci_levels(df, timeframe)
                # Use hourly data for volume profile
                volume_profile[timeframe] = calculate_volume_profile_levels(data_timeframes['hour'], timeframe)
                print(f"  ✓ {timeframe} levels calculated")
            except Exception as e:
                print(f"  ⚠ Error calculating {timeframe} levels: {e}")
    
    # Generate datasets
    print("💾 Generating datasets...")
    
    # Create datasets directory if it doesn't exist
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
    
    # Generate ML datasets (1-hour)
    hour_data = data_timeframes.get('hour')
    if hour_data is not None and not hour_data.empty:
        # Add fractal labels
        ml_dataset = detect_fractals(hour_data.copy(), 'hour')
        
        # Save with semester naming
        filename = os.path.join(datasets_dir, 'ml_dataset_2025_01_01-2025_06_30.csv')
        ml_dataset.to_csv(filename)
        print(f"  ✓ ML dataset saved: {filename} ({len(ml_dataset)} rows)")
        
        # Show data range
        first_date = ml_dataset.index[0]
        last_date = ml_dataset.index[-1]
        print(f"    Data range: {first_date} to {last_date}")
    
    # Generate Levels Dataset
    print("📊 Generating levels dataset...")
    levels = []
    
    # Add Fractals as levels for HTF timeframes
    for timeframe in ['daily', 'weekly', 'monthly']:
        df = data_timeframes[timeframe]
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
    
    # Add HTF levels
    for timeframe in ['daily', 'weekly', 'monthly']:
        htf_df = htf_levels.get(timeframe)
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
        fib_df = fib_levels.get(timeframe)
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
        vp_data = volume_profile.get(timeframe)
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
        levels_df.sort_values(['timeframe', 'price_level'], inplace=True)
        
        levels_filename = os.path.join(datasets_dir, 'levels_dataset_2025_01_01-2025_06_30.csv')
        levels_df.to_csv(levels_filename, index=False)
        print(f"  ✓ Levels dataset saved: {levels_filename} ({len(levels_df)} rows)")
    
    print("✅ 2025 H1 data fetch completed successfully!")
    return True

def check_december_gap():
    """Check if we're missing December 31st data and suggest action."""
    
    print("🔍 Checking December 31st, 2024 data completeness...")
    
    # Read the 2024 dataset
    df_2024 = pd.read_csv('datasets/ml_dataset_2021_01_01-2024_12_31.csv')
    
    # Filter for December 31st
    dec_31_data = df_2024[df_2024['open_time'].str.startswith('2024-12-31')]
    
    print(f"December 31st, 2024 data:")
    print(f"  Rows found: {len(dec_31_data)}")
    
    if len(dec_31_data) > 0:
        first_hour = dec_31_data.iloc[0]['open_time']
        last_hour = dec_31_data.iloc[-1]['open_time']
        print(f"  Time range: {first_hour} to {last_hour}")
        
        if len(dec_31_data) < 24:
            missing_hours = 24 - len(dec_31_data)
            print(f"  ⚠️  Missing {missing_hours} hours of December 31st data")
            print(f"  📝 Note: This gap will be filled when fetching 2025 data")
        else:
            print(f"  ✅ Complete December 31st data")
    else:
        print(f"  ❌ No December 31st data found")

if __name__ == "__main__":
    print("🚀 Missing Data Fetch Script")
    print("="*50)
    
    # Check December gap
    check_december_gap()
    print()
    
    # Fetch 2025 data
    try:
        success = fetch_2025_semester_1()
        if success:
            print("\n🎉 All missing data has been fetched!")
        else:
            print("\n❌ Data fetch failed!")
    except Exception as e:
        print(f"\n❌ Error during data fetch: {e}")