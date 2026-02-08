# fetch_december_31.py
# Fetch the missing December 31st, 2024 data and append to existing dataset

import pandas as pd
from datetime import datetime
from binance.client import Client
import config
from data_fetching import fetch_data
from indicators import detect_fractals
import os

def fetch_december_31_2024():
    """Fetch the missing 23 hours of December 31st, 2024."""
    
    print("🔄 Fetching missing December 31st, 2024 data...")
    
    # Initialize client
    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
    
    # Fetch December 31st to January 1st to ensure we get all December 31st hours
    start_str = "31 Dec 2024"
    end_str = "1 Jan 2025"
    
    print(f"Fetching BTCUSDT hourly data for {start_str}")
    
    # Fetch hourly data for December 31st
    hour_data = fetch_data(client, config.SYMBOL, Client.KLINE_INTERVAL_1HOUR, start_str, end_str)
    
    if hour_data.empty:
        print("❌ No data fetched for December 31st")
        return False
    
    print(f"  ✓ Fetched {len(hour_data)} hours of data")
    print(f"  📅 Range: {hour_data.index[0]} to {hour_data.index[-1]}")
    
    # Add fractal labels
    hour_data_with_fractals = detect_fractals(hour_data.copy(), 'hour')
    
    # Load existing 2024 dataset
    existing_file = 'datasets/ml_dataset_2021_01_01-2024_12_31.csv'
    existing_df = pd.read_csv(existing_file)
    
    print(f"\\n📊 Current 2024 dataset:")
    print(f"  Rows: {len(existing_df)}")
    print(f"  Last entry: {existing_df.iloc[-1]['open_time']}")
    
    # Convert new data to same format as existing
    hour_data_with_fractals.reset_index(inplace=True)
    
    # Filter out the existing December 31st 00:00 hour to avoid duplication
    existing_dec_31 = existing_df[existing_df['open_time'].str.startswith('2024-12-31')]
    if len(existing_dec_31) > 0:
        print(f"  Found {len(existing_dec_31)} existing Dec 31st entries (will avoid duplication)")
        
        # Remove existing December 31st from new data to avoid duplication
        new_data_filtered = hour_data_with_fractals[
            ~hour_data_with_fractals['open_time'].astype(str).str.startswith('2024-12-31 00:00:00')
        ]
    else:
        new_data_filtered = hour_data_with_fractals
    
    if len(new_data_filtered) == 0:
        print("✅ No new data to add (all hours already exist)")
        return True
    
    print(f"  📈 Adding {len(new_data_filtered)} new hours of December 31st data")
    
    # Combine existing data with new December 31st data
    combined_df = pd.concat([existing_df, new_data_filtered], ignore_index=True)
    
    # Sort by timestamp to ensure proper order
    combined_df['open_time'] = pd.to_datetime(combined_df['open_time'])
    combined_df = combined_df.sort_values('open_time')
    combined_df['open_time'] = combined_df['open_time'].astype(str)
    
    # Create backup of original file
    backup_file = f"datasets/ml_dataset_2021_01_01-2024_12_31_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    existing_df.to_csv(backup_file, index=False)
    print(f"  💾 Backup created: {backup_file}")
    
    # Save updated dataset
    combined_df.to_csv(existing_file, index=False)
    
    print(f"\\n✅ Updated 2024 dataset:")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  First entry: {combined_df.iloc[0]['open_time']}")
    print(f"  Last entry: {combined_df.iloc[-1]['open_time']}")
    
    # Show December 31st coverage
    dec_31_final = combined_df[combined_df['open_time'].str.startswith('2024-12-31')]
    print(f"  December 31st coverage: {len(dec_31_final)} hours")
    if len(dec_31_final) > 0:
        print(f"    First: {dec_31_final.iloc[0]['open_time']}")
        print(f"    Last: {dec_31_final.iloc[-1]['open_time']}")
    
    return True

def verify_continuity():
    """Verify data continuity after the update."""
    print("\\n🔍 Verifying data continuity...")
    
    # Load datasets
    df_2024 = pd.read_csv('datasets/ml_dataset_2021_01_01-2024_12_31.csv')
    df_2025 = pd.read_csv('datasets/ml_dataset_2025_01_01-2025_06_30.csv')
    
    # Check transition
    last_2024 = df_2024.iloc[-1]['open_time']
    first_2025 = df_2025.iloc[0]['open_time']
    
    print(f"Data transition:")
    print(f"  Last 2024: {last_2024}")
    print(f"  First 2025: {first_2025}")
    
    # Parse dates to check gap
    last_2024_dt = pd.to_datetime(last_2024)
    first_2025_dt = pd.to_datetime(first_2025)
    
    expected_next = last_2024_dt + pd.Timedelta(hours=1)
    
    if expected_next == first_2025_dt:
        print(f"  ✅ Perfect continuity (no gap)")
    else:
        gap_hours = (first_2025_dt - expected_next).total_seconds() / 3600
        if gap_hours > 0:
            print(f"  ⚠️ Gap: {gap_hours} hours missing")
        else:
            print(f"  ⚠️ Overlap: {abs(gap_hours)} hours duplicated")

if __name__ == "__main__":
    print("🚀 December 31st, 2024 Data Completion Script")
    print("="*60)
    
    try:
        success = fetch_december_31_2024()
        
        if success:
            verify_continuity()
            print("\\n🎉 December 31st data completion successful!")
        else:
            print("\\n❌ December 31st data fetch failed!")
            
    except Exception as e:
        print(f"\\n❌ Error during data fetch: {e}")
        import traceback
        traceback.print_exc()