# main.py
import pandas as pd
import logging
from binance.client import Client
import config
from data_fetching import fetch_data
from indicators import detect_fractals, calculate_htf_levels, calculate_fibonacci_levels, calculate_volume_profile_levels
from dataset_generation import generate_datasets

def fetch_and_prepare_data(client):
    """Fetch data and prepare all levels."""
    symbol = config.SYMBOL
    start_str = config.START_DATE
    end_str = config.END_DATE
    print(f"Fetching data for {symbol} from {start_str} to {end_str}")
    
    # Define timeframe mappings
    timeframe_intervals = {
        'daily': Client.KLINE_INTERVAL_1DAY,
        'weekly': Client.KLINE_INTERVAL_1WEEK,
        'monthly': Client.KLINE_INTERVAL_1MONTH
    }
    
    # Additional intervals for ML dataset and Volume Profile
    additional_intervals = {
        'minute': Client.KLINE_INTERVAL_1MINUTE,
        'hour': Client.KLINE_INTERVAL_1HOUR,
        '12hour': Client.KLINE_INTERVAL_12HOUR
    }
    
    # Fetch data for all timeframes
    data_timeframes = {}
    
    # Fetch main timeframes
    for timeframe, interval in timeframe_intervals.items():
        df = fetch_data(client, symbol, interval, start_str, end_str)
        data_timeframes[timeframe] = df
        print(f"Fetched {timeframe} data: {len(df)} rows")
        if df.empty:
            print(f"No data available for {timeframe} timeframe")
        else:
            print(f"{timeframe} data range: {df.index[0]} to {df.index[-1]}")
    
    # Fetch additional timeframes
    for name, interval in additional_intervals.items():
        df = fetch_data(client, symbol, interval, start_str, end_str)
        data_timeframes[name] = df
        print(f"Fetched {name} data: {len(df)} rows")
        if not df.empty:
            print(f"{name} data range: {df.index[0]} to {df.index[-1]}")
    
    # Initialize dictionaries for levels
    htf_levels = {}
    fib_levels = {}
    volume_profile = {}
    
    # Calculate technical levels for main timeframes
    for timeframe in timeframe_intervals.keys():
        df = data_timeframes[timeframe]
        if not df.empty:
            try:
                htf_levels[timeframe] = calculate_htf_levels(df, timeframe)
                fib_levels[timeframe] = calculate_fibonacci_levels(df, timeframe)
                # Use minute data for volume profile
                volume_profile[timeframe] = calculate_volume_profile_levels(data_timeframes['minute'], timeframe)
                print(f"Calculated levels for {timeframe}")
            except Exception as e:
                print(f"Error calculating levels for {timeframe}: {e}")
    
    return {
        'data_timeframes': data_timeframes,
        'htf_levels': htf_levels,
        'fib_levels': fib_levels,
        'volume_profile': volume_profile
    }

def main():
    """Main function to run the trading bot."""
    try:
        # Initialize Binance client
        client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
        
        # Fetch and prepare data
        prepared_data = fetch_and_prepare_data(client)
        
        # Generate datasets
        generate_datasets(prepared_data)
        
        print("Trading bot execution completed successfully")
        
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
