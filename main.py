# main.py
import pandas as pd
import logging
import argparse
import sys
import os
import config
from data_manager import DataManager

# Conditional imports for data fetching (require API access)
try:
    from binance.client import Client
    from data_fetching import fetch_data
    from indicators import detect_fractals, calculate_htf_levels, calculate_fibonacci_levels, calculate_volume_profile_levels
    from dataset_generation import generate_datasets
    BINANCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Binance API modules not available: {e}")
    print("Data fetching will be disabled. Use --features-only or install requirements.")
    BINANCE_AVAILABLE = False

# Conditional import for feature engineering
try:
    from create_ml_features import create_feature_dataset
    FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Feature engineering module not available: {e}")
    FEATURES_AVAILABLE = False

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

def run_data_pipeline(client, period_info=None):
    """Run the data fetching and basic dataset generation pipeline."""
    if period_info:
        print(f"Data pipeline would generate period: {period_info['period']}")
        print("Note: Period-specific data fetching not yet implemented")
        print("Using existing data fetching for now...")
    
    # Fetch and prepare data
    prepared_data = fetch_and_prepare_data(client)
    
    # Generate datasets
    generate_datasets(prepared_data)
    
    return True

def run_feature_pipeline(data_manager, period, sample_rows=None):
    """Run the feature engineering pipeline for a specific period."""
    period_info = data_manager.get_period_info(period)
    if not period_info:
        print(f"Error: Period {period} not found")
        return False
    
    # Check if required files exist
    required_files = ['ml_dataset', 'levels_dataset']
    missing_files = []
    
    for file_type in required_files:
        if file_type not in period_info['files']:
            missing_files.append(file_type)
        elif not os.path.exists(period_info['files'][file_type]['path']):
            missing_files.append(file_type)
    
    if missing_files:
        print(f"Error: Missing required files for period {period}: {missing_files}")
        print("Run with --data-only first to generate base datasets")
        return False
    
    # Prepare file paths
    ml_dataset_path = period_info['files']['ml_dataset']['path']
    levels_dataset_path = period_info['files']['levels_dataset']['path']
    
    # Handle sampling if requested
    if sample_rows:
        print(f"Creating sample dataset with {sample_rows} rows...")
        sample_period = data_manager.create_sample_period(period, sample_rows, f"sample_{sample_rows}")
        sample_info = data_manager.get_period_info(sample_period)
        ml_dataset_path = sample_info['files']['ml_dataset']['path']
        levels_dataset_path = sample_info['files']['levels_dataset']['path']
        output_file = f"ml_features_dataset_{sample_period}.csv"
    else:
        output_file = f"ml_features_dataset_{period}.csv"
    
    # Run feature engineering
    print(f"Running feature engineering for period {period}...")
    print(f"Input: {ml_dataset_path} ({period_info['rows']:,} rows)")
    print(f"Output: {output_file}")
    
    try:
        result_df = create_feature_dataset(ml_dataset_path, levels_dataset_path, output_file)
        print(f"Feature engineering completed successfully")
        print(f"Generated {len(result_df):,} feature rows")
        return True
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return False

def main():
    """Enhanced main function with argument parsing and period management."""
    parser = argparse.ArgumentParser(
        description="Trading Bot Pipeline with Period Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Full pipeline with default period
  python main.py --data-only                  # Only fetch data and generate basic datasets
  python main.py --features-only              # Only run feature engineering (default period)
  python main.py --period 2021_01_01-2022_12_31 --features-only
  python main.py --quick-test                 # Use small dataset for rapid testing
  python main.py --sample 1000 --features-only
  python main.py --list-periods               # Show available data periods
  python main.py --check-data                 # Validate data integrity
        """
    )
    
    # Execution mode flags
    parser.add_argument('--data-only', action='store_true',
                        help='Only fetch data and generate basic datasets (skip feature engineering)')
    parser.add_argument('--features-only', action='store_true',
                        help='Only run feature engineering (skip data fetching)')
    parser.add_argument('--full', action='store_true',
                        help='Run complete pipeline (data + features) - this is the default')
    
    # Period management
    parser.add_argument('--period', type=str,
                        help='Specific period to process (e.g., 2021_01_01-2024_12_31)')
    parser.add_argument('--all-periods', action='store_true',
                        help='Process all available periods')
    parser.add_argument('--list-periods', action='store_true',
                        help='List available data periods and exit')
    parser.add_argument('--check-data', action='store_true',
                        help='Check data integrity and show status')
    
    # Rapid development features
    parser.add_argument('--quick-test', action='store_true',
                        help='Use small dataset for rapid testing (auto-selects suitable period)')
    parser.add_argument('--sample', type=int, metavar='N',
                        help='Process only first N rows of data')
    
    args = parser.parse_args()
    
    # Initialize data manager
    try:
        data_manager = DataManager()
    except Exception as e:
        print(f"Error initializing data manager: {e}")
        return 1
    
    # Handle information requests
    if args.list_periods:
        data_manager.print_data_status()
        return 0
    
    if args.check_data:
        data_manager.print_data_status()
        issues = data_manager.validate_data_integrity()
        if any(issues.values()):
            print("\nData issues found. Please review and fix before proceeding.")
            return 1
        return 0
    
    # Determine execution mode
    run_data = True
    run_features = True
    
    if args.data_only:
        run_features = False
    elif args.features_only:
        run_data = False
    elif not args.full:
        # Default behavior: run full pipeline
        pass
    
    # Handle period selection
    periods_to_process = []
    
    if args.all_periods:
        periods_to_process = data_manager.get_available_periods()
        if not periods_to_process:
            print("No data periods found. Run with --data-only first.")
            return 1
    elif args.quick_test:
        if not args.features_only:
            print("--quick-test is typically used with --features-only")
            print("Adding --features-only automatically...")
            run_data = False
            run_features = True
        
        quick_period = data_manager.get_period_for_quick_test()
        if quick_period:
            periods_to_process = [quick_period]
            print(f"Quick test mode: using period {quick_period}")
        else:
            print("No suitable period found for quick test")
            return 1
    elif args.period:
        if args.period not in data_manager.get_available_periods():
            print(f"Error: Period {args.period} not found")
            print("Available periods:")
            for period in data_manager.get_available_periods():
                print(f"  {period}")
            return 1
        periods_to_process = [args.period]
    else:
        # Use default period
        default_period = data_manager.manifest.get('default_period')
        if default_period:
            periods_to_process = [default_period]
        elif run_features:
            print("No default period found and no period specified")
            print("Available periods:")
            for period in data_manager.get_available_periods():
                print(f"  {period}")
            return 1
    
    # Show execution plan
    print("\n" + "="*50)
    print("EXECUTION PLAN")
    print("="*50)
    print(f"Data Pipeline: {'Yes' if run_data else 'No'}")
    print(f"Feature Pipeline: {'Yes' if run_features else 'No'}")
    if periods_to_process:
        print(f"Periods: {', '.join(periods_to_process)}")
    if args.sample:
        print(f"Sample Size: {args.sample} rows")
    print("="*50 + "\n")
    
    # Check availability of required modules
    if run_data and not BINANCE_AVAILABLE:
        print("Error: Data pipeline requested but Binance modules not available")
        print("Install requirements: pip install -r requirements.txt")
        return 1
    
    if run_features and not FEATURES_AVAILABLE:
        print("Error: Feature pipeline requested but feature modules not available")
        return 1
    
    # Execute pipeline
    try:
        # Data pipeline
        if run_data:
            print("Starting data pipeline...")
            client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
            
            if periods_to_process:
                # For now, run data pipeline once regardless of periods
                # TODO: Implement period-specific data fetching
                period_info = data_manager.get_period_info(periods_to_process[0])
                success = run_data_pipeline(client, period_info)
            else:
                success = run_data_pipeline(client)
            
            if not success:
                print("Data pipeline failed")
                return 1
            
            # Refresh manifest after data generation
            data_manager = DataManager()
        
        # Feature pipeline
        if run_features:
            print("Starting feature pipeline...")
            
            for period in periods_to_process:
                print(f"\nProcessing period: {period}")
                success = run_feature_pipeline(data_manager, period, args.sample)
                
                if not success:
                    print(f"Feature pipeline failed for period {period}")
                    return 1
        
        print("\nPipeline execution completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        return 1
    except Exception as e:
        print(f"Error in pipeline execution: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
