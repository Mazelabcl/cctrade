# smart_data_fetcher.py

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import re
from binance.client import Client
from data_fetching import fetch_data
from indicators import detect_fractals, calculate_htf_levels, calculate_fibonacci_levels, calculate_volume_profile_levels
from dataset_generation import generate_datasets
import config

class SmartDataFetcher:
    """Smart data fetcher that only fetches missing periods and organizes by semester."""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = datasets_dir
        self.client = None
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)
    
    def analyze_existing_data(self) -> List[Dict]:
        """Analyze existing data files to understand coverage."""
        files = os.listdir(self.datasets_dir)
        ml_files = [f for f in files if f.startswith('ml_dataset_') and f.endswith('.csv') and '_' in f.split('_', 2)[2]]
        
        periods = []
        pattern = r'ml_dataset_(\d{4}_\d{2}_\d{2})-(\d{4}_\d{2}_\d{2})\.csv'
        
        for file in ml_files:
            match = re.match(pattern, file)
            if match:
                start_str, end_str = match.groups()
                start_date = datetime.strptime(start_str, "%Y_%m_%d")
                end_date = datetime.strptime(end_str, "%Y_%m_%d")
                
                # Check actual data coverage
                path = os.path.join(self.datasets_dir, file)
                try:
                    df = pd.read_csv(path)
                    actual_start = pd.to_datetime(df.iloc[0]['open_time']).date()
                    actual_end = pd.to_datetime(df.iloc[-1]['open_time']).date()
                    rows = len(df)
                    
                    periods.append({
                        'file': file,
                        'nominal_start': start_date.date(),
                        'nominal_end': end_date.date(),
                        'actual_start': actual_start,
                        'actual_end': actual_end,
                        'rows': rows,
                        'complete': True
                    })
                except Exception as e:
                    print(f"Error reading {file}: {e}")
        
        return sorted(periods, key=lambda x: x['actual_start'])
    
    def find_missing_periods(self, target_end_date: str = "2025-06-30") -> List[Dict]:
        """Find missing periods that need to be fetched."""
        existing_periods = self.analyze_existing_data()
        target_end = datetime.strptime(target_end_date, "%Y-%m-%d").date()
        
        missing_periods = []
        
        if not existing_periods:
            # No data at all, start from 2017
            missing_periods.append({
                'start_date': datetime(2017, 1, 1).date(),
                'end_date': target_end,
                'reason': 'no_existing_data'
            })
            return missing_periods
        
        # Check for gaps between periods
        for i in range(len(existing_periods) - 1):
            current_end = existing_periods[i]['actual_end']
            next_start = existing_periods[i + 1]['actual_start']
            
            # Check if there's a gap (more than 1 day)
            if (next_start - current_end).days > 1:
                gap_start = current_end + timedelta(days=1)
                gap_end = next_start - timedelta(days=1)
                missing_periods.append({
                    'start_date': gap_start,
                    'end_date': gap_end,
                    'reason': f'gap_between_{existing_periods[i]["file"]}_and_{existing_periods[i+1]["file"]}'
                })
        
        # Check if we need data after the last period
        last_period = existing_periods[-1]
        if last_period['actual_end'] < target_end:
            missing_start = last_period['actual_end'] + timedelta(days=1)
            missing_periods.append({
                'start_date': missing_start,
                'end_date': target_end,
                'reason': f'extension_after_{last_period["file"]}'
            })
        
        return missing_periods
    
    def create_semester_periods(self, start_date: datetime.date, end_date: datetime.date) -> List[Dict]:
        """Break down a date range into semester periods (6-month chunks)."""
        periods = []
        current_start = start_date
        
        while current_start <= end_date:
            # Calculate semester end (6 months later or target end date)
            if current_start.month <= 6:
                # First half of year: Jan-Jun
                semester_end = datetime(current_start.year, 6, 30).date()
            else:
                # Second half of year: Jul-Dec
                semester_end = datetime(current_start.year, 12, 31).date()
            
            # Don't go beyond target end date
            period_end = min(semester_end, end_date)
            
            periods.append({
                'start_date': current_start,
                'end_date': period_end,
                'filename_start': current_start.strftime("%Y_%m_%d"),
                'filename_end': period_end.strftime("%Y_%m_%d")
            })
            
            # Move to next semester
            if semester_end.month == 6:
                current_start = datetime(current_start.year, 7, 1).date()
            else:
                current_start = datetime(current_start.year + 1, 1, 1).date()
        
        return periods
    
    def fetch_missing_data(self, api_key: str, api_secret: str) -> bool:
        """Fetch all missing data periods."""
        self.client = Client(api_key, api_secret)
        
        # Find what's missing
        missing_periods = self.find_missing_periods()
        
        if not missing_periods:
            print("✅ No missing data periods found!")
            return True
        
        print(f"Found {len(missing_periods)} missing periods to fetch:")
        for period in missing_periods:
            days = (period['end_date'] - period['start_date']).days + 1
            print(f"  📅 {period['start_date']} to {period['end_date']} ({days} days) - {period['reason']}")
        
        # Break each missing period into semesters and fetch
        for missing_period in missing_periods:
            semester_periods = self.create_semester_periods(
                missing_period['start_date'], 
                missing_period['end_date']
            )
            
            for semester in semester_periods:
                print(f"\n🔄 Fetching semester: {semester['start_date']} to {semester['end_date']}")
                success = self.fetch_semester_data(semester)
                if not success:
                    print(f"❌ Failed to fetch semester {semester['start_date']} to {semester['end_date']}")
                    return False
        
        print("\n✅ All missing data fetched successfully!")
        return True
    
    def fetch_semester_data(self, semester: Dict) -> bool:
        """Fetch data for a specific semester period."""
        try:
            start_str = semester['start_date'].strftime('%d %b %Y')
            end_str = semester['end_date'].strftime('%d %b %Y')
            
            print(f"  Fetching BTCUSDT data from {start_str} to {end_str}")
            
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
            
            # Fetch main timeframes
            for timeframe, interval in timeframe_intervals.items():
                df = fetch_data(self.client, config.SYMBOL, interval, start_str, end_str)
                data_timeframes[timeframe] = df
                print(f"    ✓ {timeframe}: {len(df)} rows")
            
            # Fetch additional timeframes
            for name, interval in additional_intervals.items():
                df = fetch_data(self.client, config.SYMBOL, interval, start_str, end_str)
                data_timeframes[name] = df
                print(f"    ✓ {name}: {len(df)} rows")
            
            # Calculate technical levels
            htf_levels = {}
            fib_levels = {}
            volume_profile = {}
            
            for timeframe in timeframe_intervals.keys():
                df = data_timeframes[timeframe]
                if not df.empty:
                    try:
                        htf_levels[timeframe] = calculate_htf_levels(df, timeframe)
                        fib_levels[timeframe] = calculate_fibonacci_levels(df, timeframe)
                        # Use hourly data for volume profile since we skip minute data
                        volume_profile[timeframe] = calculate_volume_profile_levels(data_timeframes['hour'], timeframe)
                        print(f"    ✓ {timeframe} levels calculated")
                    except Exception as e:
                        print(f"    ⚠ Error calculating {timeframe} levels: {e}")
            
            # Create semester datasets
            prepared_data = {
                'data_timeframes': data_timeframes,
                'htf_levels': htf_levels,
                'fib_levels': fib_levels,
                'volume_profile': volume_profile
            }
            
            # Generate datasets with semester-specific naming
            self.generate_semester_datasets(prepared_data, semester)
            
            return True
            
        except Exception as e:
            print(f"❌ Error fetching semester data: {e}")
            return False
    
    def generate_semester_datasets(self, prepared_data: Dict, semester: Dict):
        """Generate datasets for a specific semester with proper naming."""
        try:
            # Create semester-specific file names
            period_str = f"{semester['filename_start']}-{semester['filename_end']}"
            
            # Generate ML datasets (1-hour and 12-hour)
            timeframes = {'hour': '1h', '12hour': '12h'}
            
            for tf_key, tf_label in timeframes.items():
                data = prepared_data['data_timeframes'].get(tf_key)
                if data is not None and not data.empty:
                    # Add fractal labels
                    ml_dataset = detect_fractals(data.copy(), tf_key)
                    
                    # Save with semester naming
                    filename = os.path.join(self.datasets_dir, f'ml_dataset_{period_str}.csv')
                    ml_dataset.to_csv(filename)
                    print(f"    ✓ ML dataset saved: {filename} ({len(ml_dataset)} rows)")
            
            # Generate Levels Dataset
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
            
            # Add other level types (HTF, Fibonacci, Volume Profile)
            for timeframe in ['daily', 'weekly', 'monthly']:
                # HTF levels
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
                
                # Fibonacci levels
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
                
                # Volume Profile levels
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
                levels_df.sort_values(['timeframe', 'price_level'], inplace=True)
                
                levels_filename = os.path.join(self.datasets_dir, f'levels_dataset_{period_str}.csv')
                levels_df.to_csv(levels_filename, index=False)
                print(f"    ✓ Levels dataset saved: {levels_filename} ({len(levels_df)} rows)")
        
        except Exception as e:
            print(f"❌ Error generating semester datasets: {e}")

def main():
    """Main function for smart data fetching."""
    fetcher = SmartDataFetcher()
    
    # Analyze current data
    print("📊 Analyzing existing data...")
    existing_periods = fetcher.analyze_existing_data()
    
    print("\nCurrent Data Coverage:")
    for period in existing_periods:
        print(f"  📁 {period['file']}")
        print(f"     Nominal: {period['nominal_start']} to {period['nominal_end']}")
        print(f"     Actual:  {period['actual_start']} to {period['actual_end']} ({period['rows']:,} rows)")
    
    # Find missing periods
    missing_periods = fetcher.find_missing_periods()
    
    if missing_periods:
        print(f"\n⚠️  Found {len(missing_periods)} missing periods:")
        for period in missing_periods:
            days = (period['end_date'] - period['start_date']).days + 1
            print(f"    📅 {period['start_date']} to {period['end_date']} ({days} days)")
            print(f"       Reason: {period['reason']}")
        
        # Ask to fetch missing data
        response = input(f"\n🤖 Fetch missing data? (y/n): ").lower().strip()
        if response == 'y':
            success = fetcher.fetch_missing_data(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
            if success:
                print("✅ Data fetching completed successfully!")
            else:
                print("❌ Data fetching failed!")
    else:
        print("\n✅ No missing data periods found!")

if __name__ == "__main__":
    main()