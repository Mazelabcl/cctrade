"""
Target Variable Generation for ML Models

This module creates proper target variables for fractal prediction with configurable horizons.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import config

def create_fractal_targets(
    candles_df: pd.DataFrame, 
    prediction_horizon: str = None,
    lookforward_candles: int = None
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Create multi-class fractal direction targets for ML models.
    
    Args:
        candles_df: DataFrame with OHLCV data and fractal columns
        prediction_horizon: 'hour', 'day', 'week', '15days', 'month'
        lookforward_candles: Override with specific number of candles to look ahead
        
    Returns:
        target_series: Series with fractal direction (0=no_fractal, 1=bullish, 2=bearish)
        feature_df: DataFrame with features (excluding target prediction period)
    """
    
    # Set prediction horizon
    if lookforward_candles is not None:
        horizon_candles = lookforward_candles
        horizon_name = f"custom_{lookforward_candles}"
    else:
        horizon_key = prediction_horizon or config.DEFAULT_PREDICTION_HORIZON
        horizon_candles = config.PREDICTION_HORIZONS[horizon_key]
        horizon_name = horizon_key
    
    print(f"Creating targets with prediction horizon: {horizon_name} ({horizon_candles} candles)")
    
    # Validate input data
    required_columns = ['is_fractal_up', 'is_fractal_down', 'open_time']
    missing_cols = [col for col in required_columns if col not in candles_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert time column if needed
    if not pd.api.types.is_datetime64_any_dtype(candles_df['open_time']):
        candles_df = candles_df.copy()
        candles_df['open_time'] = pd.to_datetime(candles_df['open_time'])
    
    # Calculate targets
    targets = []
    valid_indices = []
    
    # We can only create targets for candles where we have enough future data
    max_index = len(candles_df) - horizon_candles
    
    for i in range(max_index):
        # Look ahead N candles from current position
        future_start = i + 1
        future_end = i + 1 + horizon_candles
        future_candles = candles_df.iloc[future_start:future_end]
        
        # Check for fractal formation in the prediction window
        bullish_fractals = future_candles['is_fractal_up'].sum()
        bearish_fractals = future_candles['is_fractal_down'].sum()
        
        # Determine target class
        if bullish_fractals > 0 and bearish_fractals > 0:
            # Both types present - choose the first one chronologically
            first_bullish = future_candles[future_candles['is_fractal_up'] == 1].index
            first_bearish = future_candles[future_candles['is_fractal_down'] == 1].index
            
            if len(first_bullish) > 0 and len(first_bearish) > 0:
                if first_bullish[0] < first_bearish[0]:
                    target = 1  # Bullish fractal comes first
                else:
                    target = 2  # Bearish fractal comes first
            elif len(first_bullish) > 0:
                target = 1
            else:
                target = 2
        elif bullish_fractals > 0:
            target = 1  # Bullish fractal (swing low)
        elif bearish_fractals > 0:
            target = 2  # Bearish fractal (swing high)
        else:
            target = 0  # No fractal
        
        targets.append(target)
        valid_indices.append(i)
    
    # Create target series with proper index
    target_series = pd.Series(
        targets, 
        index=candles_df.iloc[valid_indices].index,
        name='fractal_direction'
    )
    
    # Create feature DataFrame (excluding the prediction window)
    feature_df = candles_df.iloc[valid_indices].copy()
    
    # Add metadata
    feature_df['prediction_horizon'] = horizon_name
    feature_df['prediction_horizon_candles'] = horizon_candles
    
    # Print target distribution
    target_counts = target_series.value_counts().sort_index()
    target_percentages = (target_counts / len(target_series) * 100).round(2)
    
    print(f"\nTarget Distribution:")
    print(f"  No Fractal (0): {target_counts.get(0, 0):,} ({target_percentages.get(0, 0):.1f}%)")
    print(f"  Bullish Fractal (1): {target_counts.get(1, 0):,} ({target_percentages.get(1, 0):.1f}%)")
    print(f"  Bearish Fractal (2): {target_counts.get(2, 0):,} ({target_percentages.get(2, 0):.1f}%)")
    print(f"  Total samples: {len(target_series):,}")
    
    return target_series, feature_df

def validate_target_balance(target_series: pd.Series, min_class_percentage: float = 5.0) -> bool:
    """
    Validate that target classes are reasonably balanced.
    
    Args:
        target_series: Target variable series
        min_class_percentage: Minimum percentage for any class
        
    Returns:
        True if classes are reasonably balanced
    """
    target_counts = target_series.value_counts()
    total_samples = len(target_series)
    
    for class_val, count in target_counts.items():
        percentage = (count / total_samples) * 100
        if percentage < min_class_percentage:
            print(f"Warning: Class {class_val} has only {percentage:.1f}% of samples (minimum: {min_class_percentage}%)")
            return False
    
    return True

def create_time_aware_splits(
    feature_df: pd.DataFrame, 
    target_series: pd.Series,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Create time-aware train/validation/test splits to prevent look-ahead bias.
    
    Args:
        feature_df: Features DataFrame
        target_series: Target variable series
        train_ratio: Proportion for training
        val_ratio: Proportion for validation (remainder goes to test)
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    
    # Sort by time to ensure proper chronological splits
    if 'open_time' in feature_df.columns:
        sorted_indices = feature_df['open_time'].sort_values().index
    else:
        sorted_indices = feature_df.index.sort_values()
    
    feature_df_sorted = feature_df.loc[sorted_indices]
    target_series_sorted = target_series.loc[sorted_indices]
    
    n_samples = len(feature_df_sorted)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Create splits
    X_train = feature_df_sorted.iloc[:train_end]
    X_val = feature_df_sorted.iloc[train_end:val_end]
    X_test = feature_df_sorted.iloc[val_end:]
    
    y_train = target_series_sorted.iloc[:train_end]
    y_val = target_series_sorted.iloc[train_end:val_end]
    y_test = target_series_sorted.iloc[val_end:]
    
    print(f"\nTime-aware data splits:")
    print(f"  Training: {len(X_train):,} samples ({train_ratio*100:.1f}%)")
    print(f"  Validation: {len(X_val):,} samples ({val_ratio*100:.1f}%)")
    print(f"  Test: {len(X_test):,} samples ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    # Show time ranges
    if 'open_time' in feature_df.columns:
        print(f"  Training period: {X_train['open_time'].min()} to {X_train['open_time'].max()}")
        print(f"  Validation period: {X_val['open_time'].min()} to {X_val['open_time'].max()}")
        print(f"  Test period: {X_test['open_time'].min()} to {X_test['open_time'].max()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def export_for_datarobot(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    output_file: str,
    include_timestamp: bool = False
) -> str:
    """
    Export data in DataRobot-ready format.
    
    Args:
        feature_df: Features DataFrame
        target_series: Target variable series
        output_file: Output CSV file path
        include_timestamp: Whether to include timestamp column
        
    Returns:
        Path to exported file
    """
    
    # Combine features and target
    ml_dataset = feature_df.copy()
    ml_dataset['fractal_direction'] = target_series
    
    # Remove metadata columns that shouldn't be features
    columns_to_remove = [
        'is_fractal_up', 'is_fractal_down',  # These are used to create target
        'prediction_horizon', 'prediction_horizon_candles'  # Metadata
    ]
    
    if not include_timestamp:
        columns_to_remove.append('open_time')
    
    for col in columns_to_remove:
        if col in ml_dataset.columns:
            ml_dataset = ml_dataset.drop(columns=[col])
    
    # Ensure target is the last column (DataRobot convention)
    target_col = ml_dataset.pop('fractal_direction')
    ml_dataset['fractal_direction'] = target_col
    
    # Save to CSV
    ml_dataset.to_csv(output_file, index=False)
    
    print(f"\nDataRobot-ready dataset exported:")
    print(f"  File: {output_file}")
    print(f"  Samples: {len(ml_dataset):,}")
    print(f"  Features: {len(ml_dataset.columns) - 1}")
    print(f"  Target: fractal_direction (0=no_fractal, 1=bullish, 2=bearish)")
    
    return output_file

if __name__ == "__main__":
    # Test with sample data
    print("Testing target variable generation...")
    
    # Load sample data
    try:
        sample_data = pd.read_csv("datasets/ml_dataset_2025_01_01-2025_06_30.csv")
        print(f"Loaded sample data: {len(sample_data)} rows")
        
        # Test different prediction horizons
        for horizon in ['hour', 'day', 'week']:
            print(f"\n{'='*50}")
            print(f"Testing {horizon} prediction horizon")
            print(f"{'='*50}")
            
            targets, features = create_fractal_targets(
                sample_data, 
                prediction_horizon=horizon
            )
            
            # Validate balance
            validate_target_balance(targets)
            
    except FileNotFoundError:
        print("Sample data not found. Run main.py --features-only first.")