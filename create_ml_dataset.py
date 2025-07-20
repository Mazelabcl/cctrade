"""
ML Dataset Creation Pipeline

This module coordinates target variable generation and feature engineering 
to create ML-ready datasets for DataRobot.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple, Dict, List
from datetime import datetime

import config
from target_variable import create_fractal_targets, validate_target_balance, export_for_datarobot
from ml_feature_engineering import create_comprehensive_features

def normalize_fractal_columns(candles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize fractal column formats to consistent naming and data types.
    
    Args:
        candles_df: DataFrame with either format of fractal columns
        
    Returns:
        DataFrame with standardized is_fractal_up/is_fractal_down columns (0/1 integers)
    """
    df = candles_df.copy()
    
    # Check which format we have
    has_format1 = all(col in df.columns for col in ['is_fractal_up', 'is_fractal_down'])
    has_format2 = all(col in df.columns for col in ['bullish_fractal', 'bearish_fractal'])
    
    if has_format2 and not has_format1:
        # Convert format 2 to format 1
        df['is_fractal_up'] = df['bullish_fractal'].astype(int)
        df['is_fractal_down'] = df['bearish_fractal'].astype(int)
        # Keep original columns for reference but use standardized names
    elif has_format1:
        # Ensure format 1 columns are integers
        df['is_fractal_up'] = df['is_fractal_up'].astype(int)
        df['is_fractal_down'] = df['is_fractal_down'].astype(int)
    else:
        raise ValueError("No valid fractal column format found")
    
    return df

def create_ml_ready_dataset(
    candles_file: str,
    levels_file: str,
    output_file: str,
    prediction_horizon: str = None,
    lookforward_candles: int = None,
    max_samples: int = None
) -> Tuple[str, Dict]:
    """
    Create a complete ML-ready dataset with targets and features.
    
    Args:
        candles_file: Path to candles CSV file
        levels_file: Path to levels CSV file  
        output_file: Path for output ML dataset
        prediction_horizon: 'hour', 'day', 'week', '15days', 'month'
        lookforward_candles: Override with specific number of candles
        max_samples: Limit dataset size for testing (None for full dataset)
        
    Returns:
        Tuple of (output_file_path, summary_stats)
    """
    
    print("="*60)
    print("ML DATASET CREATION PIPELINE")
    print("="*60)
    
    # Load data
    print("1. Loading data...")
    candles_df = pd.read_csv(candles_file)
    levels_df = pd.read_csv(levels_file)
    
    print(f"   Loaded {len(candles_df):,} candles")
    print(f"   Loaded {len(levels_df):,} levels")
    
    # Normalize fractal column format
    candles_df = normalize_fractal_columns(candles_df)
    print("   ✓ Normalized fractal column format")
    
    # Apply sample limit if specified
    if max_samples and max_samples < len(candles_df):
        print(f"   Limiting to first {max_samples:,} candles for testing")
        candles_df = candles_df.head(max_samples)
    
    # Validate data
    print("2. Validating data...")
    validation_errors = validate_input_data(candles_df, levels_df)
    if validation_errors:
        raise ValueError(f"Data validation failed: {validation_errors}")
    print("   ✓ Data validation passed")
    
    # Create comprehensive features
    print("3. Creating comprehensive features...")
    features_df = create_comprehensive_features(
        candles_df=candles_df,
        levels_df=levels_df,
        start_index=2  # Need some history for features
    )
    print(f"   ✓ Created {len(features_df):,} feature rows with {len(features_df.columns)} features")
    
    # Generate targets
    print("4. Generating target variables...")
    
    # We need to align the candles with the features (features start from index 2)
    aligned_candles = candles_df.iloc[2:2+len(features_df)].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    
    targets, _ = create_fractal_targets(
        candles_df=aligned_candles,
        prediction_horizon=prediction_horizon,
        lookforward_candles=lookforward_candles
    )
    
    # Align features with targets (targets are shorter due to prediction horizon)
    final_features = features_df.iloc[:len(targets)].copy()
    final_targets = targets.reset_index(drop=True)
    
    print(f"   ✓ Generated {len(final_targets):,} target samples")
    
    # Validate target balance
    print("5. Validating target distribution...")
    is_balanced = validate_target_balance(final_targets, min_class_percentage=2.0)
    if not is_balanced:
        print("   ⚠ Warning: Target classes may be imbalanced")
    else:
        print("   ✓ Target distribution is reasonable")
    
    # Prepare final dataset
    print("6. Preparing final ML dataset...")
    
    # Clean feature names for ML models
    final_features = clean_feature_names(final_features)
    
    # Add target to features
    ml_dataset = final_features.copy()
    ml_dataset['fractal_direction'] = final_targets
    
    # Create summary statistics
    summary_stats = create_dataset_summary(ml_dataset, final_targets)
    
    # Export dataset
    print("7. Exporting ML dataset...")
    output_path = export_for_datarobot(
        feature_df=final_features,
        target_series=final_targets,
        output_file=output_file,
        include_timestamp=True  # Keep timestamp for analysis
    )
    
    print("="*60)
    print("ML DATASET CREATION COMPLETED")
    print("="*60)
    print(f"Output file: {output_path}")
    print(f"Dataset shape: {ml_dataset.shape}")
    print(f"Ready for DataRobot upload!")
    
    return output_path, summary_stats

def validate_input_data(candles_df: pd.DataFrame, levels_df: pd.DataFrame) -> List[str]:
    """
    Validate input data quality and structure.
    
    Args:
        candles_df: Candles DataFrame
        levels_df: Levels DataFrame
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required columns in candles
    required_candle_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    fractal_cols_option1 = ['is_fractal_up', 'is_fractal_down']
    fractal_cols_option2 = ['bullish_fractal', 'bearish_fractal']
    
    missing_candle_cols = [col for col in required_candle_cols if col not in candles_df.columns]
    if missing_candle_cols:
        errors.append(f"Missing candle columns: {missing_candle_cols}")
    
    # Check for fractal columns (either format)
    has_fractal_format1 = all(col in candles_df.columns for col in fractal_cols_option1)
    has_fractal_format2 = all(col in candles_df.columns for col in fractal_cols_option2)
    
    if not has_fractal_format1 and not has_fractal_format2:
        errors.append(f"Missing fractal columns. Need either {fractal_cols_option1} or {fractal_cols_option2}")
    
    # Check required columns in levels
    required_level_cols = ['price_level', 'level_type', 'timeframe', 'created_at']
    missing_level_cols = [col for col in required_level_cols if col not in levels_df.columns]
    if missing_level_cols:
        errors.append(f"Missing level columns: {missing_level_cols}")
    
    # Check for empty DataFrames
    if len(candles_df) == 0:
        errors.append("Candles DataFrame is empty")
    if len(levels_df) == 0:
        errors.append("Levels DataFrame is empty")
    
    # Check for minimum data requirements
    if len(candles_df) < 100:
        errors.append(f"Insufficient candle data: {len(candles_df)} rows (minimum: 100)")
    
    # Check for obvious data issues
    if not errors:  # Only check if basic structure is correct
        # Check for null values in critical columns
        critical_nulls = candles_df[['open', 'high', 'low', 'close', 'volume']].isnull().sum().sum()
        if critical_nulls > 0:
            errors.append(f"Found {critical_nulls} null values in critical price/volume columns")
        
        # Check for logical price relationships
        invalid_ohlc = (candles_df['high'] < candles_df['low']).sum()
        if invalid_ohlc > 0:
            errors.append(f"Found {invalid_ohlc} candles with high < low")
        
        # Check fractal columns based on available format
        if has_fractal_format1:
            if not candles_df['is_fractal_up'].isin([0, 1]).all():
                errors.append("is_fractal_up column contains values other than 0 or 1")
            if not candles_df['is_fractal_down'].isin([0, 1]).all():
                errors.append("is_fractal_down column contains values other than 0 or 1")
        elif has_fractal_format2:
            if not candles_df['bullish_fractal'].isin([True, False]).all():
                errors.append("bullish_fractal column contains values other than True or False")
            if not candles_df['bearish_fractal'].isin([True, False]).all():
                errors.append("bearish_fractal column contains values other than True or False")
    
    return errors

def clean_feature_names(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean feature names to be ML model friendly.
    
    Args:
        features_df: Features DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    df = features_df.copy()
    
    # Replace problematic characters
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    df.columns = df.columns.str.replace('__+', '_', regex=True)  # Multiple underscores to single
    df.columns = df.columns.str.strip('_')  # Remove leading/trailing underscores
    
    # Ensure no duplicate column names
    cols = df.columns.tolist()
    seen = set()
    for i, col in enumerate(cols):
        original_col = col
        counter = 1
        while col in seen:
            col = f"{original_col}_{counter}"
            counter += 1
        seen.add(col)
        cols[i] = col
    
    df.columns = cols
    
    return df

def create_dataset_summary(ml_dataset: pd.DataFrame, targets: pd.Series) -> Dict:
    """
    Create summary statistics for the ML dataset.
    
    Args:
        ml_dataset: Complete ML dataset
        targets: Target variable series
        
    Returns:
        Dictionary with summary statistics
    """
    target_counts = targets.value_counts().sort_index()
    target_percentages = (target_counts / len(targets) * 100).round(2)
    
    # Feature statistics
    numeric_features = ml_dataset.select_dtypes(include=[np.number])
    
    summary = {
        'total_samples': len(ml_dataset),
        'total_features': len(ml_dataset.columns) - 1,  # Exclude target
        'target_distribution': {
            'no_fractal_count': target_counts.get(0, 0),
            'bullish_fractal_count': target_counts.get(1, 0),
            'bearish_fractal_count': target_counts.get(2, 0),
            'no_fractal_pct': target_percentages.get(0, 0),
            'bullish_fractal_pct': target_percentages.get(1, 0),
            'bearish_fractal_pct': target_percentages.get(2, 0)
        },
        'feature_stats': {
            'numeric_features': len(numeric_features.columns),
            'features_with_nulls': numeric_features.isnull().sum().sum(),
            'constant_features': (numeric_features.nunique() == 1).sum()
        },
        'data_quality': {
            'has_timestamp': 'timestamp' in ml_dataset.columns,
            'time_range': None,
            'missing_values_pct': (ml_dataset.isnull().sum().sum() / ml_dataset.size * 100).round(2)
        }
    }
    
    # Add time range if timestamp available
    if 'timestamp' in ml_dataset.columns:
        try:
            ts_col = pd.to_datetime(ml_dataset['timestamp'])
            summary['data_quality']['time_range'] = f"{ts_col.min()} to {ts_col.max()}"
        except:
            summary['data_quality']['time_range'] = "Unable to parse timestamps"
    
    return summary

def print_dataset_summary(summary: Dict):
    """Print formatted dataset summary."""
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    
    print(f"Total Samples: {summary['total_samples']:,}")
    print(f"Total Features: {summary['total_features']:,}")
    
    print(f"\nTarget Distribution:")
    td = summary['target_distribution']
    print(f"  No Fractal (0): {td['no_fractal_count']:,} ({td['no_fractal_pct']:.1f}%)")
    print(f"  Bullish Fractal (1): {td['bullish_fractal_count']:,} ({td['bullish_fractal_pct']:.1f}%)")
    print(f"  Bearish Fractal (2): {td['bearish_fractal_count']:,} ({td['bearish_fractal_pct']:.1f}%)")
    
    print(f"\nFeature Quality:")
    fs = summary['feature_stats']
    print(f"  Numeric Features: {fs['numeric_features']:,}")
    print(f"  Features with Nulls: {fs['features_with_nulls']:,}")
    print(f"  Constant Features: {fs['constant_features']:,}")
    
    print(f"\nData Quality:")
    dq = summary['data_quality']
    print(f"  Has Timestamp: {dq['has_timestamp']}")
    print(f"  Time Range: {dq['time_range']}")
    print(f"  Missing Values: {dq['missing_values_pct']:.1f}%")

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create ML-ready dataset")
    parser.add_argument("--candles", required=True, help="Path to candles CSV file")
    parser.add_argument("--levels", required=True, help="Path to levels CSV file")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--horizon", choices=['hour', 'day', 'week', '15days', 'month'], 
                       default='day', help="Prediction horizon")
    parser.add_argument("--max-samples", type=int, help="Limit dataset size for testing")
    
    args = parser.parse_args()
    
    # Create ML dataset
    output_path, summary = create_ml_ready_dataset(
        candles_file=args.candles,
        levels_file=args.levels,
        output_file=args.output,
        prediction_horizon=args.horizon,
        max_samples=args.max_samples
    )
    
    # Print summary
    print_dataset_summary(summary)

if __name__ == "__main__":
    main()