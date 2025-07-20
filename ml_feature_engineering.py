"""
Enhanced ML Feature Engineering

This module creates ML-optimized features with proper aggregations and interpretable representations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import config
from datetime import datetime, timedelta

def calculate_level_distances(price: float, levels_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate distance-based features to nearby levels.
    
    Args:
        price: Current price to measure from
        levels_df: DataFrame with price levels
        
    Returns:
        Dictionary of distance features
    """
    if levels_df.empty:
        return {
            'nearest_support_distance_pct': 999.0,
            'nearest_resistance_distance_pct': 999.0,
            'nearest_support_strength': 0.0,
            'nearest_resistance_strength': 0.0
        }
    
    # Separate support and resistance levels
    support_levels = levels_df[levels_df['price_level'] < price].copy()
    resistance_levels = levels_df[levels_df['price_level'] > price].copy()
    
    features = {}
    
    # Support features
    if not support_levels.empty:
        nearest_support = support_levels.loc[support_levels['price_level'].idxmax()]
        features['nearest_support_distance_pct'] = abs(price - nearest_support['price_level']) / price * 100
        features['nearest_support_strength'] = calculate_level_strength(nearest_support)
    else:
        features['nearest_support_distance_pct'] = 999.0
        features['nearest_support_strength'] = 0.0
    
    # Resistance features
    if not resistance_levels.empty:
        nearest_resistance = resistance_levels.loc[resistance_levels['price_level'].idxmin()]
        features['nearest_resistance_distance_pct'] = abs(nearest_resistance['price_level'] - price) / price * 100
        features['nearest_resistance_strength'] = calculate_level_strength(nearest_resistance)
    else:
        features['nearest_resistance_distance_pct'] = 999.0
        features['nearest_resistance_strength'] = 0.0
    
    return features

def calculate_level_strength(level_row: pd.Series) -> float:
    """
    Calculate strength score for a level based on type, timeframe, and age.
    
    Args:
        level_row: Single row from levels DataFrame
        
    Returns:
        Strength score (higher = stronger level)
    """
    base_strength = 1.0
    
    # Timeframe weights
    timeframe_weights = {
        'monthly': 3.0,
        'weekly': 2.0,
        'daily': 1.0,
        'hourly': 0.5
    }
    
    # Level type weights
    type_weights = {
        'HTF_level': 2.5,
        'VP_poc': 2.0,
        'Fib_0.618': 1.8,
        'Fib_0.5': 1.5,
        'Fib_0.75': 1.3,
        'Fractal_High': 1.2,
        'Fractal_Low': 1.2,
        'VP_vah': 1.0,
        'VP_val': 1.0
    }
    
    # Apply weights
    timeframe = level_row.get('timeframe', 'daily')
    level_type = level_row.get('level_type', 'unknown')
    
    strength = base_strength
    strength *= timeframe_weights.get(timeframe, 1.0)
    strength *= type_weights.get(level_type, 1.0)
    
    # Age decay (if created_at is available)
    if 'created_at' in level_row and pd.notna(level_row['created_at']):
        try:
            level_time = pd.to_datetime(level_row['created_at'])
            current_time = pd.Timestamp.now()
            days_old = (current_time - level_time).days
            # Decay factor - older levels get slightly less weight
            age_factor = config.LEVEL_STRENGTH_DECAY ** (days_old / 30)  # Decay per month
            strength *= age_factor
        except:
            pass  # Skip age decay if time parsing fails
    
    # Touch penalty (levels that have been touched are weaker)
    support_touches = level_row.get('support_touches', 0)
    resistance_touches = level_row.get('resistance_touches', 0)
    total_touches = support_touches + resistance_touches
    
    # Reduce strength based on touches (but don't go below 0.1)
    touch_penalty = max(0.1, 1.0 - (total_touches * 0.15))
    strength *= touch_penalty
    
    return strength

def calculate_confluence_features(price: float, levels_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate confluence zone features around current price.
    
    Args:
        price: Current price
        levels_df: DataFrame with price levels
        
    Returns:
        Dictionary of confluence features
    """
    features = {}
    
    # Define zone ranges around current price
    zone_ranges = {
        '0_5pct': 0.005,   # 0.5%
        '1_0pct': 0.01,    # 1.0%
        '1_5pct': 0.015,   # 1.5%
        '2_0pct': 0.02,    # 2.0%
    }
    
    for zone_name, zone_width in zone_ranges.items():
        zone_start = price * (1 - zone_width)
        zone_end = price * (1 + zone_width)
        
        # Find levels in this zone
        zone_levels = levels_df[
            (levels_df['price_level'] >= zone_start) & 
            (levels_df['price_level'] <= zone_end)
        ]
        
        # Count levels by type and timeframe
        features[f'levels_in_zone_{zone_name}'] = len(zone_levels)
        
        # Timeframe breakdown
        for timeframe in ['daily', 'weekly', 'monthly']:
            tf_levels = zone_levels[zone_levels['timeframe'] == timeframe]
            features[f'{timeframe}_levels_in_zone_{zone_name}'] = len(tf_levels)
        
        # Level type breakdown
        for level_type in ['HTF_level', 'VP_poc', 'Fib_0.618', 'Fractal_High', 'Fractal_Low']:
            type_levels = zone_levels[zone_levels['level_type'] == level_type]
            features[f'{level_type}_in_zone_{zone_name}'] = len(type_levels)
        
        # Weighted confluence score
        if not zone_levels.empty:
            strength_scores = zone_levels.apply(calculate_level_strength, axis=1)
            features[f'confluence_strength_zone_{zone_name}'] = strength_scores.sum()
        else:
            features[f'confluence_strength_zone_{zone_name}'] = 0.0
    
    return features

def calculate_volume_features(current_candle: Dict, volume_history: pd.Series) -> Dict[str, float]:
    """
    Calculate enhanced volume features.
    
    Args:
        current_candle: Dictionary with current candle data
        volume_history: Series of historical volume data
        
    Returns:
        Dictionary of volume features
    """
    current_volume = current_candle.get('volume', 0)
    
    if len(volume_history) < 20:
        return {
            'volume_ma_20_ratio': 1.0,
            'volume_ma_50_ratio': 1.0,
            'volume_spike': 0,
            'volume_percentile_20': 50.0,
            'volume_trend_5': 0.0
        }
    
    features = {}
    
    # Moving average ratios
    ma_20 = volume_history.tail(20).mean()
    ma_50 = volume_history.tail(50).mean() if len(volume_history) >= 50 else ma_20
    
    features['volume_ma_20_ratio'] = current_volume / ma_20 if ma_20 > 0 else 1.0
    features['volume_ma_50_ratio'] = current_volume / ma_50 if ma_50 > 0 else 1.0
    
    # Volume spike detection
    features['volume_spike'] = 1 if features['volume_ma_20_ratio'] > 2.0 else 0
    
    # Volume percentile in recent history
    recent_volumes = volume_history.tail(20)
    features['volume_percentile_20'] = (recent_volumes < current_volume).mean() * 100
    
    # Volume trend (slope of last 5 periods)
    if len(volume_history) >= 5:
        recent_5 = volume_history.tail(5).values
        x = np.arange(len(recent_5))
        slope = np.polyfit(x, recent_5, 1)[0]
        features['volume_trend_5'] = slope / recent_5.mean() if recent_5.mean() > 0 else 0.0
    else:
        features['volume_trend_5'] = 0.0
    
    return features

def calculate_price_action_features(current_candle: Dict, price_history: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate price action and momentum features.
    
    Args:
        current_candle: Dictionary with current candle data
        price_history: DataFrame with historical OHLCV data
        
    Returns:
        Dictionary of price action features
    """
    features = {}
    
    # Basic candle metrics
    o, h, l, c = current_candle['open'], current_candle['high'], current_candle['low'], current_candle['close']
    
    # Candle body and wick ratios
    total_range = h - l
    body_size = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    features['body_ratio'] = body_size / total_range if total_range > 0 else 0.0
    features['upper_wick_ratio'] = upper_wick / total_range if total_range > 0 else 0.0
    features['lower_wick_ratio'] = lower_wick / total_range if total_range > 0 else 0.0
    features['is_bullish'] = 1 if c > o else 0
    
    # Price change features
    if len(price_history) > 0:
        prev_close = price_history['close'].iloc[-1]
        features['price_change_pct'] = (c - prev_close) / prev_close * 100
        features['gap_pct'] = (o - prev_close) / prev_close * 100
    else:
        features['price_change_pct'] = 0.0
        features['gap_pct'] = 0.0
    
    # Moving average features
    if len(price_history) >= 20:
        ma_20 = price_history['close'].tail(20).mean()
        ma_50 = price_history['close'].tail(50).mean() if len(price_history) >= 50 else ma_20
        
        features['price_vs_ma20_pct'] = (c - ma_20) / ma_20 * 100
        features['price_vs_ma50_pct'] = (c - ma_50) / ma_50 * 100
        features['ma20_vs_ma50_pct'] = (ma_20 - ma_50) / ma_50 * 100 if ma_50 > 0 else 0.0
    else:
        features['price_vs_ma20_pct'] = 0.0
        features['price_vs_ma50_pct'] = 0.0
        features['ma20_vs_ma50_pct'] = 0.0
    
    # Volatility features
    if len(price_history) >= 14:
        returns = price_history['close'].pct_change().tail(14)
        features['volatility_14d'] = returns.std() * 100
        features['price_range_14d_pct'] = (price_history['high'].tail(14).max() - price_history['low'].tail(14).min()) / price_history['close'].tail(14).mean() * 100
    else:
        features['volatility_14d'] = 0.0
        features['price_range_14d_pct'] = 0.0
    
    return features

def calculate_temporal_features(timestamp: pd.Timestamp, fractal_history: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate time-based features.
    
    Args:
        timestamp: Current timestamp
        fractal_history: DataFrame with historical fractal data
        
    Returns:
        Dictionary of temporal features
    """
    features = {}
    
    # Time of day/week features
    features['hour_of_day'] = timestamp.hour
    features['day_of_week'] = timestamp.dayofweek
    features['hour_normalized'] = timestamp.hour / 23.0
    features['day_normalized'] = timestamp.dayofweek / 6.0
    
    # Trading session features
    hour = timestamp.hour
    if 0 <= hour < 8:
        session = 'asian'
    elif 8 <= hour < 16:
        session = 'european'
    else:
        session = 'american'
    
    features['session_asian'] = 1 if session == 'asian' else 0
    features['session_european'] = 1 if session == 'european' else 0
    features['session_american'] = 1 if session == 'american' else 0
    
    # Fractal timing features
    if not fractal_history.empty and 'open_time' in fractal_history.columns:
        # Find last fractals
        bullish_fractals = fractal_history[fractal_history['is_fractal_up'] == 1]
        bearish_fractals = fractal_history[fractal_history['is_fractal_down'] == 1]
        
        if not bullish_fractals.empty:
            last_bullish = bullish_fractals['open_time'].max()
            hours_since_bullish = (timestamp - last_bullish).total_seconds() / 3600
            features['hours_since_last_bullish_fractal'] = hours_since_bullish
        else:
            features['hours_since_last_bullish_fractal'] = 999.0
        
        if not bearish_fractals.empty:
            last_bearish = bearish_fractals['open_time'].max()
            hours_since_bearish = (timestamp - last_bearish).total_seconds() / 3600
            features['hours_since_last_bearish_fractal'] = hours_since_bearish
        else:
            features['hours_since_last_bearish_fractal'] = 999.0
    else:
        features['hours_since_last_bullish_fractal'] = 999.0
        features['hours_since_last_bearish_fractal'] = 999.0
    
    return features

def create_comprehensive_features(
    candles_df: pd.DataFrame,
    levels_df: pd.DataFrame,
    start_index: int = 2
) -> pd.DataFrame:
    """
    Create comprehensive ML features for the entire dataset.
    
    Args:
        candles_df: DataFrame with OHLCV and fractal data
        levels_df: DataFrame with price levels
        start_index: Index to start processing from (need history for features)
        
    Returns:
        DataFrame with comprehensive features
    """
    print("Creating comprehensive ML features...")
    
    # Initialize results DataFrame
    feature_rows = []
    
    # Convert time columns
    candles_df = candles_df.copy()
    levels_df = levels_df.copy()
    candles_df['open_time'] = pd.to_datetime(candles_df['open_time'])
    levels_df['created_at'] = pd.to_datetime(levels_df['created_at'])
    
    # Process each candle
    for i in range(start_index, len(candles_df)):
        if i % 500 == 0:
            print(f"Processing candle {i+1}/{len(candles_df)}")
        
        current_candle = candles_df.iloc[i]
        current_time = current_candle['open_time']
        current_price = current_candle['close']
        
        # Get historical data up to current point
        historical_candles = candles_df.iloc[:i]
        historical_levels = levels_df[levels_df['created_at'] <= current_time]
        
        # Calculate feature groups
        row_features = {}
        
        # Basic identifiers
        row_features['timestamp'] = current_time
        row_features['open'] = current_candle['open']
        row_features['high'] = current_candle['high']
        row_features['low'] = current_candle['low']
        row_features['close'] = current_candle['close']
        row_features['volume'] = current_candle['volume']
        
        # Level distance features
        level_features = calculate_level_distances(current_price, historical_levels)
        row_features.update(level_features)
        
        # Confluence features
        confluence_features = calculate_confluence_features(current_price, historical_levels)
        row_features.update(confluence_features)
        
        # Volume features
        volume_features = calculate_volume_features(
            current_candle.to_dict(),
            historical_candles['volume']
        )
        row_features.update(volume_features)
        
        # Price action features
        price_features = calculate_price_action_features(
            current_candle.to_dict(),
            historical_candles
        )
        row_features.update(price_features)
        
        # Temporal features
        temporal_features = calculate_temporal_features(current_time, historical_candles)
        row_features.update(temporal_features)
        
        feature_rows.append(row_features)
    
    # Create DataFrame
    features_df = pd.DataFrame(feature_rows)
    
    print(f"Created {len(features_df)} feature rows with {len(features_df.columns)} features")
    return features_df

if __name__ == "__main__":
    # Test with sample data
    print("Testing ML feature engineering...")
    
    try:
        # Load sample data
        candles_df = pd.read_csv("datasets/ml_dataset_2025_01_01-2025_06_30.csv")
        levels_df = pd.read_csv("datasets/levels_dataset_2025_01_01-2025_06_30.csv")
        
        print(f"Loaded candles: {len(candles_df)} rows")
        print(f"Loaded levels: {len(levels_df)} rows")
        
        # Create features for small sample
        sample_features = create_comprehensive_features(
            candles_df.head(100),  # Use first 100 candles for testing
            levels_df,
            start_index=2
        )
        
        print(f"\nSample features shape: {sample_features.shape}")
        print(f"Feature columns: {list(sample_features.columns)}")
        
    except FileNotFoundError:
        print("Sample data not found. Run main.py --features-only first.")