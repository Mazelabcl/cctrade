import pandas as pd
import numpy as np
from datetime import datetime
import time
from chunk_features import generate_chunk_features
from volume_features import calculate_volume_features

def calculate_level_distances(price: float, levels_df: pd.DataFrame, timestamp: datetime, max_distance: float = 1000.0) -> dict:
    """Calculate both absolute and relative distances to levels"""
    # Filter levels that existed before current timestamp
    valid_levels = levels_df[pd.to_datetime(levels_df['created_at']) <= timestamp].copy()
    
    if valid_levels.empty:
        return {
            'distance_to_nearest_level': None,
            'distance_to_above': None,
            'distance_to_below': None,
            'relative_distance_nearest': None,
            'relative_distance_above': None,
            'relative_distance_below': None,
            'nearby_level_count': 0,
            'nearby_htf_count': 0,
            'nearby_fractal_count': 0,
            'nearby_vp_count': 0
        }
    
    # Calculate all distances
    valid_levels['abs_distance'] = abs(valid_levels['price_level'] - price)
    valid_levels['rel_distance'] = valid_levels['abs_distance'] / price * 100  # as percentage
    
    # Find levels above and below
    levels_above = valid_levels[valid_levels['price_level'] > price]
    levels_below = valid_levels[valid_levels['price_level'] < price]
    
    # Get nearby levels
    nearby_levels = valid_levels[valid_levels['abs_distance'] <= max_distance]
    
    # Count nearby levels by type
    nearby_htf = nearby_levels[nearby_levels['level_type'].str.contains('HTF', na=False)]
    nearby_fractal = nearby_levels[nearby_levels['level_type'].str.contains('Fractal', na=False)]
    nearby_vp = nearby_levels[nearby_levels['level_type'].str.contains('VP', na=False)]
    
    # Calculate features
    nearest_idx = valid_levels['abs_distance'].idxmin() if not valid_levels.empty else None
    
    return {
        'distance_to_nearest_level': valid_levels.loc[nearest_idx, 'abs_distance'] if nearest_idx else None,
        'distance_to_above': None if levels_above.empty else levels_above['price_level'].min() - price,
        'distance_to_below': None if levels_below.empty else price - levels_below['price_level'].max(),
        'relative_distance_nearest': valid_levels.loc[nearest_idx, 'rel_distance'] if nearest_idx else None,
        'relative_distance_above': None if levels_above.empty else (levels_above['price_level'].min() - price) / price * 100,
        'relative_distance_below': None if levels_below.empty else (price - levels_below['price_level'].max()) / price * 100,
        'nearby_level_count': len(nearby_levels),
        'nearby_htf_count': len(nearby_htf),
        'nearby_fractal_count': len(nearby_fractal),
        'nearby_vp_count': len(nearby_vp)
    }

def detect_confluence_zones(levels_df: pd.DataFrame, threshold: float = 200.0, timestamp: datetime = None) -> dict:
    """Detect zones with enhanced type-specific features"""
    if timestamp is not None:
        levels_df = levels_df[pd.to_datetime(levels_df['created_at']) <= timestamp].copy()
    
    if levels_df.empty:
        return {
            'confluence_zones_count': 0,
            'max_levels_in_zone': 0,
            'total_levels_in_zones': 0,
            'zones_with_htf': 0,
            'zones_with_vp': 0
        }
    
    # Sort levels by price
    sorted_levels = levels_df.sort_values('price_level')
    
    confluence_zones = []
    current_zone = [(sorted_levels.iloc[0]['price_level'], sorted_levels.iloc[0]['level_type'])]
    
    # Iterate through sorted levels
    for i in range(1, len(sorted_levels)):
        current_price = sorted_levels.iloc[i]['price_level']
        current_type = sorted_levels.iloc[i]['level_type']
        prev_price = sorted_levels.iloc[i-1]['price_level']
        
        if current_price - prev_price <= threshold:
            current_zone.append((current_price, current_type))
        else:
            if len(current_zone) > 1:
                confluence_zones.append(current_zone)
            current_zone = [(current_price, current_type)]
    
    if len(current_zone) > 1:
        confluence_zones.append(current_zone)
    
    # Count zones with specific level types
    zones_with_htf = sum(1 for zone in confluence_zones 
                        if any('HTF' in level[1] for level in zone))
    zones_with_vp = sum(1 for zone in confluence_zones 
                       if any('VP' in level[1] for level in zone))
    
    return {
        'confluence_zones_count': len(confluence_zones),
        'max_levels_in_zone': max([len(zone) for zone in confluence_zones]) if confluence_zones else 0,
        'total_levels_in_zones': sum(len(zone) for zone in confluence_zones),
        'zones_with_htf': zones_with_htf,
        'zones_with_vp': zones_with_vp
    }

def calculate_time_features(levels_df: pd.DataFrame, timestamp: datetime, max_distance: float = 1000.0) -> dict:
    """Calculate time-based features for nearby levels"""
    valid_levels = levels_df[pd.to_datetime(levels_df['created_at']) <= timestamp].copy()
    
    if valid_levels.empty:
        return {
            'nearest_level_age_hours': None,
            'avg_nearby_level_age_hours': None,
            'oldest_nearby_level_age_hours': None,
            'newest_nearby_level_age_hours': None
        }
    
    # Calculate ages in hours
    valid_levels['age_hours'] = (timestamp - pd.to_datetime(valid_levels['created_at'])).dt.total_seconds() / 3600
    
    # Get nearest level age
    nearest_level_age = valid_levels.loc[valid_levels['abs_distance'].idxmin(), 'age_hours'] \
        if 'abs_distance' in valid_levels.columns else None
    
    # Get nearby levels ages
    nearby_levels = valid_levels[valid_levels['abs_distance'] <= max_distance] \
        if 'abs_distance' in valid_levels.columns else pd.DataFrame()
    
    if nearby_levels.empty:
        avg_age = None
        oldest_age = None
        newest_age = None
    else:
        avg_age = nearby_levels['age_hours'].mean()
        oldest_age = nearby_levels['age_hours'].max()
        newest_age = nearby_levels['age_hours'].min()
    
    return {
        'nearest_level_age_hours': nearest_level_age,
        'avg_nearby_level_age_hours': avg_age,
        'oldest_nearby_level_age_hours': oldest_age,
        'newest_nearby_level_age_hours': newest_age
    }

def track_level_touches(price: float, levels_df: pd.DataFrame, touch_threshold: float = 50.0) -> tuple:
    """Track level touches with enhanced features"""
    if 'touch_count' not in levels_df.columns:
        levels_df['touch_count'] = 0
        levels_df['last_touch_time'] = None
    
    # Count current touches
    mask = abs(levels_df['price_level'] - price) <= touch_threshold
    current_touches = sum(mask)
    
    # Update touch counts and times
    levels_df.loc[mask, 'touch_count'] += 1
    
    # Calculate features
    features = {
        'current_touches': current_touches,
        'max_touch_count': levels_df['touch_count'].max(),
        'avg_touch_count': levels_df['touch_count'].mean(),
        'total_touches': levels_df['touch_count'].sum(),
        'touched_htf_levels': sum((levels_df['level_type'].str.contains('HTF')) & (levels_df['touch_count'] > 0)),
        'touched_vp_levels': sum((levels_df['level_type'].str.contains('VP')) & (levels_df['touch_count'] > 0))
    }
    
    return levels_df, features

def create_ml_features(ml_data: pd.DataFrame, levels_df: pd.DataFrame, 
                      max_distance: float = 1000.0, 
                      confluence_threshold: float = 200.0,
                      touch_threshold: float = 50.0):
    """Create all features for machine learning with timing information"""
    results = ml_data.copy()
    
    # Initialize feature columns
    feature_columns = [
        'distance_to_nearest_level', 'distance_to_above', 'distance_to_below',
        'relative_distance_nearest', 'relative_distance_above', 'relative_distance_below',
        'nearby_level_count', 'nearby_htf_count', 'nearby_fractal_count', 'nearby_vp_count'
    ]
    
    # Add volume feature columns
    volume_feature_columns = [
        'volume_vs_ma20', 'volume_ma3_vs_previous_ma3', 'volume_stability'
    ]
    
    # Add chunk feature columns (for 11 chunks: -5 to +5)
    chunk_base_features = ['total_levels', 'naked_ratio', 'touched_1_3_ratio', 
                          'monthly_ratio', 'weekly_ratio', 'daily_ratio']
    chunk_feature_columns = [f'chunk_{i}_{feat}' 
                           for i in range(11) 
                           for feat in chunk_base_features]
    
    # Initialize all columns
    for col in feature_columns + volume_feature_columns + chunk_feature_columns:
        results[col] = None
    
    print("Generating features...")
    start_time = time.time()
    
    for idx in results.index:
        current_price = results.loc[idx, 'close']
        current_time = pd.to_datetime(results.loc[idx, 'open_time'])
        
        # Calculate basic level distances and counts
        level_features = calculate_level_distances(
            current_price, levels_df, current_time, max_distance
        )
        
        # Calculate volume features
        volume_features = calculate_volume_features(
            results.loc[:idx]  # Use data up to current index
        )
        
        # Calculate chunk-based features
        chunk_features = generate_chunk_features(
            current_price, levels_df
        )
        
        # Update results
        for col in feature_columns:
            results.loc[idx, col] = level_features[col]
            
        for col in volume_feature_columns:
            results.loc[idx, col] = volume_features[col]
            
        for col in chunk_feature_columns:
            results.loc[idx, col] = chunk_features[col]
        
        # Progress update every 1000 rows
        if idx % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {idx} rows in {elapsed:.2f} seconds")
    
    print(f"Feature generation completed in {time.time() - start_time:.2f} seconds")
    return results

if __name__ == "__main__":
    # Load test data
    ml_data = pd.read_csv("test_data/ml_dataset_test_v3.csv")
    levels_df = pd.read_csv("test_data/levels_dataset_test_v3.csv")
    
    # Convert timestamp columns
    ml_data['open_time'] = pd.to_datetime(ml_data['open_time'])
    levels_df['created_at'] = pd.to_datetime(levels_df['created_at'])
    
    # Create features
    print("\nProcessing features...")
    results = create_ml_features(ml_data, levels_df)
    
    # Save results
    results.to_csv("test_data/ml_features_output_v3.csv", index=False)
    
    # Print performance metrics
    print(f"\nProcessing completed in {time.time():.2f} seconds")
    print(f"Average time per candle: {(time.time()/len(ml_data))*1000:.2f} ms")
    
    # Print sample results
    print("\nSample of results (first 3 rows):")
    print("=" * 80)
    
    # Select sample columns from each feature type
    sample_cols = [
        # Level distance features
        'distance_to_nearest_level', 'distance_to_above', 'distance_to_below',
        
        # Volume features
        'volume_vs_ma20', 'volume_stability',
        
        # Chunk features (first chunk)
        'chunk_5_total_levels', 'chunk_5_naked_ratio'
    ]
    
    print(results[sample_cols].head(3).to_string())
