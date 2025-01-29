import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def create_price_chunks(price: float, chunk_size: float = 100.0) -> List[tuple]:
    """Create fixed-size price chunks above and below current price
    
    Example:
    Price: 38,186
    Will create chunks:
    37,800-37,900
    37,900-38,000
    38,000-38,100
    38,100-38,200 (current chunk)
    38,200-38,300
    38,300-38,400
    """
    # Find the current chunk's boundaries
    current_chunk_start = (price // chunk_size) * chunk_size
    
    # Create 5 chunks below and 5 above
    chunks = []
    for i in range(-5, 6):
        chunk_start = current_chunk_start + (i * chunk_size)
        chunk_end = chunk_start + chunk_size
        chunks.append((chunk_start, chunk_end))
    
    return chunks

def get_levels_in_chunk(chunk_start: float, chunk_end: float, 
                       levels_df: pd.DataFrame, max_touches: int = 3) -> pd.DataFrame:
    """Get valid levels within a price chunk"""
    # Filter levels within chunk price range
    mask = (levels_df['price_level'] >= chunk_start) & \
           (levels_df['price_level'] < chunk_end)
    
    # Add touch_count if it doesn't exist
    if 'touch_count' not in levels_df.columns:
        levels_df['touch_count'] = 0
    
    # Apply touch count filter if column exists
    if 'touch_count' in levels_df.columns:
        mask = mask & (levels_df['touch_count'] <= max_touches)
    
    return levels_df[mask]

def calculate_chunk_ratios(levels_in_chunk: pd.DataFrame) -> Optional[Dict]:
    """Calculate various ratios for levels in a chunk"""
    if levels_in_chunk.empty:
        return {
            'total_levels': 0,
            'naked_ratio': 0.0,
            'touched_1_3_ratio': 0.0,
            'monthly_ratio': 0.0,
            'weekly_ratio': 0.0,
            'daily_ratio': 0.0
        }
    
    total_levels = len(levels_in_chunk)
    
    # Touch state ratios
    naked_levels = levels_in_chunk[levels_in_chunk['touch_count'] == 0]
    touched_1_3 = levels_in_chunk[(levels_in_chunk['touch_count'] >= 1) & 
                                 (levels_in_chunk['touch_count'] <= 3)]
    
    # Timeframe ratios
    monthly_levels = levels_in_chunk[levels_in_chunk['timeframe'].str.contains('monthly', case=False)]
    weekly_levels = levels_in_chunk[levels_in_chunk['timeframe'].str.contains('weekly', case=False)]
    daily_levels = levels_in_chunk[levels_in_chunk['timeframe'].str.contains('daily', case=False)]
    
    return {
        'total_levels': total_levels,
        'naked_ratio': len(naked_levels) / total_levels if total_levels > 0 else 0.0,
        'touched_1_3_ratio': len(touched_1_3) / total_levels if total_levels > 0 else 0.0,
        'monthly_ratio': len(monthly_levels) / total_levels if total_levels > 0 else 0.0,
        'weekly_ratio': len(weekly_levels) / total_levels if total_levels > 0 else 0.0,
        'daily_ratio': len(daily_levels) / total_levels if total_levels > 0 else 0.0
    }

def generate_chunk_features(price: float, levels_df: pd.DataFrame, 
                          chunk_size: float = 100.0) -> Dict:
    """Generate all chunk-based features for a given price"""
    # Create chunks
    chunks = create_price_chunks(price, chunk_size)
    
    # Calculate features for each chunk
    all_features = {}
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_levels = get_levels_in_chunk(chunk_start, chunk_end, levels_df)
        ratios = calculate_chunk_ratios(chunk_levels)
        
        # Add chunk index to feature names
        chunk_features = {
            f'chunk_{i}_' + k: v for k, v in ratios.items()
        }
        all_features.update(chunk_features)
    
    return all_features

if __name__ == "__main__":
    # Test the functionality
    from datetime import datetime, timedelta
    
    # Create sample test data
    test_levels = pd.DataFrame({
        'price_level': [40100, 40200, 40300, 40400, 40500],
        'timeframe': ['daily', 'weekly', 'monthly', 'daily', 'weekly'],
        'touch_count': [0, 1, 2, 3, 4],
        'created_at': [datetime.now() - timedelta(days=i) for i in range(5)]
    })
    
    # Test feature generation
    test_price = 40300
    features = generate_chunk_features(test_price, test_levels)
    
    print("\nTest Results:")
    print("=" * 80)
    for name, value in features.items():
        print(f"{name}: {value}")
