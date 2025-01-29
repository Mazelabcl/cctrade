import pandas as pd
from datetime import datetime, timedelta
from chunk_features import generate_chunk_features
from volume_features import calculate_volume_features

def create_test_data():
    """Create synthetic test data"""
    # Create price levels at different ranges
    base_price = 38186  # More realistic price
    levels = []
    timeframes = ['daily', 'weekly', 'monthly']
    
    # Create levels in chunks of 100
    chunk_starts = range(37800, 38500, 100)  # 7 chunks total
    
    for chunk_start in chunk_starts:
        # Create 3 levels in each chunk
        for j in range(3):
            level_price = chunk_start + (j * 30)  # Spread levels within chunk
            levels.append({
                'price_level': level_price,
                'timeframe': timeframes[j % 3],
                'touch_count': j % 4,  # 0-3 touches
                'created_at': datetime.now() - timedelta(days=j)
            })
    
    levels_df = pd.DataFrame(levels)
    
    # Create candle data
    candles = []
    current_price = base_price
    for i in range(20):
        candles.append({
            'open_time': datetime.now() - timedelta(hours=20-i),
            'open': current_price,
            'high': current_price * 1.001,
            'low': current_price * 0.999,
            'close': current_price * (1 + 0.0002 * (i - 10)),
            'volume': 100 * (1 + 0.1 * (i % 3 - 1))  # Oscillating volume
        })
        current_price = candles[-1]['close']
    
    candles_df = pd.DataFrame(candles)
    return levels_df, candles_df

def test_features():
    """Test feature generation"""
    print("Creating test data...")
    levels_df, candles_df = create_test_data()
    
    print("\nGenerating features...")
    current_price = candles_df['close'].iloc[-1]
    
    # Generate chunk features
    chunk_features = generate_chunk_features(current_price, levels_df)
    
    # Generate volume features
    volume_features = calculate_volume_features(candles_df)
    
    # Print results
    print("\nChunk Features:")
    print("=" * 80)
    for chunk_idx in range(-5, 6):
        chunk_name = f'chunk_{chunk_idx+5}'
        print(f"\nChunk {chunk_idx} ({current_price * (1 + chunk_idx * 0.0025):.2f}):")
        for key, value in chunk_features.items():
            if chunk_name in key:
                print(f"  {key.replace(chunk_name + '_', '')}: {value:.3f}")
    
    print("\nVolume Features:")
    print("=" * 80)
    for name, value in volume_features.items():
        print(f"{name}: {value:.3f}")

if __name__ == "__main__":
    test_features()
