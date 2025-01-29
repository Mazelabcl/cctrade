import pandas as pd
import numpy as np
from chunk_features import generate_chunk_features
from volume_features import calculate_volume_features

def manual_verification():
    """Test with easily verifiable data"""
    # Load test data
    levels_df = pd.read_csv("test_data/manual_test_levels.csv")
    candles_df = pd.read_csv("test_data/manual_test_candles.csv")
    
    # Convert timestamps
    levels_df['created_at'] = pd.to_datetime(levels_df['created_at'])
    candles_df['open_time'] = pd.to_datetime(candles_df['open_time'])
    
    # Test price at 38150 (should be in 38100-38200 chunk)
    test_price = 38150
    
    print("\nManual Verification Test")
    print("=" * 80)
    print(f"Test price: {test_price}")
    
    print("\nExpected Results (Manual Calculation):")
    print("-" * 40)
    print("Chunk 38000-38100:")
    print("- Total levels: 2 (38050 HTF, 38075 VP)")
    print("- Naked ratio: 0.5 (1 naked, 1 touched)")
    print("- Daily ratio: 1.0 (both daily)")
    
    print("\nChunk 38100-38200:")
    print("- Total levels: 2 (38150 HTF, 38180 VP)")
    print("- Naked ratio: 0.5 (38180 VP is naked)")
    print("- Weekly ratio: 0.5 (38150 is weekly)")
    print("- Daily ratio: 0.5 (38180 is daily)")
    
    print("\nChunk 38200-38300:")
    print("- Total levels: 3 (38220 HTF, 38250 VP, 38280 HTF)")
    print("- Naked ratio: 0.33 (only 38280 is naked)")
    print("- Monthly ratio: 0.33 (38220)")
    print("- Weekly ratio: 0.33 (38250)")
    print("- Daily ratio: 0.33 (38280)")
    
    print("\nVolume Calculations:")
    print("-" * 40)
    volumes = candles_df['volume'].values
    print("Volumes:", volumes)
    print("\nMA3 Calculation:")
    print(f"Last 3 candles: {volumes[-3:]} -> mean = {np.mean(volumes[-3:]):.1f}")
    print(f"Previous 3 candles: {volumes[-6:-3]} -> mean = {np.mean(volumes[-6:-3]):.1f}")
    print(f"Expected MA3 ratio: {np.mean(volumes[-3:]) / np.mean(volumes[-6:-3]):.3f}")
    
    print("\nVolume Stability:")
    last_6 = volumes[-6:]
    print(f"Last 6 volumes: {last_6}")
    print(f"Std: {np.std(last_6):.1f}")
    print(f"Mean: {np.mean(last_6):.1f}")
    print(f"Expected stability (std/mean): {np.std(last_6)/np.mean(last_6):.3f}")
    
    print("\nActual Results:")
    print("-" * 40)
    chunk_features = generate_chunk_features(test_price, levels_df)
    volume_features = calculate_volume_features(candles_df)
    
    # Print chunk features
    for i in range(11):  # -5 to +5 chunks
        chunk_base = (test_price // 100) * 100 + ((i-5) * 100)
        chunk_features_filtered = {
            k.replace(f'chunk_{i}_', ''): v 
            for k, v in chunk_features.items() 
            if f'chunk_{i}_' in k
        }
        if any(v > 0 for v in chunk_features_filtered.values()):
            print(f"\nChunk {chunk_base}-{chunk_base+100}:")
            for key, value in chunk_features_filtered.items():
                print(f"  {key}: {value:.3f}")
    
    print("\nVolume Features:")
    print("-" * 40)
    for name, value in volume_features.items():
        print(f"{name}: {value:.3f}")

if __name__ == "__main__":
    manual_verification()
