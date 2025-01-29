import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from level_touch_tracker import update_level_touches
from test_nearest_levels import find_nearest_naked_levels

def create_test_data():
    """Create test levels and candles"""
    # Create test levels
    levels_data = {
        'timeframe': ['daily', 'daily', 'weekly', 'weekly', 'monthly'],
        'level_type': ['HTF', 'Fib_0.618', 'HTF', 'Fib_0.618', 'HTF'],
        'price_level': [41000, 41500, 42000, 42500, 43000],
        'support_touches': [0, 0, 0, 0, 0],
        'resistance_touches': [0, 0, 0, 0, 0]
    }
    levels_df = pd.DataFrame(levels_data)
    
    # Create test candles
    base_time = datetime(2025, 1, 1)
    candles_data = []
    
    # Test Case 1: Level starts pristine, gets 3 support touches
    price = 41100  # Just above first level
    for i in range(3):
        candles_data.append({
            'open_time': base_time + timedelta(hours=i),
            'open': price,
            'high': price + 100,
            'low': 40900,  # Touches first level as support
            'close': price,
            'volume': 1000
        })
    
    # Test Case 2: Level gets 4th support touch (becomes invalid)
    candles_data.append({
        'open_time': base_time + timedelta(hours=3),
        'open': price,
        'high': price + 100,
        'low': 40900,
        'close': price,
        'volume': 1000
    })
    
    # Test Case 3: Level has mixed touches
    price = 42100  # Around third level
    for i in range(2):
        candles_data.append({
            'open_time': base_time + timedelta(hours=4+i),
            'open': price,
            'high': 42200,  # Touches as resistance
            'low': 41900,  # Touches as support
            'close': price,
            'volume': 1000
        })
    
    candles_df = pd.DataFrame(candles_data)
    
    return levels_df, candles_df

def test_level_touches():
    """Test level touch tracking and zone validation"""
    print("\nTesting Level Touch Tracking")
    print("===========================")
    
    # Create test data
    levels_df, candles_df = create_test_data()
    
    # Process each candle
    for i in range(len(candles_df) - 2):
        candle_n2 = candles_df.iloc[i+1].to_dict()
        candle_n1 = candles_df.iloc[i+2].to_dict()
        
        print(f"\nProcessing candle at {candle_n1['open_time']}")
        
        # Find zones based on N-2
        support_zone, resistance_zone = find_nearest_naked_levels(candle_n2['close'], levels_df)
        
        # Print zone info
        if support_zone:
            print(f"Support zone: {support_zone['zone_start']:.0f} - {support_zone['zone_end']:.0f}")
            print(f"Valid levels in support zone: {len(support_zone['levels'])}")
        
        if resistance_zone:
            print(f"Resistance zone: {resistance_zone['zone_start']:.0f} - {resistance_zone['zone_end']:.0f}")
            print(f"Valid levels in resistance zone: {len(resistance_zone['levels'])}")
        
        # Update level touches based on N-1
        levels_df = update_level_touches(levels_df, candle_n1)
        
        # Print updated touch counts
        print("\nUpdated level touches:")
        for _, level in levels_df.iterrows():
            print(f"Level {level['price_level']:.0f}: S={level['support_touches']}, R={level['resistance_touches']}")

if __name__ == "__main__":
    test_level_touches()
