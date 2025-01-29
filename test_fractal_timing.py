import pandas as pd
from datetime import datetime, timedelta
from fractal_timing import FractalType, detect_fractal, update_fractal_timing

def create_test_data():
    """Create test candles with clear fractal patterns"""
    base_time = datetime(2025, 1, 1)
    candles_data = []
    
    # Create a swing high pattern
    prices = [41000, 41100, 41500, 41200, 41000]  # Clear swing high in middle
    for i, price in enumerate(prices):
        candles_data.append({
            'open_time': base_time + timedelta(hours=i),
            'open': price,
            'high': price + 50,
            'low': price - 50,
            'close': price,
            'volume': 1000
        })
    
    # Create some normal candles
    for i in range(5, 8):
        candles_data.append({
            'open_time': base_time + timedelta(hours=i),
            'open': 41000,
            'high': 41050,
            'low': 40950,
            'close': 41000,
            'volume': 1000
        })
    
    # Create a swing low pattern
    prices = [41000, 40900, 40500, 40800, 41000]  # Clear swing low in middle
    for i, price in enumerate(prices):
        candles_data.append({
            'open_time': base_time + timedelta(hours=i+8),
            'open': price,
            'high': price + 50,
            'low': price - 50,
            'close': price,
            'volume': 1000
        })
    
    return pd.DataFrame(candles_data)

def test_fractal_timing():
    """Test fractal detection and timing tracking"""
    print("\nTesting Fractal Detection and Timing")
    print("==================================")
    
    # Create test data
    candles_df = create_test_data()
    
    # Update fractal information
    updated_df = update_fractal_timing(candles_df)
    
    # Print results
    print("\nFractal Detection Results:")
    for i in range(len(updated_df)):
        row = updated_df.iloc[i]
        fractal_type = FractalType(row['fractal_type'])
        print(f"\nCandle at {row['open_time']}:")
        print(f"Price: {row['close']}")
        print(f"Fractal Type: {fractal_type.name}")
        print(f"Candles since last up: {row['candles_since_last_up']}")
        print(f"Candles since last down: {row['candles_since_last_down']}")
        if row['last_up_time']:
            print(f"Last swing low at: {row['last_up_time']}")
        if row['last_down_time']:
            print(f"Last swing high at: {row['last_down_time']}")

if __name__ == "__main__":
    test_fractal_timing()
