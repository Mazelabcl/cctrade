import pandas as pd
from time_blocks import get_utc_block
from volume_ratios import calculate_volume_ratios
from candle_ratios import analyze_candle_ratios
from swing_patterns import analyze_swing_patterns
from test_nearest_levels import find_nearest_naked_levels
from test_candle_interaction import analyze_candle_interaction

def load_test_data():
    """Load test candles and levels"""
    candles_df = pd.read_csv('test_data/test_candles.csv')
    levels_df = pd.read_csv('test_data/test_levels.csv')
    return candles_df, levels_df

def test_time_blocks():
    """Test UTC block detection"""
    print("\nTesting Time Blocks for all candles:")
    candles_df, _ = load_test_data()
    candles = candles_df.to_dict('records')
    
    for i, candle in enumerate(candles):
        result = get_utc_block({'open_time': candle['open_time']})
        print(f"Candle {candle['open_time']} -> UTC Block: {result['utc_block']}")

    # Use last candle as N-1 for remaining tests
    candle_n1 = candles[-2]
    print("\nTesting Time Blocks:")
    result = get_utc_block(candle_n1)
    print(f"N-1 Candle UTC Block: {result}")

def test_volume_ratios():
    """Test volume ratio calculations"""
    print("\nTesting Volume Ratios:")
    candles_df, _ = load_test_data()
    
    # Get volume history and N-1 candle
    volume_history = candles_df['volume']
    candle_n1 = candles_df.iloc[-2].to_dict()
    
    result = calculate_volume_ratios(candle_n1, volume_history)
    print(f"Volume Ratios: {result}")

def test_candle_ratios():
    """Test candle ratio calculations"""
    print("\nTesting Candle Ratios:")
    candles_df, _ = load_test_data()
    
    # Test N-1 candle
    candle_n1 = candles_df.iloc[-2].to_dict()
    result = analyze_candle_ratios(candle_n1)
    print(f"Candle Ratios: {result}")

def test_swing_patterns():
    """Test swing pattern detection"""
    print("\nTesting Swing Patterns:")
    candles_df, _ = load_test_data()
    
    # Get N-1, N-2, N-3 candles and print their values
    candle_n1 = candles_df.iloc[-2].to_dict()
    candle_n2 = candles_df.iloc[-3].to_dict()
    candle_n3 = candles_df.iloc[-4].to_dict()
    
    print(f"N-1 high: {candle_n1['high']}, low: {candle_n1['low']}")
    print(f"N-2 high: {candle_n2['high']}, low: {candle_n2['low']}")
    print(f"N-3 high: {candle_n3['high']}, low: {candle_n3['low']}")
    
    result = analyze_swing_patterns(candle_n1, candle_n2, candle_n3)
    print(f"Swing Patterns: {result}")

def test_nearest_levels():
    """Test nearest naked levels detection"""
    print("\nTesting Nearest Levels:")
    candles_df, levels_df = load_test_data()
    
    # Use N-2 close price
    price = candles_df.iloc[-3]['close']
    support_zone, resistance_zone = find_nearest_naked_levels(price, levels_df)
    print(f"Support Zone: {support_zone}")
    print(f"Resistance Zone: {resistance_zone}")

def test_candle_interaction():
    """Test candle interaction with levels"""
    print("\nTesting Candle Interaction:")
    candles_df, levels_df = load_test_data()
    
    # Get N-1 candle and zones
    candle_n1 = candles_df.iloc[-2].to_dict()
    price = candles_df.iloc[-3]['close']
    support_zone, resistance_zone = find_nearest_naked_levels(price, levels_df)
    
    result = analyze_candle_interaction(candle_n1, support_zone, resistance_zone)
    print(f"Candle Interaction: {result}")

if __name__ == "__main__":
    print("Starting Tests with Synthetic Data")
    print("=================================")
    
    # Run tests one by one
    test_time_blocks()
    test_volume_ratios()
    test_candle_ratios()
    test_swing_patterns()
    test_nearest_levels()
    test_candle_interaction()
