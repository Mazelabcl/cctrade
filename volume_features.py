import pandas as pd
import numpy as np
from typing import Dict

def calculate_volume_ma(volumes: np.ndarray, window: int) -> float:
    """Calculate volume moving average for the last n candles"""
    if len(volumes) < window:
        return np.mean(volumes)
    return np.mean(volumes[-window:])

def calculate_volume_features(candle_data: pd.DataFrame) -> Dict:
    """Calculate volume-related features
    
    Features:
    1. volume_vs_ma20: Current volume compared to 20-period MA
    2. volume_ma3_vs_previous_ma3: Ratio of last 3 candles vs previous 3
    3. volume_stability: Normalized standard deviation (lower = more stable)
    """
    volumes = candle_data['volume'].values
    if len(volumes) < 6:
        return {
            'volume_vs_ma20': 1.0,
            'volume_ma3_vs_previous_ma3': 1.0,
            'volume_stability': 0.0
        }
    
    # Current volume vs MA20
    current_volume = volumes[-1]
    volume_ma20 = calculate_volume_ma(volumes, 20)
    volume_vs_ma20 = current_volume / volume_ma20 if volume_ma20 > 0 else 1.0
    
    # Last 3 vs Previous 3
    last_3_mean = np.mean(volumes[-3:])
    prev_3_mean = np.mean(volumes[-6:-3])
    volume_trend = last_3_mean / prev_3_mean if prev_3_mean > 0 else 1.0
    
    # Volume stability (using coefficient of variation)
    last_6_std = np.std(volumes[-6:])
    last_6_mean = np.mean(volumes[-6:])
    stability = last_6_std / last_6_mean if last_6_mean > 0 else 0.0
    
    return {
        'volume_vs_ma20': round(volume_vs_ma20, 3),
        'volume_ma3_vs_previous_ma3': round(volume_trend, 3),
        'volume_stability': round(stability, 3)  # Lower means more stable
    }

if __name__ == "__main__":
    # Test the functionality
    test_data = pd.DataFrame({
        'volume': [100, 120, 90, 110, 130, 95, 105, 115, 125, 100]
    })
    
    features = calculate_volume_features(test_data)
    
    print("\nTest Results:")
    print("=" * 80)
    for name, value in features.items():
        print(f"{name}: {value}")
