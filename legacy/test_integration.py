import pandas as pd
import numpy as np
from datetime import datetime
import unittest

# Import all our feature modules
from fractal_timing import detect_fractal, update_fractal_timing, FractalType
from level_touch_tracker import update_level_touches
from candle_ratios import analyze_candle_ratios
from volume_ratios import calculate_volume_ratios
from time_blocks import get_utc_block

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load test data
        cls.candles_df = pd.read_csv('test_data/integration_test_candles.csv', parse_dates=['open_time'])
        cls.levels_df = pd.read_csv('test_data/integration_test_levels.csv', parse_dates=['created_at'])
        
        # Process all features
        cls.results = cls.process_features(cls.candles_df, cls.levels_df)
        
    @staticmethod
    def process_features(candles_df, levels_df):
        """Process all features for the candles dataset"""
        # Create a copy with proper index
        results = candles_df.copy().reset_index(drop=True)
        
        # 1. Detect fractals and count candles since last fractal
        results = update_fractal_timing(results)
        
        # 2. Process each candle for level touches
        level_features = []
        for idx, row in results.iterrows():
            # Update level touches
            updated_levels = update_level_touches(
                levels_df=levels_df,
                candle_n1={
                    'high': row['high'],
                    'low': row['low']
                }
            )
            
            # Extract touch counts for each level type
            touches = {}
            for _, level in updated_levels.iterrows():
                level_key = f"{level['timeframe']}_{level['level_type']}"
                touches[f'support_touches_{level_key}'] = level['support_touches']
                touches[f'resistance_touches_{level_key}'] = level['resistance_touches']
            
            level_features.append(touches)
        
        level_features_df = pd.DataFrame(level_features, index=results.index)
        results = pd.concat([results, level_features_df], axis=1)
        
        # 3. Calculate candle ratios
        candle_ratios = []
        for _, row in results.iterrows():
            ratios = analyze_candle_ratios({
                'high': row['high'],
                'low': row['low'],
                'open': row['open'],
                'close': row['close']
            })
            candle_ratios.append(ratios)
        
        candle_ratios_df = pd.DataFrame(candle_ratios, index=results.index)
        results = pd.concat([results, candle_ratios_df], axis=1)
        
        # 4. Calculate volume ratios
        volume_ratios = []
        for idx, row in results.iterrows():
            # Get volume history up to current candle
            volume_history = results.iloc[:idx+1]['volume']
            
            ratios = calculate_volume_ratios(
                candle_n1={'volume': row['volume']},
                volume_history=volume_history
            )
            volume_ratios.append(ratios)
        
        volume_ratios_df = pd.DataFrame(volume_ratios, index=results.index)
        results = pd.concat([results, volume_ratios_df], axis=1)
        
        # 5. Add time block
        time_blocks = []
        for _, row in results.iterrows():
            block = get_utc_block({
                'open_time': row['open_time']
            })
            time_blocks.append(block)
        
        time_blocks_df = pd.DataFrame(time_blocks, index=results.index)
        results = pd.concat([results, time_blocks_df], axis=1)
        
        return results

    def test_fractal_detection(self):
        """Test if fractals are detected correctly"""
        # Swing High at candle 3 (index 2)
        self.assertEqual(self.results.iloc[2]['fractal_type'], 2,  # FractalType.DOWN = 2 (Swing High)
                        "Failed to detect swing high at candle 3")
        
        # Swing Low at candle 10 (index 9)
        self.assertEqual(self.results.iloc[9]['fractal_type'], 1,  # FractalType.UP = 1 (Swing Low)
                        "Failed to detect swing low at candle 10")
        
        # Swing High at candle 13 (index 12)
        self.assertEqual(self.results.iloc[12]['fractal_type'], 2,  # FractalType.DOWN = 2 (Swing High)
                        "Failed to detect swing high at candle 13")

    def test_support_touches(self):
        """Test support zone touches"""
        # Strong support zone touches (39100-39500)
        support_touches = [7, 8, 19, 20]  # Candle indices that should touch support
        
        for idx in support_touches:
            row = self.results.iloc[idx]
            # Get all support touch columns
            support_cols = [col for col in row.index if col.startswith('support_touches_')]
            
            # Check if any support level was touched
            touches_found = False
            for col in support_cols:
                if row[col] > 0:
                    touches_found = True
                    break
                    
            self.assertTrue(touches_found, f"Failed to detect support touch at candle {idx+1}")

    def test_resistance_touches(self):
        """Test resistance zone touches"""
        # Strong resistance zone touches (40800-41000)
        resistance_touches = [12, 13, 14, 15]  # Candle indices that should touch resistance
        
        for idx in resistance_touches:
            row = self.results.iloc[idx]
            # Get all resistance touch columns
            resistance_cols = [col for col in row.index if col.startswith('resistance_touches_')]
            
            # Check if any resistance level was touched
            touches_found = False
            for col in resistance_cols:
                if row[col] > 0:
                    touches_found = True
                    break
                    
            self.assertTrue(touches_found, f"Failed to detect resistance touch at candle {idx+1}")

    def test_volume_spikes(self):
        """Test volume pattern detection"""
        # Major spike at candle 12 (600 volume)
        self.assertGreater(
            self.results.iloc[11]['volume_short_ratio'], 1.5,
            "Failed to detect volume spike at candle 12"
        )
        
        # High volume at candles 13 and 20
        high_volume_candles = [12, 19]  # indices for candles 13 and 20
        for idx in high_volume_candles:
            self.assertGreater(
                self.results.iloc[idx]['volume_short_ratio'], 1.2,
                f"Failed to detect high volume at candle {idx+1}"
            )

    def test_candle_patterns(self):
        """Test candle pattern detection"""
        # Long lower wick at candle 9 (support bounce)
        self.assertGreater(
            self.results.iloc[8]['lower_wick_ratio'], 0.3,
            "Failed to detect long lower wick at candle 9"
        )
        
        # Long upper wick at candle 15 (resistance rejection)
        self.assertGreater(
            self.results.iloc[14]['upper_wick_ratio'], 0.3,
            "Failed to detect long upper wick at candle 15"
        )
        
        # Small body at candle 14 (inside bar)
        self.assertLess(
            self.results.iloc[13]['body_total_ratio'], 0.3,
            "Failed to detect small body at candle 14"
        )

    def test_time_blocks(self):
        """Test time block assignment"""
        # Phase 1 (Asian session): blocks 0-1
        for idx in range(5):
            self.assertIn(
                self.results.iloc[idx]['utc_block'],
                [0, 1],
                f"Candle {idx+1} should be in Asian session"
            )
        
        # Phase 2 (London session): blocks 1-2
        for idx in range(5, 10):
            self.assertIn(
                self.results.iloc[idx]['utc_block'],
                [1, 2],
                f"Candle {idx+1} should be in London session"
            )

if __name__ == '__main__':
    unittest.main()
