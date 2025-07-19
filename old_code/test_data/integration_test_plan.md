# Integration Test Plan

## Test Scenario Overview
We'll create a 20-candle sequence that tests all features: fractals, level touches, zones, volume patterns, and candle structures.

## Price Levels Setup
We'll set up multiple levels around 40000:

### Support Levels (Below 40000)
1. Strong Support Zone (39400-39600)
   - Daily HTF at 39500
   - Weekly POC at 39450
   - Monthly Fib 0.618 at 39400
   - Daily Fractal at 39550

2. Weak Support Zone (38800-39000)
   - Single Daily HTF at 38900

### Resistance Levels (Above 40000)
1. Strong Resistance Zone (40400-40600)
   - Daily HTF at 40500
   - Weekly VAH at 40450
   - Monthly Fib 0.75 at 40600
   - Daily Fractal at 40550

2. Weak Resistance Zone (41000-41200)
   - Single Weekly HTF at 41100

## Candle Sequence Plan

### Phase 1: Fractal Formation (Candles 1-5)
- Candle 1: Normal candle around 40000
- Candle 2: Slight move up
- Candle 3: Higher high (potential swing high setup)
- Candle 4: Lower high
- Candle 5: Lower high (confirms candle 3 as swing high)
Expected: Fractal Down detected at candle 3

### Phase 2: Support Test (Candles 6-10)
- Candle 6: Sharp drop to 39600
- Candle 7: Tests strong support zone (39400-39600)
- Candle 8: Bounces from support with long lower wick
- Candle 9: Confirmation candle up
- Candle 10: Higher low (potential swing low setup)
Expected: Multiple support level touches, potential swing low forming

### Phase 3: Volume Spike & Resistance (Candles 11-15)
- Candle 11: Strong volume candle up (3x average)
- Candle 12: Tests resistance zone (40400-40600)
- Candle 13: Inside bar with high volume
- Candle 14: Break attempt of resistance
- Candle 15: Rejection from resistance
Expected: Volume ratio spikes, resistance level touches

### Phase 4: Pattern Completion (Candles 16-20)
- Candle 16: Lower high
- Candle 17: Lower low (confirms swing pattern)
- Candle 18: Dead cat bounce
- Candle 19: Sharp drop with high volume
- Candle 20: Final test of support
Expected: Complete swing pattern, multiple level touches

## Expected Feature Values

### Fractal Detection
- Swing High at Candle 3
- Swing Low at Candle 10
- Swing High at Candle 14

### Level Touches
- Support Zone (39400-39600): Multiple touches in Phase 2
- Resistance Zone (40400-40600): Multiple touches in Phase 3

### Volume Features
- Major spike at Candle 11 (3x normal)
- High volume at Candle 13 and 19

### Candle Patterns
- Long lower wick at Candle 8 (support rejection)
- Long upper wick at Candle 15 (resistance rejection)
- Small body at Candle 13 (indecision)

### Time Block Features
We'll space these candles across different time blocks to test UTC time features:
- Phase 1: Asian session
- Phase 2: London session
- Phase 3: NY session overlap
- Phase 4: NY session

## Test Data Generation Steps
1. Create levels dataset with above support/resistance levels
2. Generate candle data following the sequence plan
3. Include volume data with specified spikes
4. Add timestamps across different trading sessions
