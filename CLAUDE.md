# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Enhanced Pipeline System

The project now uses an enhanced pipeline with period management and rapid development features:

```bash
# Install dependencies (to virtual environment)
source .venv/bin/activate && pip install -r requirements.txt

# Show available data periods and status
python main.py --list-periods

# Check data integrity
python main.py --check-data

# Rapid development - process small sample for testing changes
python main.py --sample 100 --features-only

# Process specific time period
python main.py --period 2021_01_01-2024_12_31 --features-only

# Full pipeline (data fetching + feature engineering)
python main.py --full

# Data pipeline only (fetch and generate basic datasets)
python main.py --data-only

# Feature engineering only (use existing datasets)
python main.py --features-only

# Process all available periods
python main.py --all-periods --features-only
```

### Legacy Commands (still supported)
```bash
# Run all tests
python test_all.py

# Run specific test modules
python test_features.py
python test_integration.py
python test_fractal_timing.py
python test_level_touches.py
python test_nearest_levels.py
python test_candle_interaction.py

# Manual testing with synthetic data
python manual_test.py
```

### Configuration
- Environment variables: Set `BINANCE_API_KEY` and `BINANCE_API_SECRET` in `.env` file
- Global constants in `config.py` (symbol, date ranges, thresholds)
- Data manifest: `data_manifest.json` tracks available time periods and validates data integrity

### Data Management System

The enhanced pipeline includes a data manifest system that tracks time-chunked datasets:

**Manifest Features**:
- Automatic discovery of existing data periods in `base_data/`
- Gap and overlap detection between time periods
- File integrity validation
- Support for sample datasets for rapid development

**Rapid Development Workflow**:
1. `python main.py --list-periods` - See available data periods
2. `python main.py --sample 100 --features-only` - Test changes on small dataset
3. `python main.py --period YYYY_MM_DD-YYYY_MM_DD --features-only` - Test on specific period
4. `python main.py --all-periods --features-only` - Full production run

**Time-Chunked Data Structure**:
```
base_data/
├── ml_dataset_2017_01_01-2020_12_31.csv      # ~29K rows
├── levels_dataset_2017_01_01-2020_12_31.csv
├── ml_dataset_2021_01_01-2024_12_31.csv      # ~35K rows
├── levels_dataset_2021_01_01-2024_12_31.csv
└── data_manifest.json                        # Auto-generated
```

## High-Level Architecture

### Core Components

**Data Pipeline**:
1. **data_fetching.py**: Fetches OHLCV data from Binance API for multiple timeframes (1m, 1h, 12h, 1d, 1w, 1M)
2. **indicators.py**: Detects fractals and calculates technical levels (HTF, Fibonacci, Volume Profile)
3. **dataset_generation.py**: Creates two main datasets:
   - `ml_dataset.csv`: 1-hour candles with fractal labels for ML training
   - `levels_dataset.csv`: All technical levels across timeframes
4. **create_ml_features.py**: Feature engineering pipeline that combines candle data with level interactions

**Technical Analysis**:
- **fractal_timing.py**: 5-candle fractal detection (swing highs/lows) with timing features
- **level_touch_tracker.py**: Tracks level touches and validates levels (invalid after 4 touches)
- **feature_engineering.py**: Confluence zone analysis and level clustering
- **candle_ratios.py**: Candlestick pattern analysis (body ratios, wick ratios, patterns)
- **volume_ratios.py**: Volume analysis against moving averages
- **time_blocks.py**: Trading session detection (Asian/European/American sessions)

**Level Types**:
- **HTF Levels**: Higher timeframe structure levels from candle direction changes
- **Volume Profile**: POC (Point of Control), VAH/VAL (Value Area High/Low)
- **Fibonacci**: 0.5, 0.618, 0.75 retracement levels between swing points
- **Fractals**: Swing highs/lows from daily/weekly/monthly timeframes

### Feature Engineering Strategy

The ML pipeline generates features by analyzing candle N-1 interactions with zones derived from candle N-2:

1. **Zone Detection**: Find nearest naked support/resistance levels around N-2 close price
2. **Zone Analysis**: Create 1.5% zones around key levels, count level types by timeframe
3. **Interaction Features**: Track which levels are touched by N-1 candle's high/low
4. **Pattern Features**: Detect potential swing patterns, candle structure, volume spikes
5. **Timing Features**: Track candles since last fractal formation

### Data Structures

**Candle Data** (OHLCV + features):
```python
{
    'open_time': datetime,
    'open': float, 'high': float, 'low': float, 'close': float, 'volume': float,
    'bearish_fractal': bool, 'bullish_fractal': bool,
    # Feature columns added by create_ml_features.py
}
```

**Level Data**:
```python
{
    'price_level': float,
    'level_type': str,  # 'Fractal_Low', 'VP_poc', 'Fib_0.618', 'HTF_level'
    'timeframe': str,   # 'daily', 'weekly', 'monthly'
    'created_at': datetime,
    'source': str       # 'fractal', 'htf', 'fibonacci', 'volume_profile'
}
```

### Testing Strategy

- **test_all.py**: Integration test using synthetic data across all feature modules
- **test_data/**: Contains synthetic candle and level data for reproducible testing
- **manual_test.py**: Interactive testing with predefined scenarios
- Individual test files for each feature module (test_fractal_timing.py, etc.)

### Project Context

This is a cryptocurrency trading bot focused on Bitcoin (BTCUSDT) that uses machine learning to predict fractal formation (potential reversal points). The bot combines multiple timeframe analysis with technical level tracking to identify high-probability turning points in the market.

**Key Files**:
- `main.py`: Entry point and orchestration
- `config.py`: Global configuration and API credentials
- `create_ml_features.py`: Main feature engineering pipeline
- `project_documentation.md`: Detailed technical specifications
- `new_ml_features.md`: Implementation plan and feature descriptions

**Data Storage**:
- `base_data/`: Historical datasets split by time periods
- CSV files: Generated datasets for ML training
- HTML files: Interactive visualizations of price action and levels