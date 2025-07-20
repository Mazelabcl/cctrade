# 🤖 Trading Bot - Bitcoin Fractal Prediction System

A sophisticated cryptocurrency trading bot that uses machine learning to predict Bitcoin price fractals (potential reversal points). The system combines multi-timeframe technical analysis with advanced feature engineering to identify high-probability turning points in the market.

## 🎯 Project Overview

This project analyzes Bitcoin (BTCUSDT) price action across multiple timeframes to predict swing points using:

- **Fractal Detection**: 5-candle swing high/low patterns
- **Multi-Timeframe Analysis**: Daily, weekly, and monthly levels
- **Technical Level Tracking**: HTF levels, Fibonacci retracements, Volume Profile
- **Advanced Feature Engineering**: 80+ features combining candle patterns, volume analysis, and level interactions
- **Time-Chunked Data Management**: Efficient processing of large historical datasets

## 🚀 Quick Start

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (create .env file)
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

### Basic Usage

```bash
# See available data periods and status
python main.py --list-periods

# Quick development test (100 rows)
python main.py --sample 100 --features-only

# Process specific time period
python main.py --period 2021_01_01-2024_12_31 --features-only

# Full pipeline (data + features)
python main.py --full
```

## 🛠️ Command Line Interface

The enhanced pipeline system provides flexible execution modes:

### Information Commands
```bash
python main.py --list-periods    # Show available data periods
python main.py --check-data      # Validate data integrity
python main.py --help           # Show all options
```

### Execution Modes
```bash
python main.py                    # Full pipeline (default)
python main.py --data-only        # Only fetch data from Binance
python main.py --features-only    # Only run feature engineering
python main.py --full             # Explicit full pipeline
```

### Period Management
```bash
python main.py --period 2021_01_01-2024_12_31    # Specific period
python main.py --all-periods                      # Process all periods
```

### Rapid Development
```bash
python main.py --sample 100 --features-only      # Small sample for testing
python main.py --quick-test                      # Auto-select small dataset
```

## 📁 Project Structure

### Key Files

| File | Purpose |
|------|---------|
| `main.py` | Enhanced pipeline orchestration with CLI |
| `data_manager.py` | Time-chunked data management and validation |
| `create_ml_features.py` | Feature engineering pipeline |
| `config.py` | Global configuration and API credentials |
| `CLAUDE.md` | Detailed guidance for Claude Code assistant |

### Core Modules

| Module | Description |
|--------|-------------|
| `data_fetching.py` | Binance API data retrieval |
| `dataset_generation.py` | Basic dataset creation with fractals and levels |
| `indicators.py` | Technical analysis (fractals, HTF levels, Fibonacci, Volume Profile) |
| `fractal_timing.py` | Fractal detection and timing features |
| `level_touch_tracker.py` | Level interaction tracking |
| `candle_ratios.py` | Candlestick pattern analysis |
| `volume_ratios.py` | Volume analysis and spikes |
| `feature_engineering.py` | Zone analysis and confluence detection |

### Data Structure

```
base_data/
├── ml_dataset_2017_01_01-2020_12_31.csv      # Historical OHLCV + fractals
├── levels_dataset_2017_01_01-2020_12_31.csv  # Technical levels
├── ml_dataset_2021_01_01-2024_12_31.csv      # Recent data
├── levels_dataset_2021_01_01-2024_12_31.csv
└── data_manifest.json                        # Auto-generated metadata

Generated Features:
├── ml_features_dataset_[period].csv          # Full feature datasets
└── ml_features_dataset_[period]_sample_N.csv # Sample datasets
```

## 🧠 Technical Analysis Features

### Level Types Tracked
- **HTF Levels**: Higher timeframe structure levels from candle direction changes
- **Volume Profile**: POC (Point of Control), VAH/VAL (Value Area High/Low)
- **Fibonacci**: 0.5, 0.618, 0.75 retracement levels between swing points
- **Fractals**: Swing highs/lows from daily/weekly/monthly timeframes

### Feature Categories

1. **Zone Analysis** (40+ features)
   - Support/resistance zone detection
   - Level confluence counting by timeframe and type
   - Naked (untouched) level identification

2. **Candle Interaction** (20+ features)
   - Level touch detection for current candle
   - Support vs resistance interaction tracking
   - Pattern strength analysis

3. **Technical Patterns** (10+ features)
   - Candlestick ratios (body, wick, position)
   - Swing pattern detection
   - Volume spike analysis

4. **Timing Features** (10+ features)
   - Fractal timing and cycles
   - Trading session blocks
   - Time-based patterns

## 📊 Data Management

The system uses a manifest-based approach for handling time-chunked datasets:

### Features
- **Automatic Discovery**: Scans `base_data/` for existing periods
- **Gap Detection**: Identifies missing date ranges
- **Overlap Validation**: Detects conflicting time periods
- **Sample Generation**: Creates smaller datasets for rapid development
- **Integrity Checking**: Validates file existence and structure

### Rapid Development Workflow

1. **Explore Data**: `python main.py --list-periods`
2. **Quick Test**: `python main.py --sample 100 --features-only` (seconds)
3. **Period Test**: `python main.py --period YYYY_MM_DD-YYYY_MM_DD --features-only` (minutes)
4. **Production Run**: `python main.py --all-periods --features-only` (hours)

## 🧪 Testing

```bash
# Run all tests with synthetic data
python test_all.py

# Test specific components
python test_features.py
python test_fractal_timing.py
python test_level_touches.py
python test_integration.py
```

## ⚙️ Configuration

### Environment Variables (.env)
```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret
```

### Global Settings (config.py)
- Symbol: BTCUSDT
- Date ranges: 2017-2024
- Thresholds: Confluence, hit counts, price ranges
- Logging configuration

## 🔄 Development Workflow

### For Feature Development
```bash
# 1. Quick test on small sample
python main.py --sample 50 --features-only

# 2. Test on specific period
python main.py --period 2021_01_01-2024_12_31 --features-only

# 3. Production run when ready
python main.py --all-periods --features-only
```

### For Data Updates
```bash
# 1. Fetch new data
python main.py --data-only

# 2. Check data integrity
python main.py --check-data

# 3. Generate features
python main.py --features-only
```

## 📈 ML Pipeline Output

The feature engineering generates comprehensive datasets ready for machine learning:

- **Target Variable**: Fractal formation prediction (bullish/bearish swings)
- **Features**: 80+ engineered features combining technical analysis
- **Format**: CSV files ready for DataRobot or other ML platforms
- **Validation**: Proper time-series split avoiding look-ahead bias

## 🤝 Contributing

This project uses Claude Code assistant for development. See `CLAUDE.md` for detailed guidance on:
- Code architecture and patterns
- Development commands and workflows
- Feature engineering concepts
- Testing strategies

## 📝 License

This project is for educational and research purposes in algorithmic trading and machine learning applications in financial markets.

---

*Built with Python, pandas, Binance API, and advanced technical analysis for cryptocurrency trading research.*