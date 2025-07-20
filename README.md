# 🤖 Trading Bot - Bitcoin Fractal Prediction System

A sophisticated cryptocurrency trading bot that uses machine learning to predict Bitcoin price fractals (potential reversal points). The system simulates what a trader would do in a graphical interface by setting levels on candle charts and tracking their interactions to generate comprehensive ML datasets.

## 🎯 Project Overview

This project analyzes Bitcoin (BTCUSDT) price action across multiple timeframes to predict swing points using:

- **Machine Learning Ready**: Multi-class fractal prediction with configurable horizons (hour, day, week, month)
- **Fractal Detection**: 5-candle swing high/low patterns with timing analysis
- **Level Simulation**: Automated level setting mimicking trader behavior on charts
- **Technical Level Tracking**: HTF levels, Fibonacci retracements, Volume Profile, fractals
- **Advanced Feature Engineering**: 75+ interpretable features for ML models
- **Time-Chunked Data Management**: Efficient processing of large historical datasets
- **DataRobot Integration**: Ready-to-use CSV exports for immediate ML model training

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

### 🧠 ML Dataset Generation (New!)

```bash
# Create ML-ready dataset for DataRobot
python create_ml_dataset.py --candles "datasets/ml_dataset_2025_01_01-2025_06_30.csv" \
                            --levels "datasets/levels_dataset_2025_01_01-2025_06_30.csv" \
                            --output "ml_ready_dataset.csv" \
                            --horizon "day"

# Different prediction horizons
python create_ml_dataset.py --candles [INPUT] --levels [INPUT] --output [OUTPUT] --horizon "hour"   # Next 1 hour
python create_ml_dataset.py --candles [INPUT] --levels [INPUT] --output [OUTPUT] --horizon "day"    # Next 24 hours  
python create_ml_dataset.py --candles [INPUT] --levels [INPUT] --output [OUTPUT] --horizon "week"   # Next 7 days
python create_ml_dataset.py --candles [INPUT] --levels [INPUT] --output [OUTPUT] --horizon "month"  # Next 30 days

# Test with limited samples
python create_ml_dataset.py --candles [INPUT] --levels [INPUT] --output [OUTPUT] --max-samples 500
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
| `create_ml_features.py` | Legacy feature engineering pipeline |
| `create_ml_dataset.py` | **New ML dataset generation pipeline** |
| `target_variable.py` | **Multi-class fractal target generation** |
| `ml_feature_engineering.py` | **Enhanced ML-optimized feature engineering** |
| `config.py` | Global configuration and ML prediction settings |
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
├── ml_features_dataset_[period].csv          # Legacy feature datasets
├── ml_features_dataset_[period]_sample_N.csv # Legacy sample datasets
└── features/
    ├── ml_ready_dataset_full_2025.csv        # **New ML-ready datasets**
    ├── ml_ready_dataset_hour.csv             # Hourly prediction horizon
    └── ml_ready_dataset_day.csv              # Daily prediction horizon
```

## 🧠 Technical Analysis Features

### Level Types Tracked
- **HTF Levels**: Higher timeframe structure levels from candle direction changes
- **Volume Profile**: POC (Point of Control), VAH/VAL (Value Area High/Low)
- **Fibonacci**: 0.5, 0.618, 0.75 retracement levels between swing points
- **Fractals**: Swing highs/lows from daily/weekly/monthly timeframes

### ML Feature Categories (75 Features)

1. **Level Proximity Features** (4 features)
   - `nearest_support_distance_pct`: % distance to closest support
   - `nearest_resistance_distance_pct`: % distance to closest resistance  
   - `nearest_support_strength`: Weighted strength score of nearest support
   - `nearest_resistance_strength`: Weighted strength score of nearest resistance

2. **Confluence Zone Analysis** (36 features)
   - Level counting in 0.5%, 1.0%, 1.5%, 2.0% price zones
   - Breakdown by timeframe (daily, weekly, monthly)
   - Breakdown by level type (HTF, Volume Profile, Fibonacci, Fractals)
   - Weighted confluence strength scores

3. **Volume Analysis** (5 features)
   - Volume ratios vs 20/50 period moving averages
   - Volume spike detection (>2x average)
   - Volume percentile in recent history
   - Volume trend analysis

4. **Price Action Features** (11 features)
   - Candlestick ratios (body, upper/lower wick)
   - Price changes and gaps
   - Moving average relationships
   - Volatility measures

5. **Temporal Features** (8 features)
   - Hour of day, day of week (normalized)
   - Trading session detection (Asian/European/American)
   - Hours since last bullish/bearish fractal

6. **Target Variable** (Multi-class)
   - `fractal_direction`: 0=no_fractal, 1=bullish_fractal, 2=bearish_fractal
   - Configurable prediction horizons (1 hour to 30 days)

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

The new ML pipeline generates DataRobot-ready datasets with enhanced features:

### Target Variable Design
- **Multi-class Classification**: 0=no_fractal, 1=bullish_fractal, 2=bearish_fractal
- **Configurable Horizons**: Predict fractal formation in next N candles
  - `hour`: Next 1 candle (1 hour) - 75% no fractal, 12.5% each fractal type
  - `day`: Next 24 candles (1 day) - Balanced ~50% bullish, 50% bearish
  - `week`: Next 168 candles (7 days) - Longer-term swing prediction
  - `month`: Next 720 candles (30 days) - Major trend changes

### Dataset Quality
- **4,295+ samples** in full 2025 dataset
- **75 interpretable features** (no sparse vectors)
- **Zero missing values** with proper data validation
- **Time-aware splits** preventing look-ahead bias
- **Feature name cleaning** for ML platform compatibility

### Export Formats
- **CSV for DataRobot**: `timestamp,features...,fractal_direction`
- **Metadata included**: Time ranges, target distribution, quality metrics
- **Ready-to-upload**: No preprocessing required

### Example Output
```
features/ml_ready_dataset_full_2025.csv
- 4,295 samples × 76 columns (75 features + target)
- Time range: 2025-01-01 to 2025-06-29
- Target: 49.5% bullish, 50.5% bearish fractals (daily horizon)
- Quality: 0% missing values, minimal constant features
```

## 📖 Documentation

- **`README.md`**: This overview and quick start guide
- **`ML_USAGE_GUIDE.md`**: Comprehensive guide for ML dataset generation and model training
- **`CLAUDE.md`**: Detailed guidance for Claude Code assistant development
- **`project_documentation.md`**: Technical specifications and architecture
- **`new_ml_features.md`**: Feature engineering implementation details

## 🤝 Contributing

This project uses Claude Code assistant for development. Key documentation:
- Code architecture and patterns
- Development commands and workflows  
- Feature engineering concepts
- ML pipeline design
- Testing strategies

## 📝 License

This project is for educational and research purposes in algorithmic trading and machine learning applications in financial markets.

---

*Built with Python, pandas, Binance API, and advanced technical analysis for cryptocurrency trading research.*