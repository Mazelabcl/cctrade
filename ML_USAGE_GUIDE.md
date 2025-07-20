# 🧠 ML Usage Guide - Bitcoin Fractal Prediction

This guide covers the complete machine learning pipeline for generating datasets and training models to predict Bitcoin price fractals.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Problem](#understanding-the-problem)
3. [Dataset Generation](#dataset-generation)
4. [Feature Engineering](#feature-engineering)
5. [Target Variable Design](#target-variable-design)
6. [DataRobot Integration](#datarobot-integration)
7. [Model Training Best Practices](#model-training-best-practices)
8. [Evaluation and Validation](#evaluation-and-validation)
9. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Step 1: Generate Base Data
```bash
# Ensure you have base datasets
python main.py --list-periods

# If needed, generate new data
python main.py --period 2021_01_01-2024_12_31 --features-only
```

### Step 2: Create ML Dataset
```bash
# Create ML-ready dataset for daily predictions
python create_ml_dataset.py \
    --candles "datasets/ml_dataset_2025_01_01-2025_06_30.csv" \
    --levels "datasets/levels_dataset_2025_01_01-2025_06_30.csv" \
    --output "features/bitcoin_fractal_prediction_daily.csv" \
    --horizon "day"
```

### Step 3: Upload to DataRobot
1. Upload `features/bitcoin_fractal_prediction_daily.csv`
2. Select `fractal_direction` as target variable
3. Choose "Multi-class Classification"
4. Start Autopilot

## 🎯 Understanding the Problem

### What Are We Predicting?

We're predicting **fractal formation** in Bitcoin price action:

- **Bullish Fractal (1)**: A swing low that indicates potential price reversal upward
- **Bearish Fractal (2)**: A swing high that indicates potential price reversal downward  
- **No Fractal (0)**: No significant swing point formation

### Why This Matters

Fractals represent **potential turning points** in market trends. By predicting them, we can:
- Identify optimal entry/exit points
- Anticipate trend reversals
- Improve risk management timing
- Automate technical analysis decisions

### The Trading Context

The system simulates what a professional trader would do:
1. **Set levels** on the chart (support/resistance, Fibonacci, etc.)
2. **Track interactions** between price and these levels
3. **Analyze confluence** where multiple levels cluster together
4. **Time the market** based on fractal formation patterns

## 📊 Dataset Generation

### Available Time Periods

Check what data you have available:
```bash
python main.py --list-periods
```

Expected output:
```
Available Periods (3):
  ✓ 2017_01_01-2020_12_31: 2017-01-01 to 2020-12-31 (29,451 rows)
  ✓ 2021_01_01-2024_12_31: 2021-01-01 to 2024-12-31 (35,027 rows)  
  ✓ 2025_01_01-2025_06_30: 2025-01-01 to 2025-06-30 (4,321 rows)
```

### Creating ML Datasets

#### Basic Usage
```bash
python create_ml_dataset.py \
    --candles "datasets/ml_dataset_[PERIOD].csv" \
    --levels "datasets/levels_dataset_[PERIOD].csv" \
    --output "features/ml_dataset_[NAME].csv" \
    --horizon "[HORIZON]"
```

#### Prediction Horizons

Choose based on your trading strategy:

| Horizon | Candles | Use Case | Target Distribution |
|---------|---------|----------|-------------------|
| `hour` | 1 | Scalping, intraday | 75% no fractal, 12.5% each type |
| `day` | 24 | Day trading, swing starts | ~50% bullish, 50% bearish |
| `week` | 168 | Swing trading | Longer trend changes |
| `month` | 720 | Position trading | Major trend reversals |

#### Examples

**Hourly Predictions (Balanced Classes)**
```bash
python create_ml_dataset.py \
    --candles "datasets/ml_dataset_2025_01_01-2025_06_30.csv" \
    --levels "datasets/levels_dataset_2025_01_01-2025_06_30.csv" \
    --output "features/bitcoin_hourly_predictions.csv" \
    --horizon "hour"
```

**Daily Predictions (Swing Trading)**
```bash
python create_ml_dataset.py \
    --candles "datasets/ml_dataset_2021_01_01-2024_12_31.csv" \
    --levels "datasets/levels_dataset_2021_01_01-2024_12_31.csv" \
    --output "features/bitcoin_daily_predictions.csv" \
    --horizon "day"
```

**Testing with Limited Data**
```bash
python create_ml_dataset.py \
    --candles "datasets/ml_dataset_2025_01_01-2025_06_30.csv" \
    --levels "datasets/levels_dataset_2025_01_01-2025_06_30.csv" \
    --output "features/bitcoin_test.csv" \
    --horizon "day" \
    --max-samples 1000
```

### Dataset Output Analysis

The script provides detailed output:
```
============================================================
DATASET SUMMARY
============================================================
Total Samples: 4,295
Total Features: 75

Target Distribution:
  No Fractal (0): 0 (0.0%)
  Bullish Fractal (1): 2,124 (49.5%)
  Bearish Fractal (2): 2,171 (50.5%)

Feature Quality:
  Numeric Features: 75
  Features with Nulls: 0
  Constant Features: 12

Data Quality:
  Has Timestamp: True
  Time Range: 2025-01-01 02:00:00 to 2025-06-29 00:00:00
  Missing Values: 0.0%
```

## 🔧 Feature Engineering

### Feature Categories Overview

The system generates **75 interpretable features** organized into logical groups:

#### 1. Level Proximity (4 features)
```
nearest_support_distance_pct     # % distance to closest support level
nearest_resistance_distance_pct  # % distance to closest resistance level  
nearest_support_strength         # Weighted strength of nearest support
nearest_resistance_strength      # Weighted strength of nearest resistance
```

#### 2. Confluence Zone Analysis (36 features)
```
levels_in_zone_0_5pct            # Total levels within 0.5% of price
daily_levels_in_zone_0_5pct      # Daily timeframe levels in zone
HTF_level_in_zone_0_5pct         # Higher timeframe levels in zone
confluence_strength_zone_0_5pct   # Weighted confluence score
# ... repeated for 1.0%, 1.5%, 2.0% zones
```

#### 3. Volume Analysis (5 features)
```
volume_ma_20_ratio               # Current volume vs 20-period average
volume_spike                     # Binary: 1 if volume > 2x average
volume_percentile_20             # Volume percentile in recent history
volume_trend_5                   # Volume trend over last 5 periods
```

#### 4. Price Action (11 features)
```
body_ratio                       # Candle body size / total range
upper_wick_ratio                 # Upper wick / total range
is_bullish                       # 1 if bullish candle
price_change_pct                 # % change from previous close
price_vs_ma20_pct                # % above/below 20-period MA
volatility_14d                   # 14-day volatility measure
```

#### 5. Temporal Features (8 features)
```
hour_of_day                      # 0-23 hour
session_asian                    # 1 if Asian trading session
hours_since_last_bullish_fractal # Time since last bullish fractal
day_normalized                   # Day of week normalized 0-1
```

### Level Strength Calculation

The system calculates level strength using:

**Timeframe Weights**
- Monthly: 3.0x
- Weekly: 2.0x  
- Daily: 1.0x
- Hourly: 0.5x

**Level Type Weights**
- HTF levels: 2.5x
- Volume Profile POC: 2.0x
- Fibonacci 0.618: 1.8x
- Fibonacci 0.5: 1.5x
- Fractals: 1.2x

**Age Decay**
- Levels decay by 5% per month
- Recently created levels have higher weight

**Touch Penalty**
- Each touch reduces strength by 15%
- Levels become invalid after 4 touches

## 🎯 Target Variable Design

### Multi-Class Classification

Instead of binary classification, we use **three classes**:

```python
0: no_fractal      # No swing point formation
1: bullish_fractal # Swing low (potential upward reversal)
2: bearish_fractal # Swing high (potential downward reversal)
```

### Prediction Logic

For each candle at time `T`, we predict fractal formation in the **next N candles**:

```python
def create_target(current_candle, lookforward_candles):
    future_window = candles[T+1:T+1+lookforward_candles]
    
    if future_window.has_bullish_fractal():
        return 1  # bullish_fractal
    elif future_window.has_bearish_fractal():
        return 2  # bearish_fractal
    else:
        return 0  # no_fractal
```

### Horizon Selection Strategy

| Trading Style | Recommended Horizon | Rationale |
|---------------|-------------------|-----------|
| **Scalping** | `hour` | Quick entries, high frequency |
| **Day Trading** | `day` | Daily swing points, overnight holds |
| **Swing Trading** | `week` | Multi-day positions, trend changes |
| **Position Trading** | `month` | Long-term trend reversals |

### Class Balance Considerations

**Hourly Predictions**: More balanced classes
- 75% no fractal (most hours don't form swings)
- 12.5% bullish fractal
- 12.5% bearish fractal

**Daily Predictions**: Fractal-focused
- ~0% no fractal (daily timeframe typically shows swings)
- ~50% bullish fractal
- ~50% bearish fractal

## 🤖 DataRobot Integration

### Dataset Upload

1. **File Format**: Upload the generated CSV file
2. **Target Selection**: Choose `fractal_direction` as target variable
3. **Project Type**: Select "Multi-class Classification"
4. **Advanced Options**: Enable time-aware validation

### Recommended Settings

#### Project Configuration
```
Target: fractal_direction
Project Type: Multi-class Classification
Metric: LogLoss or AUC (multiclass)
Validation: Time-aware partitioning (80% train, 20% test)
```

#### Feature Engineering
```
Feature Engineering: Advanced
Automated Feature Engineering: Enabled
Feature Selection: Automatic
```

#### Model Selection
```
Autopilot Mode: Comprehensive
Time Limit: 24 hours (for full exploration)
Model Families: All (especially tree-based models)
```

### Expected Model Types

**Top Performing Models** (typically):
1. **XGBoost** - Excellent with confluence features
2. **Random Forest** - Good feature importance interpretation
3. **LightGBM** - Fast training, good accuracy
4. **Neural Networks** - May capture complex level interactions

**Feature Importance** (expected top features):
1. `nearest_support_distance_pct`
2. `nearest_resistance_distance_pct`
3. `confluence_strength_zone_1_0pct`
4. `hours_since_last_bullish_fractal`
5. `levels_in_zone_0_5pct`

## 📈 Model Training Best Practices

### Data Splitting Strategy

**Time-Aware Splits** (Critical):
```python
# Chronological splits to prevent look-ahead bias
train_data = data['2021-01-01':'2023-12-31']  # 75%
validation_data = data['2024-01-01':'2024-06-30']  # 15%  
test_data = data['2024-07-01':'2024-12-31']  # 10%
```

**Never use random splits** - this causes data leakage in time series!

### Model Selection Criteria

1. **Primary Metric**: LogLoss or AUC (multiclass)
2. **Secondary Metrics**: 
   - Precision/Recall for each class
   - Confusion matrix analysis
   - Feature importance stability

3. **Business Metrics**:
   - True Positive Rate for fractals (catch reversals)
   - False Positive Rate (avoid false signals)
   - Prediction timing accuracy

### Feature Engineering Tips

**Most Important Features**:
- Level proximity and strength
- Confluence zone analysis  
- Volume spike detection
- Fractal timing patterns

**Less Important Features**:
- Raw OHLC prices (model should focus on relationships)
- Absolute timestamps (use temporal patterns instead)
- Individual level counts (confluence scores better)

### Hyperparameter Guidelines

**Tree-Based Models**:
- Max depth: 6-12 (prevent overfitting)
- Learning rate: 0.01-0.1
- Early stopping: Enabled
- Feature sampling: 0.8-1.0

**Neural Networks**:
- Hidden layers: 2-4
- Neurons per layer: 50-200
- Dropout: 0.2-0.5
- Activation: ReLU or LeakyReLU

## ✅ Evaluation and Validation

### Key Metrics to Monitor

#### Classification Metrics
```
Overall Accuracy: Target >65%
Macro F1-Score: Target >0.6
Per-class Precision/Recall:
- Bullish Fractal: Precision >70%, Recall >60%
- Bearish Fractal: Precision >70%, Recall >60%
- No Fractal: Precision >80%, Recall >70%
```

#### Business Metrics
```
Signal Quality:
- Fractal Detection Rate: >60% of actual fractals caught
- False Signal Rate: <30% of predictions are false
- Timing Accuracy: Predicted fractals occur within predicted window
```

### Validation Approach

1. **Time-Series Cross-Validation**
   - Walk-forward validation with expanding window
   - Minimum 1-year training, 3-month validation periods

2. **Out-of-Sample Testing**
   - Reserve most recent 6 months for final validation
   - Never use this data during training or hyperparameter tuning

3. **Feature Stability**
   - Monitor feature importance across different time periods
   - Ensure top features remain consistent

### Model Interpretation

#### Feature Importance Analysis
```python
# Expected top features and their interpretation
top_features = {
    'nearest_support_distance_pct': 'Closer to support = higher fractal probability',
    'confluence_strength_zone_1_0pct': 'More levels = stronger signal',
    'volume_spike': 'High volume confirms fractal formation',
    'hours_since_last_fractal': 'Fractal timing patterns',
    'session_european': 'European session has different fractal characteristics'
}
```

#### Prediction Confidence
- High confidence: Confluence zones + volume spikes + proper timing
- Medium confidence: Some confluence + normal volume
- Low confidence: Isolated signals + low volume

## 🔧 Troubleshooting

### Common Issues

#### 1. Imbalanced Classes
**Problem**: Hourly predictions show 75% "no fractal"
**Solution**: 
- Use class weights in model training
- Consider daily/weekly horizons for more balanced data
- Focus on precision/recall for fractal classes

#### 2. Poor Fractal Detection
**Problem**: Model accuracy is good but misses actual fractals
**Solution**:
- Increase recall for fractal classes (adjust threshold)
- Add more temporal features
- Check if data includes sufficient fractal examples

#### 3. Feature Importance Issues
**Problem**: Unexpected features rank highest
**Solution**:
- Verify feature engineering logic
- Check for data leakage (future information)
- Ensure proper time-aware validation

#### 4. Overfitting
**Problem**: Great training accuracy, poor validation
**Solution**:
- Reduce model complexity
- Add regularization
- Ensure proper time-series splits
- Remove correlated features

### Performance Optimization

#### Dataset Size Management
```bash
# Start small for faster iteration
python create_ml_dataset.py --max-samples 1000 --horizon "hour"

# Scale up when ready
python create_ml_dataset.py --horizon "day"  # Full dataset
```

#### Feature Selection
```python
# Remove constant features (automatically flagged)
# Focus on confluence and proximity features first
# Add temporal features gradually
```

### Data Quality Checks

```bash
# Verify data integrity
python main.py --check-data

# Validate target distribution
head -1 features/your_dataset.csv  # Check column names
tail -10 features/your_dataset.csv  # Check for missing values
```

## 🎯 Success Metrics

### Model Performance Targets

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Overall Accuracy | 60% | 70% | 80% |
| Bullish Fractal Precision | 65% | 75% | 85% |
| Bearish Fractal Precision | 65% | 75% | 85% |
| Fractal Recall (combined) | 55% | 65% | 75% |

### Business Impact

**Successful Model Characteristics**:
- Catches 70%+ of significant price reversals
- Generates <25% false signals
- Provides actionable timing (within prediction window)
- Works consistently across different market conditions

---

## 📚 Additional Resources

- **`project_documentation.md`**: Technical architecture details
- **`new_ml_features.md`**: Feature engineering implementation
- **`CLAUDE.md`**: Development environment setup
- **DataRobot Documentation**: Platform-specific guides

---

*This guide provides a complete workflow for Bitcoin fractal prediction using machine learning. Start with small datasets, validate your approach, then scale to production.*