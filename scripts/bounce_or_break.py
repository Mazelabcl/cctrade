"""Bounce-or-Break model: predict if price bounces or breaks through a level.

Instead of predicting fractals on every candle, this model only evaluates
moments when price TOUCHES a D/W/M level — and predicts bounce vs break.

Uses existing backtest trades as the dataset (each trade = a touch event).
"""
import sys
import os
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def build_dataset():
    """Build bounce/break dataset from backtest trades."""
    from app import create_app
    from app.extensions import db
    from sqlalchemy import text

    app = create_app()
    with app.app_context():
        # Load all trades with backtest metadata
        sql = text("""
            SELECT
                t.entry_time, t.entry_price, t.direction, t.exit_reason,
                t.pnl_pct, t.candles_held, t.entry_volatility, t.volume_ratio,
                t.distance_to_level, t.zone_confluence, t.level_price,
                t.stop_loss, t.take_profit,
                b.level_type, b.level_source_timeframe, b.trade_execution_timeframe,
                b.strategy_name
            FROM individual_level_trades t
            JOIN individual_level_backtests b ON t.backtest_id = b.id
            WHERE b.status = 'completed'
            AND t.exit_reason IN ('TP_HIT', 'SL_HIT')
        """)
        rows = db.session.execute(sql).fetchall()
        cols = [
            'entry_time', 'entry_price', 'direction', 'exit_reason',
            'pnl_pct', 'candles_held', 'entry_volatility', 'volume_ratio',
            'distance_to_level', 'zone_confluence', 'level_price',
            'stop_loss', 'take_profit',
            'level_type', 'level_source_timeframe', 'trade_execution_timeframe',
            'strategy_name',
        ]
        df = pd.DataFrame(rows, columns=cols)
        logger.info("Loaded %d trades", len(df))

        # Parse entry_time
        df['entry_time'] = pd.to_datetime(df['entry_time'])

        # Target: 1 = bounce (TP_HIT), 0 = break (SL_HIT)
        df['bounce'] = (df['exit_reason'] == 'TP_HIT').astype(int)

        # === FEATURES ===

        # 1. Trade-level features (already computed during backtest)
        df['entry_volatility'] = df['entry_volatility'].fillna(0)
        df['volume_ratio'] = df['volume_ratio'].fillna(1)
        df['distance_to_level'] = df['distance_to_level'].fillna(0)
        df['zone_confluence'] = df['zone_confluence'].fillna(1)

        # 2. Direction encoding (1 = LONG, 0 = SHORT)
        df['is_long'] = (df['direction'] == 'LONG').astype(int)

        # 3. Risk/reward from SL/TP
        df['risk'] = abs(df['entry_price'] - df['stop_loss']) / df['entry_price']
        df['reward'] = abs(df['take_profit'] - df['entry_price']) / df['entry_price']
        df['rr_ratio'] = df['reward'] / df['risk'].replace(0, np.nan)
        df['rr_ratio'] = df['rr_ratio'].fillna(1)

        # 4. Level type encoding
        level_type_map = {
            'Fractal_support': 0, 'Fractal_resistance': 1,
            'HTF_level': 2,
            'Fib_CC': 3, 'Fib_0.25': 4, 'Fib_0.50': 5, 'Fib_0.75': 6,
            'VP_POC': 7, 'VP_VAH': 8, 'VP_VAL': 9,
        }
        df['level_type_code'] = df['level_type'].map(level_type_map).fillna(-1).astype(int)

        # 5. Level source TF encoding (higher = stronger)
        tf_map = {'daily': 1, 'weekly': 2, 'monthly': 3}
        df['level_tf_strength'] = df['level_source_timeframe'].map(tf_map).fillna(0).astype(int)

        # 6. Exec TF encoding
        exec_tf_map = {'1h': 1, '4h': 4, '6h': 6, '8h': 8, '12h': 12}
        df['exec_tf_hours'] = df['trade_execution_timeframe'].map(exec_tf_map).fillna(1).astype(int)

        # 7. Extract RR from strategy name (wick_rr_1.0, wick_rr_2.0, wick_rr_3.0)
        df['strategy_rr'] = df['strategy_name'].str.extract(r'(\d+\.?\d*)').astype(float).fillna(1)

        # 8. Price relative to level (above/below)
        df['price_vs_level'] = (df['entry_price'] - df['level_price']) / df['level_price']

        # 9. Time features
        df['hour'] = df['entry_time'].dt.hour
        df['day_of_week'] = df['entry_time'].dt.dayofweek
        df['utc_block'] = df['hour'] // 4

        # Feature columns for model
        FEATURE_COLS = [
            'entry_volatility', 'volume_ratio', 'distance_to_level',
            'zone_confluence', 'is_long', 'risk', 'rr_ratio',
            'level_type_code', 'level_tf_strength', 'exec_tf_hours',
            'strategy_rr', 'price_vs_level', 'utc_block', 'day_of_week',
        ]

        return df, FEATURE_COLS


def train_and_evaluate(df, feature_cols, exec_tf=None):
    """Train bounce/break model and evaluate."""

    # Filter by exec TF if specified
    if exec_tf:
        df = df[df['trade_execution_timeframe'] == exec_tf].copy()

    if len(df) < 100:
        logger.warning("Too few trades for %s: %d", exec_tf, len(df))
        return None

    # Sort by time for proper train/test split
    df = df.sort_values('entry_time').reset_index(drop=True)

    X = df[feature_cols].fillna(0)
    y = df['bounce']

    # Time-series split: 70% train, 15% val, 15% test
    n = len(X)
    train_end = int(n * 0.7)
    test_start = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    y_proba = rf.predict_proba(X_test_s)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    prec_bounce = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_bounce = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

    # Feature importance
    imps = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1])

    result = {
        'exec_tf': exec_tf or 'all',
        'total_trades': len(df),
        'test_trades': len(y_test),
        'bounce_rate': float(y.mean()),
        'accuracy': round(acc, 4),
        'f1_macro': round(f1, 4),
        'precision_bounce': round(prec_bounce, 4),
        'recall_bounce': round(rec_bounce, 4),
        'confusion_matrix': cm.tolist(),
        'feature_importance': [(f, round(i, 4)) for f, i in imps],
    }

    return result


def main():
    logger.info("Building bounce-or-break dataset...")
    df, feature_cols = build_dataset()

    logger.info("Dataset: %d trades, bounce rate=%.1f%%",
                len(df), df['bounce'].mean() * 100)

    results = []

    # Train on ALL data
    logger.info("\n=== ALL TFs combined ===")
    r = train_and_evaluate(df, feature_cols)
    if r:
        results.append(r)
        print_result(r)

    # Train per exec TF
    for tf in ['1h', '4h']:
        logger.info(f"\n=== Exec TF: {tf} ===")
        r = train_and_evaluate(df, feature_cols, exec_tf=tf)
        if r:
            results.append(r)
            print_result(r)

    # Train per RR ratio
    for rr in [1.0, 2.0, 3.0]:
        logger.info(f"\n=== RR {rr} only ===")
        df_rr = df[df['strategy_rr'] == rr]
        r = train_and_evaluate(df_rr, feature_cols)
        if r:
            r['exec_tf'] = f'all_rr{rr}'
            results.append(r)
            print_result(r)

    # Save results
    out_path = 'scripts/bounce_or_break_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", out_path)


def print_result(r):
    cm = r['confusion_matrix']
    print(f"\n  TF={r['exec_tf']} | {r['total_trades']} trades | bounce_rate={r['bounce_rate']:.1%}")
    print(f"  Acc={r['accuracy']:.3f} | F1={r['f1_macro']:.3f}")
    print(f"  Bounce precision={r['precision_bounce']:.1%} (when it says bounce, correct)")
    print(f"  Bounce recall={r['recall_bounce']:.1%} (of real bounces, detected)")
    print(f"  Confusion: break_correct={cm[0][0]} break_wrong={cm[0][1]} | "
          f"bounce_wrong={cm[1][0]} bounce_correct={cm[1][1]}")
    print(f"  Top features:")
    for fname, imp in r['feature_importance'][:7]:
        bar = '#' * int(imp * 100)
        print(f"    {fname:25s} {imp:.4f} {bar}")


if __name__ == '__main__':
    main()
