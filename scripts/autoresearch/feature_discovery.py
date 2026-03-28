#!/usr/bin/env python
"""AutoResearch Mode B — Feature Discovery for Fractal Prediction.

Discovers which features best predict when a fractal (swing high/low) will form.
Each experiment adds/removes/modifies features, retrains RF, measures improvement.

Usage:
    python scripts/autoresearch/feature_discovery.py --experiments 100
"""
import sys
import os
import time
import json
import copy
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Feature generators — each returns a Series aligned to the candle DataFrame
# ---------------------------------------------------------------------------

def feat_rsi(df, period=14):
    """RSI (Relative Strength Index)."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def feat_rsi_divergence(df, period=14):
    """Price making new high but RSI not — divergence signal."""
    rsi = feat_rsi(df, period)
    price_high = df['close'].rolling(period).max()
    rsi_high = rsi.rolling(period).max()
    # 1 if price at high but RSI below its high (bearish divergence)
    at_high = (df['close'] >= price_high * 0.998).astype(float)
    rsi_below = (rsi < rsi_high * 0.95).astype(float)
    return at_high * rsi_below


def feat_volume_ratio(df, period=20):
    """Volume relative to recent average."""
    avg_vol = df['volume'].rolling(period).mean()
    return (df['volume'] / avg_vol.replace(0, 1)).fillna(1)


def feat_volume_trend(df, period=10):
    """Volume trend — rising or falling over N bars."""
    return df['volume'].rolling(period).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else 0,
        raw=False
    ).fillna(0)


def feat_atr(df, period=14):
    """Average True Range."""
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(0)


def feat_atr_change(df, period=14):
    """ATR rate of change — volatility expanding or contracting."""
    atr = feat_atr(df, period)
    return (atr / atr.shift(period).replace(0, 1) - 1).fillna(0)


def feat_body_ratio(df):
    """Candle body as % of total range."""
    total = df['high'] - df['low']
    body = (df['close'] - df['open']).abs()
    return (body / total.replace(0, 1)).fillna(0)


def feat_upper_wick_ratio(df):
    """Upper wick as % of total range."""
    total = df['high'] - df['low']
    upper = df['high'] - df[['open', 'close']].max(axis=1)
    return (upper / total.replace(0, 1)).fillna(0)


def feat_lower_wick_ratio(df):
    """Lower wick as % of total range."""
    total = df['high'] - df['low']
    lower = df[['open', 'close']].min(axis=1) - df['low']
    return (lower / total.replace(0, 1)).fillna(0)


def feat_consecutive_direction(df):
    """Count of consecutive same-direction candles."""
    direction = (df['close'] > df['open']).astype(int)
    groups = (direction != direction.shift()).cumsum()
    return direction.groupby(groups).cumcount() + 1


def feat_distance_from_high(df, period=20):
    """How far price is from recent high (% drawdown)."""
    high = df['high'].rolling(period).max()
    return ((high - df['close']) / high.replace(0, 1)).fillna(0)


def feat_distance_from_low(df, period=20):
    """How far price is from recent low (% rally)."""
    low = df['low'].rolling(period).min()
    return ((df['close'] - low) / df['close'].replace(0, 1)).fillna(0)


def feat_candles_since_fractal(df, fractal_col):
    """Candles since last fractal of given type."""
    result = pd.Series(0, index=df.index, dtype=float)
    count = 0
    for i in range(len(df)):
        if df[fractal_col].iloc[i]:
            count = 0
        else:
            count += 1
        result.iloc[i] = count
    return result


def feat_momentum(df, period=10):
    """Price momentum (rate of change)."""
    return (df['close'] / df['close'].shift(period).replace(0, 1) - 1).fillna(0)


def feat_range_expansion(df, period=5):
    """Current range vs average range — detecting expansion."""
    cur_range = df['high'] - df['low']
    avg_range = cur_range.rolling(period * 4).mean()
    return (cur_range / avg_range.replace(0, 1)).fillna(1)


# Feature catalog with metadata
FEATURE_CATALOG = {
    'rsi_7': {'fn': feat_rsi, 'params': {'period': 7}, 'group': 'momentum'},
    'rsi_14': {'fn': feat_rsi, 'params': {'period': 14}, 'group': 'momentum'},
    'rsi_21': {'fn': feat_rsi, 'params': {'period': 21}, 'group': 'momentum'},
    'rsi_divergence': {'fn': feat_rsi_divergence, 'params': {'period': 14}, 'group': 'momentum'},
    'volume_ratio_10': {'fn': feat_volume_ratio, 'params': {'period': 10}, 'group': 'volume'},
    'volume_ratio_20': {'fn': feat_volume_ratio, 'params': {'period': 20}, 'group': 'volume'},
    'volume_trend': {'fn': feat_volume_trend, 'params': {'period': 10}, 'group': 'volume'},
    'atr_14': {'fn': feat_atr, 'params': {'period': 14}, 'group': 'volatility'},
    'atr_change': {'fn': feat_atr_change, 'params': {'period': 14}, 'group': 'volatility'},
    'body_ratio': {'fn': feat_body_ratio, 'params': {}, 'group': 'candle'},
    'upper_wick': {'fn': feat_upper_wick_ratio, 'params': {}, 'group': 'candle'},
    'lower_wick': {'fn': feat_lower_wick_ratio, 'params': {}, 'group': 'candle'},
    'consecutive_dir': {'fn': feat_consecutive_direction, 'params': {}, 'group': 'candle'},
    'dist_from_high_20': {'fn': feat_distance_from_high, 'params': {'period': 20}, 'group': 'structure'},
    'dist_from_high_50': {'fn': feat_distance_from_high, 'params': {'period': 50}, 'group': 'structure'},
    'dist_from_low_20': {'fn': feat_distance_from_low, 'params': {'period': 20}, 'group': 'structure'},
    'dist_from_low_50': {'fn': feat_distance_from_low, 'params': {'period': 50}, 'group': 'structure'},
    'momentum_5': {'fn': feat_momentum, 'params': {'period': 5}, 'group': 'momentum'},
    'momentum_10': {'fn': feat_momentum, 'params': {'period': 10}, 'group': 'momentum'},
    'momentum_20': {'fn': feat_momentum, 'params': {'period': 20}, 'group': 'momentum'},
    'range_expansion': {'fn': feat_range_expansion, 'params': {'period': 5}, 'group': 'volatility'},
}


def compute_features(df, feature_names):
    """Compute selected features and return DataFrame."""
    result = pd.DataFrame(index=df.index)
    for name in feature_names:
        if name not in FEATURE_CATALOG:
            continue
        entry = FEATURE_CATALOG[name]
        result[name] = entry['fn'](df, **entry['params'])
    return result


def compute_targets(df):
    """Compute fractal targets (N-2 pattern to avoid leakage)."""
    bullish = pd.Series(False, index=df.index)
    bearish = pd.Series(False, index=df.index)

    lows = df['low'].values
    highs = df['high'].values

    for i in range(2, len(df) - 2):
        # Bullish fractal (swing low): low[i] < neighbors
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            bullish.iloc[i] = True
        # Bearish fractal (swing high): high[i] > neighbors
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            bearish.iloc[i] = True

    return bullish, bearish


def evaluate_features(candles_df, feature_names, target='bullish',
                      n_trees=100, test_split=0.3):
    """Train RF with given features and return metrics."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    # Compute features
    features = compute_features(candles_df, feature_names)

    # Compute targets
    bullish, bearish = compute_targets(candles_df)

    # Use N-2 features to predict N-1 target (avoid leakage)
    X = features.shift(2)  # Features from candle N-2
    if target == 'bullish':
        y = bullish.shift(1).fillna(False).astype(int)  # Target at N-1
    else:
        y = bearish.shift(1).fillna(False).astype(int)

    # Drop NaN rows
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    if len(X) < 100:
        return {'error': 'Too few samples', 'accuracy': 0, 'f1': 0}

    # Train/test split (time-based, not random)
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if y_train.sum() < 10 or y_test.sum() < 5:
        return {'error': 'Too few positive samples', 'accuracy': 0, 'f1': 0}

    # Train
    clf = RandomForestClassifier(
        n_estimators=n_trees, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Feature importances
    importances = dict(zip(feature_names, clf.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]

    return {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'total_samples': len(X),
        'positive_rate': round(y.mean() * 100, 1),
        'features_used': len(feature_names),
        'top_features': top_features,
    }


def run_feature_discovery(n_experiments=100, exec_tf='4h'):
    """Main feature discovery loop."""
    from app import create_app
    from app.extensions import db
    from sqlalchemy import text

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'feature_discovery.jsonl')

    app = create_app()
    with app.app_context():
        # Load candle data
        print("Loading candles...", flush=True)
        rows = db.session.execute(text("""
            SELECT open_time, open, high, low, close, volume
            FROM candles WHERE timeframe = :tf
            ORDER BY open_time
        """), {'tf': exec_tf}).fetchall()

        candles_df = pd.DataFrame(rows, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
        candles_df['open_time'] = pd.to_datetime(candles_df['open_time'])
        print(f"Loaded {len(candles_df)} candles ({exec_tf})", flush=True)

        # Baseline: start with basic features
        all_features = list(FEATURE_CATALOG.keys())
        current_features = ['rsi_14', 'volume_ratio_20', 'atr_14', 'body_ratio',
                           'momentum_10', 'dist_from_high_20', 'dist_from_low_20']

        print("=" * 70, flush=True)
        print("AUTORESEARCH MODE B — Feature Discovery for Fractal Prediction", flush=True)
        print("=" * 70, flush=True)
        print(f"Feature catalog: {len(all_features)} available features", flush=True)
        print(f"Starting features: {current_features}", flush=True)
        print(flush=True)

        # Baseline
        print("Evaluating baseline (bullish)...", flush=True)
        t0 = time.time()
        baseline_bull = evaluate_features(candles_df, current_features, target='bullish')
        baseline_bear = evaluate_features(candles_df, current_features, target='bearish')
        elapsed = time.time() - t0
        print(f"Baseline ({elapsed:.1f}s):", flush=True)
        print(f"  Bullish: acc={baseline_bull['accuracy']}, f1={baseline_bull['f1']}, "
              f"prec={baseline_bull['precision']}, rec={baseline_bull['recall']}", flush=True)
        print(f"  Bearish: acc={baseline_bear['accuracy']}, f1={baseline_bear['f1']}, "
              f"prec={baseline_bear['precision']}, rec={baseline_bear['recall']}", flush=True)
        print(flush=True)

        # Combined metric: average of bullish + bearish F1
        best_score = (baseline_bull['f1'] + baseline_bear['f1']) / 2
        best_features = list(current_features)
        improvements = 0
        history = []

        for i in range(n_experiments):
            # Propose mutation
            mutation_type = random.choice(['add', 'remove', 'swap', 'add_group'])

            new_features = list(current_features)
            description = ''

            if mutation_type == 'add' and len(new_features) < 15:
                available = [f for f in all_features if f not in new_features]
                if available:
                    feat = random.choice(available)
                    new_features.append(feat)
                    description = f"add {feat}"

            elif mutation_type == 'remove' and len(new_features) > 3:
                feat = random.choice(new_features)
                new_features.remove(feat)
                description = f"remove {feat}"

            elif mutation_type == 'swap' and len(new_features) > 0:
                to_remove = random.choice(new_features)
                available = [f for f in all_features if f not in new_features]
                if available:
                    to_add = random.choice(available)
                    new_features.remove(to_remove)
                    new_features.append(to_add)
                    description = f"swap {to_remove} -> {to_add}"

            elif mutation_type == 'add_group':
                groups = set(FEATURE_CATALOG[f]['group'] for f in all_features)
                group = random.choice(list(groups))
                group_feats = [f for f in all_features if FEATURE_CATALOG[f]['group'] == group and f not in new_features]
                if group_feats:
                    feat = random.choice(group_feats)
                    new_features.append(feat)
                    description = f"add {feat} (group: {group})"

            if not description:
                print(f"[{i+1}/{n_experiments}] SKIP", flush=True)
                continue

            t0 = time.time()
            bull = evaluate_features(candles_df, new_features, target='bullish')
            bear = evaluate_features(candles_df, new_features, target='bearish')
            elapsed = time.time() - t0

            if bull.get('error') or bear.get('error'):
                print(f"[{i+1}/{n_experiments}] ERROR: {bull.get('error', bear.get('error'))}", flush=True)
                continue

            score = (bull['f1'] + bear['f1']) / 2
            improved = score > best_score

            experiment = {
                'id': i,
                'timestamp': datetime.now().isoformat(),
                'mutation': description,
                'features': new_features,
                'bullish': bull,
                'bearish': bear,
                'combined_f1': round(score, 4),
                'improved': improved,
                'best_score': best_score,
                'elapsed_sec': round(elapsed, 1),
            }
            history.append(experiment)

            with open(results_file, 'a') as f:
                f.write(json.dumps(experiment, default=str) + '\n')

            if improved:
                improvements += 1
                best_score = score
                best_features = list(new_features)
                current_features = list(new_features)
                print(f"[{i+1}/{n_experiments}] ** IMPROVED ** {description} "
                      f"-> F1={score:.4f} (bull={bull['f1']}, bear={bear['f1']}) "
                      f"({elapsed:.1f}s)", flush=True)
            else:
                print(f"[{i+1}/{n_experiments}] no gain: {description} "
                      f"-> F1={score:.4f} ({elapsed:.1f}s)", flush=True)

        # Summary
        print(flush=True)
        print("=" * 70, flush=True)
        print("FEATURE DISCOVERY SUMMARY", flush=True)
        print("=" * 70, flush=True)
        print(f"Experiments: {n_experiments}", flush=True)
        print(f"Improvements: {improvements}", flush=True)
        print(f"Baseline combined F1: {(baseline_bull['f1'] + baseline_bear['f1']) / 2:.4f}", flush=True)
        print(f"Best combined F1: {best_score:.4f}", flush=True)
        print(f"Best features ({len(best_features)}): {best_features}", flush=True)

        # Final evaluation with best features
        print(flush=True)
        print("Final evaluation with best features:", flush=True)
        final_bull = evaluate_features(candles_df, best_features, target='bullish')
        final_bear = evaluate_features(candles_df, best_features, target='bearish')
        print(f"  Bullish: acc={final_bull['accuracy']}, f1={final_bull['f1']}, "
              f"prec={final_bull['precision']}, rec={final_bull['recall']}", flush=True)
        print(f"  Bearish: acc={final_bear['accuracy']}, f1={final_bear['f1']}, "
              f"prec={final_bear['precision']}, rec={final_bear['recall']}", flush=True)
        if final_bull.get('top_features'):
            print(f"  Top bullish features: {final_bull['top_features']}", flush=True)
        if final_bear.get('top_features'):
            print(f"  Top bearish features: {final_bear['top_features']}", flush=True)

        print(flush=True)
        print(f"Results logged to: {results_file}", flush=True)

        return best_features, best_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Discovery for Fractal Prediction')
    parser.add_argument('--experiments', type=int, default=100)
    parser.add_argument('--tf', type=str, default='4h')
    args = parser.parse_args()

    run_feature_discovery(n_experiments=args.experiments, exec_tf=args.tf)
