#!/usr/bin/env python
"""AutoResearch Mode D — Fractal Prediction with Level-Context Features.

Discovers whether ML can predict fractal formation using level-context features
(confluence, distance to naked levels, TF weights) combined with technical features.

Mode B showed F1=0.10 with technical features alone. This mode adds level-context
features that the confluence scalper proved have real edge.

Usage:
    python scripts/autoresearch/fractal_predictor.py --experiments 200 --timeframe 1h
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
# Timeframe weights (higher timeframe = more significant level)
# ---------------------------------------------------------------------------
TF_WEIGHTS = {'1h': 0.25, '4h': 0.5, 'daily': 1.0, 'weekly': 2.0, 'monthly': 3.0}

# Level types that act as support vs resistance
SUPPORT_TYPES = {
    'Fractal_support', 'PrevSession_Low', 'PrevSession_25',
    'PrevSession_VP_VAL', 'VP_VAL',
}
RESISTANCE_TYPES = {
    'Fractal_resistance', 'PrevSession_High', 'PrevSession_75',
    'PrevSession_VP_VAH', 'VP_VAH',
}
# Neutral (both support and resistance depending on price position)
NEUTRAL_TYPES = {
    'HTF_level', 'Fib_CC', 'Fib_0.25', 'Fib_0.50', 'Fib_0.75',
    'PrevSession_EQ', 'PrevSession_VWAP', 'PrevSession_VP_POC', 'VP_POC',
}


# ---------------------------------------------------------------------------
# Technical feature generators (from Mode B feature_discovery.py)
# ---------------------------------------------------------------------------

def feat_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def feat_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(0)


def feat_atr_change(df, period=14):
    atr = feat_atr(df, period)
    return (atr / atr.shift(period).replace(0, 1) - 1).fillna(0)


def feat_body_ratio(df):
    total = df['high'] - df['low']
    body = (df['close'] - df['open']).abs()
    return (body / total.replace(0, 1)).fillna(0)


def feat_upper_wick_ratio(df):
    total = df['high'] - df['low']
    upper = df['high'] - df[['open', 'close']].max(axis=1)
    return (upper / total.replace(0, 1)).fillna(0)


def feat_lower_wick_ratio(df):
    total = df['high'] - df['low']
    lower = df[['open', 'close']].min(axis=1) - df['low']
    return (lower / total.replace(0, 1)).fillna(0)


def feat_volume_ratio(df, period=20):
    avg_vol = df['volume'].rolling(period).mean()
    return (df['volume'] / avg_vol.replace(0, 1)).fillna(1)


def feat_momentum(df, period=10):
    return (df['close'] / df['close'].shift(period).replace(0, 1) - 1).fillna(0)


def feat_distance_from_high(df, period=20):
    high = df['high'].rolling(period).max()
    return ((high - df['close']) / high.replace(0, 1)).fillna(0)


def feat_distance_from_low(df, period=20):
    low = df['low'].rolling(period).min()
    return ((df['close'] - low) / df['close'].replace(0, 1)).fillna(0)


def feat_rsi_divergence(df, period=14):
    rsi = feat_rsi(df, period)
    price_high = df['close'].rolling(period).max()
    rsi_high = rsi.rolling(period).max()
    at_high = (df['close'] >= price_high * 0.998).astype(float)
    rsi_below = (rsi < rsi_high * 0.95).astype(float)
    return at_high * rsi_below


def feat_consecutive_direction(df):
    direction = (df['close'] > df['open']).astype(int)
    groups = (direction != direction.shift()).cumsum()
    return direction.groupby(groups).cumcount() + 1


def feat_range_expansion(df, period=5):
    cur_range = df['high'] - df['low']
    avg_range = cur_range.rolling(period * 4).mean()
    return (cur_range / avg_range.replace(0, 1)).fillna(1)


# Technical feature catalog
TECH_FEATURES = {
    'rsi_14': {'fn': feat_rsi, 'params': {'period': 14}, 'group': 'momentum'},
    'rsi_7': {'fn': feat_rsi, 'params': {'period': 7}, 'group': 'momentum'},
    'rsi_divergence': {'fn': feat_rsi_divergence, 'params': {'period': 14}, 'group': 'momentum'},
    'volume_ratio_20': {'fn': feat_volume_ratio, 'params': {'period': 20}, 'group': 'volume'},
    'volume_ratio_10': {'fn': feat_volume_ratio, 'params': {'period': 10}, 'group': 'volume'},
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

# Level-context feature names (computed separately, not from catalog)
LEVEL_FEATURES = [
    'conf_support_count', 'conf_resistance_count',
    'conf_support_types', 'conf_resistance_types',
    'conf_support_tf_weight', 'conf_resistance_tf_weight',
    'nearest_support_dist', 'nearest_resistance_dist',
    'nearest_support_tf', 'nearest_resistance_tf',
    'naked_support_total', 'naked_resistance_total',
    'has_htf_support', 'has_htf_resistance',
    'candles_since_bullish', 'candles_since_bearish',
]

ALL_FEATURES = list(TECH_FEATURES.keys()) + LEVEL_FEATURES

# Price action features (NOT retail indicators — these are candle shape = pure PA)
PA_FEATURES = ['body_ratio', 'upper_wick', 'lower_wick', 'consecutive_dir',
               'dist_from_high_20', 'dist_from_low_20', 'range_expansion']

# Default config — levels + price action only (NO retail indicators like RSI/momentum/volume)
DEFAULT_CONFIG = {
    'features': [
        # Price action (candle shape — pure PA, not indicators)
        'body_ratio', 'upper_wick', 'lower_wick',
        'dist_from_high_20', 'dist_from_low_20',
        # Level context features (the coach's edge)
        'conf_support_count', 'conf_resistance_count',
        'conf_support_types', 'conf_resistance_types',
        'conf_support_tf_weight', 'conf_resistance_tf_weight',
        'nearest_support_dist', 'nearest_resistance_dist',
        'nearest_support_tf', 'nearest_resistance_tf',
        'naked_support_total', 'naked_resistance_total',
        'has_htf_support', 'has_htf_resistance',
        # Fractal rhythm
        'candles_since_bullish', 'candles_since_bearish',
    ],
    'zone_width': 0.01,        # 1% zone for level features
    'model': 'rf',             # 'rf', 'xgb', 'lgbm'
    'n_trees': 200,
    'max_depth': 10,
    'learning_rate': 0.1,      # for xgb/lgbm
    'test_split': 0.3,
}


# ---------------------------------------------------------------------------
# Data loading (adapted from confluence_scalper.py)
# ---------------------------------------------------------------------------

def load_data(session, timeframe='1h'):
    """Load candles and levels from DB, return cached data dict."""
    from app.services.level_trade_backtest_db import load_candles_db, load_levels_db

    print(f"Loading {timeframe} candles from DB...", flush=True)
    t0 = time.time()
    candles = load_candles_db(session, timeframe=timeframe)
    print(f"  Loaded {len(candles):,} candles ({time.time()-t0:.1f}s)", flush=True)

    # Convert to DataFrame
    df = pd.DataFrame(candles)
    df = df.sort_values('open_time').reset_index(drop=True)

    print("Loading levels from DB...", flush=True)
    t0 = time.time()
    levels = load_levels_db(session)
    print(f"  Loaded {len(levels):,} levels ({time.time()-t0:.1f}s)", flush=True)

    # Level arrays for vectorized computation
    l_prices = np.array(levels['price_level'].values, dtype=np.float64)
    l_types = levels['level_type'].values.astype(str)
    l_timeframes = levels['timeframe'].values.astype(str)
    l_tf_weights = np.array([TF_WEIGHTS.get(tf, 0) for tf in l_timeframes], dtype=np.float64)

    # Classify levels as support/resistance/neutral
    l_is_support = np.array([t in SUPPORT_TYPES or t in NEUTRAL_TYPES for t in l_types])
    l_is_resistance = np.array([t in RESISTANCE_TYPES or t in NEUTRAL_TYPES for t in l_types])
    l_is_htf = np.array([tf in ('weekly', 'monthly') for tf in l_timeframes])

    # Level type IDs for unique counting
    unique_types = sorted(set(l_types))
    type_to_id = {t: i for i, t in enumerate(unique_types)}
    l_type_ids = np.array([type_to_id[t] for t in l_types], dtype=np.int32)

    # Validity timestamps
    far_future = pd.Timestamp('2099-01-01', tz='UTC')
    structural_types = {'Fractal_support', 'Fractal_resistance', 'HTF_level',
                        'Fib_CC', 'Fib_0.25', 'Fib_0.50', 'Fib_0.75'}

    l_created = []
    l_validity_end = []
    for _, row in levels.iterrows():
        created = row.get('created_at')
        if pd.isna(created):
            created = pd.Timestamp('2017-01-01', tz='UTC')
        elif not hasattr(created, 'tz') or created.tz is None:
            created = pd.Timestamp(created, tz='UTC')
        l_created.append(created.timestamp())

        is_structural = row['level_type'] in structural_types
        if is_structural:
            end = row.get('first_touched_at', None)
        else:
            end = row.get('superseded_at', None) or row.get('invalidated_at', None)

        if pd.isna(end) or end is None:
            l_validity_end.append(far_future.timestamp())
        else:
            if not hasattr(end, 'tz') or end.tz is None:
                end = pd.Timestamp(end, tz='UTC')
            l_validity_end.append(end.timestamp())

    l_created = np.array(l_created, dtype=np.float64)
    l_validity_end = np.array(l_validity_end, dtype=np.float64)

    # Sorted prices for searchsorted
    sorted_idx = np.argsort(l_prices)
    sorted_prices = l_prices[sorted_idx]

    # Candle timestamps
    c_times = []
    for t in df['open_time']:
        if hasattr(t, 'timestamp'):
            c_times.append(t.timestamp())
        else:
            c_times.append(float(t) / 1000 if t > 1e12 else float(t))
    c_times = np.array(c_times, dtype=np.float64)

    data = {
        'df': df,
        'c_times': c_times,
        'c_closes': df['close'].values.astype(np.float64),
        'c_highs': df['high'].values.astype(np.float64),
        'c_lows': df['low'].values.astype(np.float64),
        'l_prices': l_prices,
        'l_types': l_types,
        'l_type_ids': l_type_ids,
        'l_tf_weights': l_tf_weights,
        'l_is_support': l_is_support,
        'l_is_resistance': l_is_resistance,
        'l_is_htf': l_is_htf,
        'l_created': l_created,
        'l_validity_end': l_validity_end,
        'sorted_idx': sorted_idx,
        'sorted_prices': sorted_prices,
        'n_levels': len(l_prices),
    }

    return data


# ---------------------------------------------------------------------------
# Compute level-context features (vectorized with chunking)
# ---------------------------------------------------------------------------

def compute_level_features(data, zone_width=0.01):
    """Compute level-context features for every candle.

    For each candle's close price, finds naked levels in a zone and computes
    confluence counts, distances, TF weights, etc.

    Returns DataFrame with level feature columns aligned to candle index.
    """
    n_candles = len(data['c_closes'])
    c_closes = data['c_closes']
    c_times = data['c_times']

    l_prices = data['l_prices']
    l_is_support = data['l_is_support']
    l_is_resistance = data['l_is_resistance']
    l_is_htf = data['l_is_htf']
    l_tf_weights = data['l_tf_weights']
    l_type_ids = data['l_type_ids']
    l_created = data['l_created']
    l_validity_end = data['l_validity_end']
    sorted_prices = data['sorted_prices']
    sorted_idx = data['sorted_idx']

    # Output arrays
    conf_sup_count = np.zeros(n_candles, dtype=np.float64)
    conf_res_count = np.zeros(n_candles, dtype=np.float64)
    conf_sup_types = np.zeros(n_candles, dtype=np.float64)
    conf_res_types = np.zeros(n_candles, dtype=np.float64)
    conf_sup_tf = np.zeros(n_candles, dtype=np.float64)
    conf_res_tf = np.zeros(n_candles, dtype=np.float64)
    near_sup_dist = np.full(n_candles, 0.1, dtype=np.float64)  # default 10%
    near_res_dist = np.full(n_candles, 0.1, dtype=np.float64)
    near_sup_tf = np.zeros(n_candles, dtype=np.float64)
    near_res_tf = np.zeros(n_candles, dtype=np.float64)
    naked_sup_total = np.zeros(n_candles, dtype=np.float64)
    naked_res_total = np.zeros(n_candles, dtype=np.float64)
    has_htf_sup = np.zeros(n_candles, dtype=np.float64)
    has_htf_res = np.zeros(n_candles, dtype=np.float64)

    # Process in weekly chunks for efficiency
    chunk_size = 168  # 1 week of 1h candles
    htf_zone = 0.02   # 2% zone for HTF check

    for chunk_start in range(0, n_candles, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_candles)
        chunk_time_start = c_times[chunk_start]
        chunk_time_end = c_times[chunk_end - 1]

        # Active levels for this chunk: created before chunk AND still valid
        active_mask = (l_created <= chunk_time_start) & (l_validity_end > chunk_time_start)
        active_idx = np.where(active_mask)[0]

        if len(active_idx) == 0:
            continue

        active_prices = l_prices[active_idx]
        active_support = l_is_support[active_idx]
        active_resistance = l_is_resistance[active_idx]
        active_htf = l_is_htf[active_idx]
        active_tf_weights = l_tf_weights[active_idx]
        active_type_ids = l_type_ids[active_idx]

        # Global naked counts for chunk
        n_sup = int(active_support.sum())
        n_res = int(active_resistance.sum())

        for i in range(chunk_start, chunk_end):
            price = c_closes[i]
            if price <= 0:
                continue

            naked_sup_total[i] = n_sup
            naked_res_total[i] = n_res

            # Zone boundaries
            lo = price * (1 - zone_width)
            hi = price * (1 + zone_width)

            # Find levels in zone using searchsorted on active prices
            # (We search in the global sorted array but filter to active)
            left = np.searchsorted(sorted_prices, lo, side='left')
            right = np.searchsorted(sorted_prices, hi, side='right')

            if left < right:
                zone_orig_idx = sorted_idx[left:right]
                # Filter to active only
                zone_active_mask = active_mask[zone_orig_idx]
                zone_levels = zone_orig_idx[zone_active_mask]

                if len(zone_levels) > 0:
                    z_support = l_is_support[zone_levels]
                    z_resistance = l_is_resistance[zone_levels]
                    z_tf = l_tf_weights[zone_levels]
                    z_types = l_type_ids[zone_levels]
                    z_prices = l_prices[zone_levels]
                    z_htf = l_is_htf[zone_levels]

                    # Support levels below price
                    sup_mask = z_support & (z_prices <= price)
                    if sup_mask.any():
                        conf_sup_count[i] = sup_mask.sum()
                        conf_sup_types[i] = len(set(z_types[sup_mask].tolist()))
                        conf_sup_tf[i] = z_tf[sup_mask].sum()
                        # Nearest
                        sup_dists = (price - z_prices[sup_mask]) / price
                        min_idx = sup_dists.argmin()
                        near_sup_dist[i] = sup_dists[min_idx]
                        near_sup_tf[i] = z_tf[sup_mask][min_idx]

                    # Resistance levels above price
                    res_mask = z_resistance & (z_prices >= price)
                    if res_mask.any():
                        conf_res_count[i] = res_mask.sum()
                        conf_res_types[i] = len(set(z_types[res_mask].tolist()))
                        conf_res_tf[i] = z_tf[res_mask].sum()
                        # Nearest
                        res_dists = (z_prices[res_mask] - price) / price
                        min_idx = res_dists.argmin()
                        near_res_dist[i] = res_dists[min_idx]
                        near_res_tf[i] = z_tf[res_mask][min_idx]

            # HTF check (wider zone: ±2%)
            htf_lo = price * (1 - htf_zone)
            htf_hi = price * (1 + htf_zone)
            htf_left = np.searchsorted(sorted_prices, htf_lo, side='left')
            htf_right = np.searchsorted(sorted_prices, htf_hi, side='right')
            if htf_left < htf_right:
                htf_orig = sorted_idx[htf_left:htf_right]
                htf_active = htf_orig[active_mask[htf_orig]]
                if len(htf_active) > 0:
                    htf_flags = l_is_htf[htf_active]
                    htf_prices_arr = l_prices[htf_active]
                    htf_sup = htf_flags & l_is_support[htf_active] & (htf_prices_arr <= price)
                    htf_res = htf_flags & l_is_resistance[htf_active] & (htf_prices_arr >= price)
                    has_htf_sup[i] = 1.0 if htf_sup.any() else 0.0
                    has_htf_res[i] = 1.0 if htf_res.any() else 0.0

    result = pd.DataFrame({
        'conf_support_count': conf_sup_count,
        'conf_resistance_count': conf_res_count,
        'conf_support_types': conf_sup_types,
        'conf_resistance_types': conf_res_types,
        'conf_support_tf_weight': conf_sup_tf,
        'conf_resistance_tf_weight': conf_res_tf,
        'nearest_support_dist': near_sup_dist,
        'nearest_resistance_dist': near_res_dist,
        'nearest_support_tf': near_sup_tf,
        'nearest_resistance_tf': near_res_tf,
        'naked_support_total': naked_sup_total,
        'naked_resistance_total': naked_res_total,
        'has_htf_support': has_htf_sup,
        'has_htf_resistance': has_htf_res,
    })

    return result


# ---------------------------------------------------------------------------
# Fractal detection (target variable)
# ---------------------------------------------------------------------------

def compute_fractals(df):
    """Detect 5-candle fractals. Returns (bullish_series, bearish_series)."""
    lows = df['low'].values
    highs = df['high'].values
    n = len(df)

    bullish = np.zeros(n, dtype=bool)
    bearish = np.zeros(n, dtype=bool)

    for i in range(2, n - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            bullish[i] = True
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            bearish[i] = True

    return pd.Series(bullish, index=df.index), pd.Series(bearish, index=df.index)


def compute_candles_since_fractal(bullish, bearish):
    """Compute candles since last bullish/bearish fractal."""
    n = len(bullish)
    since_bull = np.zeros(n, dtype=np.float64)
    since_bear = np.zeros(n, dtype=np.float64)
    cb, cr = 999, 999
    for i in range(n):
        if bullish.iloc[i]:
            cb = 0
        else:
            cb += 1
        if bearish.iloc[i]:
            cr = 0
        else:
            cr += 1
        since_bull[i] = min(cb, 999)
        since_bear[i] = min(cr, 999)
    return pd.Series(since_bull, index=bullish.index), pd.Series(since_bear, index=bearish.index)


# ---------------------------------------------------------------------------
# Build feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(data, config, _cache=None):
    """Build feature matrix X and target y.

    Uses cache to avoid recomputing level features when only tech features change.
    """
    df = data['df']
    zone_width = config.get('zone_width', 0.01)

    # Compute fractals (always needed for targets + candles_since features)
    if _cache and 'fractals' in _cache:
        bullish, bearish = _cache['fractals']
    else:
        bullish, bearish = compute_fractals(df)
        if _cache is not None:
            _cache['fractals'] = (bullish, bearish)

    # Compute candles since fractal
    if _cache and 'since_fractal' in _cache:
        since_bull, since_bear = _cache['since_fractal']
    else:
        since_bull, since_bear = compute_candles_since_fractal(bullish, bearish)
        if _cache is not None:
            _cache['since_fractal'] = (since_bull, since_bear)

    # Compute level features (cached by zone_width)
    cache_key = f'level_features_{zone_width}'
    if _cache and cache_key in _cache:
        level_feats = _cache[cache_key]
    else:
        print("  Computing level features...", flush=True)
        t0 = time.time()
        level_feats = compute_level_features(data, zone_width=zone_width)
        print(f"  Level features computed ({time.time()-t0:.1f}s)", flush=True)
        if _cache is not None:
            _cache[cache_key] = level_feats

    # Build X with selected features
    feature_names = config['features']
    X = pd.DataFrame(index=df.index)

    for name in feature_names:
        if name in TECH_FEATURES:
            entry = TECH_FEATURES[name]
            X[name] = entry['fn'](df, **entry['params'])
        elif name == 'candles_since_bullish':
            X[name] = since_bull
        elif name == 'candles_since_bearish':
            X[name] = since_bear
        elif name in level_feats.columns:
            X[name] = level_feats[name].values

    # Target: 3-class (0=no_fractal, 1=bullish, 2=bearish)
    # Use shift to avoid leakage: features at N-2 predict target at N
    # (fractal at N needs N-1 and N+1, so at N-2 we don't know yet)
    y = pd.Series(0, index=df.index, dtype=int)
    y[bullish] = 1
    y[bearish] = 2

    # Shift features by 2 (use N-2 features to predict N target)
    X = X.shift(2)

    # Drop NaN rows
    valid = X.notna().all(axis=1) & y.notna()
    # Also drop first/last 2 rows (fractal detection needs neighbors)
    valid.iloc[:4] = False
    valid.iloc[-2:] = False

    orig_indices = np.where(valid.values)[0]  # Original df indices for near-miss
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    return X, y, feature_names, orig_indices


# ---------------------------------------------------------------------------
# ML evaluation
# ---------------------------------------------------------------------------

def evaluate(config, data, _cache=None):
    """Train model with given config, return metrics."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score

    t0 = time.time()

    try:
        X, y, feature_names, orig_indices = build_feature_matrix(data, config, _cache)
    except Exception as e:
        return {'error': str(e), 'f1_macro': 0, 'f1_bullish': 0, 'f1_bearish': 0}

    if len(X) < 500:
        return {'error': 'Too few samples', 'f1_macro': 0, 'f1_bullish': 0, 'f1_bearish': 0}

    # Time-based train/test split
    oos_date = config.get('oos_date', None)  # e.g. '2024-01-01' for out-of-sample
    if oos_date:
        # Split by date: train before oos_date, test after
        df = data['df']
        oos_ts = pd.Timestamp(oos_date, tz='UTC')
        # Find the candle index closest to oos_date
        candle_times = df['open_time']
        if hasattr(candle_times.iloc[0], 'tz') and candle_times.iloc[0].tz is None:
            oos_ts = oos_ts.tz_localize(None)
        oos_candle_idx = (candle_times >= oos_ts).idxmax()
        # Map to position in our filtered arrays
        split_idx = int(np.searchsorted(orig_indices, oos_candle_idx))
        split_idx = max(100, min(split_idx, len(X) - 100))
    else:
        test_split = config.get('test_split', 0.3)
        split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Check minimum positive samples
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    if train_counts.get(1, 0) < 10 or train_counts.get(2, 0) < 10:
        return {'error': 'Too few positive samples in train', 'f1_macro': 0,
                'f1_bullish': 0, 'f1_bearish': 0}

    # Class weights (fractals are rare: ~5% of candles each)
    n_total = len(y_train)
    class_weights = {}
    for c in [0, 1, 2]:
        count = (y_train == c).sum()
        if count > 0:
            class_weights[c] = n_total / (3 * count)

    # Train model
    model_type = config.get('model', 'rf')
    n_trees = config.get('n_trees', 200)
    max_depth = config.get('max_depth', 10)
    lr = config.get('learning_rate', 0.1)

    if model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=5,
            class_weight=class_weights, random_state=42, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        importances = dict(zip(feature_names, clf.feature_importances_))

    elif model_type == 'xgb':
        import xgboost as xgb
        # Compute sample weights
        sample_weights = np.array([class_weights.get(c, 1.0) for c in y_train])
        clf = xgb.XGBClassifier(
            n_estimators=n_trees, max_depth=max_depth, learning_rate=lr,
            objective='multi:softmax', num_class=3,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = clf.predict(X_test)
        importances = dict(zip(feature_names, clf.feature_importances_))

    elif model_type == 'lgbm':
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=n_trees, max_depth=max_depth, learning_rate=lr,
            class_weight=class_weights, random_state=42, n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        importances = dict(zip(feature_names, clf.feature_importances_))

    else:
        return {'error': f'Unknown model: {model_type}', 'f1_macro': 0,
                'f1_bullish': 0, 'f1_bearish': 0}

    # Metrics
    f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    f1_bull = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
    f1_bear = f1_per_class[2] if len(f1_per_class) > 2 else 0.0

    # Per-class precision/recall
    prec = precision_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    rec = recall_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1, 2])

    # Top features
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:7]

    # Class distribution
    test_dist = {int(k): int(v) for k, v in y_test.value_counts().items()}
    pred_dist = {int(k): int(v) for k, v in pd.Series(y_pred).value_counts().items()}

    # --- Near-miss analysis ---
    # For each "false positive" fractal prediction, check if:
    # 1) A fractal formed within ±2 candles (timing near-miss)
    # 2) There was a wick rejection (price action reaction) in next 2 candles
    df = data['df']
    bullish_arr, bearish_arr = _cache['fractals']
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    closes = df['close'].values
    n_candles = len(df)

    # Get test set original indices
    test_orig = orig_indices[split_idx:]

    # Predicted fractal (1 or 2) but actual was 0
    pred_fractal_mask = (y_pred == 1) | (y_pred == 2)
    actual_no_fractal = (y_test.values == 0)
    false_positives = pred_fractal_mask & actual_no_fractal
    fp_indices = test_orig[false_positives]

    # Also get true positives for comparison
    true_positives = pred_fractal_mask & ~actual_no_fractal
    tp_count = int(true_positives.sum())
    fp_count = int(false_positives.sum())
    total_pred_fractal = int(pred_fractal_mask.sum())

    near_miss_fractal = 0  # Fractal formed within ±2 candles
    near_miss_reaction = 0  # Wick rejection in next 2 candles (wick > 50% of range)
    near_miss_moved_1r = 0  # Price moved favorably by at least the SL distance

    for idx in fp_indices:
        if idx + 2 >= n_candles or idx < 2:
            continue

        pred_class = y_pred[np.where(test_orig == idx)[0][0]] if len(np.where(test_orig == idx)[0]) > 0 else 0

        # 1) Check if fractal formed within ±2 candles
        window = range(max(0, idx - 2), min(n_candles, idx + 3))
        fractal_nearby = False
        for j in window:
            if j == idx:
                continue
            if bullish_arr.iloc[j] or bearish_arr.iloc[j]:
                fractal_nearby = True
                break
        if fractal_nearby:
            near_miss_fractal += 1

        # 2) Check for wick rejection in next 2 candles
        for j in [idx, idx + 1, idx + 2]:
            if j >= n_candles:
                break
            candle_range = highs[j] - lows[j]
            if candle_range <= 0:
                continue
            upper_wick = highs[j] - max(opens[j], closes[j])
            lower_wick = min(opens[j], closes[j]) - lows[j]
            max_wick = max(upper_wick, lower_wick)
            if max_wick / candle_range >= 0.5:  # Wick > 50% of range = rejection
                near_miss_reaction += 1
                break

        # 3) Check if price moved 1R (using ATR as proxy for SL)
        # A simple check: did price move 1% in the predicted direction within 2 candles?
        entry_price = closes[idx]
        if pred_class == 1:  # Predicted bullish → price should go up
            max_price = max(highs[idx+1] if idx+1 < n_candles else 0,
                           highs[idx+2] if idx+2 < n_candles else 0)
            if max_price > entry_price * 1.005:  # 0.5% move up
                near_miss_moved_1r += 1
        elif pred_class == 2:  # Predicted bearish → price should go down
            min_price = min(lows[idx+1] if idx+1 < n_candles else 999999,
                           lows[idx+2] if idx+2 < n_candles else 999999)
            if min_price < entry_price * 0.995:  # 0.5% move down
                near_miss_moved_1r += 1

    # Compute near-miss adjusted precision
    # "Useful" predictions = true positives + near-miss fractals + reactions
    useful_strict = tp_count + near_miss_fractal  # Fractal ±2 candles
    useful_practical = tp_count + near_miss_reaction  # Had price reaction
    adjusted_precision_strict = useful_strict / total_pred_fractal if total_pred_fractal > 0 else 0
    adjusted_precision_practical = useful_practical / total_pred_fractal if total_pred_fractal > 0 else 0

    elapsed = time.time() - t0

    return {
        'f1_macro': round(f1_mac, 4),
        'f1_bullish': round(float(f1_bull), 4),
        'f1_bearish': round(float(f1_bear), 4),
        'precision_0': round(float(prec[0]), 4),
        'precision_1': round(float(prec[1]), 4),
        'precision_2': round(float(prec[2]), 4),
        'recall_0': round(float(rec[0]), 4),
        'recall_1': round(float(rec[1]), 4),
        'recall_2': round(float(rec[2]), 4),
        'n_samples': len(X),
        'n_test': len(X_test),
        'test_distribution': test_dist,
        'pred_distribution': pred_dist,
        'top_features': top_features,
        'model': model_type,
        'n_features': len(feature_names),
        'elapsed_sec': round(elapsed, 1),
        # Near-miss analysis
        'total_pred_fractal': total_pred_fractal,
        'true_positives': tp_count,
        'false_positives': fp_count,
        'near_miss_fractal_2bar': near_miss_fractal,
        'near_miss_reaction': near_miss_reaction,
        'near_miss_moved_1r': near_miss_moved_1r,
        'raw_precision': round(tp_count / total_pred_fractal, 4) if total_pred_fractal > 0 else 0,
        'adj_precision_strict': round(adjusted_precision_strict, 4),
        'adj_precision_practical': round(adjusted_precision_practical, 4),
    }


def fitness(metrics):
    """Fitness = F1 macro (primary goal is prediction accuracy)."""
    if metrics.get('error'):
        return 0
    return round(metrics.get('f1_macro', 0), 4)


# ---------------------------------------------------------------------------
# Mutation system
# ---------------------------------------------------------------------------

MUTATIONS = [
    # Feature mutations
    {'name': 'add_feature', 'action': 'add', 'field': 'features', 'pool': ALL_FEATURES, 'max_items': 25},
    {'name': 'remove_feature', 'action': 'remove', 'field': 'features', 'min_items': 4},
    {'name': 'swap_feature', 'action': 'swap', 'field': 'features', 'pool': ALL_FEATURES},
    # Model mutations
    {'name': 'model_type', 'field': 'model', 'options': ['rf', 'xgb', 'lgbm']},
    # Hyperparameter mutations
    {'name': 'n_trees', 'field': 'n_trees', 'range': [50, 500], 'step': 50, 'type': 'int'},
    {'name': 'max_depth', 'field': 'max_depth', 'range': [3, 20], 'step': 1, 'type': 'int'},
    {'name': 'learning_rate', 'field': 'learning_rate', 'range': [0.01, 0.3], 'step': 0.02},
    # Zone width for level features
    {'name': 'zone_width', 'field': 'zone_width', 'range': [0.005, 0.025], 'step': 0.0025},
]


def propose_mutation(config, history=None, allowed_features=None):
    """Propose a random mutation to the config."""
    recent_fails = set()
    if history:
        for h in history[-10:]:
            if not h.get('improved'):
                recent_fails.add(h['mutation']['name'])

    candidates = [m for m in MUTATIONS if m['name'] not in recent_fails]
    if not candidates:
        candidates = MUTATIONS

    mut_def = random.choice(candidates)
    mutation = {'name': mut_def['name'], 'field': mut_def['field']}

    if 'options' in mut_def:
        current = config.get(mut_def['field'])
        options = [o for o in mut_def['options'] if o != current]
        new_val = random.choice(options) if options else current
        mutation['old'] = current
        mutation['new'] = new_val
        mutation['description'] = f"{mut_def['name']}: {current} -> {new_val}"

    elif 'range' in mut_def:
        current = config.get(mut_def['field'])
        lo, hi = mut_def['range']
        step = mut_def.get('step', 1)
        delta = random.choice([-2, -1, 1, 2]) * step
        new_val = max(lo, min(hi, current + delta))
        if mut_def.get('type') == 'int':
            new_val = int(new_val)
        else:
            new_val = round(new_val, 4)
        mutation['old'] = current
        mutation['new'] = new_val
        mutation['description'] = f"{mut_def['name']}: {current} -> {new_val}"

    elif mut_def.get('action') == 'add':
        current = config.get(mut_def['field'], [])
        pool = [f for f in mut_def['pool'] if f not in current]
        if allowed_features is not None:
            pool = [f for f in pool if f in allowed_features]
        max_items = mut_def.get('max_items', 30)
        if pool and len(current) < max_items:
            to_add = random.choice(pool)
            mutation['new'] = to_add
            mutation['description'] = f"add {to_add}"
        else:
            mutation['description'] = "no features to add"
            mutation['skip'] = True

    elif mut_def.get('action') == 'remove':
        current = config.get(mut_def['field'], [])
        min_items = mut_def.get('min_items', 3)
        if len(current) > min_items:
            to_remove = random.choice(current)
            mutation['new'] = to_remove
            mutation['description'] = f"remove {to_remove}"
        else:
            mutation['description'] = f"can't remove (min {min_items})"
            mutation['skip'] = True

    elif mut_def.get('action') == 'swap':
        current = config.get(mut_def['field'], [])
        pool = [f for f in mut_def['pool'] if f not in current]
        if allowed_features is not None:
            pool = [f for f in pool if f in allowed_features]
        if pool and len(current) > 0:
            to_remove = random.choice(current)
            to_add = random.choice(pool)
            mutation['remove'] = to_remove
            mutation['add'] = to_add
            mutation['description'] = f"swap {to_remove} -> {to_add}"
        else:
            mutation['description'] = "no features to swap"
            mutation['skip'] = True

    return mutation


def apply_mutation(config, mutation):
    """Apply mutation to config, return new config."""
    new_config = copy.deepcopy(config)
    field = mutation['field']

    if mutation['name'] in ('add_feature',):
        new_config[field] = list(config[field]) + [mutation['new']]
    elif mutation['name'] in ('remove_feature',):
        new_config[field] = [f for f in config[field] if f != mutation['new']]
    elif mutation['name'] in ('swap_feature',):
        features = [f for f in config[field] if f != mutation['remove']]
        features.append(mutation['add'])
        new_config[field] = features
    elif 'new' in mutation:
        new_config[field] = mutation['new']

    return new_config


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_fractal_predictor(n_experiments=50, timeframe='1h', levels_only=True, tag=None,
                          oos_date=None):
    """Main AutoResearch loop for fractal prediction."""
    from app import create_app
    from app.extensions import db

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    parts = [timeframe]
    if levels_only:
        parts.append('levels')
    if tag:
        parts.append(tag)
    results_file = os.path.join(results_dir, f'fractal_predictor_{"_".join(parts)}.jsonl')

    # Define allowed feature pool
    if levels_only:
        allowed_features = set(LEVEL_FEATURES + PA_FEATURES)
    else:
        allowed_features = None  # All features allowed

    app = create_app()
    with app.app_context():
        config = copy.deepcopy(DEFAULT_CONFIG)

        # If not levels_only, allow retail features back in default config
        if not levels_only:
            config['features'] = config['features'] + ['volume_ratio_20', 'momentum_10', 'atr_14']

        # Out-of-sample split
        if oos_date:
            config['oos_date'] = oos_date

        labels = [f"[{timeframe}]"]
        if levels_only:
            labels.append("[LEVELS+PA ONLY]")
        if oos_date:
            labels.append(f"[OOS: train<{oos_date}, test>={oos_date}]")
        label = ' '.join(labels)
        print("=" * 70, flush=True)
        print(f"AUTORESEARCH MODE D — Fractal Predictor {label}", flush=True)
        print("=" * 70, flush=True)
        print(f"Running {n_experiments} experiments...\n", flush=True)

        # Load data
        data = load_data(db.session, timeframe=timeframe)
        _cache = {}

        # Baseline
        print("Evaluating baseline...", flush=True)
        baseline = evaluate(config, data, _cache)
        print(f"Baseline:", flush=True)
        for k, v in sorted(baseline.items()):
            if k not in ('error', 'top_features', 'test_distribution', 'pred_distribution'):
                print(f"  {k}: {v}", flush=True)
        if 'top_features' in baseline:
            print(f"  top_features:", flush=True)
            for fname, imp in baseline['top_features']:
                print(f"    {fname}: {imp:.4f}", flush=True)
        print(flush=True)

        history = []
        best = baseline.copy()
        best_config = copy.deepcopy(config)
        improvements = 0

        for i in range(n_experiments):
            mutation = propose_mutation(config, history, allowed_features=allowed_features)

            if mutation.get('skip'):
                print(f"[{i+1}/{n_experiments}] SKIP: {mutation['description']}", flush=True)
                continue

            new_config = apply_mutation(config, mutation)

            # If zone_width changed, level features need recomputation
            metrics = evaluate(new_config, data, _cache)

            improved = fitness(metrics) > fitness(best)

            experiment = {
                'id': i,
                'timestamp': datetime.now().isoformat(),
                'mutation': mutation,
                'metrics': metrics,
                'improved': improved,
                'best_fitness': fitness(best),
            }
            history.append(experiment)

            with open(results_file, 'a') as f:
                f.write(json.dumps(experiment, default=str) + '\n')

            if improved:
                improvements += 1
                best = metrics
                config = copy.deepcopy(new_config)
                best_config = copy.deepcopy(new_config)
                print(f"[{i+1}/{n_experiments}] ** IMPROVED ** {mutation['description']} "
                      f"-> F1={fitness(metrics)} bull={metrics['f1_bullish']} "
                      f"bear={metrics['f1_bearish']} ({metrics['elapsed_sec']}s)", flush=True)
            else:
                print(f"[{i+1}/{n_experiments}] no gain: {mutation['description']} "
                      f"-> F1={fitness(metrics)} ({metrics.get('elapsed_sec', '?')}s)", flush=True)

        # Summary
        print(flush=True)
        print("=" * 70, flush=True)
        print("FRACTAL PREDICTOR SUMMARY", flush=True)
        print("=" * 70, flush=True)
        print(f"Experiments: {n_experiments}", flush=True)
        print(f"Improvements: {improvements}", flush=True)
        print(f"Baseline F1: {fitness(baseline)}", flush=True)
        print(f"Best F1: {fitness(best)}", flush=True)
        print(flush=True)
        print("Best metrics:", flush=True)
        for k, v in sorted(best.items()):
            if k not in ('error', 'top_features', 'test_distribution', 'pred_distribution'):
                print(f"  {k}: {v}", flush=True)
        if 'top_features' in best:
            print(f"  top_features:", flush=True)
            for fname, imp in best['top_features']:
                print(f"    {fname}: {imp:.4f}", flush=True)
        print(flush=True)
        print(f"Best config:", flush=True)
        print(f"  model: {best_config['model']}", flush=True)
        print(f"  features ({len(best_config['features'])}): {best_config['features']}", flush=True)
        print(f"  zone_width: {best_config['zone_width']}", flush=True)
        print(f"  n_trees: {best_config['n_trees']}", flush=True)
        print(f"  max_depth: {best_config['max_depth']}", flush=True)
        print(f"  learning_rate: {best_config['learning_rate']}", flush=True)
        print(flush=True)
        print(f"Results logged to: {results_file}", flush=True)

        return best, best_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoResearch Mode D — Fractal Predictor')
    parser.add_argument('--experiments', type=int, default=50, help='Number of experiments')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Candle timeframe (1h, 4h)')
    parser.add_argument('--all-features', action='store_true',
                        help='Allow retail indicators (RSI, momentum, volume). Default: levels+PA only')
    parser.add_argument('--tag', type=str, default=None,
                        help='Custom tag appended to results filename')
    parser.add_argument('--oos', type=str, default=None,
                        help='Out-of-sample date (e.g. 2024-01-01). Train before, test after.')
    args = parser.parse_args()

    run_fractal_predictor(n_experiments=args.experiments, timeframe=args.timeframe,
                          levels_only=not args.all_features, tag=args.tag,
                          oos_date=args.oos)
