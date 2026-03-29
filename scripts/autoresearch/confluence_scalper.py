#!/usr/bin/env python
"""AutoResearch Mode C — Confluence Scalper Discovery.

Discovers optimal confluence-based scalping parameters by mutating entry/exit
configuration and evaluating against 4.4M 1-minute BTCUSDT candles.

Entry logic: price touches a zone where multiple level types overlap (confluence).
Exit logic: multiple strategies (fixed RR, swing trail, ATR trail, etc.)
Fitness: profit_factor * sqrt(total_trades) — rewards both profitability & frequency.

Usage:
    python scripts/autoresearch/confluence_scalper.py --experiments 5
    python scripts/autoresearch/confluence_scalper.py --experiments 200
"""
import sys
import os
import time
import json
import copy
import random
import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

ALL_LEVEL_TYPES = [
    'Fractal_support', 'Fractal_resistance', 'HTF_level',
    'Fib_CC', 'Fib_0.25', 'Fib_0.50', 'Fib_0.75',
    'PrevSession_High', 'PrevSession_Low', 'PrevSession_EQ',
    'PrevSession_25', 'PrevSession_75', 'PrevSession_VWAP',
    'PrevSession_VP_POC', 'PrevSession_VP_VAH', 'PrevSession_VP_VAL',
    'VP_POC', 'VP_VAH', 'VP_VAL',
]

DEFAULT_CONFIG = {
    'score_threshold': 3,
    'zone_width': 0.01,
    'touch_tolerance': 0.003,
    'naked_only': True,
    'level_types': [
        'Fractal_support', 'Fractal_resistance', 'HTF_level',
        'Fib_CC', 'PrevSession_High', 'PrevSession_Low',
        'PrevSession_VWAP', 'PrevSession_VP_POC',
    ],
    'exit': {
        'strategy': 'swing_trail',
        'rr_ratio': 1.5,
        'swing_lookback': 5,
        'atr_multiplier': 2.0,
        'breakeven_at_rr': 1.0,
        'partial_pct': 0.5,
        'partial_rr': 2.0,
        'timeout_candles': 20,
        'sl_buffer_pct': 0.001,
    },
}

# Timeframe weights for confluence scoring
TF_WEIGHTS = {'1h': 0, '4h': 0, 'daily': 1, 'weekly': 2, 'monthly': 3}


# ---------------------------------------------------------------------------
# Indicators (copied from evaluate.py for self-containment)
# ---------------------------------------------------------------------------

def compute_atr(highs, lows, closes, period=14):
    n = len(highs)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def find_swing_lows(lows, lookback=5):
    n = len(lows)
    swing = np.copy(lows)
    for i in range(lookback, n):
        swing[i] = np.min(lows[max(0, i-lookback):i+1])
    return swing


def find_swing_highs(highs, lookback=5):
    n = len(highs)
    swing = np.copy(highs)
    for i in range(lookback, n):
        swing[i] = np.max(highs[max(0, i-lookback):i+1])
    return swing


# ---------------------------------------------------------------------------
# Data loading + caching
# ---------------------------------------------------------------------------

def load_and_cache_data(session, _cache={}):
    """Load candles + levels once, cache in memory. Returns data dict."""
    if 'loaded' in _cache:
        return _cache

    print("Loading 1m candles from DB...", flush=True)
    t0 = time.time()
    from app.services.level_trade_backtest_db import load_candles_db, load_levels_db

    candles = load_candles_db(session, timeframe='1m')
    print(f"  Loaded {len(candles):,} candles ({time.time()-t0:.1f}s)", flush=True)

    if candles.empty:
        raise RuntimeError("No 1m candles in DB")

    _cache['c_times'] = candles['open_time'].values.astype('datetime64[ns]')
    _cache['c_opens'] = candles['open'].values.astype(np.float64)
    _cache['c_highs'] = candles['high'].values.astype(np.float64)
    _cache['c_lows'] = candles['low'].values.astype(np.float64)
    _cache['c_closes'] = candles['close'].values.astype(np.float64)
    _cache['c_vols'] = candles['volume'].values.astype(np.float64)
    _cache['n_candles'] = len(candles)

    # Precompute ATR
    print("  Computing ATR(14)...", flush=True)
    _cache['atr'] = compute_atr(_cache['c_highs'], _cache['c_lows'], _cache['c_closes'], 14)

    # Swing cache (populated on demand per lookback)
    _cache['swing_cache'] = {}

    # Load levels
    print("Loading levels from DB...", flush=True)
    t1 = time.time()
    levels = load_levels_db(session)
    print(f"  Loaded {len(levels):,} levels ({time.time()-t1:.1f}s)", flush=True)

    if levels.empty:
        raise RuntimeError("No levels in DB")

    # Encode level types to int IDs
    unique_types = sorted(levels['level_type'].unique())
    type_to_id = {t: i for i, t in enumerate(unique_types)}
    id_to_type = {i: t for t, i in type_to_id.items()}
    _cache['type_to_id'] = type_to_id
    _cache['id_to_type'] = id_to_type

    _cache['l_prices'] = levels['price_level'].values.astype(np.float64)
    _cache['l_type_ids'] = np.array([type_to_id[t] for t in levels['level_type']], dtype=np.int32)
    _cache['l_types_str'] = levels['level_type'].values
    _cache['l_timeframes'] = levels['timeframe'].values

    # Timeframe weights
    _cache['l_tf_weights'] = np.array([TF_WEIGHTS.get(tf, 0) for tf in levels['timeframe']], dtype=np.float64)

    # Timestamps for lifecycle
    far_future = np.datetime64('2099-01-01')
    _cache['l_created'] = levels['created_at'].values.astype('datetime64[ns]')

    # Structural vs mobile
    is_mobile = (
        levels['level_type'].str.startswith('PrevSession') |
        levels['level_type'].str.startswith('VP_')
    ).fillna(False).values
    _cache['l_is_structural'] = ~is_mobile

    # Validity end: structural uses first_touched_at, mobile uses superseded_at
    validity_end = np.full(len(levels), far_future, dtype='datetime64[ns]')

    if 'first_touched_at' in levels.columns:
        ft = levels['first_touched_at'].values.astype('datetime64[ns]')
        has_ft = ~pd.isna(levels['first_touched_at']).values
        structural_with_ft = _cache['l_is_structural'] & has_ft
        validity_end[structural_with_ft] = ft[structural_with_ft]

    if 'superseded_at' in levels.columns:
        sa = levels['superseded_at'].values.astype('datetime64[ns]')
        has_sa = ~pd.isna(levels['superseded_at']).values
        mobile_with_sa = is_mobile & has_sa
        validity_end[mobile_with_sa] = sa[mobile_with_sa]

    _cache['l_validity_end'] = validity_end

    # Sorted level prices for searchsorted (confluence detection)
    sorted_idx = np.argsort(_cache['l_prices'])
    _cache['l_sorted_prices'] = _cache['l_prices'][sorted_idx]
    _cache['l_sorted_idx'] = sorted_idx

    _cache['loaded'] = True
    print(f"Data cached ({time.time()-t0:.1f}s total)\n", flush=True)
    return _cache


def _get_swings(data, lookback):
    """Get or compute swing highs/lows for a given lookback."""
    key = lookback
    if key not in data['swing_cache']:
        data['swing_cache'][key] = (
            find_swing_lows(data['c_lows'], lookback),
            find_swing_highs(data['c_highs'], lookback),
        )
    return data['swing_cache'][key]


# ---------------------------------------------------------------------------
# Touch detection (vectorized per-level)
# ---------------------------------------------------------------------------

def find_touch_entries(data, config):
    """Find FIRST candle that touches each active level.

    With naked_only=True (default), each level generates at most 1 entry.
    Uses per-level numpy masking with monthly price pre-check for speed.

    Returns (N, 3) int64 array: [candle_idx, level_idx, direction]
    direction: 1=LONG, -1=SHORT.
    """
    type_to_id = data['type_to_id']
    allowed_ids = set(type_to_id[lt] for lt in config['level_types']
                      if lt in type_to_id)
    if not allowed_ids:
        return np.array([], dtype=np.int64).reshape(0, 3)

    tol = config['touch_tolerance']
    naked_only = config['naked_only']

    c_times = data['c_times']
    c_highs = data['c_highs']
    c_lows = data['c_lows']
    c_closes = data['c_closes']
    n_candles = data['n_candles']

    l_prices = data['l_prices']
    l_type_ids = data['l_type_ids']
    l_created = data['l_created']
    l_validity_end = data['l_validity_end']
    l_is_structural = data['l_is_structural']

    # Pre-compute min/max price per month for fast relevance check
    MONTH = 43200  # ~30 days of 1m candles
    n_months = (n_candles + MONTH - 1) // MONTH
    month_min = np.zeros(n_months)
    month_max = np.zeros(n_months)
    for m in range(n_months):
        ms = m * MONTH
        me = min(ms + MONTH, n_candles)
        month_min[m] = c_lows[ms:me].min()
        month_max[m] = c_highs[ms:me].max()

    # Pre-filter levels by type with numpy
    allowed_arr = np.array(list(allowed_ids), dtype=np.int32)
    type_mask = np.isin(l_type_ids, allowed_arr)
    fl_idx = np.where(type_mask)[0]

    if len(fl_idx) == 0:
        return np.array([], dtype=np.int64).reshape(0, 3)

    # Adjust validity for non-naked structural levels
    if not naked_only:
        fl_validity = np.where(
            l_is_structural[fl_idx],
            np.datetime64('2099-01-01'),
            l_validity_end[fl_idx],
        )
    else:
        fl_validity = l_validity_end[fl_idx]

    fl_prices = l_prices[fl_idx]
    fl_created = l_created[fl_idx]
    n_filtered = len(fl_idx)

    # Chunked processing with first-touch tracking
    # Once a level is touched, it's removed from future chunks
    CHUNK = 10080  # 1 week
    consumed = np.zeros(n_filtered, dtype=bool)  # True = already touched
    entries = []

    n_chunks = (n_candles + CHUNK - 1) // CHUNK
    for ci in range(n_chunks):
        cs = ci * CHUNK
        ce = min(cs + CHUNK, n_candles)
        chunk_start_t = c_times[cs]
        chunk_end_t = c_times[ce - 1]

        # Price range of this chunk
        chunk_lo = c_lows[cs:ce].min()
        chunk_hi = c_highs[cs:ce].max()
        margin = chunk_hi * tol * 2

        # Active = not consumed, correct type, time-valid, price-relevant
        active_mask = (
            ~consumed &
            (fl_created <= chunk_end_t) &
            (fl_validity > chunk_start_t) &
            (fl_prices >= chunk_lo - margin) &
            (fl_prices <= chunk_hi + margin)
        )
        active_local = np.where(active_mask)[0]
        if len(active_local) == 0:
            continue

        n_active = len(active_local)
        al_prices = fl_prices[active_local]
        al_created = fl_created[active_local]
        al_validity = fl_validity[active_local]

        # Candle arrays
        chunk_lows = c_lows[cs:ce]
        chunk_highs = c_highs[cs:ce]
        chunk_closes = c_closes[cs:ce]
        chunk_times = c_times[cs:ce]
        n_chunk = ce - cs

        # 2D broadcasting: (n_chunk, n_active)
        prices_r = al_prices[np.newaxis, :]
        tol_up = prices_r * (1 + tol)
        tol_dn = prices_r * (1 - tol)

        lows_c = chunk_lows[:, np.newaxis]
        highs_c = chunk_highs[:, np.newaxis]
        closes_c = chunk_closes[:, np.newaxis]

        long_touch = (lows_c <= tol_up) & (closes_c > prices_r)
        short_touch = (highs_c >= tol_dn) & (closes_c < prices_r)

        # Time validity per (candle, level)
        times_c = chunk_times[:, np.newaxis]
        created_r = al_created[np.newaxis, :]
        validity_r = al_validity[np.newaxis, :]
        time_ok = (created_r <= times_c) & (validity_r > times_c)

        long_valid = long_touch & time_ok
        short_valid = short_touch & time_ok

        any_touch = long_valid | short_valid

        # For each level, find FIRST candle that touches it in this chunk
        for j in range(n_active):
            col = any_touch[:, j]
            if not col.any():
                continue

            first_row = int(np.argmax(col))  # argmax returns first True
            candle_idx = cs + first_row

            if long_valid[first_row, j]:
                entries.append((candle_idx, int(fl_idx[active_local[j]]), 1))
            else:
                entries.append((candle_idx, int(fl_idx[active_local[j]]), -1))

            consumed[active_local[j]] = True

    if not entries:
        return np.array([], dtype=np.int64).reshape(0, 3)

    result = np.array(entries, dtype=np.int64)
    result = result[result[:, 0].argsort()]
    return result


# ---------------------------------------------------------------------------
# Confluence scoring
# ---------------------------------------------------------------------------

def score_and_deduplicate(entries, data, config):
    """Score entries by confluence (nearby levels touched at similar time),
    filter by threshold, and apply cooldown.

    Confluence = how many OTHER levels are near the same price and touched
    within a small time window. Uses searchsorted on sorted level prices.

    Returns filtered (entries, scores) arrays.
    """
    if len(entries) == 0:
        return entries, np.array([], dtype=np.float64)

    threshold = config['score_threshold']
    timeout = config['exit'].get('timeout_candles', 20)
    zone_width = config['zone_width']

    l_prices = data['l_prices']
    l_type_ids = data['l_type_ids']
    l_tf_weights = data['l_tf_weights']
    sorted_prices = data['l_sorted_prices']
    sorted_idx = data['l_sorted_idx']

    n = len(entries)
    scores = np.zeros(n, dtype=np.float64)

    for i in range(n):
        li = int(entries[i, 1])
        lp = l_prices[li]

        # Find all levels within zone using searchsorted
        lo = lp * (1 - zone_width)
        hi = lp * (1 + zone_width)
        left = np.searchsorted(sorted_prices, lo, side='left')
        right = np.searchsorted(sorted_prices, hi, side='right')

        if left >= right:
            scores[i] = 1.0
            continue

        # Count unique level types in zone (regardless of active status —
        # if the level exists near this price, it's confluence)
        zone_orig = sorted_idx[left:right]
        unique_types = len(set(l_type_ids[zone_orig].tolist()))

        # TF bonus
        tf_bonus = float(l_tf_weights[zone_orig].max())

        scores[i] = unique_types + tf_bonus * 0.5

    # Filter by threshold
    mask = scores >= threshold
    entries = entries[mask]
    scores = scores[mask]

    if len(entries) == 0:
        return entries, scores

    # Cooldown: keep first entry, skip next `timeout` candles
    keep = []
    last_candle = -timeout - 1
    for i in range(len(entries)):
        ci = int(entries[i, 0])
        if ci - last_candle > timeout:
            keep.append(i)
            last_candle = ci

    if not keep:
        return np.array([], dtype=np.int64).reshape(0, 3), np.array([], dtype=np.float64)

    keep = np.array(keep)
    return entries[keep], scores[keep]


# ---------------------------------------------------------------------------
# Exit simulation (copied from evaluate.py + fixed_rr)
# ---------------------------------------------------------------------------

def _simulate_exit(strategy, direction, entry, sl, risk, idx,
                   highs, lows, closes, atr, sw_lows, sw_highs,
                   timeout, rr_ratio, be_rr, atr_mult, partial_pct, partial_rr):
    """Simulate a single trade. Returns pnl in R units."""
    n = len(highs)
    end = min(idx + timeout, n)
    d = 1 if direction == 1 else -1  # 1=LONG, -1=SHORT

    if strategy == 'fixed_rr':
        tp = entry + d * risk * rr_ratio
        for j in range(idx + 1, end):
            if direction == 1:  # LONG
                if lows[j] <= sl:
                    return -1.0
                if highs[j] >= tp:
                    return rr_ratio
            else:  # SHORT
                if highs[j] >= sl:
                    return -1.0
                if lows[j] <= tp:
                    return rr_ratio

    elif strategy == 'swing_trail':
        sl_cur = sl
        for j in range(idx + 1, end):
            if direction == 1:
                if lows[j] <= sl_cur:
                    return round((sl_cur - entry) / risk, 2)
                sl_cur = max(sl_cur, sw_lows[j])
            else:
                if highs[j] >= sl_cur:
                    return round((entry - sl_cur) / risk, 2)
                sl_cur = min(sl_cur, sw_highs[j])

    elif strategy == 'breakeven_trail':
        sl_cur = sl
        reached = False
        for j in range(idx + 1, end):
            if direction == 1:
                if lows[j] <= sl_cur:
                    return round((sl_cur - entry) / risk, 2)
                if highs[j] >= entry + be_rr * risk:
                    reached = True
                if reached:
                    sl_cur = max(sl_cur, max(entry, sw_lows[j]))
            else:
                if highs[j] >= sl_cur:
                    return round((entry - sl_cur) / risk, 2)
                if lows[j] <= entry - be_rr * risk:
                    reached = True
                if reached:
                    sl_cur = min(sl_cur, min(entry, sw_highs[j]))

    elif strategy == 'atr_trail':
        sl_cur = sl
        for j in range(idx + 1, end):
            cur_atr = atr[j] if atr[j] > 0 else risk
            if direction == 1:
                if lows[j] <= sl_cur:
                    return round((sl_cur - entry) / risk, 2)
                sl_cur = max(sl_cur, highs[j] - atr_mult * cur_atr)
            else:
                if highs[j] >= sl_cur:
                    return round((entry - sl_cur) / risk, 2)
                sl_cur = min(sl_cur, lows[j] + atr_mult * cur_atr)

    elif strategy == 'partial':
        sl_cur = sl
        partial_taken = False
        partial_pnl = 0.0
        for j in range(idx + 1, end):
            if direction == 1:
                if lows[j] <= sl_cur:
                    rem = (sl_cur - entry) / risk
                    if partial_taken:
                        return round(partial_pnl + rem * (1 - partial_pct), 2)
                    return round(rem, 2)
                if not partial_taken and highs[j] >= entry + partial_rr * risk:
                    partial_taken = True
                    partial_pnl = partial_rr * partial_pct
                    sl_cur = max(sl_cur, entry)
                elif partial_taken:
                    sl_cur = max(sl_cur, sw_lows[j])
            else:
                if highs[j] >= sl_cur:
                    rem = (entry - sl_cur) / risk
                    if partial_taken:
                        return round(partial_pnl + rem * (1 - partial_pct), 2)
                    return round(rem, 2)
                if not partial_taken and lows[j] <= entry - partial_rr * risk:
                    partial_taken = True
                    partial_pnl = partial_rr * partial_pct
                    sl_cur = min(sl_cur, entry)
                elif partial_taken:
                    sl_cur = min(sl_cur, sw_highs[j])

    # Timeout — close at market
    last = closes[min(end - 1, n - 1)]
    if direction == 1:
        pnl = (last - entry) / risk
    else:
        pnl = (entry - last) / risk
    return round(pnl, 2)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate(config, session=None, _cache={}):
    """Evaluate a confluence scalper configuration. Returns metrics dict."""
    data = load_and_cache_data(session, _cache)

    c_highs = data['c_highs']
    c_lows = data['c_lows']
    c_closes = data['c_closes']
    c_times = data['c_times']
    atr = data['atr']
    n_candles = data['n_candles']

    exit_cfg = config['exit']
    strategy = exit_cfg['strategy']
    swing_lb = exit_cfg.get('swing_lookback', 5)
    timeout = exit_cfg.get('timeout_candles', 20)
    rr_ratio = exit_cfg.get('rr_ratio', 1.5)
    be_rr = exit_cfg.get('breakeven_at_rr', 1.0)
    atr_mult = exit_cfg.get('atr_multiplier', 2.0)
    partial_pct = exit_cfg.get('partial_pct', 0.5)
    partial_rr = exit_cfg.get('partial_rr', 2.0)
    sl_buffer = exit_cfg.get('sl_buffer_pct', 0.001)

    # Get swing data
    sw_lows, sw_highs = _get_swings(data, swing_lb)

    # Phase 1: Find all touch entries (vectorized chunked broadcasting)
    t_phase = time.time()
    entries = find_touch_entries(data, config)
    t1 = time.time() - t_phase
    if len(entries) == 0:
        return {'error': 'No touch entries found', 'total_r': 0, 'fitness': 0}

    # Phase 2+3: Score by confluence + deduplicate with cooldown
    t_phase = time.time()
    entries, scores = score_and_deduplicate(entries, data, config)
    t2 = time.time() - t_phase
    if len(entries) == 0:
        return {'error': 'No entries after scoring/dedup', 'total_r': 0, 'fitness': 0}

    # Phase 4: Simulate exits
    t_phase = time.time()
    results = []
    for i in range(len(entries)):
        ci = int(entries[i, 0])
        direction = int(entries[i, 2])

        entry_price = c_closes[ci]
        if direction == 1:  # LONG
            sl = c_lows[ci] * (1 - sl_buffer)
        else:  # SHORT
            sl = c_highs[ci] * (1 + sl_buffer)

        risk = abs(entry_price - sl)
        if risk == 0 or risk / entry_price < 0.0001:
            continue

        pnl_r = _simulate_exit(
            strategy, direction, entry_price, sl, risk, ci,
            c_highs, c_lows, c_closes, atr, sw_lows, sw_highs,
            timeout, rr_ratio, be_rr, atr_mult, partial_pct, partial_rr,
        )
        results.append(pnl_r)

    if not results:
        return {'error': 'No valid trades', 'total_r': 0, 'fitness': 0}

    results = np.array(results)
    wins = results[results > 0]
    losses = results[results <= 0]
    gross_win = float(wins.sum()) if len(wins) > 0 else 0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0
    pf = gross_win / gross_loss if gross_loss > 0 else 99.99

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for r in results:
        if r <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    # Sharpe of R values
    sharpe = float(results.mean() / results.std()) if results.std() > 0 else 0

    # Trades per day
    if len(entries) > 1:
        first_time = c_times[int(entries[0, 0])]
        last_time = c_times[int(entries[-1, 0])]
        days = max((last_time - first_time) / np.timedelta64(1, 'D'), 1)
        trades_per_day = len(results) / days
    else:
        trades_per_day = 0

    total_trades = len(results)
    metrics = {
        'total_r': round(float(results.sum()), 1),
        'profit_factor': round(pf, 2),
        'win_rate': round(len(wins) / total_trades * 100, 1),
        'avg_r': round(float(results.mean()), 2),
        'median_r': round(float(np.median(results)), 2),
        'total_trades': total_trades,
        'trades_per_day': round(trades_per_day, 2),
        'max_consecutive_losses': max_consec,
        'sharpe_r': round(sharpe, 3),
    }
    t3 = time.time() - t_phase
    metrics['fitness'] = round(pf * math.sqrt(total_trades), 2) if total_trades >= 10 else 0
    metrics['_timing'] = f"touch={t1:.1f}s score={t2:.1f}s sim={t3:.1f}s"
    return metrics


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def fitness(metrics):
    return metrics.get('fitness', 0)


# ---------------------------------------------------------------------------
# Mutation system (adapted from agent.py)
# ---------------------------------------------------------------------------

def get_nested(d, key_path):
    keys = key_path.split('.')
    for k in keys:
        d = d[k]
    return d


def set_nested(d, key_path, value):
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


MUTATIONS = [
    {'name': 'score_threshold', 'field': 'score_threshold', 'range': [2, 8], 'step': 0.5},
    {'name': 'zone_width', 'field': 'zone_width', 'range': [0.005, 0.03], 'step': 0.002},
    {'name': 'touch_tolerance', 'field': 'touch_tolerance', 'range': [0.001, 0.01], 'step': 0.001},
    {'name': 'naked_only', 'field': 'naked_only', 'options': [True, False]},
    {'name': 'exit_strategy', 'field': 'exit.strategy',
     'options': ['fixed_rr', 'swing_trail', 'breakeven_trail', 'atr_trail', 'partial']},
    {'name': 'rr_ratio', 'field': 'exit.rr_ratio', 'range': [0.5, 3.0], 'step': 0.25},
    {'name': 'swing_lookback', 'field': 'exit.swing_lookback', 'range': [2, 10], 'step': 1, 'type': 'int'},
    {'name': 'atr_multiplier', 'field': 'exit.atr_multiplier', 'range': [0.5, 3.0], 'step': 0.25},
    {'name': 'timeout_candles', 'field': 'exit.timeout_candles', 'range': [5, 50], 'step': 5, 'type': 'int'},
    {'name': 'breakeven_rr', 'field': 'exit.breakeven_at_rr', 'range': [0.5, 3.0], 'step': 0.25},
    {'name': 'partial_pct', 'field': 'exit.partial_pct', 'range': [0.2, 0.8], 'step': 0.1},
    {'name': 'partial_rr', 'field': 'exit.partial_rr', 'range': [1.0, 5.0], 'step': 0.5},
    {'name': 'add_level_type', 'field': 'level_types', 'action': 'add', 'pool': ALL_LEVEL_TYPES},
    {'name': 'remove_level_type', 'field': 'level_types', 'action': 'remove', 'min_items': 2},
]


def propose_mutation(config, history=None):
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
        current = get_nested(config, mut_def['field'])
        options = [o for o in mut_def['options'] if o != current]
        new_val = random.choice(options) if options else current
        mutation['old'] = current
        mutation['new'] = new_val
        mutation['description'] = f"{mut_def['name']}: {current} -> {new_val}"

    elif 'range' in mut_def:
        current = get_nested(config, mut_def['field'])
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
        current = get_nested(config, mut_def['field'])
        pool = [t for t in mut_def['pool'] if t not in current]
        if pool:
            to_add = random.choice(pool)
            mutation['new'] = to_add
            mutation['description'] = f"add {to_add}"
        else:
            mutation['description'] = "no types to add"
            mutation['skip'] = True

    elif mut_def.get('action') == 'remove':
        current = get_nested(config, mut_def['field'])
        min_items = mut_def.get('min_items', 1)
        if len(current) > min_items:
            to_remove = random.choice(current)
            mutation['new'] = to_remove
            mutation['description'] = f"remove {to_remove}"
        else:
            mutation['description'] = "can't remove (min 2)"
            mutation['skip'] = True

    return mutation


def apply_mutation(config, mutation):
    """Apply a mutation to a config copy."""
    new_config = copy.deepcopy(config)
    if mutation.get('skip'):
        return new_config

    if mutation['name'] == 'add_level_type':
        new_config['level_types'].append(mutation['new'])
    elif mutation['name'] == 'remove_level_type':
        new_config['level_types'].remove(mutation['new'])
    elif 'new' in mutation:
        set_nested(new_config, mutation['field'], mutation['new'])

    return new_config


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_confluence_scalper(n_experiments=50):
    """Main AutoResearch loop for confluence scalper discovery."""
    from app import create_app
    from app.extensions import db

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'confluence_scalper.jsonl')

    app = create_app()
    with app.app_context():
        config = copy.deepcopy(DEFAULT_CONFIG)

        print("=" * 70, flush=True)
        print("AUTORESEARCH MODE C — Confluence Scalper Discovery", flush=True)
        print("=" * 70, flush=True)
        print(f"Running {n_experiments} experiments...\n", flush=True)

        # Baseline
        print("Evaluating baseline...", flush=True)
        t0 = time.time()
        _cache = {}
        baseline = evaluate(config, db.session, _cache)
        elapsed = time.time() - t0
        print(f"Baseline ({elapsed:.1f}s):", flush=True)
        for k, v in sorted(baseline.items()):
            if k != 'error':
                print(f"  {k}: {v}", flush=True)
        print(flush=True)

        history = []
        best = baseline.copy()
        best_config = copy.deepcopy(config)
        improvements = 0

        for i in range(n_experiments):
            mutation = propose_mutation(config, history)

            if mutation.get('skip'):
                print(f"[{i+1}/{n_experiments}] SKIP: {mutation['description']}", flush=True)
                continue

            new_config = apply_mutation(config, mutation)

            t0 = time.time()
            metrics = evaluate(new_config, db.session, _cache)
            elapsed = time.time() - t0

            improved = fitness(metrics) > fitness(best)

            experiment = {
                'id': i,
                'timestamp': datetime.now().isoformat(),
                'mutation': mutation,
                'metrics': metrics,
                'improved': improved,
                'best_fitness': fitness(best),
                'elapsed_sec': round(elapsed, 1),
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
                      f"-> fitness={fitness(metrics)} PF={metrics['profit_factor']} "
                      f"trades={metrics['total_trades']} WR={metrics['win_rate']}% "
                      f"({elapsed:.1f}s)", flush=True)
            else:
                print(f"[{i+1}/{n_experiments}] no gain: {mutation['description']} "
                      f"-> fitness={fitness(metrics)} ({elapsed:.1f}s)", flush=True)

        # Summary
        print(flush=True)
        print("=" * 70, flush=True)
        print("CONFLUENCE SCALPER SUMMARY", flush=True)
        print("=" * 70, flush=True)
        print(f"Experiments: {n_experiments}", flush=True)
        print(f"Improvements: {improvements}", flush=True)
        print(f"Baseline fitness: {fitness(baseline)}", flush=True)
        print(f"Best fitness: {fitness(best)}", flush=True)
        print(flush=True)
        print("Best metrics:", flush=True)
        for k, v in sorted(best.items()):
            if k != 'error':
                print(f"  {k}: {v}", flush=True)
        print(flush=True)
        print("Best config:", flush=True)
        print(f"  score_threshold: {best_config['score_threshold']}", flush=True)
        print(f"  zone_width: {best_config['zone_width']}", flush=True)
        print(f"  touch_tolerance: {best_config['touch_tolerance']}", flush=True)
        print(f"  naked_only: {best_config['naked_only']}", flush=True)
        print(f"  level_types ({len(best_config['level_types'])}): {best_config['level_types']}", flush=True)
        print(f"  exit: {best_config['exit']}", flush=True)
        print(flush=True)
        print(f"Results logged to: {results_file}", flush=True)

        return best, best_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoResearch Mode C — Confluence Scalper')
    parser.add_argument('--experiments', type=int, default=50, help='Number of experiments')
    args = parser.parse_args()

    run_confluence_scalper(n_experiments=args.experiments)
