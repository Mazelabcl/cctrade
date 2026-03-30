#!/usr/bin/env python
"""MFE (Maximum Favorable Excursion) Analysis.

For each trade from the confluence scalper, calculates:
- MFE: how far price moved in our favor (best possible exit)
- MAE: how far price moved against us (worst drawdown before recovery)
- Actual exit R vs optimal exit R
- "Left on table" = MFE - actual exit

This tells us: are our exit strategies capturing the move, or leaving profit?

Usage:
    python scripts/autoresearch/mfe_analysis.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import copy
import time
import numpy as np
from scripts.autoresearch.confluence_scalper import (
    load_and_cache_data, find_touch_entries, score_and_deduplicate,
    _simulate_exit, _get_swings,
)


BEST_15M_CONFIG = {
    'score_threshold': 3,
    'zone_width': 0.01,
    'touch_tolerance': 0.001,
    'naked_only': True,
    'scoring_mode': 'unique_types',
    'entry_mode': 'touch',
    'session_filter': 'all',
    'level_types': [
        'Fractal_support', 'Fractal_resistance',
        'PrevSession_VWAP', 'PrevSession_VP_POC',
    ],
    'exit': {
        'strategy': 'breakeven_trail',
        'rr_ratio': 1.5,
        'swing_lookback': 9,
        'atr_multiplier': 2.0,
        'breakeven_at_rr': 1.0,
        'partial_pct': 0.5,
        'partial_rr': 2.0,
        'timeout_candles': 35,
        'sl_buffer_pct': 0.001,
    },
}


def compute_mfe_mae(direction, entry, sl, risk, idx, highs, lows, closes, n_candles,
                    windows=[10, 20, 35, 50, 100]):
    """Compute MFE and MAE for a trade across multiple lookahead windows."""
    results = {}
    for w in windows:
        end = min(idx + w, n_candles)
        if idx + 1 >= end:
            continue

        if direction == 1:  # LONG
            max_price = np.max(highs[idx+1:end])
            min_price = np.min(lows[idx+1:end])
            mfe_r = (max_price - entry) / risk
            mae_r = (entry - min_price) / risk
        else:  # SHORT
            min_price = np.min(lows[idx+1:end])
            max_price = np.max(highs[idx+1:end])
            mfe_r = (entry - min_price) / risk
            mae_r = (max_price - entry) / risk

        results[f'mfe_{w}'] = round(mfe_r, 2)
        results[f'mae_{w}'] = round(mae_r, 2)

    return results


def run_mfe_analysis():
    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        print("=" * 70, flush=True)
        print("MFE/MAE ANALYSIS", flush=True)
        print("=" * 70, flush=True)

        timeframes = ['15m', '1h', '4h']
        sessions = ['all', 'us', 'us_eu']

        for tf in timeframes:
            _cache = {}
            data = load_and_cache_data(db.session, _cache, timeframe=tf)

            c_highs = data['c_highs']
            c_lows = data['c_lows']
            c_closes = data['c_closes']
            c_times = data['c_times']
            atr = data['atr']
            n_candles = data['n_candles']

            for session in sessions:
                config = copy.deepcopy(BEST_15M_CONFIG)
                config['session_filter'] = session
                if tf in ('1h', '4h'):
                    config['exit']['timeout_candles'] = 50

                exit_cfg = config['exit']
                sl_buffer = exit_cfg.get('sl_buffer_pct', 0.001)
                swing_lb = exit_cfg.get('swing_lookback', 9)
                timeout = exit_cfg.get('timeout_candles', 35)

                sw_lows, sw_highs = _get_swings(data, swing_lb)

                # Get entries
                entries = find_touch_entries(data, config)
                if len(entries) == 0:
                    continue
                entries, scores = score_and_deduplicate(entries, data, config)
                if len(entries) == 0:
                    continue

                # Session filter (inline to match evaluate())
                session_filter = config.get('session_filter', 'all')
                if session_filter != 'all':
                    SESSION_HOURS = {
                        'asia': set(range(0, 8)),
                        'eu': set(range(8, 14)),
                        'us': set(range(14, 21)),
                        'us_eu': set(range(8, 21)),
                    }
                    allowed = SESSION_HOURS.get(session_filter, set(range(24)))
                    keep = np.zeros(len(entries), dtype=bool)
                    for i in range(len(entries)):
                        ci = int(entries[i, 0])
                        ct = c_times[ci]
                        hour = int((ct - ct.astype('datetime64[D]')) / np.timedelta64(1, 'h'))
                        keep[i] = hour in allowed
                    entries = entries[keep]
                    scores = scores[keep]

                if len(entries) == 0:
                    continue

                # Compute MFE/MAE + actual exit for each trade
                mfes = {f'mfe_{w}': [] for w in [10, 20, 35, 50, 100]}
                maes = {f'mae_{w}': [] for w in [10, 20, 35, 50, 100]}
                actual_exits = []
                actual_exits_be = []
                actual_exits_swing = []

                for i in range(len(entries)):
                    ci = int(entries[i, 0])
                    direction = int(entries[i, 2])

                    entry_price = c_closes[ci]
                    if direction == 1:
                        sl = c_lows[ci] * (1 - sl_buffer)
                    else:
                        sl = c_highs[ci] * (1 + sl_buffer)
                    risk = abs(entry_price - sl)
                    if risk <= 0 or risk / entry_price > 0.05:
                        continue

                    # MFE/MAE
                    mm = compute_mfe_mae(direction, entry_price, sl, risk, ci,
                                        c_highs, c_lows, c_closes, n_candles)
                    for k, v in mm.items():
                        if k in mfes:
                            mfes[k].append(v)
                        elif k in maes:
                            maes[k].append(v)

                    # Actual exits with different strategies
                    pnl_be = _simulate_exit(
                        'breakeven_trail', direction, entry_price, sl, risk, ci,
                        c_highs, c_lows, c_closes, atr, sw_lows, sw_highs,
                        timeout, 1.5, 1.0, 2.0, 0.5, 2.0)
                    actual_exits_be.append(pnl_be)

                    pnl_swing = _simulate_exit(
                        'swing_trail', direction, entry_price, sl, risk, ci,
                        c_highs, c_lows, c_closes, atr, sw_lows, sw_highs,
                        timeout, 1.5, 1.0, 2.0, 0.5, 2.0)
                    actual_exits_swing.append(pnl_swing)

                n_trades = len(mfes.get('mfe_35', []))
                if n_trades == 0:
                    continue

                print(f"\n{'='*70}", flush=True)
                print(f"  {tf} | {session} | {n_trades} trades", flush=True)
                print(f"{'='*70}", flush=True)

                # MFE stats
                print(f"\n  MFE (max favorable excursion in R):", flush=True)
                print(f"  {'Window':>8s} | {'Median':>7s} | {'Mean':>7s} | {'P75':>7s} | {'P90':>7s} | {'Max':>7s}", flush=True)
                print(f"  {'-'*55}", flush=True)
                for w in [10, 20, 35, 50, 100]:
                    k = f'mfe_{w}'
                    if k in mfes and mfes[k]:
                        arr = np.array(mfes[k])
                        print(f"  {w:>6d}v | {np.median(arr):7.2f} | {np.mean(arr):7.2f} | "
                              f"{np.percentile(arr, 75):7.2f} | {np.percentile(arr, 90):7.2f} | "
                              f"{np.max(arr):7.2f}", flush=True)

                # MAE stats
                print(f"\n  MAE (max adverse excursion in R):", flush=True)
                print(f"  {'Window':>8s} | {'Median':>7s} | {'Mean':>7s} | {'P75':>7s} | {'P90':>7s}", flush=True)
                print(f"  {'-'*45}", flush=True)
                for w in [10, 20, 35, 50, 100]:
                    k = f'mae_{w}'
                    if k in maes and maes[k]:
                        arr = np.array(maes[k])
                        print(f"  {w:>6d}v | {np.median(arr):7.2f} | {np.mean(arr):7.2f} | "
                              f"{np.percentile(arr, 75):7.2f} | {np.percentile(arr, 90):7.2f}", flush=True)

                # Compare actual vs optimal
                be_arr = np.array(actual_exits_be)
                sw_arr = np.array(actual_exits_swing)
                mfe35 = np.array(mfes.get('mfe_35', [0]))
                mfe50 = np.array(mfes.get('mfe_50', [0]))

                n = min(len(be_arr), len(mfe35))
                left_on_table_be = mfe35[:n] - be_arr[:n]
                left_on_table_sw = mfe35[:n] - sw_arr[:n]

                print(f"\n  Exit comparison (avg R per trade):", flush=True)
                print(f"    Breakeven trail:  {np.mean(be_arr):+.3f}R (capturing {np.mean(be_arr[:n])/np.mean(mfe35[:n])*100:.0f}% of MFE-35)", flush=True)
                print(f"    Swing trail:      {np.mean(sw_arr):+.3f}R (capturing {np.mean(sw_arr[:n])/np.mean(mfe35[:n])*100:.0f}% of MFE-35)", flush=True)
                print(f"    MFE-35 optimal:   {np.mean(mfe35):+.3f}R (perfect exit)", flush=True)
                print(f"    MFE-50 optimal:   {np.mean(mfe50):+.3f}R (if held longer)", flush=True)
                if len(mfes.get('mfe_100', [])) > 0:
                    mfe100 = np.array(mfes['mfe_100'])
                    print(f"    MFE-100 optimal:  {np.mean(mfe100):+.3f}R (much longer)", flush=True)

                print(f"\n  Left on table (MFE35 - actual):", flush=True)
                print(f"    vs breakeven_trail: {np.mean(left_on_table_be):.2f}R avg", flush=True)
                print(f"    vs swing_trail:     {np.mean(left_on_table_sw):.2f}R avg", flush=True)

                # Are these range trades or trend trades?
                mfe35_arr = np.array(mfes.get('mfe_35', []))
                range_trades = np.sum(mfe35_arr < 2.0)
                trend_trades = np.sum(mfe35_arr >= 5.0)
                mid_trades = n - range_trades - trend_trades
                print(f"\n  Trade nature (based on MFE-35):", flush=True)
                print(f"    Range (<2R MFE):  {range_trades} ({range_trades/n*100:.0f}%)", flush=True)
                print(f"    Medium (2-5R):    {mid_trades} ({mid_trades/n*100:.0f}%)", flush=True)
                print(f"    Trend (5R+ MFE):  {trend_trades} ({trend_trades/n*100:.0f}%)", flush=True)


if __name__ == '__main__':
    run_mfe_analysis()
