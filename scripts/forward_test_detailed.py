#!/usr/bin/env python
"""Forward Test Detailed — shows which levels were touched and trade mechanics.

For each trade in 2026, shows:
- Which specific levels were in the confluence zone
- Entry price, SL, trail evolution
- Bar-by-bar what happened (when did BE activate, when did trail move, etc.)

Usage:
    python scripts/forward_test_detailed.py
    python scripts/forward_test_detailed.py --top 10  (show only top 10 by P&L)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import numpy as np
from datetime import datetime


def run_detailed(top_n=None, start_date='2026-01-01'):
    from app import create_app
    from app.extensions import db
    from scripts.autoresearch.confluence_scalper import (
        load_and_cache_data, find_touch_entries, score_and_deduplicate,
        _get_swings,
    )
    from sqlalchemy import text

    app = create_app()
    with app.app_context():
        print("=" * 80)
        print(f"FORWARD TEST DETAILED — Trade-by-trade analysis")
        print("=" * 80)

        cache = {}
        data = load_and_cache_data(db.session, cache, timeframe='15m')
        c_times = data['c_times']
        c_highs = data['c_highs']
        c_lows = data['c_lows']
        c_closes = data['c_closes']
        atr = data['atr']
        n_candles = data['n_candles']

        # Level data for looking up which levels were touched
        l_prices = data['l_prices']
        l_types = data.get('l_types_str', None)
        if l_types is None:
            from app.services.level_trade_backtest_db import load_levels_db
            levels_df = load_levels_db(db.session)
            l_types = levels_df['level_type'].values.astype(str)
            l_timeframes = levels_df['timeframe'].values.astype(str)
            l_prices_full = levels_df['price_level'].values.astype(np.float64)
        else:
            l_timeframes = data.get('l_timeframes', np.array(['?'] * len(l_types)))
            l_prices_full = l_prices

        times_epoch = c_times.astype('datetime64[s]').astype(np.int64)
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        fwd_start = np.searchsorted(times_epoch, start_ts)

        config = {
            'score_threshold': 3, 'zone_width': 0.01, 'touch_tolerance': 0.003,
            'naked_only': True, 'scoring_mode': 'unique_types', 'entry_mode': 'touch',
            'session_filter': 'all',
            'level_types': ['Fractal_support', 'Fractal_resistance',
                            'PrevSession_VWAP', 'PrevSession_VP_POC'],
            'exit': {'strategy': 'breakeven_trail', 'rr_ratio': 1.5, 'swing_lookback': 10,
                     'atr_multiplier': 2.0, 'breakeven_at_rr': 2.75, 'partial_pct': 0.5,
                     'partial_rr': 2.0, 'timeout_candles': 45, 'sl_buffer_pct': 0.001},
        }

        entries = find_touch_entries(data, config)
        entries, scores = score_and_deduplicate(entries, data, config)

        fwd_mask = entries[:, 0] >= fwd_start
        entries_fwd = entries[fwd_mask]
        scores_fwd = scores[fwd_mask]

        exit_cfg = config['exit']
        sl_buffer = exit_cfg['sl_buffer_pct']
        be_rr = exit_cfg['breakeven_at_rr']
        timeout = exit_cfg['timeout_candles']
        swing_lb = exit_cfg['swing_lookback']

        sw_lows, sw_highs = _get_swings(data, swing_lb)

        # Sorted prices for zone lookup
        sorted_idx = np.argsort(l_prices_full)
        sorted_prices = l_prices_full[sorted_idx]

        trades = []

        for i in range(len(entries_fwd)):
            ci = int(entries_fwd[i, 0])
            li = int(entries_fwd[i, 1])
            direction = int(entries_fwd[i, 2])

            entry_price = c_closes[ci]
            if direction == 1:
                sl = c_lows[ci] * (1 - sl_buffer)
            else:
                sl = c_highs[ci] * (1 + sl_buffer)

            risk = abs(entry_price - sl)
            if risk <= 0 or risk / entry_price > 0.05:
                continue
            if ci + timeout >= n_candles:
                continue

            # Find levels in zone
            zone_width = config['zone_width']
            level_price = l_prices_full[li] if li < len(l_prices_full) else entry_price
            lo_zone = level_price * (1 - zone_width)
            hi_zone = level_price * (1 + zone_width)
            left = np.searchsorted(sorted_prices, lo_zone, side='left')
            right = np.searchsorted(sorted_prices, hi_zone, side='right')

            zone_levels = []
            if left < right:
                for idx in sorted_idx[left:right]:
                    if idx < len(l_types):
                        zone_levels.append({
                            'type': l_types[idx],
                            'tf': l_timeframes[idx] if idx < len(l_timeframes) else '?',
                            'price': round(float(l_prices_full[idx]), 2),
                        })

            # Simulate exit bar-by-bar
            sl_cur = sl
            be_reached = False
            exit_bar = -1
            exit_price = entry_price
            exit_reason = 'TIMEOUT'
            trail_history = []

            for j in range(1, min(timeout, n_candles - ci)):
                k = ci + j
                sw_low = sw_lows[k]
                sw_high = sw_highs[k]

                if direction == 1:
                    if c_lows[k] <= sl_cur:
                        exit_bar = j
                        exit_price = sl_cur
                        exit_reason = 'TRAIL_HIT' if be_reached else 'SL_HIT'
                        break
                    if c_highs[k] >= entry_price + be_rr * risk:
                        be_reached = True
                    if be_reached:
                        new_sl = max(sl_cur, max(entry_price, sw_low))
                        if new_sl != sl_cur:
                            trail_history.append({'bar': j, 'sl': round(new_sl, 2)})
                        sl_cur = new_sl
                else:
                    if c_highs[k] >= sl_cur:
                        exit_bar = j
                        exit_price = sl_cur
                        exit_reason = 'TRAIL_HIT' if be_reached else 'SL_HIT'
                        break
                    if c_lows[k] <= entry_price - be_rr * risk:
                        be_reached = True
                    if be_reached:
                        new_sl = min(sl_cur, min(entry_price, sw_high))
                        if new_sl != sl_cur:
                            trail_history.append({'bar': j, 'sl': round(new_sl, 2)})
                        sl_cur = new_sl

            if exit_bar < 0:
                exit_bar = min(timeout, n_candles - ci - 1)
                exit_price = c_closes[ci + exit_bar]
                exit_reason = 'TIMEOUT'

            pnl_r = round((exit_price - entry_price) / risk if direction == 1
                         else (entry_price - exit_price) / risk, 2)

            # Get timestamp
            t = c_times[ci]
            time_str = str(np.datetime_as_string(t, unit='m'))

            exit_t = c_times[ci + exit_bar]
            exit_time_str = str(np.datetime_as_string(exit_t, unit='m'))

            # MFE
            end_mfe = min(ci + timeout, n_candles)
            if direction == 1:
                mfe = round((np.max(c_highs[ci+1:end_mfe]) - entry_price) / risk, 2)
                mfe_price = round(float(np.max(c_highs[ci+1:end_mfe])), 2)
            else:
                mfe = round((entry_price - np.min(c_lows[ci+1:end_mfe])) / risk, 2)
                mfe_price = round(float(np.min(c_lows[ci+1:end_mfe])), 2)

            trades.append({
                'time': time_str,
                'exit_time': exit_time_str,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'entry': round(entry_price, 2),
                'sl_initial': round(sl, 2),
                'risk_pct': round(risk / entry_price * 100, 3),
                'exit_price': round(exit_price, 2),
                'exit_reason': exit_reason,
                'exit_bar': exit_bar,
                'be_reached': be_reached,
                'pnl_r': pnl_r,
                'mfe_r': mfe,
                'mfe_price': mfe_price,
                'zone_levels': zone_levels,
                'trail_moves': len(trail_history),
                'trail_history': trail_history[:5],  # First 5 trail moves
                'confluence_score': int(scores_fwd[i]),
            })

        # Sort by time
        trades.sort(key=lambda t: t['time'])

        if top_n:
            trades_show = sorted(trades, key=lambda t: -abs(t['pnl_r']))[:top_n]
            trades_show.sort(key=lambda t: t['time'])
            print(f"\nShowing top {top_n} trades by |P&L|:\n")
        else:
            trades_show = trades
            print(f"\nAll {len(trades)} trades:\n")

        for idx, t in enumerate(trades_show):
            won = t['pnl_r'] > 0
            icon = '+' if won else 'X'
            print(f"{'='*80}")
            print(f"[{icon}] Trade #{idx+1}: {t['direction']} at {t['entry']:,.2f} "
                  f"({t['time']})")
            print(f"    SL: {t['sl_initial']:,.2f} ({t['risk_pct']:.3f}%)")
            print(f"    Exit: {t['exit_price']:,.2f} ({t['exit_reason']}) "
                  f"after {t['exit_bar']} bars ({t['exit_time']})")
            print(f"    P&L: {t['pnl_r']:+.2f}R | MFE: {t['mfe_r']:.1f}R "
                  f"(price reached {t['mfe_price']:,.2f})")
            if t['be_reached']:
                print(f"    Breakeven activated, trail moved {t['trail_moves']} times")
                if t['trail_history']:
                    for th in t['trail_history']:
                        print(f"      bar {th['bar']}: SL -> {th['sl']:,.2f}")

            # Confluence zone levels
            print(f"    Confluence ({t['confluence_score']} score):")
            types_shown = set()
            for lv in t['zone_levels']:
                key = f"{lv['type']}_{lv['tf']}"
                if key not in types_shown:
                    print(f"      {lv['type']:25s} {lv['tf']:8s} @ {lv['price']:>12,.2f}")
                    types_shown.add(key)

        # Summary stats
        pnls = [t['pnl_r'] for t in trades]
        mfes = [t['mfe_r'] for t in trades]
        print(f"\n{'='*80}")
        print(f"SUMMARY: {len(trades)} trades | WR {sum(1 for p in pnls if p > 0)/len(pnls)*100:.0f}% | "
              f"PF {sum(p for p in pnls if p > 0)/abs(sum(p for p in pnls if p <= 0)):.2f} | "
              f"Total {sum(pnls):+.1f}R")
        print(f"Avg MFE: {np.mean(mfes):.1f}R | "
              f"MFE captured: {np.mean(pnls)/np.mean(mfes)*100:.0f}%")

        # Exit reason distribution
        reasons = {}
        for t in trades:
            reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1
        print(f"Exit reasons: {reasons}")

        # Level type frequency in winning vs losing trades
        win_types = {}
        loss_types = {}
        for t in trades:
            target = win_types if t['pnl_r'] > 0 else loss_types
            for lv in t['zone_levels']:
                key = f"{lv['type']} ({lv['tf']})"
                target[key] = target.get(key, 0) + 1

        print(f"\nLevel types in WINNING trades:")
        for k, v in sorted(win_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  {k:40s}: {v}")
        print(f"\nLevel types in LOSING trades:")
        for k, v in sorted(loss_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  {k:40s}: {v}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Forward Test Detailed')
    parser.add_argument('--top', type=int, default=None, help='Show only top N trades by |P&L|')
    parser.add_argument('--start', type=str, default='2026-01-01')
    args = parser.parse_args()
    run_detailed(top_n=args.top, start_date=args.start)
