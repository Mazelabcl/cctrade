#!/usr/bin/env python
"""Sniper Backtest — 15m signal + 1m entry with improved exit.

Takes confluence scalper entries on 15m, then zooms into 1m to find
the optimal entry point (extreme+1 bar). Uses the tighter SL from 1m
for larger position size with same dollar risk.

Compares:
- Entry A: 15m close (current system)
- Entry B: 1m extreme+1 bar (sniper entry, SL at 1m extreme)

Both use the improved exit: breakeven_trail, be_rr=2.75, timeout=45 (scaled to 1m).

Usage:
    python scripts/autoresearch/sniper_backtest.py
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


BEST_CONFIG = {
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
        'swing_lookback': 10,
        'atr_multiplier': 2.0,
        'breakeven_at_rr': 2.75,
        'partial_pct': 0.5,
        'partial_rr': 2.0,
        'timeout_candles': 45,
        'sl_buffer_pct': 0.001,
    },
}


def compute_atr_array(highs, lows, closes, period=14):
    n = len(highs)
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        if i < period:
            atr[i] = tr
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr) / period
    return atr


def find_swing_lows(lows, lookback=10):
    n = len(lows)
    result = np.copy(lows)
    for i in range(lookback, n):
        result[i] = np.min(lows[max(0, i-lookback):i+1])
    return result


def find_swing_highs(highs, lookback=10):
    n = len(highs)
    result = np.copy(highs)
    for i in range(lookback, n):
        result[i] = np.max(highs[max(0, i-lookback):i+1])
    return result


def run_sniper_backtest():
    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        print("=" * 70, flush=True)
        print("SNIPER BACKTEST", flush=True)
        print("15m signal + 1m extreme+1 entry vs 15m close entry", flush=True)
        print("Exit: breakeven_trail, be_rr=2.75, timeout=45(15m)/675(1m)", flush=True)
        print("=" * 70, flush=True)

        # Load 15m data for signals
        print("\nLoading 15m data...", flush=True)
        cache_15m = {}
        data_15m = load_and_cache_data(db.session, cache_15m, timeframe='15m')
        c_times_15m = data_15m['c_times']
        c_closes_15m = data_15m['c_closes']
        c_highs_15m = data_15m['c_highs']
        c_lows_15m = data_15m['c_lows']
        atr_15m = data_15m['atr']
        n_15m = data_15m['n_candles']

        # Load 1m data for sniper entry + exit simulation
        print("Loading 1m data...", flush=True)
        cache_1m = {}
        data_1m = load_and_cache_data(db.session, cache_1m, timeframe='1m')
        c_times_1m = data_1m['c_times']
        c_closes_1m = data_1m['c_closes']
        c_highs_1m = data_1m['c_highs']
        c_lows_1m = data_1m['c_lows']
        atr_1m = data_1m['atr']
        n_1m = data_1m['n_candles']

        # Precompute 1m swings for exit simulation
        print("Computing 1m swings and ATR...", flush=True)
        # Scale swing lookback: 10 bars on 15m = ~150 bars on 1m
        sw_lows_1m = find_swing_lows(c_lows_1m, lookback=150)
        sw_highs_1m = find_swing_highs(c_highs_1m, lookback=150)

        # Also precompute 15m swings for the 15m baseline
        sw_lows_15m, sw_highs_15m = _get_swings(data_15m, 10)

        # Get 15m scalper entries
        config = copy.deepcopy(BEST_CONFIG)
        entries = find_touch_entries(data_15m, config)
        entries, scores = score_and_deduplicate(entries, data_15m, config)
        print(f"15m entries: {len(entries)}", flush=True)

        # Convert timestamps for matching
        times_15m_epoch = c_times_15m.astype('datetime64[s]').astype(np.int64)
        times_1m_epoch = c_times_1m.astype('datetime64[s]').astype(np.int64)

        exit_cfg = config['exit']
        sl_buffer = exit_cfg['sl_buffer_pct']
        be_rr = exit_cfg['breakeven_at_rr']
        timeout_15m = exit_cfg['timeout_candles']
        timeout_1m = timeout_15m * 15  # Scale to 1m bars
        swing_lb = exit_cfg['swing_lookback']
        rr_ratio = exit_cfg['rr_ratio']
        atr_mult = exit_cfg['atr_multiplier']
        partial_pct = exit_cfg['partial_pct']
        partial_rr = exit_cfg['partial_rr']

        # Simulate trades both ways
        trades_15m = []  # Entry A: 15m close
        trades_1m = []   # Entry B: 1m extreme+1

        for idx in range(len(entries)):
            ci_15m = int(entries[idx, 0])
            direction = int(entries[idx, 2])

            # --- Entry A: 15m close (current system) ---
            entry_15m = c_closes_15m[ci_15m]
            if direction == 1:
                sl_15m = c_lows_15m[ci_15m] * (1 - sl_buffer)
            else:
                sl_15m = c_highs_15m[ci_15m] * (1 + sl_buffer)
            risk_15m = abs(entry_15m - sl_15m)

            if risk_15m <= 0 or risk_15m / entry_15m > 0.05:
                continue

            pnl_15m = _simulate_exit(
                'breakeven_trail', direction, entry_15m, sl_15m, risk_15m, ci_15m,
                c_highs_15m, c_lows_15m, c_closes_15m, atr_15m, sw_lows_15m, sw_highs_15m,
                timeout_15m, rr_ratio, be_rr, atr_mult, partial_pct, partial_rr
            )

            trades_15m.append({
                'pnl_r': pnl_15m,
                'risk_pct': risk_15m / entry_15m * 100,
                'entry': entry_15m,
            })

            # --- Entry B: 1m extreme+1 (sniper) ---
            t_15m = times_15m_epoch[ci_15m]
            i_start_1m = np.searchsorted(times_1m_epoch, t_15m)

            if i_start_1m + 20 >= n_1m:
                # Can't do 1m analysis, use 15m fallback
                trades_1m.append(trades_15m[-1])
                continue

            # Look at the 15 bars of 1m within this 15m candle
            h15 = c_highs_1m[i_start_1m:i_start_1m + 15]
            l15 = c_lows_1m[i_start_1m:i_start_1m + 15]
            c15 = c_closes_1m[i_start_1m:i_start_1m + 15]

            if len(h15) < 15:
                trades_1m.append(trades_15m[-1])
                continue

            # Find extreme bar
            if direction == 1:
                extreme_idx = np.argmin(l15)
                extreme_price = l15[extreme_idx]
                sl_1m = extreme_price * (1 - sl_buffer)
            else:
                extreme_idx = np.argmax(h15)
                extreme_price = h15[extreme_idx]
                sl_1m = extreme_price * (1 + sl_buffer)

            # Entry: close of bar after extreme (or last bar if extreme is bar 14)
            if extreme_idx + 1 < 15:
                entry_bar = extreme_idx + 1
            else:
                entry_bar = 14
            entry_1m = c15[entry_bar]

            # Calculate risk from 1m entry
            risk_1m = abs(entry_1m - sl_1m)
            if risk_1m <= 0 or risk_1m / entry_1m > 0.05:
                # Fallback to 15m
                trades_1m.append(trades_15m[-1])
                continue

            # Simulate exit on 1m candles from the entry point
            entry_1m_global = i_start_1m + entry_bar
            if entry_1m_global + timeout_1m >= n_1m:
                trades_1m.append(trades_15m[-1])
                continue

            pnl_1m = _simulate_exit(
                'breakeven_trail', direction, entry_1m, sl_1m, risk_1m, entry_1m_global,
                c_highs_1m, c_lows_1m, c_closes_1m, atr_1m, sw_lows_1m, sw_highs_1m,
                timeout_1m, rr_ratio, be_rr, atr_mult, partial_pct, partial_rr
            )

            trades_1m.append({
                'pnl_r': pnl_1m,
                'risk_pct': risk_1m / entry_1m * 100,
                'entry': entry_1m,
            })

        # === RESULTS ===
        print(f"\n{'='*70}", flush=True)
        print(f"RESULTS ({len(trades_15m)} trades)", flush=True)
        print(f"{'='*70}", flush=True)

        risk_per_trade = 10  # USD
        comm_rate = 0.0006
        months = 3053 / 30  # ~101 months of data

        for label, trades in [("15m CLOSE entry (baseline)", trades_15m),
                               ("1m SNIPER entry (extreme+1)", trades_1m)]:
            pnls = np.array([t['pnl_r'] for t in trades])
            risks = np.array([t['risk_pct'] for t in trades])
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]

            total_r = float(pnls.sum())
            wr = len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0
            pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 999
            avg_r = float(pnls.mean())
            avg_win = float(wins.mean()) if len(wins) > 0 else 0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0

            # Max consecutive losses
            max_cl = 0
            cur = 0
            for p in pnls:
                if p <= 0:
                    cur += 1
                    max_cl = max(max_cl, cur)
                else:
                    cur = 0

            # Commission
            avg_risk_pct = np.mean(risks) / 100
            avg_pos = risk_per_trade / avg_risk_pct if avg_risk_pct > 0 else 0
            comm_per = avg_pos * comm_rate
            gross = total_r * risk_per_trade
            total_comm = comm_per * len(trades)
            net = gross - total_comm
            net_mo = net / months

            print(f"\n--- {label} ---", flush=True)
            print(f"  Trades: {len(trades)}", flush=True)
            print(f"  Win rate: {wr:.1f}%", flush=True)
            print(f"  Profit Factor: {pf:.2f}", flush=True)
            print(f"  Total R: {total_r:.1f}", flush=True)
            print(f"  Avg R: {avg_r:+.3f}", flush=True)
            print(f"  Avg win: {avg_win:.2f}R, Avg loss: {avg_loss:.2f}R", flush=True)
            print(f"  Max consec losses: {max_cl}", flush=True)
            print(f"  --- $10 risk ---", flush=True)
            print(f"  Avg SL: {avg_risk_pct*100:.3f}%", flush=True)
            print(f"  Avg position: ${avg_pos:.0f}", flush=True)
            print(f"  Commission/trade: ${comm_per:.2f}", flush=True)
            print(f"  Gross profit: ${gross:.0f}", flush=True)
            print(f"  Total commission: ${total_comm:.0f}", flush=True)
            print(f"  NET profit: ${net:.0f}", flush=True)
            print(f"  NET/month: ${net_mo:.0f}", flush=True)

        # Direct comparison
        pnls_15m = np.array([t['pnl_r'] for t in trades_15m])
        pnls_1m = np.array([t['pnl_r'] for t in trades_1m])
        risks_15m = np.array([t['risk_pct'] for t in trades_15m])
        risks_1m = np.array([t['risk_pct'] for t in trades_1m])

        # Per-trade dollar profit comparison
        avg_risk_15m = np.mean(risks_15m) / 100
        avg_risk_1m = np.mean(risks_1m) / 100
        pos_15m = risk_per_trade / avg_risk_15m
        pos_1m = risk_per_trade / avg_risk_1m

        dollar_per_r_15m = risk_per_trade  # $10 per R
        dollar_per_r_1m = risk_per_trade   # Same $10 risk, but position is bigger

        print(f"\n{'='*70}", flush=True)
        print(f"POSITION SIZE COMPARISON", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  15m entry avg SL: {avg_risk_15m*100:.3f}% -> position ${pos_15m:.0f}", flush=True)
        print(f"  1m entry avg SL:  {avg_risk_1m*100:.3f}% -> position ${pos_1m:.0f}", flush=True)
        print(f"  Position multiplier: {pos_1m/pos_15m:.2f}x", flush=True)
        print(f"  Same R earned = {pos_1m/pos_15m:.2f}x more dollars", flush=True)


if __name__ == '__main__':
    run_sniper_backtest()
