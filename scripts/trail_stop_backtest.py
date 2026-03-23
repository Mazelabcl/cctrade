#!/usr/bin/env python
"""Trail Stop Backtest — test different exit strategies on existing trade entries.

Uses the same entries from IndividualLevelTrade but replaces the fixed-RR exit
with smarter exit strategies:

  A. Break-even: move SL to entry after reaching 1:1, then trail by swing lows
  B. ATR trail: SL follows price - 1.5*ATR
  C. Partial TP: close 50% at 2:1, trail the rest with break-even stop

Usage:
    python scripts/trail_stop_backtest.py --tf 4h
    python scripts/trail_stop_backtest.py --tf 1h --level-types Fractal_support Fractal_resistance
"""
import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def compute_atr(highs, lows, closes, period=14):
    """Compute ATR array."""
    n = len(highs)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
    atr = np.zeros(n)
    if n >= period:
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def find_swing_lows(lows, lookback=5):
    """Find swing low at each index: lowest low in last N candles."""
    n = len(lows)
    swing = np.copy(lows)
    for i in range(lookback, n):
        swing[i] = np.min(lows[max(0, i-lookback):i+1])
    return swing


def find_swing_highs(highs, lookback=5):
    """Find swing high at each index: highest high in last N candles."""
    n = len(highs)
    swing = np.copy(highs)
    for i in range(lookback, n):
        swing[i] = np.max(highs[max(0, i-lookback):i+1])
    return swing


def run_trail_backtest(session, exec_tf='4h', strategy='wick_rr_1.0',
                       level_types=None):
    """Run trail stop backtest on existing trade entries."""
    from app.services.level_trade_backtest_db import load_candles_db
    from sqlalchemy import text

    # Load candles
    candles = load_candles_db(session, timeframe=exec_tf)
    if candles.empty:
        print(f"No candles for {exec_tf}")
        return {}

    c_times = candles['open_time'].values
    c_highs = candles['high'].values.astype(np.float64)
    c_lows = candles['low'].values.astype(np.float64)
    c_closes = candles['close'].values.astype(np.float64)

    # Precompute ATR and swing levels
    atr = compute_atr(c_highs, c_lows, c_closes, period=14)
    swing_lows = find_swing_lows(c_lows, lookback=5)
    swing_highs = find_swing_highs(c_highs, lookback=5)

    time_to_idx = {}
    for idx, t in enumerate(c_times):
        time_to_idx[pd.Timestamp(t)] = idx

    # Load trade entries
    sql = """
        SELECT t.id, t.entry_time, t.entry_price, t.stop_loss, t.direction,
               t.exit_reason, t.pnl_pct, t.take_profit,
               b.level_type, b.level_source_timeframe
        FROM individual_level_trades t
        JOIN individual_level_backtests b ON t.backtest_id = b.id
        WHERE b.trade_execution_timeframe = :tf
        AND b.strategy_name = :strategy
        AND b.status = 'completed'
        AND t.exit_reason IN ('TP_HIT', 'SL_HIT')
    """
    rows = session.execute(text(sql), {'tf': exec_tf, 'strategy': strategy}).fetchall()

    if level_types:
        rows = [r for r in rows if r[8] in level_types]

    print(f"Testing {len(rows)} trades ({exec_tf})...", flush=True)

    # Results per strategy
    strategies = {
        'original_1:1': [],
        'breakeven_trail': [],
        'atr_trail_1.5': [],
        'atr_trail_2.0': [],
        'partial_50pct_2:1': [],
        'swing_trail': [],
    }

    t0 = time.time()
    for trade_idx, row in enumerate(rows):
        entry_time = pd.Timestamp(row[1])
        entry_price = float(row[2])
        stop_loss = float(row[3])
        direction = row[4]
        original_exit = row[5]
        original_pnl = float(row[6]) if row[6] else 0
        level_type = row[8]
        level_tf = row[9]

        risk = abs(entry_price - stop_loss)
        if risk == 0:
            continue

        idx = time_to_idx.get(entry_time)
        if idx is None or idx >= len(candles) - 1:
            continue

        # Original result (for comparison)
        strategies['original_1:1'].append({
            'pnl_r': 1.0 if original_exit == 'TP_HIT' else -1.0,
            'level_type': level_type,
            'direction': direction,
        })

        # --- Strategy A: Break-even + Swing Trail ---
        sl_be = stop_loss
        reached_1to1 = False
        result_be = {'pnl_r': 0, 'level_type': level_type, 'direction': direction}
        for j in range(idx + 1, min(idx + 500, len(candles))):
            if direction == 'LONG':
                # Check SL
                if c_lows[j] <= sl_be:
                    pnl = (sl_be - entry_price) / risk
                    result_be['pnl_r'] = round(pnl, 2)
                    break
                # Check if reached 1:1
                if c_highs[j] >= entry_price + risk:
                    reached_1to1 = True
                # Update SL
                if reached_1to1:
                    new_sl = max(entry_price, swing_lows[j])
                    sl_be = max(sl_be, new_sl)
            else:
                if c_highs[j] >= sl_be:
                    pnl = (entry_price - sl_be) / risk
                    result_be['pnl_r'] = round(pnl, 2)
                    break
                if c_lows[j] <= entry_price - risk:
                    reached_1to1 = True
                if reached_1to1:
                    new_sl = min(entry_price, swing_highs[j])
                    sl_be = min(sl_be, new_sl)
        else:
            # Timeout: close at last price
            pnl = (c_closes[min(idx+499, len(candles)-1)] - entry_price) / risk if direction == 'LONG' \
                else (entry_price - c_closes[min(idx+499, len(candles)-1)]) / risk
            result_be['pnl_r'] = round(pnl, 2)
        strategies['breakeven_trail'].append(result_be)

        # --- Strategy B: ATR Trail (1.5x) ---
        for atr_mult, strat_name in [(1.5, 'atr_trail_1.5'), (2.0, 'atr_trail_2.0')]:
            sl_atr = stop_loss
            result_atr = {'pnl_r': 0, 'level_type': level_type, 'direction': direction}
            for j in range(idx + 1, min(idx + 500, len(candles))):
                current_atr = atr[j] if atr[j] > 0 else risk
                if direction == 'LONG':
                    if c_lows[j] <= sl_atr:
                        result_atr['pnl_r'] = round((sl_atr - entry_price) / risk, 2)
                        break
                    trail = c_highs[j] - atr_mult * current_atr
                    sl_atr = max(sl_atr, trail)
                else:
                    if c_highs[j] >= sl_atr:
                        result_atr['pnl_r'] = round((entry_price - sl_atr) / risk, 2)
                        break
                    trail = c_lows[j] + atr_mult * current_atr
                    sl_atr = min(sl_atr, trail)
            else:
                pnl = (c_closes[min(idx+499, len(candles)-1)] - entry_price) / risk if direction == 'LONG' \
                    else (entry_price - c_closes[min(idx+499, len(candles)-1)]) / risk
                result_atr['pnl_r'] = round(pnl, 2)
            strategies[strat_name].append(result_atr)

        # --- Strategy C: Partial 50% at 2:1, trail rest with break-even ---
        sl_p = stop_loss
        partial_taken = False
        partial_pnl = 0.0
        result_p = {'pnl_r': 0, 'level_type': level_type, 'direction': direction}
        for j in range(idx + 1, min(idx + 500, len(candles))):
            if direction == 'LONG':
                if c_lows[j] <= sl_p:
                    remaining_pnl = (sl_p - entry_price) / risk
                    if partial_taken:
                        result_p['pnl_r'] = round(partial_pnl + remaining_pnl * 0.5, 2)
                    else:
                        result_p['pnl_r'] = round(remaining_pnl, 2)
                    break
                if not partial_taken and c_highs[j] >= entry_price + 2 * risk:
                    partial_taken = True
                    partial_pnl = 2.0 * 0.5  # 50% closed at 2:1
                    sl_p = max(sl_p, entry_price)  # move to break-even
                elif partial_taken:
                    sl_p = max(sl_p, swing_lows[j])
            else:
                if c_highs[j] >= sl_p:
                    remaining_pnl = (entry_price - sl_p) / risk
                    if partial_taken:
                        result_p['pnl_r'] = round(partial_pnl + remaining_pnl * 0.5, 2)
                    else:
                        result_p['pnl_r'] = round(remaining_pnl, 2)
                    break
                if not partial_taken and c_lows[j] <= entry_price - 2 * risk:
                    partial_taken = True
                    partial_pnl = 2.0 * 0.5
                    sl_p = min(sl_p, entry_price)
                elif partial_taken:
                    sl_p = min(sl_p, swing_highs[j])
        else:
            pnl = (c_closes[min(idx+499, len(candles)-1)] - entry_price) / risk if direction == 'LONG' \
                else (entry_price - c_closes[min(idx+499, len(candles)-1)]) / risk
            if partial_taken:
                result_p['pnl_r'] = round(partial_pnl + pnl * 0.5, 2)
            else:
                result_p['pnl_r'] = round(pnl, 2)
        strategies['partial_50pct_2:1'].append(result_p)

        # --- Strategy D: Pure Swing Trail (no break-even requirement) ---
        sl_sw = stop_loss
        result_sw = {'pnl_r': 0, 'level_type': level_type, 'direction': direction}
        for j in range(idx + 1, min(idx + 500, len(candles))):
            if direction == 'LONG':
                if c_lows[j] <= sl_sw:
                    result_sw['pnl_r'] = round((sl_sw - entry_price) / risk, 2)
                    break
                # Trail from candle 3 onwards (give it room)
                if j > idx + 3:
                    sl_sw = max(sl_sw, swing_lows[j])
            else:
                if c_highs[j] >= sl_sw:
                    result_sw['pnl_r'] = round((entry_price - sl_sw) / risk, 2)
                    break
                if j > idx + 3:
                    sl_sw = min(sl_sw, swing_highs[j])
        else:
            pnl = (c_closes[min(idx+499, len(candles)-1)] - entry_price) / risk if direction == 'LONG' \
                else (entry_price - c_closes[min(idx+499, len(candles)-1)]) / risk
            result_sw['pnl_r'] = round(pnl, 2)
        strategies['swing_trail'].append(result_sw)

        if (trade_idx + 1) % 5000 == 0:
            print(f"  {trade_idx+1}/{len(rows)} trades ({time.time()-t0:.0f}s)...", flush=True)

    print(f"Done in {time.time()-t0:.0f}s", flush=True)
    return strategies


def print_results(strategies, level_types_filter=None):
    """Print comparison of all strategies."""
    print("\n" + "=" * 80)
    print("TRAIL STOP BACKTEST RESULTS")
    if level_types_filter:
        print(f"Filtered to: {', '.join(level_types_filter)}")
    print("=" * 80)

    print(f"\n{'Strategy':25s} {'Trades':>7s} {'WR%':>6s} {'Avg R':>7s} {'Med R':>7s} "
          f"{'Total R':>8s} {'PF':>6s}")
    print("-" * 75)

    for name, trades in strategies.items():
        if not trades:
            continue
        df = pd.DataFrame(trades)
        wins = df[df['pnl_r'] > 0]
        losses = df[df['pnl_r'] <= 0]
        wr = len(wins) / len(df) * 100 if len(df) > 0 else 0
        avg_r = df['pnl_r'].mean()
        med_r = df['pnl_r'].median()
        total_r = df['pnl_r'].sum()
        gross_wins = wins['pnl_r'].sum() if not wins.empty else 0
        gross_losses = abs(losses['pnl_r'].sum()) if not losses.empty else 0.001
        pf = gross_wins / gross_losses

        print(f"{name:25s} {len(df):>7d} {wr:>5.1f}% {avg_r:>6.2f}R {med_r:>6.2f}R "
              f"{total_r:>7.0f}R {pf:>6.2f}")

    # By level type for best strategy
    print("\n--- By Level Type (breakeven_trail) ---")
    if 'breakeven_trail' in strategies and strategies['breakeven_trail']:
        df = pd.DataFrame(strategies['breakeven_trail'])
        by_type = df.groupby('level_type').agg(
            trades=('pnl_r', 'count'),
            wr=('pnl_r', lambda x: (x > 0).mean() * 100),
            avg_r=('pnl_r', 'mean'),
            total_r=('pnl_r', 'sum'),
        ).sort_values('avg_r', ascending=False)

        print(f"{'Type':25s} {'Trades':>7s} {'WR%':>6s} {'Avg R':>7s} {'Total R':>8s}")
        print("-" * 55)
        for lt, row in by_type.iterrows():
            print(f"{lt:25s} {row['trades']:>7.0f} {row['wr']:>5.1f}% "
                  f"{row['avg_r']:>6.2f}R {row['total_r']:>7.0f}R")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trail Stop Backtest')
    parser.add_argument('--tf', default='4h', help='Execution timeframe')
    parser.add_argument('--level-types', nargs='+', default=None,
                        help='Filter to specific level types')
    parser.add_argument('--output', default=None, help='Output JSON file')
    args = parser.parse_args()

    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        strategies = run_trail_backtest(
            db.session,
            exec_tf=args.tf,
            level_types=args.level_types,
        )

        # Print all trades
        print_results(strategies, args.level_types)

        # Print Fractals-only view
        if not args.level_types:
            print("\n\n" + "=" * 80)
            print("FRACTAL TRADES ONLY")
            print("=" * 80)
            fractal_strats = {}
            for name, trades in strategies.items():
                fractal_strats[name] = [t for t in trades
                                         if t['level_type'] in ('Fractal_support', 'Fractal_resistance')]
            print_results(fractal_strats, ['Fractal_support', 'Fractal_resistance'])

        if args.output:
            # Save serializable version
            out = {name: trades for name, trades in strategies.items()}
            with open(args.output, 'w') as f:
                json.dump(out, f, default=str)
            print(f"\nSaved to {args.output}")
