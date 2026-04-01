#!/usr/bin/env python
"""Forward Test — simulate the validated strategy on unseen 2026 data.

Runs the 15m confluence scalper on data from 2026-01-01 to present.
This data was NOT used in any AutoResearch experiment.

Usage:
    python scripts/forward_test.py
    python scripts/forward_test.py --risk 50
    python scripts/forward_test.py --risk 300
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from datetime import datetime


def run_forward_test(risk_per_trade=10, start_date='2026-01-01'):
    from app import create_app
    from app.extensions import db
    from sqlalchemy import text
    from scripts.autoresearch.confluence_scalper import (
        load_and_cache_data, find_touch_entries, score_and_deduplicate,
        _simulate_exit, _get_swings,
    )
    import copy

    app = create_app()
    with app.app_context():
        print("=" * 70)
        print(f"FORWARD TEST — Unseen Data (>= {start_date})")
        print(f"Risk per trade: ${risk_per_trade}")
        print("Strategy: 15m confluence scalper (validated config)")
        print("=" * 70)

        # Load all 15m data
        cache = {}
        data = load_and_cache_data(db.session, cache, timeframe='15m')
        c_times = data['c_times']
        c_highs = data['c_highs']
        c_lows = data['c_lows']
        c_closes = data['c_closes']
        atr = data['atr']
        n_candles = data['n_candles']

        # Find the forward test boundary
        times_epoch = c_times.astype('datetime64[s]').astype(np.int64)
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        fwd_start = np.searchsorted(times_epoch, start_ts)

        print(f"\nData: {n_candles:,} total candles")
        print(f"Forward test starts at candle {fwd_start} ({start_date})")
        print(f"Forward test candles: {n_candles - fwd_start:,}")

        # Validated config
        config = {
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

        # Get entries
        print("\nFinding touch entries...")
        entries = find_touch_entries(data, config)
        entries, scores = score_and_deduplicate(entries, data, config)

        # Filter to forward test period only
        fwd_mask = entries[:, 0] >= fwd_start
        entries_fwd = entries[fwd_mask]
        scores_fwd = scores[fwd_mask]
        print(f"Entries in forward period: {len(entries_fwd)}")

        # Simulate trades
        exit_cfg = config['exit']
        sl_buffer = exit_cfg['sl_buffer_pct']
        swing_lb = exit_cfg['swing_lookback']
        timeout = exit_cfg['timeout_candles']
        be_rr = exit_cfg['breakeven_at_rr']
        strategy = exit_cfg['strategy']

        sw_lows, sw_highs = _get_swings(data, swing_lb)

        trades = []
        for i in range(len(entries_fwd)):
            ci = int(entries_fwd[i, 0])
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

            pnl = _simulate_exit(
                strategy, direction, entry_price, sl, risk, ci,
                c_highs, c_lows, c_closes, atr, sw_lows, sw_highs,
                timeout, 1.5, be_rr, 2.0, 0.5, 2.0
            )

            # Get timestamp
            t = c_times[ci]
            if hasattr(t, 'isoformat'):
                time_str = str(t)[:16]
            else:
                time_str = str(np.datetime_as_string(t, unit='m'))

            trades.append({
                'time': time_str,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'entry': round(entry_price, 2),
                'sl': round(sl, 2),
                'risk_pct': round(risk / entry_price * 100, 3),
                'pnl_r': round(pnl, 2),
                'pnl_usd': round(pnl * risk_per_trade, 2),
            })

        if not trades:
            print("No trades in forward period!")
            return

        # Results
        n = len(trades)
        pnls = np.array([t['pnl_r'] for t in trades])
        pnls_usd = np.array([t['pnl_usd'] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        total_r = float(pnls.sum())
        wr = len(wins) / n * 100
        pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 999

        # Commission
        avg_risk_pct = np.mean([t['risk_pct'] for t in trades]) / 100
        avg_pos = risk_per_trade / avg_risk_pct if avg_risk_pct > 0 else 0
        comm_per = avg_pos * 0.0006
        gross = total_r * risk_per_trade
        total_comm = comm_per * n
        net = gross - total_comm

        # Period
        first_time = trades[0]['time']
        last_time = trades[-1]['time']

        # Max consecutive losses
        max_cl = 0
        cur = 0
        for p in pnls:
            if p <= 0:
                cur += 1
                max_cl = max(max_cl, cur)
            else:
                cur = 0

        # Equity curve
        cumulative = np.cumsum(pnls_usd)

        print(f"\n{'='*70}")
        print(f"FORWARD TEST RESULTS ({first_time} to {last_time})")
        print(f"{'='*70}")
        print(f"  Trades: {n}")
        print(f"  Win rate: {wr:.1f}%")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Total R: {total_r:+.1f}")
        print(f"  Avg R: {pnls.mean():+.3f}")
        if len(wins) > 0:
            print(f"  Avg win: {wins.mean():.2f}R (${wins.mean()*risk_per_trade:.2f})")
        if len(losses) > 0:
            print(f"  Avg loss: {losses.mean():.2f}R (${losses.mean()*risk_per_trade:.2f})")
        print(f"  Max consecutive losses: {max_cl}")
        print(f"")
        print(f"  --- ${risk_per_trade} risk per trade ---")
        print(f"  Avg position size: ${avg_pos:.0f}")
        print(f"  Commission/trade: ${comm_per:.2f}")
        print(f"  Gross profit: ${gross:.2f}")
        print(f"  Total commission: ${total_comm:.2f}")
        print(f"  NET profit: ${net:.2f}")
        print(f"  Peak equity: ${cumulative.max():.2f}")
        print(f"  Max drawdown: ${cumulative.min():.2f}")

        # Trade log
        print(f"\n--- TRADE LOG ---")
        print(f"  {'Time':>20s} | {'Dir':>5s} | {'Entry':>10s} | {'SL':>10s} | {'P&L R':>7s} | {'P&L $':>8s} | {'Cumul $':>8s}")
        print(f"  {'-'*80}")
        cum = 0
        for t in trades:
            cum += t['pnl_usd']
            marker = ' **' if t['pnl_r'] > 2 else (' !!' if t['pnl_r'] < -0.9 else '')
            print(f"  {t['time']:>20s} | {t['direction']:>5s} | {t['entry']:>10.2f} | "
                  f"{t['sl']:>10.2f} | {t['pnl_r']:>+6.2f} | {t['pnl_usd']:>+7.2f} | "
                  f"{cum:>+7.2f}{marker}")

        # Compare with backtest expectation
        print(f"\n--- COMPARISON vs BACKTEST ---")
        print(f"  Backtest (2017-2025): PF 2.23, WR 35.9%, $156/mo")
        months = max(1, (n_candles - fwd_start) / (4 * 24 * 30))
        print(f"  Forward ({start_date}-now, {months:.1f} months): PF {pf:.2f}, WR {wr:.1f}%, ${net/months:.0f}/mo")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Forward Test')
    parser.add_argument('--risk', type=float, default=10, help='USD risk per trade')
    parser.add_argument('--start', type=str, default='2026-01-01', help='Start date')
    args = parser.parse_args()
    run_forward_test(risk_per_trade=args.risk, start_date=args.start)
