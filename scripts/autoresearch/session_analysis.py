#!/usr/bin/env python
"""Session Analysis — Test scalper performance by trading session.

Quick analysis: runs the best scalper config on each session independently
to see where the edge concentrates.

Usage:
    python scripts/autoresearch/session_analysis.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import copy
import numpy as np
from scripts.autoresearch.confluence_scalper import evaluate


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

SESSIONS = ['all', 'us', 'eu', 'asia', 'us_eu']
TIMEFRAMES = ['15m', '1h', '4h']


def run_session_analysis():
    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        print("=" * 70, flush=True)
        print("SESSION ANALYSIS — Edge by Trading Session", flush=True)
        print("=" * 70, flush=True)

        _cache = {}

        for tf in TIMEFRAMES:
            print(f"\n{'='*70}", flush=True)
            print(f"TIMEFRAME: {tf}", flush=True)
            print(f"{'='*70}", flush=True)

            results = {}
            for session in SESSIONS:
                config = copy.deepcopy(BEST_15M_CONFIG)
                config['session_filter'] = session

                # Adjust config for timeframe
                if tf in ('1h', '4h'):
                    config['exit']['timeout_candles'] = 50

                metrics = evaluate(config, db.session, _cache, timeframe=tf)

                if metrics.get('error'):
                    print(f"  {session:10s} | ERROR: {metrics['error']}", flush=True)
                    continue

                # Commission analysis
                risk = 10
                trades = metrics['total_trades']
                total_r = metrics['total_r']
                avg_r = total_r / trades if trades > 0 else 0
                pf = metrics['profit_factor']
                wr = metrics['win_rate']

                # Estimate SL % from timeframe
                sl_pct = {'15m': 0.0045, '1h': 0.006, '4h': 0.012}.get(tf, 0.005)
                pos_size = risk / sl_pct
                comm_per_trade = pos_size * 0.0006
                gross = total_r * risk
                total_comm = comm_per_trade * trades
                net = gross - total_comm
                days = {'15m': 3053, '1h': 3053, '4h': 3053}.get(tf, 3053)
                months = days / 30

                results[session] = {
                    'trades': trades, 'pf': pf, 'wr': wr, 'total_r': total_r,
                    'net': net, 'net_month': net / months,
                    'trades_day': trades / days,
                }

                print(f"  {session:10s} | PF {pf:5.2f} | WR {wr:5.1f}% | "
                      f"trades {trades:5d} ({trades/days:.1f}/d) | "
                      f"R {total_r:8.1f} | net ${net:8.0f} (${net/months:.0f}/mo)",
                      flush=True)

            # Summary
            if results:
                best_session = max(results.items(), key=lambda x: x[1]['pf'])
                print(f"\n  BEST: {best_session[0]} (PF {best_session[1]['pf']:.2f})", flush=True)


if __name__ == '__main__':
    run_session_analysis()
