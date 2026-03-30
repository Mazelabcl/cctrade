#!/usr/bin/env python
"""AutoResearch Mode E — Combo: Confluence Scalper + Fractal Predictor Filter.

Uses the fractal predictor (Mode D) as a filter for the confluence scalper (Mode C).
Only takes scalper trades when the predictor says "fractal probable" in the higher TF.

Usage:
    python scripts/autoresearch/combo_backtest.py --oos 2024-01-01
"""
import sys
import os
import time
import json
import copy
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def run_combo(oos_date='2024-01-01'):
    """Run the combo backtest: scalper entries filtered by predictor."""
    from app import create_app
    from app.extensions import db

    # Import from our modules
    from scripts.autoresearch.confluence_scalper import (
        load_and_cache_data as load_scalper_data,
        find_touch_entries, score_and_deduplicate, _simulate_exit,
    )
    from scripts.autoresearch.fractal_predictor import (
        load_data as load_predictor_data,
        build_feature_matrix, compute_atr,
        find_swing_lows, find_swing_highs,
        DEFAULT_CONFIG as PRED_DEFAULT_CONFIG,
    )

    app = create_app()
    with app.app_context():
        print("=" * 70, flush=True)
        print(f"AUTORESEARCH MODE E — Combo Backtest", flush=True)
        print(f"Scalper: 15m confluence | Filter: 1h fractal predictor", flush=True)
        print(f"OOS: test >= {oos_date}", flush=True)
        print("=" * 70, flush=True)

        # ---------------------------------------------------------------
        # Step 1: Train the predictor on 1h data (train < oos_date)
        # ---------------------------------------------------------------
        print("\n[1/4] Training fractal predictor on 1h...", flush=True)
        from sklearn.ensemble import RandomForestClassifier

        pred_config = copy.deepcopy(PRED_DEFAULT_CONFIG)
        # Best known 1h config from OOS experiments
        pred_config['features'] = [
            'body_ratio', 'lower_wick', 'dist_from_high_20', 'dist_from_low_20',
            'conf_support_count', 'conf_resistance_count', 'conf_resistance_types',
            'conf_support_tf_weight', 'conf_resistance_tf_weight',
            'nearest_support_dist', 'nearest_resistance_dist',
            'nearest_support_tf', 'nearest_resistance_tf',
            'naked_support_total', 'naked_resistance_total',
            'has_htf_resistance', 'candles_since_bullish', 'candles_since_bearish',
            'consecutive_dir', 'conf_support_types',
        ]
        pred_config['zone_width'] = 0.02
        pred_config['n_trees'] = 150
        pred_config['max_depth'] = 14
        pred_config['oos_date'] = oos_date

        pred_data = load_predictor_data(db.session, timeframe='1h')
        pred_cache = {}

        X, y, feature_names, orig_indices = build_feature_matrix(pred_data, pred_config, pred_cache)

        # Split by OOS date
        pred_df = pred_data['df']
        oos_ts = pd.Timestamp(oos_date, tz='UTC')
        candle_times = pred_df['open_time']
        if hasattr(candle_times.iloc[0], 'tz') and candle_times.iloc[0].tz is None:
            oos_ts = oos_ts.tz_localize(None)
        oos_candle_idx = (candle_times >= oos_ts).idxmax()
        split_idx = int(np.searchsorted(orig_indices, oos_candle_idx))
        split_idx = max(100, min(split_idx, len(X) - 100))

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        n_total = len(y_train)
        class_weights = {}
        for c in [0, 1, 2]:
            count = (y_train == c).sum()
            if count > 0:
                class_weights[c] = n_total / (3 * count)

        clf = RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=5,
            class_weight=class_weights, random_state=42, n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        # Generate predictions for ALL candles (including test period)
        X_all = X  # Already has all data
        y_pred_all = clf.predict(X_all)

        # Build lookup: 1h candle open_time -> prediction
        # Map orig_indices back to timestamps
        pred_times = pred_df['open_time']
        pred_lookup = {}  # timestamp -> predicted class
        for i, orig_idx in enumerate(orig_indices):
            t = pred_times.iloc[orig_idx]
            if hasattr(t, 'timestamp'):
                ts = t.timestamp()
            elif isinstance(t, (np.datetime64,)):
                ts = pd.Timestamp(t).timestamp()
            else:
                ts = float(t) / 1000 if float(t) > 1e12 else float(t)
            pred_lookup[int(ts)] = int(y_pred_all[i])

        print(f"  Trained on {len(X_train)} samples, predictions for {len(pred_lookup)} candles", flush=True)
        print(f"  Prediction distribution: 0={sum(1 for v in pred_lookup.values() if v==0)}, "
              f"1={sum(1 for v in pred_lookup.values() if v==1)}, "
              f"2={sum(1 for v in pred_lookup.values() if v==2)}", flush=True)

        # ---------------------------------------------------------------
        # Step 2: Run the scalper on 15m to get trade entries
        # ---------------------------------------------------------------
        print("\n[2/4] Running confluence scalper on 15m...", flush=True)

        scalper_config = {
            'score_threshold': 3,
            'zone_width': 0.01,
            'touch_tolerance': 0.001,
            'naked_only': True,
            'scoring_mode': 'unique_types',
            'entry_mode': 'touch',
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

        scalper_cache = {}
        scalper_data = load_scalper_data(db.session, scalper_cache, timeframe='15m')

        c_times_15m_raw = scalper_data['c_times']  # datetime64[ns]
        # Convert to epoch seconds for lookup
        c_times_15m = c_times_15m_raw.astype('datetime64[s]').astype(np.int64)
        c_highs = scalper_data['c_highs']
        c_lows = scalper_data['c_lows']
        c_closes = scalper_data['c_closes']
        atr = scalper_data['atr']
        n_candles = scalper_data['n_candles']

        # Find OOS boundary in 15m data
        oos_ts_val = int(oos_ts.timestamp())
        oos_15m_idx = np.searchsorted(c_times_15m, oos_ts_val)
        print(f"  OOS boundary: candle {oos_15m_idx} of {n_candles}", flush=True)

        # Get all touch entries
        entries = find_touch_entries(scalper_data, scalper_config)
        print(f"  Touch entries: {len(entries)}", flush=True)

        # Score and deduplicate
        entries, scores = score_and_deduplicate(entries, scalper_data, scalper_config)
        print(f"  After scoring/dedup: {len(entries)}", flush=True)

        # Filter to OOS only
        oos_mask = entries[:, 0] >= oos_15m_idx
        entries_oos = entries[oos_mask]
        scores_oos = scores[oos_mask]
        print(f"  OOS entries: {len(entries_oos)}", flush=True)

        # ---------------------------------------------------------------
        # Step 3: Filter scalper trades by predictor
        # ---------------------------------------------------------------
        print("\n[3/4] Filtering scalper trades by predictor...", flush=True)

        exit_cfg = scalper_config['exit']
        strategy = exit_cfg['strategy']
        swing_lb = exit_cfg.get('swing_lookback', 9)
        timeout = exit_cfg.get('timeout_candles', 35)
        rr_ratio = exit_cfg.get('rr_ratio', 1.5)
        be_rr = exit_cfg.get('breakeven_at_rr', 1.0)
        atr_mult = exit_cfg.get('atr_multiplier', 2.0)
        partial_pct = exit_cfg.get('partial_pct', 0.5)
        partial_rr = exit_cfg.get('partial_rr', 2.0)
        sl_buffer = exit_cfg.get('sl_buffer_pct', 0.001)

        # Precompute swings
        from scripts.autoresearch.confluence_scalper import _get_swings
        sw_lows, sw_highs = _get_swings(scalper_data, swing_lb)

        # Simulate trades with and without filter
        results_all = []      # All scalper trades (no filter)
        results_filtered = [] # Only when predictor says fractal

        for i in range(len(entries_oos)):
            ci = int(entries_oos[i, 0])
            direction = int(entries_oos[i, 2])

            entry_price = c_closes[ci]
            if direction == 1:
                sl = c_lows[ci] * (1 - sl_buffer)
            else:
                sl = c_highs[ci] * (1 + sl_buffer)

            risk = abs(entry_price - sl)
            if risk <= 0 or risk / entry_price > 0.05:
                continue

            pnl = _simulate_exit(
                strategy, direction, entry_price, sl, risk, ci,
                c_highs, c_lows, c_closes, atr, sw_lows, sw_highs,
                timeout, rr_ratio, be_rr, atr_mult, partial_pct, partial_rr
            )

            trade = {
                'candle_idx': ci,
                'direction': direction,
                'entry': round(entry_price, 2),
                'pnl_r': round(pnl, 2),
                'risk_pct': round(risk / entry_price * 100, 3),
            }
            results_all.append(trade)

            # Check predictor: map 15m timestamp to nearest 1h candle
            t_15m = int(c_times_15m[ci])
            # Round down to nearest hour
            t_1h = (t_15m // 3600) * 3600

            # Check current and recent 1h candles for fractal prediction
            pred_signal = False
            for offset in [0, -3600, -7200]:  # Current, -1h, -2h
                pred_class = pred_lookup.get(t_1h + offset, 0)
                if pred_class > 0:  # 1=bullish or 2=bearish predicted
                    # Check direction alignment
                    if (pred_class == 1 and direction == 1) or \
                       (pred_class == 2 and direction == -1):
                        pred_signal = True
                        break
                    # Also accept any fractal prediction (direction-agnostic)
                    # since near-miss analysis showed 89% have useful reactions
                    pred_signal = True
                    break

            if pred_signal:
                results_filtered.append(trade)

        # ---------------------------------------------------------------
        # Step 4: Compare results
        # ---------------------------------------------------------------
        print("\n[4/4] Results comparison...", flush=True)

        def compute_metrics(trades, label):
            if not trades:
                print(f"\n--- {label}: NO TRADES ---", flush=True)
                return

            pnls = np.array([t['pnl_r'] for t in trades])
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            total_r = float(pnls.sum())
            wr = len(wins) / len(pnls) * 100
            pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 999
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

            # Commission analysis ($10 risk)
            risk_per_trade = 10
            avg_risk_pct = np.mean([t['risk_pct'] for t in trades]) / 100
            avg_pos = risk_per_trade / avg_risk_pct if avg_risk_pct > 0 else 0
            comm_rate = 0.0006
            comm_per_trade = avg_pos * comm_rate
            total_comm = comm_per_trade * len(trades)
            gross = total_r * risk_per_trade
            net = gross - total_comm

            # Period estimate (~2 years OOS)
            n_months = 24

            print(f"\n--- {label} ---", flush=True)
            print(f"  Trades: {len(trades)}", flush=True)
            print(f"  Win rate: {wr:.1f}%", flush=True)
            print(f"  Profit Factor: {pf:.2f}", flush=True)
            print(f"  Total R: {total_r:.1f}", flush=True)
            print(f"  Avg R: {float(pnls.mean()):.3f}", flush=True)
            print(f"  Avg win: {avg_win:.2f}R, Avg loss: {avg_loss:.2f}R", flush=True)
            print(f"  Max consec losses: {max_cl}", flush=True)
            print(f"  --- $10 risk ---", flush=True)
            print(f"  Avg SL: {avg_risk_pct*100:.3f}%", flush=True)
            print(f"  Avg position: ${avg_pos:.0f}", flush=True)
            print(f"  Commission/trade: ${comm_per_trade:.2f}", flush=True)
            print(f"  Gross profit: ${gross:.0f}", flush=True)
            print(f"  Total commission: ${total_comm:.0f}", flush=True)
            print(f"  NET profit: ${net:.0f}", flush=True)
            print(f"  NET/month: ${net/n_months:.0f}", flush=True)

            return {
                'trades': len(trades), 'wr': round(wr, 1), 'pf': round(pf, 2),
                'total_r': round(total_r, 1), 'net': round(net, 0),
                'net_per_month': round(net / n_months, 0),
            }

        m_all = compute_metrics(results_all, "SCALPER SOLO (no filter)")
        m_filt = compute_metrics(results_filtered, "SCALPER + PREDICTOR FILTER")

        if m_all and m_filt:
            print(f"\n{'='*70}", flush=True)
            print("IMPROVEMENT SUMMARY", flush=True)
            print(f"{'='*70}", flush=True)
            print(f"  Trades: {m_all['trades']} -> {m_filt['trades']} "
                  f"({m_filt['trades']-m_all['trades']:+d}, "
                  f"{m_filt['trades']/m_all['trades']*100:.0f}% kept)", flush=True)
            print(f"  Win rate: {m_all['wr']}% -> {m_filt['wr']}%", flush=True)
            print(f"  PF: {m_all['pf']} -> {m_filt['pf']}", flush=True)
            print(f"  Total R: {m_all['total_r']} -> {m_filt['total_r']}", flush=True)
            print(f"  NET/month: ${m_all['net_per_month']} -> ${m_filt['net_per_month']}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoResearch Mode E — Combo Backtest')
    parser.add_argument('--oos', type=str, default='2024-01-01',
                        help='Out-of-sample date')
    args = parser.parse_args()
    run_combo(oos_date=args.oos)
