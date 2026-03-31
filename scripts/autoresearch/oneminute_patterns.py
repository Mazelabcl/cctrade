#!/usr/bin/env python
"""1-Minute Pattern Analysis — What happens inside each 15m touch?

For each 15m scalper entry, analyzes the 15 bars of 1m to classify:
1. SFP (strict): breaks prior extreme, closes back
2. Volume spike: bar with 2x+ avg volume
3. Pin bar: wick > 66% of range on rejection side
4. Engulfing: bar that engulfs the previous bar
5. Clean reversal: just reverses without any special pattern

Then cross-references: SFP+volume, SFP+pin, volume+pin, etc.
And measures OUTCOME: how much did price move favorably after each pattern?

Usage:
    python scripts/autoresearch/oneminute_patterns.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import copy
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
        'swing_lookback': 10,
        'atr_multiplier': 2.0,
        'breakeven_at_rr': 2.75,
        'partial_pct': 0.5,
        'partial_rr': 2.0,
        'timeout_candles': 45,
        'sl_buffer_pct': 0.001,
    },
}


def run_pattern_analysis():
    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        print("=" * 70, flush=True)
        print("1-MINUTE PATTERN ANALYSIS", flush=True)
        print("What patterns exist inside 15m touches? How do they combine?", flush=True)
        print("=" * 70, flush=True)

        # Load data
        cache_15m = {}
        data_15m = load_and_cache_data(db.session, cache_15m, timeframe='15m')
        c_times_15m = data_15m['c_times']
        c_closes_15m = data_15m['c_closes']
        c_highs_15m = data_15m['c_highs']
        c_lows_15m = data_15m['c_lows']

        cache_1m = {}
        data_1m = load_and_cache_data(db.session, cache_1m, timeframe='1m')
        c_times_1m = data_1m['c_times']
        c_closes_1m = data_1m['c_closes']
        c_highs_1m = data_1m['c_highs']
        c_lows_1m = data_1m['c_lows']
        c_opens_1m = data_1m['c_opens'] if 'c_opens' in data_1m else c_closes_1m
        c_vols_1m = data_1m['c_vols']
        atr_1m = data_1m['atr']
        n_1m = data_1m['n_candles']

        # Precompute 1m swings for exit simulation
        from scripts.autoresearch.sniper_backtest import find_swing_lows, find_swing_highs
        sw_lows_1m = find_swing_lows(c_lows_1m, lookback=150)
        sw_highs_1m = find_swing_highs(c_highs_1m, lookback=150)

        # Get 15m entries
        config = copy.deepcopy(BEST_15M_CONFIG)
        entries = find_touch_entries(data_15m, config)
        entries, scores = score_and_deduplicate(entries, data_15m, config)
        print(f"15m entries: {len(entries)}", flush=True)

        times_15m_epoch = c_times_15m.astype('datetime64[s]').astype(np.int64)
        times_1m_epoch = c_times_1m.astype('datetime64[s]').astype(np.int64)

        sl_buffer = 0.001

        # Collectors for each trade
        trades = []

        for idx in range(len(entries)):
            ci_15m = int(entries[idx, 0])
            direction = int(entries[idx, 2])

            t_15m = times_15m_epoch[ci_15m]
            i_start = np.searchsorted(times_1m_epoch, t_15m)

            if i_start + 30 >= n_1m:
                continue

            # 15 bars within the 15m candle + 15 bars after
            h = c_highs_1m[i_start:i_start + 30]
            l = c_lows_1m[i_start:i_start + 30]
            c = c_closes_1m[i_start:i_start + 30]
            o = c_opens_1m[i_start:i_start + 30] if 'c_opens' in data_1m else c
            v = c_vols_1m[i_start:i_start + 30]

            if len(h) < 30:
                continue

            # The 15 bars of the touch candle
            h15 = h[:15]
            l15 = l[:15]
            c15 = c[:15]
            o15 = o[:15]
            v15 = v[:15]

            # Find extreme
            if direction == 1:
                extreme_idx = int(np.argmin(l15))
                extreme_price = l15[extreme_idx]
                sl = extreme_price * (1 - sl_buffer)
            else:
                extreme_idx = int(np.argmax(h15))
                extreme_price = h15[extreme_idx]
                sl = extreme_price * (1 + sl_buffer)

            # === PATTERN DETECTION (within 15 bars) ===

            # 1. SFP (strict): bar 5-14 breaks first-5 extreme, closes back
            has_sfp = False
            sfp_bar = -1
            if direction == 1:
                ref_low = np.min(l15[:5])
                for j in range(5, 15):
                    if l15[j] < ref_low and c15[j] > ref_low:
                        has_sfp = True
                        sfp_bar = j
                        break
            else:
                ref_high = np.max(h15[:5])
                for j in range(5, 15):
                    if h15[j] > ref_high and c15[j] < ref_high:
                        has_sfp = True
                        sfp_bar = j
                        break

            # 2. Volume spike: any bar with volume > 2x average of first 5
            avg_vol = np.mean(v15[:5]) if np.mean(v15[:5]) > 0 else 1
            vol_spikes = v15 / avg_vol
            has_vol_spike = bool(np.any(vol_spikes > 2.0))
            vol_spike_bar = int(np.argmax(vol_spikes)) if has_vol_spike else -1
            max_vol_ratio = float(np.max(vol_spikes))
            # Is vol spike near extreme?
            vol_at_extreme = bool(abs(vol_spike_bar - extreme_idx) <= 2) if has_vol_spike else False

            # 3. Pin bar: any bar near extreme with wick > 66% of range on rejection side
            has_pin = False
            pin_bar = -1
            for j in range(max(0, extreme_idx - 2), min(15, extreme_idx + 3)):
                rng = h15[j] - l15[j]
                if rng <= 0:
                    continue
                if direction == 1:
                    lower_wick = min(o15[j], c15[j]) - l15[j]
                    if lower_wick / rng > 0.66:
                        has_pin = True
                        pin_bar = j
                        break
                else:
                    upper_wick = h15[j] - max(o15[j], c15[j])
                    if upper_wick / rng > 0.66:
                        has_pin = True
                        pin_bar = j
                        break

            # 4. Engulfing: bar after extreme engulfs the extreme bar
            has_engulfing = False
            if extreme_idx + 1 < 15:
                j = extreme_idx + 1
                if direction == 1:
                    # Bullish engulfing: next bar's body engulfs extreme's body
                    if c15[j] > o15[j] and c15[j] > max(o15[extreme_idx], c15[extreme_idx]):
                        has_engulfing = True
                else:
                    if c15[j] < o15[j] and c15[j] < min(o15[extreme_idx], c15[extreme_idx]):
                        has_engulfing = True

            # 5. Immediate reversal: bar after extreme already goes our way
            has_reversal = False
            if extreme_idx + 1 < 15:
                if direction == 1:
                    has_reversal = c15[extreme_idx + 1] > extreme_price
                else:
                    has_reversal = c15[extreme_idx + 1] < extreme_price

            # === OUTCOME: simulate trade from extreme+1 on 1m ===
            entry_bar = min(extreme_idx + 1, 14)
            entry_price = c15[entry_bar]
            risk = abs(entry_price - sl)
            if risk <= 0 or risk / entry_price > 0.05:
                continue

            gi = i_start + entry_bar
            if gi + 675 >= n_1m:
                continue

            pnl = _simulate_exit(
                'breakeven_trail', direction, entry_price, sl, risk, gi,
                c_highs_1m, c_lows_1m, c_closes_1m, atr_1m, sw_lows_1m, sw_highs_1m,
                675, 1.5, 2.75, 2.0, 0.5, 2.0)

            # MFE (max favorable in next 45 bars of 15m = 675 bars of 1m)
            end_mfe = min(gi + 675, n_1m)
            if direction == 1:
                mfe = (np.max(c_highs_1m[gi+1:end_mfe]) - entry_price) / risk
            else:
                mfe = (entry_price - np.min(c_lows_1m[gi+1:end_mfe])) / risk

            # Count signals
            n_signals = sum([has_sfp, has_vol_spike, has_pin, has_engulfing])

            trades.append({
                'direction': direction,
                'pnl_r': round(pnl, 2),
                'mfe_r': round(float(mfe), 2),
                'risk_pct': round(risk / entry_price * 100, 4),
                'extreme_bar': extreme_idx,
                'has_sfp': has_sfp,
                'has_vol_spike': has_vol_spike,
                'vol_at_extreme': vol_at_extreme,
                'max_vol_ratio': round(max_vol_ratio, 1),
                'has_pin': has_pin,
                'has_engulfing': has_engulfing,
                'has_reversal': has_reversal,
                'n_signals': n_signals,
            })

        # === RESULTS ===
        n = len(trades)
        print(f"\n{'='*70}", flush=True)
        print(f"RESULTS ({n} trades)", flush=True)
        print(f"{'='*70}", flush=True)

        pnls = np.array([t['pnl_r'] for t in trades])
        mfes = np.array([t['mfe_r'] for t in trades])

        # Overall
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 999
        print(f"\nOverall: PF {pf:.2f}, WR {len(wins)/n*100:.1f}%, "
              f"avg R {pnls.mean():+.3f}, avg MFE {mfes.mean():.1f}R", flush=True)

        # === Pattern frequencies ===
        print(f"\n--- PATTERN FREQUENCIES ---", flush=True)
        patterns = {
            'SFP': [t['has_sfp'] for t in trades],
            'Vol spike (2x)': [t['has_vol_spike'] for t in trades],
            'Vol at extreme': [t['vol_at_extreme'] for t in trades],
            'Pin bar': [t['has_pin'] for t in trades],
            'Engulfing': [t['has_engulfing'] for t in trades],
            'Immediate reversal': [t['has_reversal'] for t in trades],
        }
        for name, flags in patterns.items():
            count = sum(flags)
            print(f"  {name:25s}: {count:5d} ({count/n*100:5.1f}%)", flush=True)

        # === Performance by individual pattern ===
        print(f"\n--- PERFORMANCE BY PATTERN ---", flush=True)
        print(f"  {'Pattern':25s} | {'Count':>6s} | {'WR':>6s} | {'PF':>7s} | {'Avg R':>7s} | {'Avg MFE':>7s}", flush=True)
        print(f"  {'-'*75}", flush=True)

        def print_stats(name, mask):
            mask = np.array(mask)
            if mask.sum() == 0:
                return
            sub_pnls = pnls[mask]
            sub_mfes = mfes[mask]
            sub_wins = sub_pnls[sub_pnls > 0]
            sub_losses = sub_pnls[sub_pnls <= 0]
            sub_pf = float(sub_wins.sum() / abs(sub_losses.sum())) if sub_losses.sum() != 0 else 999
            sub_wr = len(sub_wins) / len(sub_pnls) * 100
            print(f"  {name:25s} | {int(mask.sum()):6d} | {sub_wr:5.1f}% | {sub_pf:7.2f} | {sub_pnls.mean():+6.3f} | {sub_mfes.mean():7.1f}", flush=True)

        for name, flags in patterns.items():
            print_stats(f"YES {name}", flags)
            print_stats(f"NO  {name}", [not f for f in flags])

        # === Combinations ===
        print(f"\n--- SIGNAL COMBINATIONS ---", flush=True)
        print(f"  {'Combo':35s} | {'Count':>6s} | {'WR':>6s} | {'PF':>7s} | {'Avg R':>7s} | {'MFE':>5s}", flush=True)
        print(f"  {'-'*80}", flush=True)

        combos = {
            'SFP + Vol spike': [t['has_sfp'] and t['has_vol_spike'] for t in trades],
            'SFP + Pin': [t['has_sfp'] and t['has_pin'] for t in trades],
            'SFP + Engulfing': [t['has_sfp'] and t['has_engulfing'] for t in trades],
            'Vol spike + Pin': [t['has_vol_spike'] and t['has_pin'] for t in trades],
            'Vol at extreme + Pin': [t['vol_at_extreme'] and t['has_pin'] for t in trades],
            'SFP + Vol + Pin (triple)': [t['has_sfp'] and t['has_vol_spike'] and t['has_pin'] for t in trades],
            'Any 2+ signals': [t['n_signals'] >= 2 for t in trades],
            'Any 3+ signals': [t['n_signals'] >= 3 for t in trades],
            'Zero signals (nothing)': [t['n_signals'] == 0 for t in trades],
            'Only reversal (no pattern)': [t['has_reversal'] and t['n_signals'] == 0 for t in trades],
        }

        for name, flags in combos.items():
            mask = np.array(flags)
            if mask.sum() == 0:
                continue
            sub_pnls = pnls[mask]
            sub_mfes = mfes[mask]
            sub_wins = sub_pnls[sub_pnls > 0]
            sub_losses = sub_pnls[sub_pnls <= 0]
            sub_pf = float(sub_wins.sum() / abs(sub_losses.sum())) if sub_losses.sum() != 0 else 999
            sub_wr = len(sub_wins) / len(sub_pnls) * 100
            print(f"  {name:35s} | {int(mask.sum()):6d} | {sub_wr:5.1f}% | {sub_pf:7.2f} | {sub_pnls.mean():+6.3f} | {sub_mfes.mean():5.1f}", flush=True)

        # === By number of signals ===
        print(f"\n--- BY NUMBER OF SIGNALS ---", flush=True)
        for ns in range(5):
            mask = np.array([t['n_signals'] == ns for t in trades])
            if mask.sum() == 0:
                continue
            sub_pnls = pnls[mask]
            sub_mfes = mfes[mask]
            sub_wins = sub_pnls[sub_pnls > 0]
            sub_losses = sub_pnls[sub_pnls <= 0]
            sub_pf = float(sub_wins.sum() / abs(sub_losses.sum())) if sub_losses.sum() != 0 else 999
            sub_wr = len(sub_wins) / len(sub_pnls) * 100
            print(f"  {ns} signals: {int(mask.sum()):5d} trades | WR {sub_wr:5.1f}% | PF {sub_pf:7.2f} | "
                  f"avg R {sub_pnls.mean():+6.3f} | MFE {sub_mfes.mean():5.1f}", flush=True)

        # === Volume analysis deeper ===
        print(f"\n--- VOLUME DEPTH ---", flush=True)
        vol_ratios = np.array([t['max_vol_ratio'] for t in trades])
        for threshold in [1.5, 2.0, 3.0, 5.0]:
            mask = vol_ratios >= threshold
            if mask.sum() == 0:
                continue
            sub_pnls = pnls[mask]
            sub_wins = sub_pnls[sub_pnls > 0]
            sub_losses = sub_pnls[sub_pnls <= 0]
            sub_pf = float(sub_wins.sum() / abs(sub_losses.sum())) if sub_losses.sum() != 0 else 999
            print(f"  Vol >= {threshold:.1f}x: {int(mask.sum()):5d} trades | WR {len(sub_wins)/len(sub_pnls)*100:5.1f}% | "
                  f"PF {sub_pf:5.2f} | avg R {sub_pnls.mean():+.3f}", flush=True)


if __name__ == '__main__':
    run_pattern_analysis()
