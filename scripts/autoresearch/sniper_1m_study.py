#!/usr/bin/env python
"""Sniper 1m Study — Analyze 1m price action around scalper entries.

For each confluence scalper entry on 15m, zooms into 1m candles to study:
- Is there an SFP (stop hunt pattern) detectable on 1m?
- What's the optimal 1m entry vs the 15m close entry?
- How tight can the SL be if entering on 1m?
- Volume patterns around the touch

Usage:
    python scripts/autoresearch/sniper_1m_study.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import copy
import time
import numpy as np
from scripts.autoresearch.confluence_scalper import (
    load_and_cache_data, find_touch_entries, score_and_deduplicate,
)


BEST_15M_CONFIG = {
    'score_threshold': 3,
    'zone_width': 0.01,
    'touch_tolerance': 0.001,
    'naked_only': True,
    'scoring_mode': 'unique_types',
    'entry_mode': 'touch',
    'session_filter': 'us_eu',
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


def run_sniper_study():
    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        print("=" * 70, flush=True)
        print("SNIPER 1m STUDY", flush=True)
        print("Analyzing 1m price action around 15m scalper entries", flush=True)
        print("=" * 70, flush=True)

        # Load 15m data for scalper entries
        print("\nLoading 15m data...", flush=True)
        cache_15m = {}
        data_15m = load_and_cache_data(db.session, cache_15m, timeframe='15m')
        c_times_15m = data_15m['c_times']
        c_closes_15m = data_15m['c_closes']
        c_highs_15m = data_15m['c_highs']
        c_lows_15m = data_15m['c_lows']

        # Load 1m data
        print("Loading 1m data...", flush=True)
        cache_1m = {}
        data_1m = load_and_cache_data(db.session, cache_1m, timeframe='1m')
        c_times_1m = data_1m['c_times']
        c_closes_1m = data_1m['c_closes']
        c_highs_1m = data_1m['c_highs']
        c_lows_1m = data_1m['c_lows']
        c_vols_1m = data_1m['c_vols']
        n_1m = data_1m['n_candles']

        # Get 15m scalper entries
        config = copy.deepcopy(BEST_15M_CONFIG)
        entries = find_touch_entries(data_15m, config)
        entries, scores = score_and_deduplicate(entries, data_15m, config)

        # Apply session filter
        session_hours = set(range(8, 21))  # us_eu
        keep = np.zeros(len(entries), dtype=bool)
        for i in range(len(entries)):
            ci = int(entries[i, 0])
            ct = c_times_15m[ci]
            hour = int((ct - ct.astype('datetime64[D]')) / np.timedelta64(1, 'h'))
            keep[i] = hour in session_hours
        entries = entries[keep]
        scores = scores[keep]

        print(f"\n15m entries (US+EU session): {len(entries)}", flush=True)

        # Convert 15m timestamps to epoch for matching with 1m
        times_15m_epoch = c_times_15m.astype('datetime64[s]').astype(np.int64)
        times_1m_epoch = c_times_1m.astype('datetime64[s]').astype(np.int64)

        # For each 15m entry, analyze the surrounding 1m candles
        # Window: 15 candles before (the 15m candle) + 30 candles after
        sfp_found_count = 0
        sfp_not_found = 0
        optimal_entries = []  # How much closer to SL we could enter on 1m
        volume_spikes = 0
        wick_rejections = 0
        entry_improvements = []  # R improvement from 1m entry vs 15m close

        sl_buffer = 0.001
        sample_count = 0
        max_samples = 2000  # Limit for speed

        for idx in range(min(len(entries), max_samples)):
            ci_15m = int(entries[idx, 0])
            direction = int(entries[idx, 2])

            # 15m entry details
            entry_15m = c_closes_15m[ci_15m]
            t_15m = times_15m_epoch[ci_15m]

            if direction == 1:
                sl_15m = c_lows_15m[ci_15m] * (1 - sl_buffer)
            else:
                sl_15m = c_highs_15m[ci_15m] * (1 + sl_buffer)
            risk_15m = abs(entry_15m - sl_15m)
            if risk_15m <= 0:
                continue

            # Find corresponding 1m candles
            # The 15m candle spans 15 minutes, so look at 1m candles in that window
            i_start_1m = np.searchsorted(times_1m_epoch, t_15m)
            # Look at the 15m candle (15 bars) + 30 bars after
            i_end_1m = min(i_start_1m + 45, n_1m)
            i_pre_1m = max(i_start_1m - 5, 0)  # 5 bars before for context

            if i_start_1m >= n_1m or i_end_1m <= i_start_1m:
                continue

            sample_count += 1

            # 1m candles in the touch window
            h_1m = c_highs_1m[i_start_1m:i_end_1m]
            l_1m = c_lows_1m[i_start_1m:i_end_1m]
            c_1m = c_closes_1m[i_start_1m:i_end_1m]
            v_1m = c_vols_1m[i_start_1m:i_end_1m]

            if len(h_1m) < 15:
                continue

            # --- Analysis 1: SFP detection on 1m ---
            # For LONG: look for a candle that makes a lower low then closes above
            # For SHORT: look for a candle that makes a higher high then closes below
            sfp_detected = False
            sfp_entry = None

            # Reference point: the level price (approximated by the touch)
            level_price = entry_15m  # Close enough

            for j in range(min(30, len(h_1m))):
                if direction == 1:  # LONG
                    # SFP: price goes below prior low, then closes above it
                    if j > 0 and l_1m[j] < min(l_1m[:j]) and c_1m[j] > min(l_1m[:j]):
                        sfp_detected = True
                        sfp_entry = c_1m[j]
                        break
                else:  # SHORT
                    if j > 0 and h_1m[j] > max(h_1m[:j]) and c_1m[j] < max(h_1m[:j]):
                        sfp_detected = True
                        sfp_entry = c_1m[j]
                        break

            if sfp_detected:
                sfp_found_count += 1
            else:
                sfp_not_found += 1

            # --- Analysis 2: Optimal 1m entry (closest to SL) ---
            if direction == 1:  # LONG: best entry is the lowest close above SL
                valid_closes = c_1m[c_1m > sl_15m]
                if len(valid_closes) > 0:
                    best_1m_entry = np.min(valid_closes)
                else:
                    best_1m_entry = entry_15m
            else:  # SHORT: best entry is the highest close below SL
                valid_closes = c_1m[c_1m < sl_15m]
                if len(valid_closes) > 0:
                    best_1m_entry = np.max(valid_closes)
                else:
                    best_1m_entry = entry_15m

            # How much tighter is the 1m SL?
            if direction == 1:
                risk_1m = abs(best_1m_entry - sl_15m)
                improvement = (risk_15m - risk_1m) / risk_15m if risk_15m > 0 else 0
            else:
                risk_1m = abs(best_1m_entry - sl_15m)
                improvement = (risk_15m - risk_1m) / risk_15m if risk_15m > 0 else 0
            entry_improvements.append(improvement)

            # --- Analysis 3: Volume spike at touch ---
            avg_vol = np.mean(v_1m[:5]) if len(v_1m) >= 5 else 1
            touch_vol = np.max(v_1m[:15]) if len(v_1m) >= 15 else 0
            if avg_vol > 0 and touch_vol > avg_vol * 2:
                volume_spikes += 1

            # --- Analysis 4: Wick rejection on 1m ---
            for j in range(min(15, len(h_1m))):
                candle_range = h_1m[j] - l_1m[j]
                if candle_range <= 0:
                    continue
                if direction == 1:
                    lower_wick = min(c_1m[j], c_closes_1m[i_start_1m + j] if i_start_1m + j > 0 else c_1m[j]) - l_1m[j]
                    if lower_wick / candle_range > 0.6:
                        wick_rejections += 1
                        break
                else:
                    upper_wick = h_1m[j] - max(c_1m[j], c_closes_1m[i_start_1m + j] if i_start_1m + j > 0 else c_1m[j])
                    if upper_wick / candle_range > 0.6:
                        wick_rejections += 1
                        break

        # Results
        n = sample_count
        if n == 0:
            print("No samples to analyze!", flush=True)
            return

        print(f"\n{'='*70}", flush=True)
        print(f"RESULTS ({n} trades analyzed)", flush=True)
        print(f"{'='*70}", flush=True)

        print(f"\n1. SFP Detection on 1m:", flush=True)
        print(f"   SFP found:     {sfp_found_count} ({sfp_found_count/n*100:.1f}%)", flush=True)
        print(f"   SFP not found: {sfp_not_found} ({sfp_not_found/n*100:.1f}%)", flush=True)

        print(f"\n2. Entry Improvement (1m vs 15m):", flush=True)
        imp = np.array(entry_improvements)
        print(f"   Avg SL reduction:    {np.mean(imp)*100:.1f}%", flush=True)
        print(f"   Median SL reduction: {np.median(imp)*100:.1f}%", flush=True)
        print(f"   P75 SL reduction:    {np.percentile(imp, 75)*100:.1f}%", flush=True)
        print(f"   > 30% tighter SL:    {np.sum(imp > 0.3)/n*100:.1f}% of trades", flush=True)
        print(f"   > 50% tighter SL:    {np.sum(imp > 0.5)/n*100:.1f}% of trades", flush=True)
        pos_mult = 1 / (1 - np.mean(imp)) if np.mean(imp) < 1 else 1
        print(f"   Position size multiplier: {pos_mult:.2f}x (same risk, bigger position)", flush=True)

        print(f"\n3. Volume Patterns:", flush=True)
        print(f"   Volume spike (2x avg) at touch: {volume_spikes} ({volume_spikes/n*100:.1f}%)", flush=True)

        print(f"\n4. Wick Rejection on 1m:", flush=True)
        print(f"   Wick rejection (>60% wick): {wick_rejections} ({wick_rejections/n*100:.1f}%)", flush=True)

        print(f"\n5. Implications:", flush=True)
        print(f"   If entering on 1m SFP instead of 15m close:", flush=True)
        print(f"   - {sfp_found_count/n*100:.0f}% of trades would have SFP confirmation", flush=True)
        print(f"   - SL would be {np.mean(imp)*100:.0f}% tighter on avg", flush=True)
        print(f"   - Position size {pos_mult:.1f}x larger (same $ risk)", flush=True)
        print(f"   - Profit per R would be {pos_mult:.1f}x more $", flush=True)


if __name__ == '__main__':
    run_sniper_study()
