#!/usr/bin/env python
"""Sniper 1m Study v2 — Rigorous analysis of 1m structure around scalper entries.

v1 had issues: volume/wick metrics were trivially high, SFP window too wide.
v2 focuses on actionable questions:

1. WHICH 1m bar makes the extreme (low for LONG, high for SHORT)?
   - Is it bar 1? bar 7? bar 15? After the 15m close?
   - Distribution of extreme timing

2. SFP with strict definition:
   - Must break the LOW of the first 5 bars (not just any prior low)
   - Must close ABOVE that low (for LONG)
   - Window: only first 15 bars (within the 15m candle)

3. Entry comparison:
   - Entry A: 15m close (current system)
   - Entry B: limit order at the level price
   - Entry C: 1m SFP close (after stop hunt)
   - Entry D: 1m close of extreme bar +1 (bar after the low)
   - Compare SL and R for each

4. Volume concentration:
   - Which 1m bar has the highest volume?
   - Is it the same bar as the extreme? Before? After?

Usage:
    python scripts/autoresearch/sniper_1m_v2.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import copy
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


def run_sniper_v2():
    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        print("=" * 70, flush=True)
        print("SNIPER 1m STUDY v2 — Rigorous Analysis", flush=True)
        print("=" * 70, flush=True)

        # Load data
        print("\nLoading 15m data...", flush=True)
        cache_15m = {}
        data_15m = load_and_cache_data(db.session, cache_15m, timeframe='15m')
        c_times_15m = data_15m['c_times']
        c_closes_15m = data_15m['c_closes']
        c_highs_15m = data_15m['c_highs']
        c_lows_15m = data_15m['c_lows']

        print("Loading 1m data...", flush=True)
        cache_1m = {}
        data_1m = load_and_cache_data(db.session, cache_1m, timeframe='1m')
        c_times_1m = data_1m['c_times']
        c_closes_1m = data_1m['c_closes']
        c_highs_1m = data_1m['c_highs']
        c_lows_1m = data_1m['c_lows']
        c_vols_1m = data_1m['c_vols']
        n_1m = data_1m['n_candles']

        # Get 15m entries (US+EU)
        config = copy.deepcopy(BEST_15M_CONFIG)
        entries = find_touch_entries(data_15m, config)
        entries, scores = score_and_deduplicate(entries, data_15m, config)

        # Session filter
        session_hours = set(range(8, 21))
        keep = np.zeros(len(entries), dtype=bool)
        for i in range(len(entries)):
            ci = int(entries[i, 0])
            ct = c_times_15m[ci]
            hour = int((ct - ct.astype('datetime64[D]')) / np.timedelta64(1, 'h'))
            keep[i] = hour in session_hours
        entries = entries[keep]

        print(f"15m entries (US+EU): {len(entries)}", flush=True)

        times_15m_epoch = c_times_15m.astype('datetime64[s]').astype(np.int64)
        times_1m_epoch = c_times_1m.astype('datetime64[s]').astype(np.int64)

        sl_buffer = 0.001

        # Collectors
        extreme_bar_positions = []       # Which bar (0-14) has the extreme
        extreme_is_last_5 = 0            # Extreme in bar 10-14 (end of candle)
        extreme_is_first_5 = 0           # Extreme in bar 0-4
        extreme_is_middle = 0            # Extreme in bar 5-9

        sfp_strict_count = 0             # SFP within first 15 bars, breaks first-5 extreme
        sfp_strict_bar = []              # Which bar the SFP happens

        vol_peak_bar = []                # Which bar has max volume
        vol_peak_is_extreme = 0          # Vol peak == extreme bar
        vol_peak_near_extreme = 0        # Vol peak within ±2 bars of extreme

        # Entry comparisons (SL in R)
        entry_15m_risks = []             # Risk with 15m close entry
        entry_extreme_plus1_risks = []   # Risk entering 1 bar after extreme
        entry_sfp_risks = []             # Risk entering on SFP close
        entry_level_risks = []           # Risk with limit at level price

        # Post-extreme movement (does price reverse after the extreme?)
        reversal_1bar = 0                # Price moves favorably 1 bar after extreme
        reversal_3bar = 0                # Favorable in 3 bars
        reversal_5bar = 0                # Favorable in 5 bars

        n_analyzed = 0
        max_samples = 2000

        for idx in range(min(len(entries), max_samples)):
            ci_15m = int(entries[idx, 0])
            direction = int(entries[idx, 2])

            entry_15m = c_closes_15m[ci_15m]
            t_15m = times_15m_epoch[ci_15m]

            # Map to 1m
            i_start_1m = np.searchsorted(times_1m_epoch, t_15m)
            if i_start_1m + 20 >= n_1m:
                continue

            # The 15 bars of 1m that compose this 15m candle
            h15 = c_highs_1m[i_start_1m:i_start_1m + 15]
            l15 = c_lows_1m[i_start_1m:i_start_1m + 15]
            c15 = c_closes_1m[i_start_1m:i_start_1m + 15]
            v15 = c_vols_1m[i_start_1m:i_start_1m + 15]

            if len(h15) < 15:
                continue

            n_analyzed += 1

            # --- 1. Which bar is the extreme? ---
            if direction == 1:  # LONG: extreme = lowest low
                extreme_idx = np.argmin(l15)
                extreme_price = l15[extreme_idx]
            else:  # SHORT: extreme = highest high
                extreme_idx = np.argmax(h15)
                extreme_price = h15[extreme_idx]

            extreme_bar_positions.append(int(extreme_idx))
            if extreme_idx < 5:
                extreme_is_first_5 += 1
            elif extreme_idx >= 10:
                extreme_is_last_5 += 1
            else:
                extreme_is_middle += 1

            # --- 2. Strict SFP ---
            # For LONG: after seeing the low of first 5 bars, does a later bar
            # break below that low and then close above it?
            if direction == 1:
                ref_low = np.min(l15[:5])
                sfp_found = False
                for j in range(5, 15):
                    if l15[j] < ref_low and c15[j] > ref_low:
                        sfp_strict_count += 1
                        sfp_strict_bar.append(j)
                        sfp_found = True
                        break
            else:
                ref_high = np.max(h15[:5])
                sfp_found = False
                for j in range(5, 15):
                    if h15[j] > ref_high and c15[j] < ref_high:
                        sfp_strict_count += 1
                        sfp_strict_bar.append(j)
                        sfp_found = True
                        break

            # --- 3. Volume peak ---
            vol_peak = np.argmax(v15)
            vol_peak_bar.append(int(vol_peak))
            if vol_peak == extreme_idx:
                vol_peak_is_extreme += 1
            if abs(vol_peak - extreme_idx) <= 2:
                vol_peak_near_extreme += 1

            # --- 4. Entry comparison ---
            # SL is always at extreme - buffer (LONG) or extreme + buffer (SHORT)
            if direction == 1:
                sl = extreme_price * (1 - sl_buffer)
                risk_15m = abs(entry_15m - sl)
                # Entry on bar after extreme
                if extreme_idx + 1 < 15:
                    entry_after = c15[extreme_idx + 1]
                    risk_after = abs(entry_after - sl)
                else:
                    entry_after = entry_15m
                    risk_after = risk_15m
                # Risk with SFP entry
                if sfp_found:
                    sfp_bar_idx = sfp_strict_bar[-1]
                    entry_sfp = c15[sfp_bar_idx]
                    risk_sfp = abs(entry_sfp - sl)
                else:
                    risk_sfp = risk_15m
            else:
                sl = extreme_price * (1 + sl_buffer)
                risk_15m = abs(entry_15m - sl)
                if extreme_idx + 1 < 15:
                    entry_after = c15[extreme_idx + 1]
                    risk_after = abs(entry_after - sl)
                else:
                    entry_after = entry_15m
                    risk_after = risk_15m
                if sfp_found:
                    sfp_bar_idx = sfp_strict_bar[-1]
                    entry_sfp = c15[sfp_bar_idx]
                    risk_sfp = abs(entry_sfp - sl)
                else:
                    risk_sfp = risk_15m

            if risk_15m > 0:
                entry_15m_risks.append(risk_15m / entry_15m * 100)
                entry_extreme_plus1_risks.append(risk_after / entry_15m * 100)
                if sfp_found:
                    entry_sfp_risks.append(risk_sfp / entry_15m * 100)

            # --- 5. Post-extreme reversal ---
            # After the extreme bar, does price move in our direction?
            post_start = i_start_1m + extreme_idx + 1
            if post_start + 5 < n_1m:
                post_closes = c_closes_1m[post_start:post_start + 5]
                if direction == 1:
                    if len(post_closes) >= 1 and post_closes[0] > extreme_price:
                        reversal_1bar += 1
                    if len(post_closes) >= 3 and np.max(post_closes[:3]) > extreme_price * 1.001:
                        reversal_3bar += 1
                    if len(post_closes) >= 5 and np.max(post_closes[:5]) > extreme_price * 1.002:
                        reversal_5bar += 1
                else:
                    if len(post_closes) >= 1 and post_closes[0] < extreme_price:
                        reversal_1bar += 1
                    if len(post_closes) >= 3 and np.min(post_closes[:3]) < extreme_price * 0.999:
                        reversal_3bar += 1
                    if len(post_closes) >= 5 and np.min(post_closes[:5]) < extreme_price * 0.998:
                        reversal_5bar += 1

        # === RESULTS ===
        n = n_analyzed
        print(f"\n{'='*70}", flush=True)
        print(f"RESULTS ({n} trades analyzed)", flush=True)
        print(f"{'='*70}", flush=True)

        # 1. Extreme bar distribution
        print(f"\n1. WHERE is the extreme within the 15m candle?", flush=True)
        print(f"   (bar 0 = first minute, bar 14 = last minute)", flush=True)
        bars = np.array(extreme_bar_positions)
        print(f"   First 5 min (bar 0-4):  {extreme_is_first_5} ({extreme_is_first_5/n*100:.1f}%)", flush=True)
        print(f"   Middle (bar 5-9):       {extreme_is_middle} ({extreme_is_middle/n*100:.1f}%)", flush=True)
        print(f"   Last 5 min (bar 10-14): {extreme_is_last_5} ({extreme_is_last_5/n*100:.1f}%)", flush=True)
        print(f"   Median bar: {np.median(bars):.0f}", flush=True)
        print(f"   Mean bar:   {np.mean(bars):.1f}", flush=True)

        # Bar-by-bar histogram
        print(f"\n   Bar-by-bar distribution:", flush=True)
        for b in range(15):
            count = np.sum(bars == b)
            bar_chart = '#' * int(count / n * 100)
            print(f"   bar {b:2d}: {count:4d} ({count/n*100:5.1f}%) {bar_chart}", flush=True)

        # 2. Strict SFP
        print(f"\n2. STRICT SFP (breaks first-5-bar extreme, closes back):", flush=True)
        print(f"   Found: {sfp_strict_count} ({sfp_strict_count/n*100:.1f}%)", flush=True)
        if sfp_strict_bar:
            sfp_bars = np.array(sfp_strict_bar)
            print(f"   SFP typically on bar: {np.median(sfp_bars):.0f} (median), {np.mean(sfp_bars):.1f} (mean)", flush=True)

        # 3. Volume peak
        print(f"\n3. VOLUME PEAK:", flush=True)
        vp = np.array(vol_peak_bar)
        print(f"   Peak IS the extreme bar: {vol_peak_is_extreme} ({vol_peak_is_extreme/n*100:.1f}%)", flush=True)
        print(f"   Peak within +/-2 of extreme: {vol_peak_near_extreme} ({vol_peak_near_extreme/n*100:.1f}%)", flush=True)
        print(f"   Peak median bar: {np.median(vp):.0f}", flush=True)

        # 4. Entry comparison
        print(f"\n4. ENTRY COMPARISON (SL = extreme - buffer):", flush=True)
        r15 = np.array(entry_15m_risks)
        rafter = np.array(entry_extreme_plus1_risks)
        print(f"   15m close entry:     avg SL = {np.mean(r15):.3f}% (current system)", flush=True)
        print(f"   Extreme+1 bar entry: avg SL = {np.mean(rafter):.3f}% "
              f"({(1-np.mean(rafter)/np.mean(r15))*100:.0f}% tighter)", flush=True)
        if entry_sfp_risks:
            rsfp = np.array(entry_sfp_risks)
            print(f"   SFP close entry:     avg SL = {np.mean(rsfp):.3f}% "
                  f"({(1-np.mean(rsfp)/np.mean(r15))*100:.0f}% tighter)", flush=True)

        # Position size multiplier
        mult_after = np.mean(r15) / np.mean(rafter) if np.mean(rafter) > 0 else 1
        print(f"\n   Position multiplier (extreme+1): {mult_after:.2f}x", flush=True)
        if entry_sfp_risks:
            mult_sfp = np.mean(r15) / np.mean(rsfp) if np.mean(rsfp) > 0 else 1
            print(f"   Position multiplier (SFP):       {mult_sfp:.2f}x", flush=True)

        # 5. Post-extreme reversal
        print(f"\n5. POST-EXTREME REVERSAL (does price move our way after the extreme?):", flush=True)
        print(f"   1 bar after:  {reversal_1bar} ({reversal_1bar/n*100:.1f}%)", flush=True)
        print(f"   3 bars after: {reversal_3bar} ({reversal_3bar/n*100:.1f}%) (moved +0.1%)", flush=True)
        print(f"   5 bars after: {reversal_5bar} ({reversal_5bar/n*100:.1f}%) (moved +0.2%)", flush=True)


if __name__ == '__main__':
    run_sniper_v2()
