#!/usr/bin/env python
"""Real-Time 1m Touch Detection System.

Simulates what would happen in real-time trading on 1m candles:
1. Pre-calculate naked levels of confluence (fractals, VWAP, VP_POC)
2. Each 1m candle: does price touch a naked level?
3. If yes + volume >= threshold: enter on close, SL at wick extreme
4. Exit: breakeven trail

No future information used — each decision uses only past data.

Usage:
    python scripts/autoresearch/realtime_1m.py
    python scripts/autoresearch/realtime_1m.py --vol-threshold 3.0
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import argparse
import numpy as np
from scripts.autoresearch.confluence_scalper import (
    load_and_cache_data, _simulate_exit,
)
from scripts.autoresearch.sniper_backtest import find_swing_lows, find_swing_highs


def run_realtime_1m(vol_threshold=3.0, session_filter='all'):
    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        print("=" * 70, flush=True)
        print("REAL-TIME 1m TOUCH DETECTION SYSTEM", flush=True)
        print(f"Vol threshold: {vol_threshold}x | Session: {session_filter}", flush=True)
        print("No future information — pure real-time simulation", flush=True)
        print("=" * 70, flush=True)

        # Load 1m candle data
        print("\nLoading 1m data...", flush=True)
        cache = {}
        data = load_and_cache_data(db.session, cache, timeframe='1m')
        c_times = data['c_times']
        c_highs = data['c_highs']
        c_lows = data['c_lows']
        c_closes = data['c_closes']
        c_vols = data['c_vols']
        atr = data['atr']
        n_candles = data['n_candles']

        # Precompute swings for exit
        print("Computing swings...", flush=True)
        sw_lows = find_swing_lows(c_lows, lookback=150)
        sw_highs = find_swing_highs(c_highs, lookback=150)

        # Precompute volume MA (rolling 20-bar average)
        print("Computing volume MA...", flush=True)
        vol_ma = np.zeros(n_candles)
        vol_window = 20
        for i in range(vol_window, n_candles):
            vol_ma[i] = np.mean(c_vols[i-vol_window:i])
        # Fill early bars
        for i in range(1, vol_window):
            vol_ma[i] = np.mean(c_vols[:i])

        # Load levels
        print("Loading levels...", flush=True)
        from app.services.level_trade_backtest_db import load_levels_db
        import pandas as pd

        levels_df = load_levels_db(db.session)
        print(f"  {len(levels_df)} total levels loaded", flush=True)

        # Filter to key level types (fractals + VWAP + VP_POC)
        key_types = {'Fractal_support', 'Fractal_resistance',
                     'PrevSession_VWAP', 'PrevSession_VP_POC'}
        levels_df = levels_df[levels_df['level_type'].isin(key_types)].copy()
        print(f"  {len(levels_df)} key levels (fractals + VWAP + VP_POC)", flush=True)

        # Prepare level arrays
        l_prices = levels_df['price_level'].values.astype(np.float64)
        l_types = levels_df['level_type'].values.astype(str)

        # Validity: created_at and first_touched_at/superseded_at
        far_future = pd.Timestamp('2099-01-01', tz='UTC').timestamp()
        structural = {'Fractal_support', 'Fractal_resistance'}

        l_created = []
        l_validity_end = []
        for _, row in levels_df.iterrows():
            created = row.get('created_at')
            if pd.isna(created):
                created = pd.Timestamp('2017-01-01', tz='UTC')
            elif not hasattr(created, 'tz') or created.tz is None:
                created = pd.Timestamp(created, tz='UTC')
            l_created.append(created.timestamp())

            is_struct = row['level_type'] in structural
            if is_struct:
                end = row.get('first_touched_at', None)
            else:
                end = row.get('superseded_at', None) or row.get('invalidated_at', None)
            if pd.isna(end) or end is None:
                l_validity_end.append(far_future)
            else:
                if not hasattr(end, 'tz') or end.tz is None:
                    end = pd.Timestamp(end, tz='UTC')
                l_validity_end.append(end.timestamp())

        l_created = np.array(l_created, dtype=np.float64)
        l_validity_end = np.array(l_validity_end, dtype=np.float64)

        # Sorted prices for searchsorted
        sorted_idx = np.argsort(l_prices)
        sorted_prices = l_prices[sorted_idx]

        # Convert candle times to epoch
        c_times_epoch = c_times.astype('datetime64[s]').astype(np.int64).astype(np.float64)

        # Session hours
        session_hours = None
        if session_filter == 'us':
            session_hours = set(range(14, 21))
        elif session_filter == 'us_eu':
            session_hours = set(range(8, 21))
        elif session_filter == 'eu':
            session_hours = set(range(8, 14))

        # ===================================================================
        # REAL-TIME SIMULATION
        # Process each 1m candle sequentially — only use past data
        # ===================================================================
        print("\nRunning real-time simulation...", flush=True)
        t0 = time.time()

        touch_tolerance = 0.003  # 0.3%
        sl_buffer = 0.001
        cooldown = 0  # Minimum bars between trades
        zone_width = 0.01  # 1% zone for confluence check
        min_confluence = 2  # Need at least 2 level types in zone
        be_rr = 4.77  # Equivalent to 2.75 on 15m SL
        timeout = 675  # 45 bars of 15m = 675 bars of 1m

        trades = []
        last_trade_bar = -cooldown - 1
        consumed_levels = set()  # Track first-touch per level

        # Process in chunks for speed (check active levels per chunk)
        chunk_size = 10080  # 1 week
        n_chunks = (n_candles + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            cs = chunk_idx * chunk_size
            ce = min(cs + chunk_size, n_candles)
            chunk_time = c_times_epoch[cs]

            # Active levels for this chunk
            active_mask = (l_created <= chunk_time) & (l_validity_end > chunk_time)
            active_idx = np.where(active_mask)[0]
            # Remove consumed
            active_idx = np.array([i for i in active_idx if i not in consumed_levels])

            if len(active_idx) == 0:
                continue

            active_prices = l_prices[active_idx]
            active_types = l_types[active_idx]

            for i in range(cs, ce):
                if i - last_trade_bar < cooldown:
                    continue
                if i + timeout >= n_candles:
                    break

                # Session filter
                if session_hours is not None:
                    ct = c_times[i]
                    hour = int((ct - ct.astype('datetime64[D]')) / np.timedelta64(1, 'h'))
                    if hour not in session_hours:
                        continue

                price = c_closes[i]
                low = c_lows[i]
                high = c_highs[i]

                # Volume check first (fast filter)
                if vol_ma[i] <= 0:
                    continue
                vol_ratio = c_vols[i] / vol_ma[i]
                if vol_ratio < vol_threshold:
                    continue

                # Check if this candle touches any active level
                # LONG touch: low <= level * (1+tol) AND close > level
                # SHORT touch: high >= level * (1-tol) AND close < level
                touch_found = False
                touch_direction = 0
                touch_level_idx = -1
                touch_level_price = 0
                zone_types = set()

                for j_idx, j in enumerate(active_idx):
                    lp = active_prices[j_idx]
                    lt = active_types[j_idx]

                    # LONG touch
                    if low <= lp * (1 + touch_tolerance) and price > lp:
                        # Check confluence: count unique types in zone
                        lo_zone = lp * (1 - zone_width)
                        hi_zone = lp * (1 + zone_width)
                        left = np.searchsorted(sorted_prices, lo_zone, side='left')
                        right = np.searchsorted(sorted_prices, hi_zone, side='right')
                        if left < right:
                            zone_orig = sorted_idx[left:right]
                            # Filter to active
                            zone_active = [z for z in zone_orig if active_mask[z] and z not in consumed_levels]
                            zone_types = set(l_types[z] for z in zone_active)

                        if len(zone_types) >= min_confluence:
                            touch_found = True
                            touch_direction = 1
                            touch_level_idx = j
                            touch_level_price = lp
                            break

                    # SHORT touch
                    elif high >= lp * (1 - touch_tolerance) and price < lp:
                        lo_zone = lp * (1 - zone_width)
                        hi_zone = lp * (1 + zone_width)
                        left = np.searchsorted(sorted_prices, lo_zone, side='left')
                        right = np.searchsorted(sorted_prices, hi_zone, side='right')
                        if left < right:
                            zone_orig = sorted_idx[left:right]
                            zone_active = [z for z in zone_orig if active_mask[z] and z not in consumed_levels]
                            zone_types = set(l_types[z] for z in zone_active)

                        if len(zone_types) >= min_confluence:
                            touch_found = True
                            touch_direction = -1
                            touch_level_idx = j
                            touch_level_price = lp
                            break

                if not touch_found:
                    continue

                # Mark level as consumed
                consumed_levels.add(touch_level_idx)

                # Entry on this candle's close
                entry_price = price

                # SL at the wick extreme of THIS candle + buffer
                if touch_direction == 1:
                    sl = low * (1 - sl_buffer)
                else:
                    sl = high * (1 + sl_buffer)

                risk = abs(entry_price - sl)
                if risk <= 0 or risk / entry_price > 0.03:  # Max 3% SL
                    continue

                # Simulate exit
                pnl = _simulate_exit(
                    'breakeven_trail', touch_direction, entry_price, sl, risk, i,
                    c_highs, c_lows, c_closes, atr, sw_lows, sw_highs,
                    timeout, 1.5, be_rr, 2.0, 0.5, 2.0
                )

                trades.append({
                    'candle_idx': i,
                    'direction': touch_direction,
                    'entry': round(entry_price, 2),
                    'sl': round(sl, 2),
                    'risk_pct': round(risk / entry_price * 100, 4),
                    'pnl_r': round(pnl, 2),
                    'vol_ratio': round(vol_ratio, 1),
                    'confluence_types': len(zone_types),
                })

                last_trade_bar = i + timeout // 3  # Cooldown: 1/3 of timeout

        elapsed = time.time() - t0
        print(f"Simulation done ({elapsed:.1f}s)", flush=True)

        # === RESULTS ===
        if not trades:
            print("No trades generated!", flush=True)
            return

        n = len(trades)
        pnls = np.array([t['pnl_r'] for t in trades])
        risks = np.array([t['risk_pct'] for t in trades])
        vols = np.array([t['vol_ratio'] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        total_r = float(pnls.sum())
        wr = len(wins) / n * 100
        pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 999
        avg_r = float(pnls.mean())

        # Max consecutive losses
        max_cl = 0
        cur = 0
        for p in pnls:
            if p <= 0:
                cur += 1
                max_cl = max(max_cl, cur)
            else:
                cur = 0

        # Commission analysis
        risk_per_trade = 10
        avg_risk_pct = np.mean(risks) / 100
        avg_pos = risk_per_trade / avg_risk_pct if avg_risk_pct > 0 else 0
        comm_rate = 0.0006
        comm_per = avg_pos * comm_rate
        gross = total_r * risk_per_trade
        total_comm = comm_per * n
        net = gross - total_comm
        days = n_candles / (24 * 60)
        months = days / 30

        print(f"\n{'='*70}", flush=True)
        print(f"RESULTS — Real-Time 1m System (vol >= {vol_threshold}x, session={session_filter})", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Trades: {n}", flush=True)
        print(f"  Win rate: {wr:.1f}%", flush=True)
        print(f"  Profit Factor: {pf:.2f}", flush=True)
        print(f"  Total R: {total_r:.1f}", flush=True)
        print(f"  Avg R: {avg_r:+.3f}", flush=True)
        print(f"  Avg win: {wins.mean():.2f}R, Avg loss: {losses.mean():.2f}R" if len(wins) > 0 and len(losses) > 0 else "", flush=True)
        print(f"  Max consec losses: {max_cl}", flush=True)
        print(f"  Trades/day: {n/days:.2f}", flush=True)
        print(f"  Avg vol ratio at entry: {vols.mean():.1f}x", flush=True)
        print(f"  --- $10 risk ---", flush=True)
        print(f"  Avg SL: {avg_risk_pct*100:.3f}%", flush=True)
        print(f"  Avg position: ${avg_pos:.0f}", flush=True)
        print(f"  Commission/trade: ${comm_per:.2f}", flush=True)
        print(f"  Gross profit: ${gross:.0f}", flush=True)
        print(f"  Total commission: ${total_comm:.0f}", flush=True)
        print(f"  NET profit: ${net:.0f}", flush=True)
        print(f"  NET/month: ${net/months:.0f}", flush=True)
        print(f"  Period: {days:.0f} days ({months:.0f} months)", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-Time 1m Touch System')
    parser.add_argument('--vol-threshold', type=float, default=3.0,
                        help='Minimum volume ratio for entry (default: 3.0)')
    parser.add_argument('--session', type=str, default='all',
                        help='Session filter: all, us, eu, us_eu')
    args = parser.parse_args()
    run_realtime_1m(vol_threshold=args.vol_threshold, session_filter=args.session)
