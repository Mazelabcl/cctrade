"""Level-to-Level TP Backtest.

Instead of fixed RR ratios, TP = next opposing level.
- LONG entry at support level → TP = next resistance level above
- SHORT entry at resistance level → TP = next support level below

SL remains the same: entry candle's wick extreme + buffer.

This mimics Daniel's actual trading: "level to level trading".
"""
import sys
import os
import json
import logging
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

TOUCH_TOLERANCE_PCT = 0.005
SL_BUFFER_PCT = 0.001
TIMEOUT_CANDLES = 100


def run_l2l_backtest(session, exec_tf='4h', symbol='BTCUSDT'):
    """Run level-to-level backtest."""
    from app.services.level_trade_backtest_db import _query_to_df

    # Load candles
    candles = _query_to_df(session,
        "SELECT open_time, open, high, low, close, volume "
        "FROM candles WHERE symbol = :sym AND timeframe = :tf "
        "ORDER BY open_time ASC",
        {'sym': symbol, 'tf': exec_tf})
    candles['open_time'] = pd.to_datetime(candles['open_time'])
    logger.info("Loaded %d candles (%s)", len(candles), exec_tf)

    # Load D/W/M levels
    levels = _query_to_df(session,
        "SELECT id, price_level, level_type, timeframe, created_at, first_touched_at "
        "FROM levels "
        "WHERE timeframe IN ('daily', 'weekly', 'monthly') "
        "ORDER BY created_at ASC")
    levels['created_at'] = pd.to_datetime(levels['created_at'])
    levels['first_touched_at'] = pd.to_datetime(levels['first_touched_at'])
    logger.info("Loaded %d D/W/M levels", len(levels))

    # Numpy arrays
    c_times = candles['open_time'].values
    c_highs = candles['high'].values.astype(np.float64)
    c_lows = candles['low'].values.astype(np.float64)
    c_closes = candles['close'].values.astype(np.float64)
    c_opens = candles['open'].values.astype(np.float64)
    c_volumes = candles['volume'].values.astype(np.float64)

    lev_prices = levels['price_level'].values.astype(np.float64)
    lev_created = levels['created_at'].values
    lev_touched = levels['first_touched_at'].values
    lev_types = levels['level_type'].values
    lev_tfs = levels['timeframe'].values
    lev_ids = levels['id'].values
    n_levels = len(lev_prices)

    # Track which levels have been consumed
    level_consumed = np.zeros(n_levels, dtype=bool)

    trades = []
    position = None
    tol = TOUCH_TOLERANCE_PCT

    for i in range(len(candles)):
        ct = c_times[i]
        hi = c_highs[i]
        lo = c_lows[i]
        cl = c_closes[i]
        op = c_opens[i]

        # Manage open position
        if position is not None:
            position['candles_held'] += 1
            exited = False

            if position['direction'] == 'LONG':
                if lo <= position['sl']:
                    position['exit_time'] = ct
                    position['exit_price'] = position['sl']
                    position['exit_reason'] = 'SL_HIT'
                    exited = True
                elif hi >= position['tp']:
                    position['exit_time'] = ct
                    position['exit_price'] = position['tp']
                    position['exit_reason'] = 'TP_HIT'
                    exited = True
            else:  # SHORT
                if hi >= position['sl']:
                    position['exit_time'] = ct
                    position['exit_price'] = position['sl']
                    position['exit_reason'] = 'SL_HIT'
                    exited = True
                elif lo <= position['tp']:
                    position['exit_time'] = ct
                    position['exit_price'] = position['tp']
                    position['exit_reason'] = 'TP_HIT'
                    exited = True

            if not exited and position['candles_held'] >= TIMEOUT_CANDLES:
                position['exit_time'] = ct
                position['exit_price'] = cl
                position['exit_reason'] = 'TIMEOUT'
                exited = True

            if exited:
                # Compute PnL
                if position['direction'] == 'LONG':
                    pnl = position['exit_price'] - position['entry_price']
                else:
                    pnl = position['entry_price'] - position['exit_price']
                position['pnl'] = pnl
                position['pnl_pct'] = pnl / position['entry_price']
                position['rr_achieved'] = abs(pnl) / position['risk'] if position['risk'] > 0 else 0
                if pnl < 0:
                    position['rr_achieved'] = -position['rr_achieved']
                trades.append(position)
                position = None

            continue  # Don't look for entries while in position

        # Look for entry signals
        # Only consider levels created before this candle and not consumed
        valid_mask = (lev_created < ct) & ~level_consumed
        dist = np.abs(lev_prices - cl) / cl
        nearby_mask = valid_mask & (dist < 0.15)
        candidates = np.where(nearby_mask)[0]

        if candidates.size == 0:
            continue

        # Sort by distance (closest first)
        candidates = candidates[np.argsort(dist[candidates])]

        for j in candidates:
            lp = lev_prices[j]
            lp_tol = lp * tol

            # Entry signal
            if lo <= lp + lp_tol and cl > lp:
                direction = 'LONG'
                sl = lo * (1.0 - SL_BUFFER_PCT)
                risk = cl - sl
            elif hi >= lp - lp_tol and cl < lp:
                direction = 'SHORT'
                sl = hi * (1.0 + SL_BUFFER_PCT)
                risk = sl - cl
            else:
                continue

            if risk <= 0:
                continue

            # Find next level in opposite direction for TP
            if direction == 'LONG':
                # TP = next level ABOVE entry price
                above_mask = valid_mask & (lev_prices > cl * 1.002)  # at least 0.2% above
                above_idx = np.where(above_mask)[0]
                if above_idx.size == 0:
                    continue  # No opposing level → skip trade
                # Closest level above
                tp_idx = above_idx[np.argmin(lev_prices[above_idx] - cl)]
                tp = lev_prices[tp_idx]
            else:
                # TP = next level BELOW entry price
                below_mask = valid_mask & (lev_prices < cl * 0.998)  # at least 0.2% below
                below_idx = np.where(below_mask)[0]
                if below_idx.size == 0:
                    continue  # No opposing level → skip trade
                # Closest level below
                tp_idx = below_idx[np.argmin(cl - lev_prices[below_idx])]
                tp = lev_prices[tp_idx]

            # Calculate actual RR of this trade
            reward = abs(tp - cl)
            actual_rr = reward / risk if risk > 0 else 0

            # Skip if RR < 0.5 (not worth the risk)
            if actual_rr < 0.5:
                continue

            # Mark ALL touched levels for this candle (including entry level)
            active = (lev_created < ct) & ~level_consumed
            hit = active & (lev_prices * (1.0 + tol) >= lo) & (lev_prices * (1.0 - tol) <= hi)
            level_consumed[:] |= hit
            level_consumed[j] = True

            position = {
                'entry_time': ct,
                'entry_price': float(cl),
                'direction': direction,
                'sl': float(sl),
                'tp': float(tp),
                'risk': float(risk),
                'reward': float(reward),
                'actual_rr': float(actual_rr),
                'level_price': float(lp),
                'level_type': str(lev_types[j]),
                'level_tf': str(lev_tfs[j]),
                'tp_level_type': str(lev_types[tp_idx]),
                'tp_level_tf': str(lev_tfs[tp_idx]),
                'candles_held': 0,
                'entry_wick_ratio': float((hi - max(op, cl)) / (hi - lo)) if direction == 'SHORT' and hi > lo else
                                    float((min(op, cl) - lo) / (hi - lo)) if direction == 'LONG' and hi > lo else 0,
                'entry_volume': float(c_volumes[i]),
            }
            break
        else:
            # No entry found — still mark any levels this candle touched
            active = (lev_created < ct) & ~level_consumed
            hit = active & (lev_prices * (1.0 + tol) >= lo) & (lev_prices * (1.0 - tol) <= hi)
            level_consumed[:] |= hit

    # Close remaining position
    if position is not None:
        position['exit_time'] = c_times[-1]
        position['exit_price'] = float(c_closes[-1])
        position['exit_reason'] = 'TIMEOUT'
        if position['direction'] == 'LONG':
            pnl = position['exit_price'] - position['entry_price']
        else:
            pnl = position['entry_price'] - position['exit_price']
        position['pnl'] = pnl
        position['pnl_pct'] = pnl / position['entry_price']
        position['rr_achieved'] = abs(pnl) / position['risk'] if position['risk'] > 0 else 0
        if pnl < 0:
            position['rr_achieved'] = -position['rr_achieved']
        trades.append(position)

    return trades


def analyze_results(trades):
    """Analyze level-to-level backtest results."""
    if not trades:
        print("No trades generated")
        return

    df = pd.DataFrame(trades)
    df['win'] = df['pnl'] > 0

    print(f"\n{'='*60}")
    print(f"LEVEL-TO-LEVEL BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total trades: {len(df)}")
    print(f"Win rate: {df['win'].mean():.1%}")
    print(f"Avg RR achieved: {df['rr_achieved'].mean():.2f}")
    print(f"Avg PnL%: {df['pnl_pct'].mean():.3%}")
    print(f"Total PnL%: {df['pnl_pct'].sum():.1%}")

    # Profit factor
    wins = df[df['pnl'] > 0]['pnl'].sum()
    losses = abs(df[df['pnl'] <= 0]['pnl'].sum())
    pf = wins / losses if losses > 0 else float('inf')
    print(f"Profit Factor: {pf:.2f}")

    # By exit reason
    print(f"\nBy exit reason:")
    for reason in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
        sub = df[df['exit_reason'] == reason]
        if len(sub) > 0:
            print(f"  {reason:10s}: {len(sub):>5} ({len(sub)/len(df):.0%}) avg_rr={sub['rr_achieved'].mean():+.2f}")

    # By entry level type
    print(f"\nBy entry level type:")
    for lt in df['level_type'].unique():
        sub = df[df['level_type'] == lt]
        if len(sub) >= 5:
            wr = sub['win'].mean()
            avg_rr = sub['rr_achieved'].mean()
            print(f"  {lt:30s}: {len(sub):>4} trades, WR={wr:.1%}, avg_RR={avg_rr:+.2f}")

    # By entry level TF
    print(f"\nBy entry level TF:")
    for tf in ['daily', 'weekly', 'monthly']:
        sub = df[df['level_tf'] == tf]
        if len(sub) >= 5:
            wr = sub['win'].mean()
            avg_rr = sub['rr_achieved'].mean()
            print(f"  {tf:10s}: {len(sub):>4} trades, WR={wr:.1%}, avg_RR={avg_rr:+.2f}")

    # By actual RR buckets
    print(f"\nBy actual RR of trade setup:")
    for low, high, label in [(0.5, 1.0, '0.5-1x'), (1.0, 2.0, '1-2x'), (2.0, 3.0, '2-3x'),
                              (3.0, 5.0, '3-5x'), (5.0, 100, '>5x')]:
        sub = df[(df['actual_rr'] >= low) & (df['actual_rr'] < high)]
        if len(sub) >= 5:
            wr = sub['win'].mean()
            avg_pnl = sub['pnl_pct'].mean()
            print(f"  RR {label:>6s}: {len(sub):>4} trades, WR={wr:.1%}, avg_pnl={avg_pnl:+.2%}")

    # By direction
    print(f"\nBy direction:")
    for d in ['LONG', 'SHORT']:
        sub = df[df['direction'] == d]
        if len(sub) >= 5:
            wr = sub['win'].mean()
            print(f"  {d:6s}: {len(sub):>4} trades, WR={wr:.1%}, avg_RR={sub['rr_achieved'].mean():+.2f}")

    # Best combos: entry level_type × entry level_tf
    print(f"\nBest combos (entry level × TF, min 5 trades):")
    combo = df.groupby(['level_type', 'level_tf']).agg(
        trades=('win', 'count'),
        wr=('win', 'mean'),
        avg_rr=('rr_achieved', 'mean'),
        avg_pnl=('pnl_pct', 'mean'),
    ).reset_index()
    combo = combo[combo['trades'] >= 5].sort_values('wr', ascending=False)
    for _, row in combo.head(15).iterrows():
        print(f"  {row['level_type']:25s} {row['level_tf']:8s}: "
              f"{row['trades']:>3.0f} trades, WR={row['wr']:.1%}, "
              f"avg_RR={row['avg_rr']:+.2f}, pnl={row['avg_pnl']:+.2%}")

    return df


def main():
    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        logger.info("Running level-to-level backtest on 4h...")
        start = time.time()
        trades = run_l2l_backtest(db.session, exec_tf='4h')
        elapsed = time.time() - start
        logger.info("Done in %.0fs, %d trades", elapsed, len(trades))

        result_df = analyze_results(trades)

        # Save to JSON
        out_path = 'scripts/level_to_level_results.json'
        with open(out_path, 'w') as f:
            json.dump(trades, f, indent=2, default=str)
        logger.info("Saved %d trades to %s", len(trades), out_path)


if __name__ == '__main__':
    main()
