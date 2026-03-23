"""Maximum Favorable Excursion (MFE) Analysis.

For each trade in the DB, removes the TP and lets the trade run
until the SL is hit or the dataset ends. Measures:
- max_rr: maximum RR achieved before SL hit
- candles_to_max: how many candles to reach the max
- max_price: highest/lowest price reached (depending on direction)
- final_exit: 'SL_HIT' or 'DATASET_END'

This answers: "How much money are we leaving on the table with fixed RR?"

Usage:
    python scripts/mfe_analysis.py [--exec_tf 4h] [--limit 5000]
"""
import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def run_mfe_analysis(session, exec_tf='4h', strategy='wick_rr_1.0',
                     limit=None, level_types=None):
    """Run MFE analysis on existing trades.

    Instead of re-simulating entries, we use the already-computed trades
    from IndividualLevelTrade and just re-simulate the EXIT by removing
    the TP and letting the trade run until SL hit.
    """
    from app.services.level_trade_backtest_db import load_candles_db

    # Load candles
    candles = load_candles_db(session, timeframe=exec_tf)
    if candles.empty:
        print(f"No candles found for {exec_tf}")
        return []

    c_times = candles['open_time'].values
    c_highs = candles['high'].values.astype(np.float64)
    c_lows = candles['low'].values.astype(np.float64)
    c_closes = candles['close'].values.astype(np.float64)

    # Build time->index map
    time_to_idx = {}
    for idx, t in enumerate(c_times):
        time_to_idx[pd.Timestamp(t)] = idx

    # Load trades from DB
    from sqlalchemy import text
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
    params = {'tf': exec_tf, 'strategy': strategy}

    if level_types:
        # Can't parameterize IN clause easily, so filter after
        pass

    result = session.execute(text(sql), params)
    rows = result.fetchall()

    if level_types:
        rows = [r for r in rows if r[8] in level_types]

    if limit:
        rows = rows[:limit]

    print(f"Analyzing {len(rows)} trades ({exec_tf}, {strategy})...", flush=True)

    mfe_results = []
    batch_size = 5000
    t0 = time.time()

    for trade_idx, row in enumerate(rows):
        trade_id = row[0]
        entry_time = pd.Timestamp(row[1])
        entry_price = float(row[2])
        stop_loss = float(row[3])
        direction = row[4]
        original_exit = row[5]
        original_pnl = float(row[6]) if row[6] else 0
        original_tp = float(row[7]) if row[7] else None
        level_type = row[8]
        level_tf = row[9]

        risk = abs(entry_price - stop_loss)
        if risk == 0:
            continue

        # Find the candle index for entry
        idx = time_to_idx.get(entry_time)
        if idx is None:
            continue

        # Let it run: scan forward until SL hit or dataset end
        max_favorable = 0.0  # max profit in price terms
        max_price = entry_price
        candles_to_max = 0
        exit_reason = 'DATASET_END'
        exit_candle = len(candles) - 1
        exit_price = c_closes[-1]

        # Track drawdown from max
        max_adverse = 0.0  # max drawdown from entry

        for j in range(idx + 1, len(candles)):
            if direction == 'LONG':
                # Check SL
                if c_lows[j] <= stop_loss:
                    exit_reason = 'SL_HIT'
                    exit_candle = j
                    exit_price = stop_loss
                    break

                # Track max favorable
                high = c_highs[j]
                favorable = high - entry_price
                if favorable > max_favorable:
                    max_favorable = favorable
                    max_price = high
                    candles_to_max = j - idx

                # Track max adverse from entry
                adverse = entry_price - c_lows[j]
                if adverse > max_adverse:
                    max_adverse = adverse

            else:  # SHORT
                # Check SL
                if c_highs[j] >= stop_loss:
                    exit_reason = 'SL_HIT'
                    exit_candle = j
                    exit_price = stop_loss
                    break

                # Track max favorable
                low = c_lows[j]
                favorable = entry_price - low
                if favorable > max_favorable:
                    max_favorable = favorable
                    max_price = low
                    candles_to_max = j - idx

                # Track max adverse from entry
                adverse = c_highs[j] - entry_price
                if adverse > max_adverse:
                    max_adverse = adverse

        max_rr = max_favorable / risk if risk > 0 else 0
        max_adverse_rr = max_adverse / risk if risk > 0 else 0
        total_candles = exit_candle - idx

        # Original RR achieved
        if original_tp and original_exit == 'TP_HIT':
            original_rr = abs(original_tp - entry_price) / risk
        elif original_exit == 'SL_HIT':
            original_rr = -1.0
        else:
            original_rr = 0

        mfe_results.append({
            'trade_id': trade_id,
            'entry_time': str(entry_time),
            'entry_price': entry_price,
            'direction': direction,
            'risk': round(risk, 2),
            'risk_pct': round(risk / entry_price * 100, 3),
            'level_type': level_type,
            'level_tf': level_tf,
            'original_exit': original_exit,
            'original_rr': round(original_rr, 2),
            'max_rr': round(max_rr, 2),
            'max_price': round(max_price, 2),
            'candles_to_max': candles_to_max,
            'max_adverse_rr': round(max_adverse_rr, 2),
            'final_exit': exit_reason,
            'total_candles': total_candles,
            'money_left': round(max_rr - max(original_rr, 0), 2),
        })

        if (trade_idx + 1) % batch_size == 0:
            elapsed = time.time() - t0
            print(f"  Processed {trade_idx + 1}/{len(rows)} trades ({elapsed:.0f}s)...",
                  flush=True)

    elapsed = time.time() - t0
    print(f"MFE analysis complete: {len(mfe_results)} trades in {elapsed:.0f}s", flush=True)
    return mfe_results


def print_summary(results):
    """Print MFE summary statistics."""
    if not results:
        print("No results to summarize")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("MFE ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nTotal trades: {len(df)}")
    print(f"SL hit (let-it-run): {(df['final_exit'] == 'SL_HIT').sum()} "
          f"({(df['final_exit'] == 'SL_HIT').mean()*100:.1f}%)")
    print(f"Dataset end (never hit SL): {(df['final_exit'] == 'DATASET_END').sum()} "
          f"({(df['final_exit'] == 'DATASET_END').mean()*100:.1f}%)")

    print(f"\n--- Max RR Achieved ---")
    print(f"Mean:   {df['max_rr'].mean():.1f}:1")
    print(f"Median: {df['max_rr'].median():.1f}:1")
    print(f"P75:    {df['max_rr'].quantile(0.75):.1f}:1")
    print(f"P90:    {df['max_rr'].quantile(0.90):.1f}:1")
    print(f"P95:    {df['max_rr'].quantile(0.95):.1f}:1")
    print(f"Max:    {df['max_rr'].max():.1f}:1")

    print(f"\n--- How many trades reach each RR? ---")
    for rr in [1, 2, 3, 5, 10, 15, 20, 30, 50]:
        pct = (df['max_rr'] >= rr).mean() * 100
        count = (df['max_rr'] >= rr).sum()
        print(f"  >= {rr:2d}:1 => {count:6d} trades ({pct:5.1f}%)")

    print(f"\n--- Money Left on Table (max_rr - original_rr) ---")
    winners = df[df['original_exit'] == 'TP_HIT']
    if not winners.empty:
        print(f"Winners only ({len(winners)} trades):")
        print(f"  Mean money left: {winners['money_left'].mean():.1f}R")
        print(f"  Median:          {winners['money_left'].median():.1f}R")
        print(f"  Max:             {winners['money_left'].max():.1f}R")

    print(f"\n--- Candles to Max ---")
    print(f"Mean:   {df['candles_to_max'].mean():.0f} candles")
    print(f"Median: {df['candles_to_max'].median():.0f} candles")
    print(f"P90:    {df['candles_to_max'].quantile(0.90):.0f} candles")

    print(f"\n--- By Level Type ---")
    by_type = df.groupby('level_type').agg(
        trades=('max_rr', 'count'),
        avg_max_rr=('max_rr', 'mean'),
        median_max_rr=('max_rr', 'median'),
        pct_above_5=('max_rr', lambda x: (x >= 5).mean() * 100),
        avg_candles=('candles_to_max', 'mean'),
    ).sort_values('avg_max_rr', ascending=False)

    print(f"{'Type':25s} {'Trades':>7s} {'Avg RR':>7s} {'Med RR':>7s} {'>=5:1':>6s} {'Candles':>8s}")
    print("-" * 65)
    for lt, row in by_type.iterrows():
        print(f"{lt:25s} {row['trades']:>7.0f} {row['avg_max_rr']:>6.1f}:1 "
              f"{row['median_max_rr']:>6.1f}:1 {row['pct_above_5']:>5.1f}% "
              f"{row['avg_candles']:>7.0f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MFE Analysis')
    parser.add_argument('--exec_tf', default='4h', help='Execution timeframe')
    parser.add_argument('--strategy', default='wick_rr_1.0', help='Strategy name')
    parser.add_argument('--limit', type=int, default=None, help='Max trades to analyze')
    parser.add_argument('--output', default='scripts/mfe_results.json', help='Output file')
    args = parser.parse_args()

    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        results = run_mfe_analysis(
            db.session,
            exec_tf=args.exec_tf,
            strategy=args.strategy,
            limit=args.limit,
        )

        print_summary(results)

        # Save to JSON
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
