#!/usr/bin/env python
"""Build candles for any timeframe from 1-minute data.

Since we have 4.4M 1-minute candles, we can construct any TF
by aggregating (OHLCV rules: first open, max high, min low, last close, sum volume).

Usage:
    python scripts/build_candles_from_1min.py --tf 15m
    python scripts/build_candles_from_1min.py --tf 30m
    python scripts/build_candles_from_1min.py --tf 15m 30m 2h
    python scripts/build_candles_from_1min.py --all  # 5m, 15m, 30m, 2h, 3h
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Map TF label to number of 1-minute candles
TF_MINUTES = {
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '2h': 120,
    '3h': 180,
}

ALL_TFS = list(TF_MINUTES.keys())


def build_candles(session, target_tf, symbol='BTCUSDT'):
    """Build candles for target_tf from 1-minute data."""
    from sqlalchemy import text
    import pandas as pd
    import numpy as np

    minutes = TF_MINUTES.get(target_tf)
    if not minutes:
        print(f"ERROR: Unknown TF '{target_tf}'. Supported: {list(TF_MINUTES.keys())}")
        return 0

    # Check if candles already exist for this TF
    existing = session.execute(text(
        "SELECT COUNT(*) FROM candles WHERE timeframe = :tf AND symbol = :sym"
    ), {'tf': target_tf, 'sym': symbol}).fetchone()[0]
    if existing > 0:
        print(f"  {target_tf}: {existing:,d} candles already exist. Deleting to rebuild...")
        session.execute(text(
            "DELETE FROM candles WHERE timeframe = :tf AND symbol = :sym"
        ), {'tf': target_tf, 'sym': symbol})
        session.commit()

    # Load all 1-minute candles
    print(f"  Loading 1-minute candles...", flush=True)
    t0 = time.time()
    rows = session.execute(text(
        "SELECT open_time, open, high, low, close, volume "
        "FROM candles WHERE timeframe = '1m' AND symbol = :sym "
        "ORDER BY open_time ASC"
    ), {'sym': symbol}).fetchall()

    if not rows:
        print("  ERROR: No 1-minute candles found!")
        return 0

    df = pd.DataFrame(rows, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
    df['open_time'] = pd.to_datetime(df['open_time'])
    print(f"  Loaded {len(df):,d} 1-min candles in {time.time()-t0:.0f}s", flush=True)

    # Floor timestamps to target TF boundaries
    freq = f'{minutes}min'
    df['bucket'] = df['open_time'].dt.floor(freq)

    # Aggregate
    print(f"  Aggregating to {target_tf}...", flush=True)
    agg = df.groupby('bucket').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        count=('open', 'count'),
    ).reset_index()

    # Filter: only keep complete candles (have all N minutes)
    complete = agg[agg['count'] == minutes].copy()
    incomplete = len(agg) - len(complete)
    if incomplete > 0:
        print(f"  Dropped {incomplete} incomplete candles "
              f"(need {minutes} 1-min candles each)", flush=True)

    # Insert into DB
    print(f"  Inserting {len(complete):,d} candles...", flush=True)
    t0 = time.time()

    batch = []
    for _, row in complete.iterrows():
        batch.append({
            'symbol': symbol,
            'timeframe': target_tf,
            'open_time': str(row['bucket']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
        })

    # Batch insert
    if batch:
        session.execute(text(
            "INSERT INTO candles (symbol, timeframe, open_time, open, high, low, close, volume) "
            "VALUES (:symbol, :timeframe, :open_time, :open, :high, :low, :close, :volume)"
        ), batch)
        session.commit()

    elapsed = time.time() - t0
    print(f"  Inserted {len(batch):,d} {target_tf} candles in {elapsed:.0f}s", flush=True)
    return len(batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build candles from 1-minute data')
    parser.add_argument('--tf', nargs='+', help='Target timeframe(s)')
    parser.add_argument('--all', action='store_true', help='Build all supported TFs')
    args = parser.parse_args()

    tfs = ALL_TFS if args.all else (args.tf or [])
    if not tfs:
        print("ERROR: Specify --tf or --all")
        print(f"Supported: {ALL_TFS}")
        sys.exit(1)

    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        print("=" * 60)
        print("BUILDING CANDLES FROM 1-MINUTE DATA")
        print("=" * 60)

        for tf in tfs:
            print(f"\n--- {tf} ---", flush=True)
            t0 = time.time()
            count = build_candles(db.session, tf)
            print(f"  Done: {count:,d} candles ({time.time()-t0:.0f}s)", flush=True)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        from sqlalchemy import text
        rows = db.session.execute(text(
            "SELECT timeframe, COUNT(*), MIN(open_time), MAX(open_time) "
            "FROM candles GROUP BY timeframe ORDER BY COUNT(*) DESC"
        )).fetchall()
        for tf, cnt, mn, mx in rows:
            print(f"  {tf:6s}  {cnt:>9,d}  [{mn[:10]} to {mx[:10]}]")
