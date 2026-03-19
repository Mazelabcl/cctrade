"""Fetch 1-minute BTCUSDT candles from Binance in monthly chunks.

Uses sqlite3 directly (not SQLAlchemy) to avoid DB lock issues with scheduler.
Progress is printed per month.

Usage:
    python scripts/fetch_1min.py
"""
import os
import sqlite3
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fetch_1min():
    # Get API keys directly from sqlite (avoid Flask app + scheduler lock)
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'instance', 'tradebot.db')
    tmp_conn = sqlite3.connect(db_path, timeout=30)
    row = tmp_conn.execute("SELECT value FROM settings WHERE key='binance_api_key'").fetchone()
    api_key = row[0] if row else ''
    row = tmp_conn.execute("SELECT value FROM settings WHERE key='binance_api_secret'").fetchone()
    api_secret = row[0] if row else ''
    tmp_conn.close()

    if not api_key:
        print('ERROR: No Binance API key configured')
        return

    from binance.client import Client
    client = Client(api_key, api_secret)

    # Use sqlite3 directly with WAL mode and busy timeout
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 60000")

    symbol = 'BTCUSDT'
    interval = '1m'
    start = datetime(2017, 8, 1)
    end = datetime(2026, 1, 1)

    total_inserted = 0
    overall_start = time.time()

    current = start
    while current < end:
        if current.month == 12:
            next_month = datetime(current.year + 1, 1, 1)
        else:
            next_month = datetime(current.year, current.month + 1, 1)

        range_start = current.strftime('%d %b %Y')
        range_end = min(next_month, end).strftime('%d %b %Y')

        # Check existing count
        existing = conn.execute(
            "SELECT count(*) FROM candles WHERE symbol=? AND timeframe=? "
            "AND open_time >= ? AND open_time < ?",
            (symbol, interval, current.strftime('%Y-%m-%d'),
             min(next_month, end).strftime('%Y-%m-%d'))
        ).fetchone()[0]

        expected = int((min(next_month, end) - current).total_seconds() / 60)
        if existing >= expected * 0.99:
            print(f'{current.strftime("%Y-%m")}: {existing:>6,} already fetched, skipping')
            current = next_month
            continue

        chunk_start = time.time()
        try:
            klines = client.get_historical_klines(symbol, interval, range_start, range_end)
        except Exception as exc:
            print(f'{current.strftime("%Y-%m")}: ERROR fetching: {exc}')
            current = next_month
            continue

        if not klines:
            print(f'{current.strftime("%Y-%m")}: no klines returned')
            current = next_month
            continue

        rows = []
        for k in klines:
            open_time = pd.to_datetime(k[0], unit='ms').to_pydatetime()
            rows.append((
                symbol, interval, str(open_time),
                float(k[1]), float(k[2]), float(k[3]), float(k[4]),
                float(k[5]), float(k[7]), int(k[8]),
            ))

        conn.executemany(
            'INSERT OR IGNORE INTO candles '
            '(symbol, timeframe, open_time, "open", high, low, close, volume, quote_volume, num_trades) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            rows
        )
        conn.commit()

        elapsed = time.time() - chunk_start
        total_inserted += len(rows)
        print(f'{current.strftime("%Y-%m")}: {len(rows):>6,} candles in {elapsed:.1f}s (total: {total_inserted:>10,})')

        current = next_month

    total_elapsed = time.time() - overall_start
    print(f'\nDone! {total_inserted:,} candles in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)')

    count = conn.execute("SELECT count(*) FROM candles WHERE timeframe='1m'").fetchone()[0]
    print(f'Total 1min candles in DB: {count:,}')
    conn.close()


if __name__ == '__main__':
    fetch_1min()
