"""Binance API data fetcher service.

Wraps legacy/data_fetching.py logic, writes directly to SQLite.
"""
import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy.orm import Session

from ..models import Candle, PipelineRun

logger = logging.getLogger(__name__)


def fetch_candles(db: Session, symbol: str = 'BTCUSDT',
                  interval: str = '1h',
                  start_str: str = '1 Jan 2017',
                  end_str: str = '30 Jun 2025',
                  api_key: str = '',
                  api_secret: str = '') -> int:
    """Fetch candles from Binance and persist to database.

    Returns number of new candles inserted.
    """
    try:
        from binance.client import Client
    except ImportError:
        logger.error("python-binance not installed")
        raise

    if not api_key or not api_secret:
        raise ValueError("Binance API key and secret are required")

    run = PipelineRun(
        pipeline_type='data_fetch',
        status='running',
        started_at=datetime.now(timezone.utc),
        metadata_json={'symbol': symbol, 'interval': interval},
    )
    db.add(run)
    db.commit()

    try:
        client = Client(api_key, api_secret)

        # Check for historical gap (earliest candle later than requested start)
        earliest = (
            db.query(Candle)
            .filter_by(symbol=symbol, timeframe=interval)
            .order_by(Candle.open_time.asc())
            .first()
        )
        # Find latest candle so we only fetch new data going forward
        latest = (
            db.query(Candle)
            .filter_by(symbol=symbol, timeframe=interval)
            .order_by(Candle.open_time.desc())
            .first()
        )

        # Build list of ranges to fetch: historical gap + forward from latest
        ranges = []
        if earliest and earliest.open_time.strftime('%Y-%m-%d') > start_str[:10]:
            # Gap: from configured start up to the earliest stored candle
            ranges.append((start_str, earliest.open_time.strftime('%d %b %Y %H:%M:%S')))
        if latest:
            ranges.append((latest.open_time.strftime('%d %b %Y %H:%M:%S'), end_str))
        else:
            ranges.append((start_str, end_str))

        all_klines = []
        for range_start, range_end in ranges:
            chunk = client.get_historical_klines(symbol, interval, range_start, range_end)
            all_klines.extend(chunk)

        klines = all_klines

        if not klines:
            run.status = 'completed'
            run.finished_at = datetime.now(timezone.utc)
            run.rows_processed = 0
            db.commit()
            return 0

        inserted = 0
        for k in klines:
            open_time = pd.to_datetime(k[0], unit='ms')

            exists = db.query(Candle.id).filter_by(
                symbol=symbol, timeframe=interval, open_time=open_time,
            ).first()
            if exists:
                continue

            candle = Candle(
                symbol=symbol,
                timeframe=interval,
                open_time=open_time,
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                quote_volume=float(k[7]),
                num_trades=int(k[8]),
            )
            db.add(candle)
            inserted += 1

        db.commit()

        run.status = 'completed'
        run.finished_at = datetime.now(timezone.utc)
        run.rows_processed = inserted
        run.period_start = pd.to_datetime(klines[0][0], unit='ms')
        run.period_end = pd.to_datetime(klines[-1][0], unit='ms')
        db.commit()

        logger.info("Fetched %d new candles for %s %s", inserted, symbol, interval)
        return inserted

    except Exception as exc:
        run.status = 'failed'
        run.finished_at = datetime.now(timezone.utc)
        run.error_message = str(exc)
        db.commit()
        raise
