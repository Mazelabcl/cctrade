"""Periodic data fetching task.

Can be triggered manually via API or scheduled by APScheduler.
"""
import logging

from flask import current_app
from ..extensions import db
from ..services.data_fetcher import fetch_candles

logger = logging.getLogger(__name__)


def sync_candle_data():
    """Fetch latest candles from Binance for all configured timeframes."""
    api_key = current_app.config.get('BINANCE_API_KEY', '')
    api_secret = current_app.config.get('BINANCE_API_SECRET', '')

    if not api_key or not api_secret:
        logger.warning("Binance API credentials not configured, skipping data sync")
        return 0

    symbol = current_app.config.get('SYMBOL', 'BTCUSDT')
    total = 0

    for interval in ['1h']:
        try:
            count = fetch_candles(
                db.session,
                symbol=symbol,
                interval=interval,
                api_key=api_key,
                api_secret=api_secret,
            )
            total += count
            logger.info("Synced %d candles for %s %s", count, symbol, interval)
        except Exception as e:
            logger.error("Failed to sync %s %s: %s", symbol, interval, e)

    return total
