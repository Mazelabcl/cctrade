"""Periodic data fetching task.

Can be triggered manually via API or scheduled by APScheduler.
"""
import logging
from datetime import datetime, timezone

from flask import current_app
from ..extensions import db
from ..services.data_fetcher import fetch_candles

logger = logging.getLogger(__name__)

DEFAULT_TIMEFRAMES = '1h'


def get_last_fetch_time() -> datetime | None:
    """Get last sync time from DB setting (persists across restarts)."""
    try:
        from ..models.setting import get_setting
        ts = get_setting('last_sync_at')
        if ts:
            return datetime.fromisoformat(ts)
    except Exception:
        pass
    return None


def sync_candle_data():
    """Fetch latest candles from Binance for all configured timeframes.

    Reads API keys and timeframes from DB settings first, falling back to Flask config.
    """
    from ..models.setting import set_setting

    # DB settings take priority, fall back to Flask config (.env)
    try:
        from ..models.setting import get_setting
        api_key = get_setting('binance_api_key') or current_app.config.get('BINANCE_API_KEY', '')
        api_secret = get_setting('binance_api_secret') or current_app.config.get('BINANCE_API_SECRET', '')
        timeframes_str = get_setting('sync_timeframes', DEFAULT_TIMEFRAMES)
    except Exception:
        api_key = current_app.config.get('BINANCE_API_KEY', '')
        api_secret = current_app.config.get('BINANCE_API_SECRET', '')
        timeframes_str = DEFAULT_TIMEFRAMES

    if not api_key or not api_secret:
        logger.warning("Binance API credentials not configured, skipping data sync")
        return 0

    symbol = current_app.config.get('SYMBOL', 'BTCUSDT')
    timeframes = [tf.strip() for tf in timeframes_str.split(',') if tf.strip()]
    total = 0

    for interval in timeframes:
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

    # Persist sync metadata to DB
    now = datetime.now(timezone.utc)
    try:
        set_setting('last_sync_at', now.isoformat())
        set_setting('last_sync_candles', str(total))
    except Exception:
        pass

    return total
