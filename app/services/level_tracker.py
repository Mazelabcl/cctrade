"""Level touch tracking service.

Counts how many times price action touches each level.
A touch occurs when candle.low <= level.price <= candle.high.
"""
import logging

from sqlalchemy.orm import Session

from ..models import Candle, Level

logger = logging.getLogger(__name__)


def update_level_touches(db: Session, candle: Candle,
                         invalidate_on_first_touch: bool = False) -> int:
    """Update touch counts for all active levels based on a single candle.

    A touch is counted when the candle's range (low to high) includes
    the level price. No direction classification — just a simple count.

    Args:
        invalidate_on_first_touch: When True, levels are invalidated on the
            first price touch (used by backtesting signal generator).

    Returns the number of levels that were touched.
    """
    active_levels = (
        db.query(Level)
        .filter(Level.invalidated_at.is_(None))
        .filter(Level.created_at < candle.open_time)
        .all()
    )

    touched = 0
    for level in active_levels:
        if not (candle.low <= level.price_level <= candle.high):
            continue

        level.support_touches += 1

        if level.first_touched_at is None:
            level.first_touched_at = candle.open_time

        touched += 1
        if invalidate_on_first_touch:
            level.invalidated_at = candle.open_time

    db.commit()
    return touched


def run_touch_tracking(db: Session, timeframe: str = '1h',
                       symbol: str = 'BTCUSDT') -> dict:
    """Run touch tracking across all candles sequentially.

    Returns summary with total touches.
    """
    candles = (
        db.query(Candle)
        .filter_by(symbol=symbol, timeframe=timeframe)
        .order_by(Candle.open_time)
        .all()
    )

    total_touches = 0
    for candle in candles:
        total_touches += update_level_touches(db, candle)

    logger.info("Touch tracking: %d total touches", total_touches)
    return {
        'total_touches': total_touches,
    }
