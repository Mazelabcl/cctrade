"""Level touch tracking and invalidation service.

Preserved from legacy/level_touch_tracker.py with invalidation logic added.
"""
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..models import Candle, Level

logger = logging.getLogger(__name__)

HIT_COUNT_THRESHOLD = 4  # Levels are invalidated after this many total touches


def update_level_touches(db: Session, candle: Candle,
                         invalidation_threshold: int = HIT_COUNT_THRESHOLD,
                         invalidate_on_first_touch: bool = False) -> int:
    """Update touch counts for all active levels based on a single candle.

    Args:
        invalidate_on_first_touch: When True, levels are invalidated on the
            first price touch (used by backtesting signal generator).
            Default False preserves the original threshold-based behaviour.

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
        was_touched = False

        if candle.low <= level.price_level:
            level.support_touches += 1
            was_touched = True

        if candle.high >= level.price_level:
            level.resistance_touches += 1
            was_touched = True

        if was_touched:
            touched += 1
            if invalidate_on_first_touch:
                level.invalidated_at = candle.open_time
            else:
                total = level.support_touches + level.resistance_touches
                if total >= invalidation_threshold:
                    level.invalidated_at = candle.open_time

    db.commit()
    return touched


def run_touch_tracking(db: Session, timeframe: str = '1h',
                       symbol: str = 'BTCUSDT') -> dict:
    """Run touch tracking across all candles sequentially.

    Returns summary with total touches and invalidations.
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

    invalidated = db.query(Level).filter(Level.invalidated_at.isnot(None)).count()

    logger.info("Touch tracking: %d touches, %d levels invalidated", total_touches, invalidated)
    return {
        'total_touches': total_touches,
        'invalidated_count': invalidated,
    }
