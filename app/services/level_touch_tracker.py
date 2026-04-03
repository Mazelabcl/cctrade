"""Level Touch Tracker — marks levels as touched when price crosses them.

CRITICAL: This must run after every candle sync to keep first_touched_at
accurate. Without this, levels appear "naked" forever even after price
has crossed them, leading to false trade signals.

A level is "touched" when a candle's [low, high] range overlaps the level
price within a tolerance of 0.3%.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from ..extensions import db
from ..models.level import Level
from ..models.candle import Candle
from sqlalchemy import text

logger = logging.getLogger(__name__)

TOUCH_TOLERANCE = 0.003  # 0.3% — same as confluence scalper

# Structural level types that use first_touched_at
STRUCTURAL_TYPES = [
    'Fractal_support', 'Fractal_resistance', 'HTF_level',
    'Fib_CC', 'Fib_0.25', 'Fib_0.50', 'Fib_0.75',
]


def update_touched_levels(session=None, timeframe='15m', since=None):
    """Check all naked structural levels against candles and mark touched ones.

    Args:
        session: DB session (uses db.session if None)
        timeframe: Candle timeframe to check against (default 15m for highest granularity)
        since: Only check candles after this datetime (None = check all)

    Returns:
        Number of levels updated
    """
    if session is None:
        session = db.session

    # Load naked structural levels (D-W-M only)
    naked_levels = session.query(Level).filter(
        Level.invalidated_at.is_(None),
        Level.first_touched_at.is_(None),
        Level.level_type.in_(STRUCTURAL_TYPES),
        Level.timeframe.in_(['daily', 'weekly', 'monthly']),
    ).all()

    if not naked_levels:
        logger.info("No naked structural levels to check")
        return 0

    # Load candles
    query = (
        "SELECT open_time, high, low FROM candles "
        "WHERE timeframe = :tf "
    )
    params = {'tf': timeframe}

    if since:
        query += "AND open_time >= :since "
        params['since'] = since

    query += "ORDER BY open_time ASC"

    rows = session.execute(text(query), params).fetchall()
    if not rows:
        logger.info("No candles to check against")
        return 0

    # Convert to numpy for speed
    c_times = pd.to_datetime([r[0] for r in rows], format='mixed').values
    c_highs = np.array([r[1] for r in rows], dtype=np.float64)
    c_lows = np.array([r[2] for r in rows], dtype=np.float64)
    n_candles = len(c_highs)

    updated = 0

    for level in naked_levels:
        lp = level.price_level
        created = level.created_at
        if created is None:
            continue

        # Find first candle after level creation
        created_np = np.datetime64(pd.Timestamp(created))
        start_idx = np.searchsorted(c_times, created_np)

        if start_idx >= n_candles:
            continue

        # Check if any candle's range overlaps the level price (with tolerance)
        hi_threshold = lp * (1 + TOUCH_TOLERANCE)
        lo_threshold = lp * (1 - TOUCH_TOLERANCE)

        touched_mask = (c_lows[start_idx:] <= hi_threshold) & \
                       (c_highs[start_idx:] >= lo_threshold)

        if touched_mask.any():
            first_touch_idx = start_idx + np.argmax(touched_mask)
            touch_time = pd.Timestamp(c_times[first_touch_idx])

            if touch_time > pd.Timestamp(created):
                level.first_touched_at = touch_time.to_pydatetime()
                level.support_touches = (level.support_touches or 0) + 1
                updated += 1

    session.commit()
    logger.info("Updated %d levels with first_touched_at (checked %d naked levels "
                "against %d candles)", updated, len(naked_levels), n_candles)
    return updated
