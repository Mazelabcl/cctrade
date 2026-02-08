"""Accuracy tracking — backfill actual_class on predictions when truth is known.

After new candles arrive and fractals are detected, we can determine
whether past predictions were correct.
"""
import logging
from sqlalchemy.orm import Session

from ..models import Candle, Prediction

logger = logging.getLogger(__name__)

# Map prediction horizons to candle lookahead counts
HORIZON_CANDLES = {
    'hour': 1,
    'day': 24,
    'week': 168,
    'month': 720,
}


def backfill_actuals(db: Session, timeframe: str = '1h',
                     symbol: str = 'BTCUSDT') -> int:
    """Fill in actual_class for predictions where the outcome is now known.

    Returns number of predictions updated.
    """
    # Get predictions without actual_class
    pending = (
        db.query(Prediction)
        .filter(Prediction.actual_class.is_(None))
        .all()
    )

    if not pending:
        return 0

    # Build a candle lookup ordered by time
    candles = (
        db.query(Candle)
        .filter_by(symbol=symbol, timeframe=timeframe)
        .order_by(Candle.open_time)
        .all()
    )
    candle_by_id = {c.id: c for c in candles}
    candle_list = candles
    id_to_idx = {c.id: i for i, c in enumerate(candle_list)}

    updated = 0
    for pred in pending:
        candle = candle_by_id.get(pred.candle_id)
        if not candle:
            continue

        idx = id_to_idx.get(pred.candle_id)
        if idx is None:
            continue

        # Get prediction horizon from the model
        model = pred.model
        if not model:
            continue
        horizon_key = model.prediction_horizon or 'day'
        lookahead = HORIZON_CANDLES.get(horizon_key, 24)

        future_start = idx + 1
        future_end = idx + 1 + lookahead

        # Check if we have enough future candles
        if future_end > len(candle_list):
            continue

        # Determine actual class
        future_candles = candle_list[future_start:future_end]
        bullish_found = any(c.bullish_fractal for c in future_candles)
        bearish_found = any(c.bearish_fractal for c in future_candles)

        if bullish_found and bearish_found:
            first_b = next(i for i, c in enumerate(future_candles) if c.bullish_fractal)
            first_d = next(i for i, c in enumerate(future_candles) if c.bearish_fractal)
            actual = 1 if first_b < first_d else 2
        elif bullish_found:
            actual = 1
        elif bearish_found:
            actual = 2
        else:
            actual = 0

        pred.actual_class = actual
        updated += 1

    if updated:
        db.commit()
        logger.info("Backfilled actual_class for %d predictions", updated)

    return updated
