"""Target variable generation for ML models.

Preserved from legacy/target_variable.py. Creates fractal direction targets
with configurable prediction horizons.
"""
import logging
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..models import Candle

logger = logging.getLogger(__name__)

PREDICTION_HORIZONS = {
    'hour': 1,
    'day': 24,
    'week': 168,
    '15days': 360,
    'month': 720,
}


def create_fractal_targets(
    db: Session,
    prediction_horizon: str = 'day',
    timeframe: str = '1h',
    symbol: str = 'BTCUSDT',
) -> pd.DataFrame:
    """Create multi-class fractal direction targets.

    Returns DataFrame with columns: candle_id, fractal_direction (0/1/2).
    """
    horizon_candles = PREDICTION_HORIZONS.get(prediction_horizon, 24)

    candles = (
        db.query(Candle)
        .filter_by(symbol=symbol, timeframe=timeframe)
        .order_by(Candle.open_time)
        .all()
    )

    if not candles:
        return pd.DataFrame(columns=['candle_id', 'fractal_direction'])

    bullish = [c.bullish_fractal for c in candles]
    bearish = [c.bearish_fractal for c in candles]
    ids = [c.id for c in candles]

    targets = []
    max_idx = len(candles) - horizon_candles

    for i in range(max_idx):
        future_start = i + 1
        future_end = i + 1 + horizon_candles

        b_count = sum(bullish[future_start:future_end])
        d_count = sum(bearish[future_start:future_end])

        if b_count > 0 and d_count > 0:
            # Both present — choose first chronologically
            first_b = next((j for j in range(future_start, future_end) if bullish[j]), future_end)
            first_d = next((j for j in range(future_start, future_end) if bearish[j]), future_end)
            target = 1 if first_b < first_d else 2
        elif b_count > 0:
            target = 1
        elif d_count > 0:
            target = 2
        else:
            target = 0

        targets.append({'candle_id': ids[i], 'fractal_direction': target})

    df = pd.DataFrame(targets)

    if not df.empty:
        counts = df['fractal_direction'].value_counts().sort_index()
        total = len(df)
        logger.info(
            "Targets (%s/%d candles): no_fractal=%d (%.1f%%), bullish=%d (%.1f%%), bearish=%d (%.1f%%)",
            prediction_horizon, horizon_candles,
            counts.get(0, 0), counts.get(0, 0) / total * 100,
            counts.get(1, 0), counts.get(1, 0) / total * 100,
            counts.get(2, 0), counts.get(2, 0) / total * 100,
        )

    return df
