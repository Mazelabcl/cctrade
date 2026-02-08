"""Background feature computation task.

Can be triggered manually via API or by the automated pipeline.
"""
import logging

from ..extensions import db
from ..services.feature_engine import compute_features

logger = logging.getLogger(__name__)


def compute_all_features(timeframe: str = '1h', symbol: str = 'BTCUSDT') -> int:
    """Compute features for all candles missing features."""
    try:
        count = compute_features(db.session, timeframe=timeframe, symbol=symbol)
        logger.info("Computed %d features for %s %s", count, symbol, timeframe)
        return count
    except Exception as e:
        logger.error("Feature computation failed: %s", e)
        raise
