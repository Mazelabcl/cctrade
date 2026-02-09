"""Convert ML predictions into actionable trade signals.

Combines fractal predictions with level proximity and ATR-based
stop-loss / take-profit sizing to produce LONG / SHORT / FLAT signals.
"""
import logging
from dataclasses import dataclass

import pandas as pd
from sqlalchemy.orm import Session

from ..config import Config
from ..models import Candle, Feature, Level, Prediction

logger = logging.getLogger(__name__)

SIGNAL_LONG = 'LONG'
SIGNAL_SHORT = 'SHORT'
SIGNAL_FLAT = 'FLAT'


@dataclass
class TradeSignal:
    candle_id: int
    signal: str            # LONG / SHORT / FLAT
    confidence: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    atr: float | None
    reason: str


def generate_signal(
    prediction: Prediction,
    candle: Candle,
    feature: Feature | None,
    nearest_support: float | None,
    nearest_resistance: float | None,
    confidence_threshold: float = Config.BACKTEST_CONFIDENCE_THRESHOLD,
    level_proximity_pct: float = Config.BACKTEST_LEVEL_PROXIMITY_PCT,
    atr_sl_mult: float = Config.BACKTEST_ATR_SL_MULT,
    atr_tp_mult: float = Config.BACKTEST_ATR_TP_MULT,
) -> TradeSignal:
    """Generate a single trade signal from a prediction.

    Returns a TradeSignal with direction, SL, and TP.
    """
    close = candle.close
    atr = feature.atr_14 if feature and feature.atr_14 else None

    # Default FLAT
    flat = TradeSignal(
        candle_id=candle.id, signal=SIGNAL_FLAT, confidence=prediction.confidence or 0,
        entry_price=close, stop_loss=None, take_profit=None, atr=atr, reason='',
    )

    # Check confidence
    conf = prediction.confidence or 0
    if conf < confidence_threshold:
        flat.reason = f'low_confidence ({conf:.3f} < {confidence_threshold})'
        return flat

    # Bullish fractal prediction → LONG
    if prediction.predicted_class == 1:
        if nearest_support is not None:
            dist = abs(close - nearest_support) / close if close else 1
            if dist > level_proximity_pct:
                flat.reason = f'support_too_far ({dist:.3f} > {level_proximity_pct})'
                return flat
        sl = close - atr * atr_sl_mult if atr else None
        tp = close + atr * atr_tp_mult if atr else None
        return TradeSignal(
            candle_id=candle.id, signal=SIGNAL_LONG, confidence=conf,
            entry_price=close, stop_loss=sl, take_profit=tp, atr=atr,
            reason='bullish_fractal_near_support',
        )

    # Bearish fractal prediction → SHORT
    if prediction.predicted_class == 2:
        if nearest_resistance is not None:
            dist = abs(nearest_resistance - close) / close if close else 1
            if dist > level_proximity_pct:
                flat.reason = f'resistance_too_far ({dist:.3f} > {level_proximity_pct})'
                return flat
        sl = close + atr * atr_sl_mult if atr else None
        tp = close - atr * atr_tp_mult if atr else None
        return TradeSignal(
            candle_id=candle.id, signal=SIGNAL_SHORT, confidence=conf,
            entry_price=close, stop_loss=sl, take_profit=tp, atr=atr,
            reason='bearish_fractal_near_resistance',
        )

    # predicted_class == 0 (no fractal)
    flat.reason = 'no_fractal_predicted'
    return flat


def generate_signals_batch(
    db: Session,
    model_id: int,
    confidence_threshold: float = Config.BACKTEST_CONFIDENCE_THRESHOLD,
    level_proximity_pct: float = Config.BACKTEST_LEVEL_PROXIMITY_PCT,
) -> list[TradeSignal]:
    """Generate signals for all predictions belonging to a model.

    Returns a list of TradeSignal objects ordered by candle open_time.
    """
    predictions = (
        db.query(Prediction)
        .filter_by(model_id=model_id)
        .join(Candle, Prediction.candle_id == Candle.id)
        .order_by(Candle.open_time)
        .all()
    )

    if not predictions:
        return []

    # Pre-load active levels as DataFrame for proximity checks
    levels = db.query(Level).filter(Level.invalidated_at.is_(None)).all()
    support_prices = sorted(
        [l.price_level for l in levels], reverse=True,
    )
    resistance_prices = sorted(
        [l.price_level for l in levels],
    )

    signals = []
    for pred in predictions:
        candle = db.query(Candle).get(pred.candle_id)
        feature = db.query(Feature).filter_by(candle_id=pred.candle_id).first()

        # Find nearest support (below or at close) and resistance (above close)
        nearest_sup = None
        for p in support_prices:
            if p <= candle.close:
                nearest_sup = p
                break
        nearest_res = None
        for p in resistance_prices:
            if p > candle.close:
                nearest_res = p
                break

        sig = generate_signal(
            pred, candle, feature, nearest_sup, nearest_res,
            confidence_threshold=confidence_threshold,
            level_proximity_pct=level_proximity_pct,
        )
        signals.append(sig)

    logger.info(
        "Generated %d signals for model %d: %d LONG, %d SHORT, %d FLAT",
        len(signals), model_id,
        sum(1 for s in signals if s.signal == SIGNAL_LONG),
        sum(1 for s in signals if s.signal == SIGNAL_SHORT),
        sum(1 for s in signals if s.signal == SIGNAL_FLAT),
    )
    return signals
