"""Backtrader data feed built from SQLite Candle + Feature + Prediction tables.

Extends ``bt.feeds.PandasData`` with extra lines for signal, confidence,
stop-loss, take-profit, ATR, and fractal labels.
"""
import logging

import backtrader as bt
import pandas as pd
from sqlalchemy.orm import Session

from ...config import Config
from ...models import Candle, Feature, Prediction
from ..signal_generator import (
    generate_signal, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_FLAT,
)

logger = logging.getLogger(__name__)

# Map signal strings to numeric codes for backtrader lines
SIGNAL_MAP = {SIGNAL_FLAT: 0, SIGNAL_LONG: 1, SIGNAL_SHORT: -1}


class FractalPandasData(bt.feeds.PandasData):
    """PandasData subclass with extra lines for the fractal strategy."""

    lines = ('signal', 'confidence', 'stop_loss', 'take_profit', 'atr',
             'bullish_fractal', 'bearish_fractal')

    params = (
        ('signal', -1),
        ('confidence', -1),
        ('stop_loss', -1),
        ('take_profit', -1),
        ('atr', -1),
        ('bullish_fractal', -1),
        ('bearish_fractal', -1),
    )


def build_backtest_dataframe(
    db: Session,
    model_id: int,
    timeframe: str = '1h',
    symbol: str = 'BTCUSDT',
    confidence_threshold: float = Config.BACKTEST_CONFIDENCE_THRESHOLD,
    level_proximity_pct: float = Config.BACKTEST_LEVEL_PROXIMITY_PCT,
) -> pd.DataFrame:
    """Join Candle + Feature + Prediction and generate signals.

    Returns a DataFrame indexed by ``open_time`` with OHLCV, signal columns,
    and fractal labels — ready to feed into ``FractalPandasData``.
    """
    from ...models import Level

    # Load candles
    candles = (
        db.query(Candle)
        .filter_by(symbol=symbol, timeframe=timeframe)
        .order_by(Candle.open_time)
        .all()
    )
    if not candles:
        return pd.DataFrame()

    candle_map = {c.id: c for c in candles}

    # Load features keyed by candle_id
    feature_map = {}
    features = db.query(Feature).filter(
        Feature.candle_id.in_(list(candle_map.keys()))
    ).all()
    for f in features:
        feature_map[f.candle_id] = f

    # Load predictions for the given model
    predictions = (
        db.query(Prediction)
        .filter_by(model_id=model_id)
        .all()
    )
    pred_map = {p.candle_id: p for p in predictions}

    # Load active levels for proximity checks
    levels = db.query(Level).filter(Level.invalidated_at.is_(None)).all()
    support_prices = sorted([l.price_level for l in levels], reverse=True)
    resistance_prices = sorted([l.price_level for l in levels])

    rows = []
    for candle in candles:
        pred = pred_map.get(candle.id)
        feat = feature_map.get(candle.id)

        sig_code = 0
        conf = 0.0
        sl = 0.0
        tp = 0.0
        atr_val = feat.atr_14 if feat and feat.atr_14 else 0.0

        if pred:
            # Find nearest levels
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

            trade_sig = generate_signal(
                pred, candle, feat, nearest_sup, nearest_res,
                confidence_threshold=confidence_threshold,
                level_proximity_pct=level_proximity_pct,
            )
            sig_code = SIGNAL_MAP.get(trade_sig.signal, 0)
            conf = trade_sig.confidence
            sl = trade_sig.stop_loss or 0.0
            tp = trade_sig.take_profit or 0.0

        rows.append({
            'open_time': candle.open_time,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume,
            'signal': sig_code,
            'confidence': conf,
            'stop_loss': sl,
            'take_profit': tp,
            'atr': atr_val,
            'bullish_fractal': 1 if candle.bullish_fractal else 0,
            'bearish_fractal': 1 if candle.bearish_fractal else 0,
        })

    df = pd.DataFrame(rows)
    df.set_index('open_time', inplace=True)
    df.index = pd.to_datetime(df.index)

    logger.info(
        "Built backtest DataFrame: %d rows, %d signals (L=%d, S=%d)",
        len(df),
        (df['signal'] != 0).sum(),
        (df['signal'] == 1).sum(),
        (df['signal'] == -1).sum(),
    )
    return df
