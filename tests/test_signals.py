"""Tests for the trading signal generator."""
from datetime import datetime
from unittest.mock import MagicMock

from app.services.signal_generator import (
    generate_signal, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_FLAT, TradeSignal,
)


def _mock_prediction(predicted_class, confidence):
    p = MagicMock()
    p.predicted_class = predicted_class
    p.confidence = confidence
    return p


def _mock_candle(candle_id, close):
    c = MagicMock()
    c.id = candle_id
    c.close = close
    return c


def _mock_feature(atr):
    f = MagicMock()
    f.atr_14 = atr
    return f


def test_long_signal_near_support():
    """Bullish prediction near support → LONG."""
    pred = _mock_prediction(predicted_class=1, confidence=0.65)
    candle = _mock_candle(1, close=42000.0)
    feature = _mock_feature(atr=500.0)
    sig = generate_signal(pred, candle, feature,
                          nearest_support=41800.0, nearest_resistance=43000.0)
    assert sig.signal == SIGNAL_LONG
    assert sig.stop_loss == 42000.0 - 500.0 * 1.5
    assert sig.take_profit == 42000.0 + 500.0 * 3.0


def test_short_signal_near_resistance():
    """Bearish prediction near resistance → SHORT."""
    pred = _mock_prediction(predicted_class=2, confidence=0.70)
    candle = _mock_candle(1, close=42900.0)
    feature = _mock_feature(atr=400.0)
    sig = generate_signal(pred, candle, feature,
                          nearest_support=41000.0, nearest_resistance=43000.0)
    assert sig.signal == SIGNAL_SHORT
    assert sig.stop_loss == 42900.0 + 400.0 * 1.5
    assert sig.take_profit == 42900.0 - 400.0 * 3.0


def test_flat_on_low_confidence():
    """Low confidence → FLAT regardless of class."""
    pred = _mock_prediction(predicted_class=1, confidence=0.40)
    candle = _mock_candle(1, close=42000.0)
    feature = _mock_feature(atr=500.0)
    sig = generate_signal(pred, candle, feature,
                          nearest_support=42000.0, nearest_resistance=43000.0)
    assert sig.signal == SIGNAL_FLAT
    assert 'low_confidence' in sig.reason


def test_flat_on_no_fractal():
    """No fractal predicted → FLAT."""
    pred = _mock_prediction(predicted_class=0, confidence=0.80)
    candle = _mock_candle(1, close=42000.0)
    feature = _mock_feature(atr=500.0)
    sig = generate_signal(pred, candle, feature,
                          nearest_support=42000.0, nearest_resistance=43000.0)
    assert sig.signal == SIGNAL_FLAT
    assert 'no_fractal' in sig.reason


def test_flat_support_too_far():
    """Bullish prediction but support is too far → FLAT."""
    pred = _mock_prediction(predicted_class=1, confidence=0.65)
    candle = _mock_candle(1, close=42000.0)
    feature = _mock_feature(atr=500.0)
    # Support at 40000 → distance = 2000/42000 ≈ 4.76% > 2%
    sig = generate_signal(pred, candle, feature,
                          nearest_support=40000.0, nearest_resistance=44000.0)
    assert sig.signal == SIGNAL_FLAT
    assert 'support_too_far' in sig.reason


def test_flat_resistance_too_far():
    """Bearish prediction but resistance is too far → FLAT."""
    pred = _mock_prediction(predicted_class=2, confidence=0.65)
    candle = _mock_candle(1, close=42000.0)
    feature = _mock_feature(atr=500.0)
    # Resistance at 45000 → distance = 3000/42000 ≈ 7.1% > 2%
    sig = generate_signal(pred, candle, feature,
                          nearest_support=40000.0, nearest_resistance=45000.0)
    assert sig.signal == SIGNAL_FLAT
    assert 'resistance_too_far' in sig.reason


def test_long_no_support_level():
    """Bullish prediction with no support level → still LONG (no proximity check)."""
    pred = _mock_prediction(predicted_class=1, confidence=0.65)
    candle = _mock_candle(1, close=42000.0)
    feature = _mock_feature(atr=500.0)
    sig = generate_signal(pred, candle, feature,
                          nearest_support=None, nearest_resistance=43000.0)
    assert sig.signal == SIGNAL_LONG


def test_signal_without_atr():
    """Signal generated without ATR → SL/TP are None."""
    pred = _mock_prediction(predicted_class=1, confidence=0.65)
    candle = _mock_candle(1, close=42000.0)
    feature = _mock_feature(atr=None)
    sig = generate_signal(pred, candle, feature,
                          nearest_support=42000.0, nearest_resistance=43000.0)
    assert sig.signal == SIGNAL_LONG
    assert sig.stop_loss is None
    assert sig.take_profit is None
