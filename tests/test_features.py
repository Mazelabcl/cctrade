"""Tests for feature engineering pipeline."""
from datetime import datetime
from app.extensions import db as _db
from app.models import Candle, Level, Feature
from app.services.feature_engine import (
    _candle_ratios, _volume_ratios, _utc_block, _find_nearest_distances,
    _compute_zone_features, _compute_atr, _compute_momentum,
    compute_features,
)
import numpy as np
import pandas as pd


def test_candle_ratios_bullish():
    """Bullish candle: close > open."""
    uwr, lwr, btr, bpr = _candle_ratios(100, 110, 90, 105)
    assert 0 <= uwr <= 1
    assert 0 <= lwr <= 1
    assert 0 <= btr <= 1
    assert 0 <= bpr <= 1
    # Upper wick: (110 - 105) / 20 = 0.25
    assert abs(uwr - 0.25) < 0.001
    # Lower wick: (100 - 90) / 20 = 0.5
    assert abs(lwr - 0.5) < 0.001


def test_candle_ratios_doji():
    """Doji candle: open == close."""
    uwr, lwr, btr, bpr = _candle_ratios(100, 110, 90, 100)
    assert abs(btr) < 0.001  # no body


def test_candle_ratios_zero_range():
    """Candle with no range."""
    uwr, lwr, btr, bpr = _candle_ratios(100, 100, 100, 100)
    assert uwr == 0.0
    assert lwr == 0.0
    assert btr == 1.0
    assert bpr == 0.5


def test_volume_ratios_short():
    """Volume ratio with 6+ candles of history."""
    history = np.array([100, 100, 100, 100, 100, 100])
    short, long = _volume_ratios(200, history)
    assert abs(short - 2.0) < 0.01
    assert long == 0.0  # not enough for long ratio


def test_volume_ratios_long():
    """Volume ratio with 168+ candles of history."""
    history = np.ones(200) * 100
    short, long = _volume_ratios(150, history)
    assert abs(short - 1.5) < 0.01
    assert abs(long - 1.5) < 0.01


def test_volume_ratios_empty():
    """Volume ratio with insufficient history."""
    short, long = _volume_ratios(100, np.array([]))
    assert short == 0.0
    assert long == 0.0


def test_utc_block():
    assert _utc_block(datetime(2024, 1, 1, 0)) == 0
    assert _utc_block(datetime(2024, 1, 1, 5)) == 1
    assert _utc_block(datetime(2024, 1, 1, 23)) == 5


def test_find_nearest_distances_basic():
    """Distance to nearest support/resistance."""
    levels_df = pd.DataFrame({
        'price_level': [40000, 41000, 42000, 43000, 44000],
        'level_type': ['Fractal_Low'] * 5,
        'timeframe': ['daily'] * 5,
        'source': ['fractal'] * 5,
        'support_touches': [0] * 5,
        'resistance_touches': [0] * 5,
    })
    sup_dist, res_dist = _find_nearest_distances(42500, levels_df)
    assert sup_dist is not None
    assert res_dist is not None
    assert sup_dist > 0
    assert res_dist > 0


def test_find_nearest_distances_empty():
    """Empty levels DataFrame returns None."""
    sup_dist, res_dist = _find_nearest_distances(42500, pd.DataFrame())
    assert sup_dist is None
    assert res_dist is None


def test_compute_zone_features_basic():
    """Zone features with levels in range."""
    levels_df = pd.DataFrame({
        'price_level': [41500, 41800, 42200, 42500],
        'level_type': ['Fractal_Low', 'HTF', 'Fractal_High', 'HTF'],
        'timeframe': ['daily', 'daily', 'daily', 'daily'],
        'support_touches': [0, 1, 0, 0],
        'resistance_touches': [0, 0, 0, 1],
    })
    cache = {('Fractal_Low', 'daily'): 0.6, ('HTF', 'daily'): 0.7}
    result = _compute_zone_features(42000, levels_df, cache, zone_width=0.015)
    assert result['support_confluence_score'] > 0
    assert result['resistance_confluence_score'] > 0
    assert 0 <= result['support_liquidity_consumed'] <= 1
    assert 0 <= result['resistance_liquidity_consumed'] <= 1


def test_compute_zone_features_empty():
    """Empty levels returns zeros."""
    result = _compute_zone_features(42000, pd.DataFrame(), {})
    assert result['support_confluence_score'] == 0.0
    assert result['resistance_confluence_score'] == 0.0


def test_compute_atr_basic():
    """ATR should be positive with typical OHLC data."""
    n = 20
    highs = np.array([100 + i + 5 for i in range(n)], dtype=float)
    lows = np.array([100 + i - 5 for i in range(n)], dtype=float)
    closes = np.array([100 + i for i in range(n)], dtype=float)
    atr = _compute_atr(highs, lows, closes, period=14)
    assert atr is not None
    assert atr > 0


def test_compute_atr_insufficient():
    assert _compute_atr(np.array([1.0]), np.array([1.0]), np.array([1.0]), period=14) is None


def test_compute_momentum():
    """Momentum for rising prices should be positive."""
    closes = np.arange(100.0, 120.0)
    mom = _compute_momentum(closes, period=12)
    assert mom is not None
    assert mom > 0


def test_compute_momentum_insufficient():
    assert _compute_momentum(np.array([1.0, 2.0]), period=12) is None


def test_compute_features_basic(app, sample_candles, sample_levels):
    """Compute features on sample candles."""
    with app.app_context():
        count = compute_features(_db.session)
        # 10 candles, skip first 2 → up to 8 features
        assert count > 0
        features = _db.session.query(Feature).all()
        assert len(features) == count
        # Check a feature has valid ratios
        f = features[0]
        assert f.upper_wick_ratio is not None
        assert f.volume_short_ratio is not None
        assert f.utc_block is not None
        # Check targets are set
        assert f.target_bullish is not None
        assert f.target_bearish is not None
        assert f.target_bullish in (0, 1)
        assert f.target_bearish in (0, 1)


def test_compute_features_idempotent(app, sample_candles, sample_levels):
    """Running compute_features twice doesn't duplicate."""
    with app.app_context():
        count1 = compute_features(_db.session)
        count2 = compute_features(_db.session)
        assert count1 > 0
        assert count2 == 0  # all already computed
