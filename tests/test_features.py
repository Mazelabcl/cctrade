"""Tests for feature engineering pipeline."""
from datetime import datetime
from app.extensions import db as _db
from app.models import Candle, Level, Feature
from app.services.feature_engine import (
    _candle_ratios, _volume_ratios, _utc_block, _find_nearest_levels,
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


def test_find_nearest_levels_support():
    """Support zone detection."""
    levels_df = pd.DataFrame({
        'price_level': [40000, 41000, 42000, 43000, 44000],
        'level_type': ['Fractal_Low'] * 5,
        'timeframe': ['daily'] * 5,
        'source': ['fractal'] * 5,
        'support_touches': [0] * 5,
        'resistance_touches': [0] * 5,
    })
    sup, res = _find_nearest_levels(42500, levels_df)
    assert sup is not None
    assert res is not None
    assert sup['distance_pct'] > 0
    assert res['distance_pct'] > 0


def test_find_nearest_levels_empty():
    """Empty levels DataFrame returns None."""
    sup, res = _find_nearest_levels(42500, pd.DataFrame())
    assert sup is None
    assert res is None


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


def test_compute_features_idempotent(app, sample_candles, sample_levels):
    """Running compute_features twice doesn't duplicate."""
    with app.app_context():
        count1 = compute_features(_db.session)
        count2 = compute_features(_db.session)
        assert count1 > 0
        assert count2 == 0  # all already computed
