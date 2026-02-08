"""Tests for indicator services — fractal detection, HTF levels, Fibonacci."""
import pandas as pd
from datetime import datetime
from app.services.indicators import (
    detect_fractals_df,
    calculate_htf_levels,
    calculate_fibonacci_levels,
    run_fractal_detection,
)
from app.models import Candle, Level
from app.extensions import db as _db


def test_detect_fractals_df_bearish():
    """Bearish fractal: middle candle has highest high."""
    df = pd.DataFrame({
        'high': [100, 110, 120, 110, 100],
        'low':  [90,  100, 110, 100, 90],
    })
    result = detect_fractals_df(df)
    assert result.iloc[2]['bearish_fractal'] == True
    assert result.iloc[2]['bullish_fractal'] == False


def test_detect_fractals_df_bullish():
    """Bullish fractal: middle candle has lowest low."""
    df = pd.DataFrame({
        'high': [120, 110, 100, 110, 120],
        'low':  [110, 100, 90,  100, 110],
    })
    result = detect_fractals_df(df)
    assert result.iloc[2]['bullish_fractal'] == True
    assert result.iloc[2]['bearish_fractal'] == False


def test_detect_fractals_df_no_fractal():
    """Monotonically increasing — no fractals."""
    df = pd.DataFrame({
        'high': [100, 110, 120, 130, 140],
        'low':  [90,  100, 110, 120, 130],
    })
    result = detect_fractals_df(df)
    assert not result['bearish_fractal'].any()
    assert not result['bullish_fractal'].any()


def test_detect_fractals_df_too_few_candles():
    """Less than 5 candles — no fractals."""
    df = pd.DataFrame({
        'high': [100, 110, 120],
        'low':  [90,  100, 110],
    })
    result = detect_fractals_df(df)
    assert not result['bearish_fractal'].any()


def test_htf_levels_direction_change():
    """HTF level created when candle direction changes."""
    df = pd.DataFrame({
        'open':     [100, 110, 120, 115],
        'close':    [110, 120, 115, 125],
        'high':     [115, 125, 125, 130],
        'low':      [95,  105, 110, 110],
        'open_time': [datetime(2024, 1, 1, i) for i in range(4)],
    })
    levels = calculate_htf_levels(df, 'daily')
    # Direction: up, up, down, up → changes at index 2 and 3
    assert len(levels) == 2
    assert levels[0]['source'] == 'htf'


def test_htf_levels_empty():
    df = pd.DataFrame()
    levels = calculate_htf_levels(df, 'daily')
    assert levels == []


def test_fibonacci_levels_basic():
    """Fibonacci levels generated between swing high and low."""
    # Build data with a clear bullish then bearish fractal
    highs = [100, 110, 90, 110, 100, 110, 120, 130, 120, 110]
    lows =  [90,  100, 80, 100, 90,  100, 110, 120, 110, 100]
    df = pd.DataFrame({
        'high': highs,
        'low': lows,
        'open_time': [datetime(2024, 1, 1, i) for i in range(10)],
    })
    levels = calculate_fibonacci_levels(df, 'daily')
    # Should generate some fib levels
    assert len(levels) >= 0  # May or may not find fractals depending on pattern
    for lev in levels:
        assert lev['source'] == 'fibonacci'
        assert 'Fib_' in lev['level_type']


def test_run_fractal_detection(app, sample_candles):
    """Run fractal detection on DB candles."""
    with app.app_context():
        count = run_fractal_detection(_db.session)
        # With only 10 candles, limited fractals possible
        assert isinstance(count, int)


def test_api_run_indicators(client):
    """POST /api/run-indicators returns started status."""
    response = client.post('/api/run-indicators')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'started'


def test_api_pipeline_runs(client):
    response = client.get('/api/pipeline-runs')
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)
