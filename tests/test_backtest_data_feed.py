"""Tests for backtest data feed construction."""
import pandas as pd
import backtrader as bt

from app.services.backtest.data_feed import (
    FractalPandasData, SIGNAL_MAP,
)


def test_fractal_pandas_data_has_extra_lines():
    """FractalPandasData should define extra lines."""
    expected = {'signal', 'confidence', 'stop_loss', 'take_profit',
                'atr', 'bullish_fractal', 'bearish_fractal'}
    actual = set(FractalPandasData.lines._getlines())
    assert expected.issubset(actual)


def test_fractal_pandas_data_loads_dataframe():
    """FractalPandasData can load a simple DataFrame without errors."""
    dates = pd.date_range('2024-01-01', periods=10, freq='h')
    df = pd.DataFrame({
        'open': range(10),
        'high': range(1, 11),
        'low': range(10),
        'close': range(10),
        'volume': [100] * 10,
        'signal': [0] * 10,
        'confidence': [0.0] * 10,
        'stop_loss': [0.0] * 10,
        'take_profit': [0.0] * 10,
        'atr': [100.0] * 10,
        'bullish_fractal': [0] * 10,
        'bearish_fractal': [0] * 10,
    }, index=dates)

    cerebro = bt.Cerebro()
    data = FractalPandasData(dataname=df)
    cerebro.adddata(data)
    # Just verify it doesn't raise
    cerebro.run()


def test_signal_map_values():
    """Signal map should have expected numeric codes."""
    assert SIGNAL_MAP['FLAT'] == 0
    assert SIGNAL_MAP['LONG'] == 1
    assert SIGNAL_MAP['SHORT'] == -1
