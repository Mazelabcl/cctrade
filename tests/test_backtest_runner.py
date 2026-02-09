"""Tests for the backtest runner."""
from datetime import datetime
from app.extensions import db as _db
from app.models import Candle, Level, Feature, MLModel, Prediction, BacktestResult


def _seed_backtest_data(db):
    """Create enough data for a minimal backtest run."""
    candles = []
    for i in range(50):
        c = Candle(
            symbol='BTCUSDT', timeframe='1h',
            open_time=datetime(2024, 1, 1 + i // 24, i % 24),
            open=42000.0 + i * 10,
            high=42200.0 + i * 10,
            low=41800.0 + i * 10,
            close=42100.0 + i * 10,
            volume=100.0,
            bullish_fractal=(i == 10),
            bearish_fractal=(i == 20),
        )
        db.session.add(c)
        candles.append(c)
    db.session.commit()

    # Add a level near the price range
    level = Level(
        price_level=42000.0, level_type='Fractal_Low',
        timeframe='daily', source='fractal',
        created_at=datetime(2024, 1, 1),
    )
    db.session.add(level)
    db.session.commit()

    # Create a model record
    model = MLModel(
        name='test_model', algorithm='random_forest', version=1,
        prediction_horizon='day', file_path='test.joblib',
        feature_names=['test'], accuracy=0.5,
    )
    db.session.add(model)
    db.session.commit()

    # Create features with ATR
    for c in candles:
        f = Feature(candle_id=c.id, computed_at=datetime(2024, 1, 1),
                    atr_14=500.0)
        db.session.add(f)
    db.session.commit()

    # Create predictions — a few bullish signals
    for c in candles[10:15]:
        p = Prediction(
            model_id=model.id, candle_id=c.id,
            predicted_class=1, prob_no_fractal=0.2,
            prob_bullish=0.65, prob_bearish=0.15,
            confidence=0.65,
        )
        db.session.add(p)
    db.session.commit()

    return model


def test_run_backtest_basic(app):
    """Run a basic backtest and verify result is persisted."""
    with app.app_context():
        model = _seed_backtest_data(_db)

        from app.services.backtest.runner import run_backtest
        result = run_backtest(
            _db.session, model_id=model.id,
            initial_cash=100000, commission=0.001,
        )

        assert isinstance(result, BacktestResult)
        assert result.id is not None
        assert result.model_id == model.id
        assert result.initial_cash == 100000
        assert result.final_value is not None
        assert result.total_return is not None
        assert result.trade_log is not None

        # Verify persisted in DB
        stored = _db.session.query(BacktestResult).filter_by(id=result.id).first()
        assert stored is not None
        assert stored.model_id == model.id


def test_run_backtest_updates_model(app):
    """Backtest should update MLModel with backtest metrics."""
    with app.app_context():
        model = _seed_backtest_data(_db)

        from app.services.backtest.runner import run_backtest
        run_backtest(_db.session, model_id=model.id)

        updated = _db.session.get(MLModel, model.id)
        assert updated.backtest_return is not None
        assert updated.total_trades is not None


def test_run_backtest_invalid_model(app):
    """Backtest with non-existent model should fail."""
    import pytest
    with app.app_context():
        from app.services.backtest.runner import run_backtest
        with pytest.raises(ValueError, match="not found"):
            run_backtest(_db.session, model_id=9999)
