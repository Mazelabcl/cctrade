"""Tests for ML training, prediction, target building, and accuracy tracking."""
from datetime import datetime, timezone
from app.extensions import db as _db
from app.models import Candle, Level, Feature, MLModel, Prediction
from app.services.target_builder import create_fractal_targets
from app.tasks.accuracy_tracker import backfill_actuals


def _create_candles_with_fractals(n=50):
    """Create n candles with periodic fractals for testing."""
    candles = []
    for i in range(n):
        c = Candle(
            symbol='BTCUSDT',
            timeframe='1h',
            open_time=datetime(2024, 1, 1 + i // 24, i % 24),
            open=42000.0 + i * 10,
            high=42100.0 + i * 10,
            low=41900.0 + i * 10,
            close=42050.0 + i * 10,
            volume=100.0 + i,
            bullish_fractal=(i % 10 == 3),
            bearish_fractal=(i % 10 == 7),
        )
        _db.session.add(c)
        candles.append(c)
    _db.session.commit()
    return candles


def _create_features_for_candles(candles):
    """Create features for all candles."""
    for c in candles:
        f = Feature(
            candle_id=c.id,
            computed_at=datetime.now(timezone.utc),
            upper_wick_ratio=0.1,
            lower_wick_ratio=0.1,
            body_total_ratio=0.8,
            body_position_ratio=0.5,
            volume_short_ratio=1.0,
            volume_long_ratio=1.0,
            utc_block=c.open_time.hour // 4,
            candles_since_last_up=3,
            candles_since_last_down=3,
            support_distance_pct=0.01,
            resistance_distance_pct=0.01,
            support_daily_count=1,
            resistance_daily_count=1,
        )
        _db.session.add(f)
    _db.session.commit()


def test_create_fractal_targets(app):
    """Target builder creates targets from candles."""
    with app.app_context():
        _create_candles_with_fractals(50)
        targets = create_fractal_targets(_db.session, prediction_horizon='hour')
        assert len(targets) > 0
        assert 'candle_id' in targets.columns
        assert 'fractal_direction' in targets.columns
        assert set(targets['fractal_direction'].unique()).issubset({0, 1, 2})


def test_create_fractal_targets_empty(app):
    """Target builder with no candles returns empty."""
    with app.app_context():
        targets = create_fractal_targets(_db.session)
        assert targets.empty


def test_create_fractal_targets_day_horizon(app):
    """Day horizon creates fewer targets (needs 24 future candles)."""
    with app.app_context():
        _create_candles_with_fractals(50)
        hour_targets = create_fractal_targets(_db.session, prediction_horizon='hour')
        day_targets = create_fractal_targets(_db.session, prediction_horizon='day')
        assert len(hour_targets) > len(day_targets)


def test_backfill_actuals_no_predictions(app):
    """Backfill with no predictions returns 0."""
    with app.app_context():
        count = backfill_actuals(_db.session)
        assert count == 0


def test_backfill_actuals_fills_in(app):
    """Backfill fills in actual_class when future candles exist."""
    with app.app_context():
        candles = _create_candles_with_fractals(50)

        # Create a model record (minimal)
        model = MLModel(
            name='test_model', algorithm='random_forest', version=1,
            prediction_horizon='hour', file_path='/tmp/test.joblib',
            feature_names=['a'], created_at=datetime.now(timezone.utc),
        )
        _db.session.add(model)
        _db.session.commit()

        # Create predictions for first few candles
        for c in candles[:10]:
            pred = Prediction(
                model_id=model.id, candle_id=c.id,
                predicted_class=1, confidence=0.8,
                created_at=datetime.now(timezone.utc),
            )
            _db.session.add(pred)
        _db.session.commit()

        count = backfill_actuals(_db.session)
        assert count == 10

        # Check actual_class was set
        preds = _db.session.query(Prediction).all()
        for p in preds:
            assert p.actual_class is not None
            assert p.actual_class in (0, 1, 2)


def test_api_predictions_endpoint(client):
    """GET /api/predictions returns empty list."""
    response = client.get('/api/predictions')
    assert response.status_code == 200
    assert response.get_json() == []


def test_api_predictions_overlay(client):
    """GET /api/predictions/overlay returns empty list."""
    response = client.get('/api/predictions/overlay')
    assert response.status_code == 200
    assert response.get_json() == []


def test_api_run_pipeline(client):
    """POST /api/run-pipeline returns started."""
    response = client.post('/api/run-pipeline')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'started'


def test_api_backfill_actuals(client):
    """POST /api/backfill-actuals returns ok."""
    response = client.post('/api/backfill-actuals')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'


def test_api_health_enhanced(client, sample_candles):
    """Health endpoint includes candle and model counts."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'
    assert data['candles'] == 10
    assert data['models'] == 0


def test_api_fetch_data(client):
    """POST /api/fetch-data returns started."""
    response = client.post('/api/fetch-data')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'started'
