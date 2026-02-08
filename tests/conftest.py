import pytest
from datetime import datetime
from app import create_app
from app.extensions import db as _db
from app.models import Candle, Level


@pytest.fixture(scope='session')
def app():
    """Create application for testing."""
    app = create_app('testing')
    return app


@pytest.fixture(autouse=True)
def setup_db(app):
    """Create tables before each test, drop after."""
    with app.app_context():
        _db.create_all()
        yield _db
        _db.session.remove()
        _db.drop_all()


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def sample_candles(app):
    """Create sample candle data for testing."""
    with app.app_context():
        candles = []
        for i in range(10):
            c = Candle(
                symbol='BTCUSDT',
                timeframe='1h',
                open_time=datetime(2024, 1, 1, i),
                open=42000.0 + i * 100,
                high=42200.0 + i * 100,
                low=41900.0 + i * 100,
                close=42100.0 + i * 100,
                volume=100.0 + i * 10,
                bearish_fractal=(i == 5),
                bullish_fractal=(i == 3),
            )
            _db.session.add(c)
            candles.append(c)
        _db.session.commit()
        return candles


@pytest.fixture
def sample_levels(app):
    """Create sample level data for testing."""
    with app.app_context():
        levels = []
        for price, ltype, tf, src in [
            (42000.0, 'Fractal_Low', 'daily', 'fractal'),
            (42500.0, 'HTF_level', 'weekly', 'htf'),
            (41800.0, 'Fib_0.618', 'daily', 'fibonacci'),
            (43000.0, 'VP_poc', 'daily', 'volume_profile'),
        ]:
            level = Level(
                price_level=price,
                level_type=ltype,
                timeframe=tf,
                source=src,
                created_at=datetime(2024, 1, 1),
            )
            _db.session.add(level)
            levels.append(level)
        _db.session.commit()
        return levels
