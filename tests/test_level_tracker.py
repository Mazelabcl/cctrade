"""Tests for level touch tracking."""
from datetime import datetime
from app.extensions import db as _db
from app.models import Candle, Level
from app.services.level_tracker import update_level_touches


def _make_candle(db, hour, low, high, open_time=None):
    c = Candle(
        symbol='BTCUSDT', timeframe='1h',
        open_time=open_time or datetime(2024, 1, 2, hour),
        open=low + 50, high=high, low=low,
        close=high - 50, volume=100.0,
    )
    db.session.add(c)
    db.session.commit()
    return c


def _make_level(db, price, created_at=None):
    level = Level(
        price_level=price,
        level_type='Fractal_Low',
        timeframe='daily',
        source='fractal',
        created_at=created_at or datetime(2024, 1, 1),
    )
    db.session.add(level)
    db.session.commit()
    return level


def test_touch_increments_counter(app):
    """A candle whose range covers the level increments support_touches."""
    with app.app_context():
        level = _make_level(_db, 42000.0)
        c1 = _make_candle(_db, 1, low=41900.0, high=42100.0)
        touched = update_level_touches(_db.session, c1)
        assert touched == 1
        assert level.support_touches == 1
        assert level.first_touched_at == c1.open_time
        assert level.invalidated_at is None  # no invalidation


def test_multiple_touches_no_invalidation(app):
    """Multiple touches increment counter but never invalidate."""
    with app.app_context():
        level = _make_level(_db, 42000.0)
        for hour in range(1, 6):
            c = _make_candle(_db, hour, low=41900.0, high=42100.0)
            update_level_touches(_db.session, c)
        assert level.support_touches == 5
        assert level.invalidated_at is None


def test_candle_not_reaching_level(app):
    """Candle range that doesn't cover level price — no touch."""
    with app.app_context():
        level = _make_level(_db, 42000.0)
        c = _make_candle(_db, 1, low=42100.0, high=42400.0)
        touched = update_level_touches(_db.session, c)
        assert touched == 0
        assert level.support_touches == 0


def test_first_touch_invalidation(app):
    """Level is invalidated immediately on first touch when flag is set."""
    with app.app_context():
        level = _make_level(_db, 42000.0)
        candle = _make_candle(_db, 1, low=41900.0, high=42100.0)

        touched = update_level_touches(
            _db.session, candle, invalidate_on_first_touch=True,
        )
        assert touched == 1
        assert level.invalidated_at == candle.open_time


def test_level_created_after_candle_not_touched(app):
    """Levels created after a candle's open_time are not touched."""
    with app.app_context():
        candle = _make_candle(_db, 1, low=41900.0, high=42100.0)
        level = _make_level(_db, 42000.0, created_at=datetime(2024, 1, 3))

        touched = update_level_touches(
            _db.session, candle, invalidate_on_first_touch=True,
        )
        assert touched == 0
        assert level.invalidated_at is None


def test_already_invalidated_level_skipped(app):
    """Already invalidated levels are not processed again."""
    with app.app_context():
        level = _make_level(_db, 42000.0)
        level.invalidated_at = datetime(2024, 1, 1, 12)
        _db.session.commit()

        candle = _make_candle(_db, 1, low=41900.0, high=42100.0)
        touched = update_level_touches(
            _db.session, candle, invalidate_on_first_touch=True,
        )
        assert touched == 0
