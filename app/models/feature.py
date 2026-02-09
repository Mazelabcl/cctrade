from datetime import datetime
from ..extensions import db


class Feature(db.Model):
    __tablename__ = 'features'

    id = db.Column(db.Integer, primary_key=True)
    candle_id = db.Column(db.Integer, db.ForeignKey('candles.id'), nullable=False, unique=True)
    computed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # Support zone
    support_distance_pct = db.Column(db.Float)
    support_zone_start = db.Column(db.Float)
    support_zone_end = db.Column(db.Float)
    support_daily_count = db.Column(db.Integer)
    support_weekly_count = db.Column(db.Integer)
    support_monthly_count = db.Column(db.Integer)
    support_fib618_count = db.Column(db.Integer)
    support_naked_count = db.Column(db.Integer)
    total_support_touches = db.Column(db.Integer)

    # Resistance zone
    resistance_distance_pct = db.Column(db.Float)
    resistance_zone_start = db.Column(db.Float)
    resistance_zone_end = db.Column(db.Float)
    resistance_daily_count = db.Column(db.Integer)
    resistance_weekly_count = db.Column(db.Integer)
    resistance_monthly_count = db.Column(db.Integer)
    resistance_fib618_count = db.Column(db.Integer)
    resistance_naked_count = db.Column(db.Integer)
    total_resistance_touches = db.Column(db.Integer)

    # Candle ratios
    upper_wick_ratio = db.Column(db.Float)
    lower_wick_ratio = db.Column(db.Float)
    body_total_ratio = db.Column(db.Float)
    body_position_ratio = db.Column(db.Float)

    # Volume ratios
    volume_short_ratio = db.Column(db.Float)
    volume_long_ratio = db.Column(db.Float)

    # Timing
    utc_block = db.Column(db.Integer)
    candles_since_last_up = db.Column(db.Integer)
    candles_since_last_down = db.Column(db.Integer)

    # Momentum / Volatility indicators
    rsi_14 = db.Column(db.Float)
    macd_line = db.Column(db.Float)
    macd_signal = db.Column(db.Float)
    macd_histogram = db.Column(db.Float)
    bollinger_width = db.Column(db.Float)
    atr_14 = db.Column(db.Float)
    momentum_12 = db.Column(db.Float)

    # Vectors stored as JSON
    support_level_vector = db.Column(db.JSON)
    resistance_level_vector = db.Column(db.JSON)
    support_touched_vector = db.Column(db.JSON)
    resistance_touched_vector = db.Column(db.JSON)

    def __repr__(self):
        return f'<Feature candle_id={self.candle_id}>'

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
