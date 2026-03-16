from datetime import datetime
from ..extensions import db


class Feature(db.Model):
    __tablename__ = 'features'

    id = db.Column(db.Integer, primary_key=True)
    candle_id = db.Column(db.Integer, db.ForeignKey('candles.id'), nullable=False, unique=True)
    computed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # Target: was N-1 a fractal? (binary columns, both can be 1 for dual fractals)
    target_bullish = db.Column(db.Integer)   # 1 if N-1 is swing low, 0 otherwise
    target_bearish = db.Column(db.Integer)   # 1 if N-1 is swing high, 0 otherwise

    # Zone distances (from N-2 close to nearest level)
    support_distance_pct = db.Column(db.Float)
    resistance_distance_pct = db.Column(db.Float)

    # Zone quality (from backtest win_rates)
    support_confluence_score = db.Column(db.Float)
    resistance_confluence_score = db.Column(db.Float)
    support_liquidity_consumed = db.Column(db.Float)
    resistance_liquidity_consumed = db.Column(db.Float)

    # Candle ratios (N-1 shape)
    upper_wick_ratio = db.Column(db.Float)
    lower_wick_ratio = db.Column(db.Float)
    body_total_ratio = db.Column(db.Float)
    body_position_ratio = db.Column(db.Float)

    # Volume ratios (N-1)
    volume_short_ratio = db.Column(db.Float)
    volume_long_ratio = db.Column(db.Float)

    # Timing
    utc_block = db.Column(db.Integer)
    candles_since_last_up = db.Column(db.Integer)
    candles_since_last_down = db.Column(db.Integer)

    # Volatility / Momentum
    atr_14 = db.Column(db.Float)
    momentum_12 = db.Column(db.Float)

    def __repr__(self):
        return f'<Feature candle_id={self.candle_id}>'

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
