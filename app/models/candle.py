from datetime import datetime
from ..extensions import db


class Candle(db.Model):
    __tablename__ = 'candles'

    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, default='BTCUSDT')
    timeframe = db.Column(db.String(10), nullable=False)
    open_time = db.Column(db.DateTime, nullable=False)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=False)
    quote_volume = db.Column(db.Float)
    num_trades = db.Column(db.Integer)
    bearish_fractal = db.Column(db.Boolean, default=False)
    bullish_fractal = db.Column(db.Boolean, default=False)

    feature = db.relationship('Feature', backref='candle', uselist=False, lazy='select')
    predictions = db.relationship('Prediction', backref='candle', lazy='dynamic')

    __table_args__ = (
        db.UniqueConstraint('symbol', 'timeframe', 'open_time', name='uq_candle_tf_time'),
        db.Index('idx_candles_tf_time', 'timeframe', 'open_time'),
    )

    def __repr__(self):
        return f'<Candle {self.symbol} {self.timeframe} {self.open_time}>'

    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'open_time': self.open_time.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'bearish_fractal': self.bearish_fractal,
            'bullish_fractal': self.bullish_fractal,
        }
