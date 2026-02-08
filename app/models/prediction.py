from datetime import datetime, timezone
from ..extensions import db


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('ml_models.id'), nullable=False)
    candle_id = db.Column(db.Integer, db.ForeignKey('candles.id'), nullable=False)
    predicted_class = db.Column(db.Integer, nullable=False)
    prob_no_fractal = db.Column(db.Float)
    prob_bullish = db.Column(db.Float)
    prob_bearish = db.Column(db.Float)
    confidence = db.Column(db.Float)
    actual_class = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, nullable=False,
                           default=lambda: datetime.now(timezone.utc))

    # relationship defined on MLModel side via backref='model'

    def __repr__(self):
        return f'<Prediction model={self.model_id} candle={self.candle_id} class={self.predicted_class}>'

    def to_dict(self):
        return {
            'id': self.id,
            'model_id': self.model_id,
            'candle_id': self.candle_id,
            'predicted_class': self.predicted_class,
            'prob_no_fractal': self.prob_no_fractal,
            'prob_bullish': self.prob_bullish,
            'prob_bearish': self.prob_bearish,
            'confidence': self.confidence,
            'actual_class': self.actual_class,
            'created_at': self.created_at.isoformat(),
        }
