from datetime import datetime
from ..extensions import db


class Level(db.Model):
    __tablename__ = 'levels'

    id = db.Column(db.Integer, primary_key=True)
    price_level = db.Column(db.Float, nullable=False)
    level_type = db.Column(db.String(50), nullable=False)
    timeframe = db.Column(db.String(20), nullable=False)
    source = db.Column(db.String(30), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    invalidated_at = db.Column(db.DateTime)
    support_touches = db.Column(db.Integer, default=0)
    resistance_touches = db.Column(db.Integer, default=0)
    metadata_json = db.Column('metadata', db.JSON)

    __table_args__ = (
        db.Index('idx_levels_price', 'price_level'),
        db.Index('idx_levels_created', 'created_at'),
        db.Index('idx_levels_active', 'invalidated_at',
                 sqlite_where=db.text('invalidated_at IS NULL')),
    )

    def __repr__(self):
        return f'<Level {self.level_type} {self.price_level} {self.timeframe}>'

    def to_dict(self):
        return {
            'id': self.id,
            'price_level': self.price_level,
            'level_type': self.level_type,
            'timeframe': self.timeframe,
            'source': self.source,
            'created_at': self.created_at.isoformat(),
            'invalidated_at': self.invalidated_at.isoformat() if self.invalidated_at else None,
            'support_touches': self.support_touches,
            'resistance_touches': self.resistance_touches,
            'metadata': self.metadata_json,
        }
