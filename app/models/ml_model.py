from datetime import datetime, timezone
from ..extensions import db


class MLModel(db.Model):
    __tablename__ = 'ml_models'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    algorithm = db.Column(db.String(50), nullable=False)
    version = db.Column(db.Integer, nullable=False)
    prediction_horizon = db.Column(db.String(20), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    feature_names = db.Column(db.JSON, nullable=False)

    # Metrics
    accuracy = db.Column(db.Float)
    precision_macro = db.Column(db.Float)
    recall_macro = db.Column(db.Float)
    f1_macro = db.Column(db.Float)
    roc_auc = db.Column(db.Float)

    # Backtest metrics (populated after backtesting)
    sharpe_ratio = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    profit_factor = db.Column(db.Float)
    win_rate = db.Column(db.Float)
    total_trades = db.Column(db.Integer)
    backtest_return = db.Column(db.Float)

    # Metadata
    train_rows = db.Column(db.Integer)
    train_period = db.Column(db.String(50))
    hyperparameters = db.Column(db.JSON)
    training_duration_sec = db.Column(db.Float)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, nullable=False,
                           default=lambda: datetime.now(timezone.utc))
    is_active = db.Column(db.Boolean, default=False)

    predictions = db.relationship('Prediction', backref='model', lazy='dynamic')

    def __repr__(self):
        return f'<MLModel {self.name} v{self.version} ({self.algorithm})>'

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'algorithm': self.algorithm,
            'version': self.version,
            'prediction_horizon': self.prediction_horizon,
            'accuracy': self.accuracy,
            'precision_macro': self.precision_macro,
            'recall_macro': self.recall_macro,
            'f1_macro': self.f1_macro,
            'roc_auc': self.roc_auc,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.profit_factor,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'backtest_return': self.backtest_return,
            'train_rows': self.train_rows,
            'train_period': self.train_period,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,
        }
