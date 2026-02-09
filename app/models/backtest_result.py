from datetime import datetime, timezone
from ..extensions import db


class BacktestResult(db.Model):
    __tablename__ = 'backtest_results'

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('ml_models.id'), nullable=False)

    # Config
    initial_cash = db.Column(db.Float, nullable=False)
    commission = db.Column(db.Float)
    risk_per_trade = db.Column(db.Float)
    confidence_threshold = db.Column(db.Float)
    level_proximity_pct = db.Column(db.Float)
    atr_sl_mult = db.Column(db.Float)
    atr_tp_mult = db.Column(db.Float)

    # Results
    final_value = db.Column(db.Float)
    total_return = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    total_trades = db.Column(db.Integer)
    win_rate = db.Column(db.Float)
    profit_factor = db.Column(db.Float)
    avg_win = db.Column(db.Float)
    avg_loss = db.Column(db.Float)
    sqn = db.Column(db.Float)

    # Trade log (JSON list of trade dicts)
    trade_log = db.Column(db.JSON)

    created_at = db.Column(db.DateTime, nullable=False,
                           default=lambda: datetime.now(timezone.utc))

    model = db.relationship('MLModel', backref='backtest_results')

    def __repr__(self):
        return f'<BacktestResult model={self.model_id} return={self.total_return}>'

    def to_dict(self):
        return {
            'id': self.id,
            'model_id': self.model_id,
            'initial_cash': self.initial_cash,
            'commission': self.commission,
            'risk_per_trade': self.risk_per_trade,
            'final_value': self.final_value,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'sqn': self.sqn,
            'trade_log': self.trade_log,
            'created_at': self.created_at.isoformat(),
        }
