from datetime import datetime, timezone
from ..extensions import db


class IndividualLevelBacktest(db.Model):
    """Backtest results for a single level type/timeframe combination."""
    __tablename__ = 'individual_level_backtests'

    id = db.Column(db.Integer, primary_key=True)

    # Level configuration
    level_type = db.Column(db.String(50), nullable=False)  # "HTF", "Fractal_High", "Fib_0.618", etc.
    level_source_timeframe = db.Column(db.String(10), nullable=False)  # "1h", "4h", "1d", "1w"
    trade_execution_timeframe = db.Column(db.String(10), nullable=False)  # "1h", "4h", "1d"

    # Strategy configuration
    strategy_name = db.Column(db.String(50), nullable=False)  # "fixed_percent", "atr_based", etc.
    parameters = db.Column(db.JSON)  # {sl_pct: 2, tp_pct: 4, timeout: 50, ...}

    # Date range
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)

    # Status
    status = db.Column(db.String(20), nullable=False, default='pending')  # pending, running, completed, failed

    # Performance metrics
    total_trades = db.Column(db.Integer, default=0)
    winning_trades = db.Column(db.Integer, default=0)
    losing_trades = db.Column(db.Integer, default=0)
    win_rate = db.Column(db.Float)  # %
    profit_factor = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)  # %
    total_pnl = db.Column(db.Float)  # $
    avg_win = db.Column(db.Float)  # $
    avg_loss = db.Column(db.Float)  # $
    avg_trade_duration = db.Column(db.Float)  # candles

    # Timestamps
    created_at = db.Column(db.DateTime, nullable=False,
                           default=lambda: datetime.now(timezone.utc))
    finished_at = db.Column(db.DateTime)
    error_message = db.Column(db.Text)

    # Relationships
    trades = db.relationship('IndividualLevelTrade', backref='backtest', lazy='dynamic',
                             cascade='all, delete-orphan')

    __table_args__ = (
        db.Index('idx_ilb_level_type', 'level_type'),
        db.Index('idx_ilb_source_tf', 'level_source_timeframe'),
        db.Index('idx_ilb_strategy', 'strategy_name'),
        db.Index('idx_ilb_status', 'status'),
        db.Index('idx_ilb_created', 'created_at'),
    )

    def __repr__(self):
        return f'<IndividualLevelBacktest {self.level_type}_{self.level_source_timeframe} {self.strategy_name}>'

    def to_dict(self):
        return {
            'id': self.id,
            'level_type': self.level_type,
            'level_source_timeframe': self.level_source_timeframe,
            'trade_execution_timeframe': self.trade_execution_timeframe,
            'strategy_name': self.strategy_name,
            'parameters': self.parameters,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'status': self.status,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_pnl': self.total_pnl,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'avg_trade_duration': self.avg_trade_duration,
            'created_at': self.created_at.isoformat(),
            'finished_at': self.finished_at.isoformat() if self.finished_at else None,
            'error_message': self.error_message,
        }


class IndividualLevelTrade(db.Model):
    """Individual trade record from a level backtest."""
    __tablename__ = 'individual_level_trades'

    id = db.Column(db.Integer, primary_key=True)
    backtest_id = db.Column(db.Integer, db.ForeignKey('individual_level_backtests.id'), nullable=False)
    level_id = db.Column(db.Integer, db.ForeignKey('levels.id'))  # The specific level that triggered this trade

    # Entry
    entry_time = db.Column(db.DateTime, nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    direction = db.Column(db.String(10), nullable=False)  # "LONG" or "SHORT"
    stop_loss = db.Column(db.Float, nullable=False)
    take_profit = db.Column(db.Float, nullable=False)

    # Exit
    exit_time = db.Column(db.DateTime)
    exit_price = db.Column(db.Float)
    exit_reason = db.Column(db.String(20))  # "TP_HIT", "SL_HIT", "TIMEOUT"

    # Results
    pnl = db.Column(db.Float)  # $
    pnl_pct = db.Column(db.Float)  # %
    candles_held = db.Column(db.Integer)

    # Features for ML analysis
    entry_volatility = db.Column(db.Float)  # ATR at entry
    volume_ratio = db.Column(db.Float)  # Volume vs MA at entry
    distance_to_level = db.Column(db.Float)  # % distance from exact level
    zone_confluence = db.Column(db.Integer)  # Number of levels in zone

    # Additional metadata
    metadata_json = db.Column('metadata', db.JSON)

    # Relationships
    level = db.relationship('Level', backref='backtest_trades')

    __table_args__ = (
        db.Index('idx_ilt_backtest', 'backtest_id'),
        db.Index('idx_ilt_level', 'level_id'),
        db.Index('idx_ilt_entry_time', 'entry_time'),
        db.Index('idx_ilt_direction', 'direction'),
        db.Index('idx_ilt_exit_reason', 'exit_reason'),
    )

    def __repr__(self):
        return f'<IndividualLevelTrade {self.direction} @{self.entry_price} pnl={self.pnl}>'

    def to_dict(self):
        return {
            'id': self.id,
            'backtest_id': self.backtest_id,
            'level_id': self.level_id,
            'entry_time': self.entry_time.isoformat(),
            'entry_price': self.entry_price,
            'direction': self.direction,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'candles_held': self.candles_held,
            'entry_volatility': self.entry_volatility,
            'volume_ratio': self.volume_ratio,
            'distance_to_level': self.distance_to_level,
            'zone_confluence': self.zone_confluence,
            'metadata': self.metadata_json,
        }
