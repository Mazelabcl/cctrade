"""Paper trade model — tracks simulated trades from the confluence scalper."""
from datetime import datetime
from ..extensions import db


class PaperTrade(db.Model):
    __tablename__ = 'paper_trades'

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Signal
    signal_time = db.Column(db.DateTime, nullable=False)
    candle_id = db.Column(db.Integer, db.ForeignKey('candles.id'), nullable=True)
    direction = db.Column(db.String(5), nullable=False)  # LONG / SHORT
    entry_price = db.Column(db.Float, nullable=False)
    stop_loss = db.Column(db.Float, nullable=False)
    risk_pct = db.Column(db.Float)  # SL distance as % of entry
    confluence_score = db.Column(db.Integer)  # Unique level types in zone
    level_types_json = db.Column(db.Text)  # JSON list of level types in zone

    # Trail state
    breakeven_reached = db.Column(db.Boolean, default=False)
    current_trail_sl = db.Column(db.Float)  # Current trailing SL (updated each candle)
    bars_since_entry = db.Column(db.Integer, default=0)

    # Outcome (filled when trade closes)
    exit_time = db.Column(db.DateTime)
    exit_price = db.Column(db.Float)
    exit_reason = db.Column(db.String(20))  # SL_HIT, TRAIL_HIT, TIMEOUT, MANUAL
    pnl_r = db.Column(db.Float)  # P&L in R units
    status = db.Column(db.String(10), default='open')  # open, closed

    __table_args__ = (
        db.Index('idx_paper_trades_status', 'status'),
        db.Index('idx_paper_trades_signal_time', 'signal_time'),
    )

    @property
    def risk(self):
        """Absolute risk (entry to SL distance)."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def is_long(self):
        return self.direction == 'LONG'

    @property
    def current_pnl_r(self):
        """Current unrealized P&L in R (for open trades, needs current price)."""
        if self.status == 'closed':
            return self.pnl_r
        return None  # Caller must provide current price

    def compute_pnl(self, exit_price):
        """Compute P&L in R units given an exit price."""
        if self.risk <= 0:
            return 0
        if self.is_long:
            return round((exit_price - self.entry_price) / self.risk, 2)
        else:
            return round((self.entry_price - exit_price) / self.risk, 2)

    def __repr__(self):
        return (f'<PaperTrade {self.id} {self.direction} {self.entry_price:.0f} '
                f'{self.status} {self.pnl_r or "open"}R>')
