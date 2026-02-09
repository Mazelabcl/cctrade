"""Backtrader strategy that trades fractal signals with bracket orders."""
import logging
from datetime import datetime

import backtrader as bt

logger = logging.getLogger(__name__)


class FractalStrategy(bt.Strategy):
    """Reads signal / confidence / SL / TP from data feed lines.

    - Enters LONG on signal == 1 with bracket order (SL + TP).
    - Enters SHORT on signal == -1 with bracket order (SL + TP).
    - Position sizing based on ATR and risk percentage.
    """

    params = (
        ('risk_pct', 0.02),          # risk per trade as fraction of portfolio
        ('confidence_threshold', 0.55),
    )

    def __init__(self):
        self.signal = self.data.signal
        self.confidence = self.data.confidence
        self.sl_price = self.data.stop_loss
        self.tp_price = self.data.take_profit
        self.atr = self.data.atr
        self.trade_log = []
        self._pending_entry = None

    def next(self):
        if self.position:
            return  # already in a trade

        sig = int(self.signal[0])
        conf = self.confidence[0]
        sl = self.sl_price[0]
        tp = self.tp_price[0]
        atr_val = self.atr[0]

        if sig == 0 or conf < self.p.confidence_threshold:
            return
        if sl == 0 or tp == 0 or atr_val == 0:
            return

        # Position size: risk_pct of portfolio / risk_per_unit
        portfolio_value = self.broker.getvalue()
        risk_amount = portfolio_value * self.p.risk_pct
        price = self.data.close[0]

        risk_per_unit = abs(price - sl)
        if risk_per_unit <= 0:
            return

        size = risk_amount / risk_per_unit

        if sig == 1:  # LONG
            self.buy_bracket(
                size=size,
                stopprice=sl,
                limitprice=tp,
            )
            self._pending_entry = {
                'direction': 'LONG',
                'entry_time': self.data.datetime.datetime(0),
                'entry_price': price,
                'size': size,
                'sl': sl,
                'tp': tp,
                'atr': atr_val,
                'confidence': conf,
            }

        elif sig == -1:  # SHORT
            self.sell_bracket(
                size=size,
                stopprice=sl,
                limitprice=tp,
            )
            self._pending_entry = {
                'direction': 'SHORT',
                'entry_time': self.data.datetime.datetime(0),
                'entry_price': price,
                'size': size,
                'sl': sl,
                'tp': tp,
                'atr': atr_val,
                'confidence': conf,
            }

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        entry = self._pending_entry or {}
        self.trade_log.append({
            'direction': entry.get('direction', ''),
            'entry_time': str(entry.get('entry_time', '')),
            'entry_price': entry.get('entry_price', 0),
            'exit_time': str(self.data.datetime.datetime(0)),
            'exit_price': trade.price,
            'pnl': trade.pnl,
            'pnlcomm': trade.pnlcomm,
            'size': entry.get('size', 0),
            'sl': entry.get('sl', 0),
            'tp': entry.get('tp', 0),
            'atr': entry.get('atr', 0),
            'confidence': entry.get('confidence', 0),
        })
        self._pending_entry = None

    def get_trade_analysis(self) -> dict:
        """Compute custom trade metrics from the trade log."""
        if not self.trade_log:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
            }

        wins = [t for t in self.trade_log if t['pnlcomm'] > 0]
        losses = [t for t in self.trade_log if t['pnlcomm'] <= 0]

        total_wins = sum(t['pnlcomm'] for t in wins) if wins else 0
        total_losses = abs(sum(t['pnlcomm'] for t in losses)) if losses else 0

        return {
            'total_trades': len(self.trade_log),
            'win_rate': len(wins) / len(self.trade_log) if self.trade_log else 0.0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'avg_win': total_wins / len(wins) if wins else 0.0,
            'avg_loss': -total_losses / len(losses) if losses else 0.0,
        }
