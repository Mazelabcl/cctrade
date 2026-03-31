"""Confluence Signal Service — detect trade signals on live 15m candles.

Uses the validated AutoResearch config:
- Level types: Fractal_support, Fractal_resistance, PrevSession_VWAP, PrevSession_VP_POC
- Confluence: 3+ unique types in ±1% zone
- Entry: close of touch candle
- SL: wick extreme + buffer
- Exit: breakeven trail, be_rr=2.75, timeout=45 candles, swing_lookback=10
"""
import json
import logging
from datetime import datetime, timedelta

from ..extensions import db
from ..models.candle import Candle
from ..models.level import Level
from ..models.paper_trade import PaperTrade
from sqlalchemy import and_, or_, func as sa_func

logger = logging.getLogger(__name__)

# Validated config from AutoResearch
SCALPER_CONFIG = {
    'level_types': ['Fractal_support', 'Fractal_resistance',
                    'PrevSession_VWAP', 'PrevSession_VP_POC'],
    'score_threshold': 3,       # Min unique level types in zone
    'zone_width': 0.01,         # 1% zone for confluence
    'touch_tolerance': 0.003,   # 0.3% wick tolerance
    'naked_only': True,
    'sl_buffer_pct': 0.001,     # 0.1% buffer on SL
    'be_rr': 2.75,              # Move SL to BE after +2.75R
    'timeout_candles': 45,      # Max bars before timeout exit
    'swing_lookback': 10,       # Bars for swing trail
    'cooldown_candles': 15,     # Min bars between trades
    'timeframe': '15m',
}


def check_for_signal(session):
    """Check the latest 15m candle for a confluence touch signal.

    Returns PaperTrade if signal found, None otherwise.
    """
    config = SCALPER_CONFIG

    # Get the most recent completed 15m candle
    latest_candle = (
        session.query(Candle)
        .filter(Candle.timeframe == config['timeframe'])
        .order_by(Candle.open_time.desc())
        .first()
    )
    if not latest_candle:
        return None

    # Check cooldown: skip if we have a recent trade
    recent_trade = (
        session.query(PaperTrade)
        .filter(PaperTrade.status == 'open')
        .first()
    )
    if recent_trade:
        return None  # Already have an open trade

    last_closed = (
        session.query(PaperTrade)
        .filter(PaperTrade.status == 'closed')
        .order_by(PaperTrade.signal_time.desc())
        .first()
    )
    if last_closed:
        cooldown_minutes = config['cooldown_candles'] * 15
        if latest_candle.open_time - last_closed.signal_time < timedelta(minutes=cooldown_minutes):
            return None

    # Check if we already signaled on this candle
    existing = (
        session.query(PaperTrade)
        .filter(PaperTrade.signal_time == latest_candle.open_time)
        .first()
    )
    if existing:
        return None

    # Load active naked levels of the configured types
    levels = (
        session.query(Level)
        .filter(
            Level.level_type.in_(config['level_types']),
            Level.invalidated_at.is_(None),
            Level.created_at <= latest_candle.open_time,
        )
        .all()
    )

    if config['naked_only']:
        levels = [l for l in levels if l.first_touched_at is None]

    if not levels:
        return None

    # Check for touch
    price = latest_candle.close
    low = latest_candle.low
    high = latest_candle.high
    tol = config['touch_tolerance']
    zone_width = config['zone_width']

    for level in levels:
        lp = level.price_level
        direction = None

        # LONG touch: wick went down to level, closed above
        if low <= lp * (1 + tol) and price > lp:
            direction = 'LONG'
        # SHORT touch: wick went up to level, closed below
        elif high >= lp * (1 - tol) and price < lp:
            direction = 'SHORT'

        if not direction:
            continue

        # Score confluence: count unique level types in zone
        zone_lo = lp * (1 - zone_width)
        zone_hi = lp * (1 + zone_width)
        zone_levels = [l for l in levels if zone_lo <= l.price_level <= zone_hi]
        zone_types = set(l.level_type for l in zone_levels)

        if len(zone_types) < config['score_threshold']:
            continue

        # Signal found! Create paper trade
        sl_buffer = config['sl_buffer_pct']
        if direction == 'LONG':
            sl = low * (1 - sl_buffer)
        else:
            sl = high * (1 + sl_buffer)

        risk = abs(price - sl)
        if risk <= 0 or risk / price > 0.05:
            continue

        trade = PaperTrade(
            signal_time=latest_candle.open_time,
            candle_id=latest_candle.id,
            direction=direction,
            entry_price=round(price, 2),
            stop_loss=round(sl, 2),
            risk_pct=round(risk / price * 100, 3),
            confluence_score=len(zone_types),
            level_types_json=json.dumps(sorted(zone_types)),
            current_trail_sl=round(sl, 2),
            status='open',
        )

        session.add(trade)
        session.commit()

        logger.info(f"Paper trade signal: {direction} at {price:.2f}, "
                    f"SL {sl:.2f}, confluence {len(zone_types)} types: {zone_types}")
        return trade

    return None


def update_open_trades(session):
    """Check open paper trades against recent candles for SL/trail/timeout."""
    config = SCALPER_CONFIG

    open_trades = (
        session.query(PaperTrade)
        .filter(PaperTrade.status == 'open')
        .all()
    )

    if not open_trades:
        return []

    closed = []

    for trade in open_trades:
        # Get candles since entry
        candles = (
            session.query(Candle)
            .filter(
                Candle.timeframe == config['timeframe'],
                Candle.open_time > trade.signal_time,
            )
            .order_by(Candle.open_time.asc())
            .all()
        )

        if not candles:
            continue

        trade.bars_since_entry = len(candles)
        sl_cur = trade.current_trail_sl
        risk = trade.risk
        be_rr = config['be_rr']
        entry = trade.entry_price

        # Compute swing lows/highs for trail
        lookback = config['swing_lookback']
        all_candles = (
            session.query(Candle)
            .filter(
                Candle.timeframe == config['timeframe'],
                Candle.open_time >= trade.signal_time - timedelta(hours=lookback),
            )
            .order_by(Candle.open_time.asc())
            .all()
        )

        for i, candle in enumerate(candles):
            # Get recent candles for swing calculation
            candle_idx_in_all = next((j for j, c in enumerate(all_candles) if c.id == candle.id), -1)
            if candle_idx_in_all >= 0:
                recent = all_candles[max(0, candle_idx_in_all - lookback):candle_idx_in_all + 1]
                sw_low = min(c.low for c in recent) if recent else candle.low
                sw_high = max(c.high for c in recent) if recent else candle.high
            else:
                sw_low = candle.low
                sw_high = candle.high

            if trade.is_long:
                # Check SL hit
                if candle.low <= sl_cur:
                    _close_trade(trade, candle, sl_cur, 'SL_HIT' if not trade.breakeven_reached else 'TRAIL_HIT')
                    closed.append(trade)
                    break

                # Check breakeven activation
                if candle.high >= entry + be_rr * risk:
                    trade.breakeven_reached = True

                # Trail: after BE reached, trail with swing lows (at least entry)
                if trade.breakeven_reached:
                    sl_cur = max(sl_cur, max(entry, sw_low))
            else:
                # SHORT
                if candle.high >= sl_cur:
                    _close_trade(trade, candle, sl_cur, 'SL_HIT' if not trade.breakeven_reached else 'TRAIL_HIT')
                    closed.append(trade)
                    break

                if candle.low <= entry - be_rr * risk:
                    trade.breakeven_reached = True

                if trade.breakeven_reached:
                    sl_cur = min(sl_cur, min(entry, sw_high))

            trade.current_trail_sl = round(sl_cur, 2)

        # Check timeout
        if trade.status == 'open' and len(candles) >= config['timeout_candles']:
            last_candle = candles[-1]
            _close_trade(trade, last_candle, last_candle.close, 'TIMEOUT')
            closed.append(trade)

    session.commit()

    for t in closed:
        logger.info(f"Paper trade closed: {t.direction} entry={t.entry_price:.2f} "
                    f"exit={t.exit_price:.2f} reason={t.exit_reason} pnl={t.pnl_r:+.2f}R")

    return closed


def _close_trade(trade, candle, exit_price, reason):
    """Close a paper trade with given exit price and reason."""
    trade.exit_time = candle.open_time
    trade.exit_price = round(exit_price, 2)
    trade.exit_reason = reason
    trade.pnl_r = trade.compute_pnl(exit_price)
    trade.status = 'closed'


def get_paper_trade_stats(session):
    """Compute summary stats for all paper trades."""
    trades = session.query(PaperTrade).all()

    if not trades:
        return {
            'total_trades': 0, 'open_trades': 0, 'closed_trades': 0,
            'win_rate': 0, 'profit_factor': 0, 'total_r': 0, 'avg_r': 0,
            'net_per_month': 0, 'best_trade': 0, 'worst_trade': 0,
        }

    open_trades = [t for t in trades if t.status == 'open']
    closed_trades = [t for t in trades if t.status == 'closed']

    if not closed_trades:
        return {
            'total_trades': len(trades), 'open_trades': len(open_trades),
            'closed_trades': 0, 'win_rate': 0, 'profit_factor': 0,
            'total_r': 0, 'avg_r': 0, 'net_per_month': 0,
            'best_trade': 0, 'worst_trade': 0,
        }

    pnls = [t.pnl_r for t in closed_trades if t.pnl_r is not None]
    if not pnls:
        return {
            'total_trades': len(trades), 'open_trades': len(open_trades),
            'closed_trades': len(closed_trades), 'win_rate': 0,
            'profit_factor': 0, 'total_r': 0, 'avg_r': 0,
            'net_per_month': 0, 'best_trade': 0, 'worst_trade': 0,
        }

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_r = sum(pnls)
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999

    # Estimate monthly (based on date range)
    if len(closed_trades) >= 2:
        first = min(t.signal_time for t in closed_trades)
        last = max(t.signal_time for t in closed_trades)
        days = max((last - first).days, 1)
        months = max(days / 30, 1)
    else:
        months = 1

    # Commission estimate ($10 risk)
    risk_per_trade = 10
    avg_risk_pct = sum(t.risk_pct for t in closed_trades if t.risk_pct) / len(closed_trades) / 100
    avg_pos = risk_per_trade / avg_risk_pct if avg_risk_pct > 0 else 0
    comm_per_trade = avg_pos * 0.0006
    gross_profit = total_r * risk_per_trade
    total_comm = comm_per_trade * len(closed_trades)
    net = gross_profit - total_comm

    return {
        'total_trades': len(trades),
        'open_trades': len(open_trades),
        'closed_trades': len(closed_trades),
        'win_rate': round(len(wins) / len(pnls) * 100, 1),
        'profit_factor': pf,
        'total_r': round(total_r, 1),
        'avg_r': round(total_r / len(pnls), 3),
        'net_per_month': round(net / months, 0),
        'best_trade': round(max(pnls), 2),
        'worst_trade': round(min(pnls), 2),
        'gross_profit_10usd': round(gross_profit, 0),
        'total_commission': round(total_comm, 0),
        'net_profit_10usd': round(net, 0),
    }
