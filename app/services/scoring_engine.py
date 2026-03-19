"""Scoring engine — rates trade setups like a coach's checklist.

Instead of ML predictions, scores each "touch event" (price arrives at level)
using weighted factors derived from backtest win rates and market context.

Score = sum of factor scores. Higher = stronger setup.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score weights — calibrated from actual backtest win rates
# ---------------------------------------------------------------------------

# Default win rate for level types not in backtest (e.g., PrevSession, VWAP)
DEFAULT_WIN_RATE = 0.35


def build_level_scores_from_db(db: Session) -> dict:
    """Build level type scores from actual backtest win rates.

    Returns dict: (level_type, timeframe) -> avg_win_rate (0-1 scale)
    Falls back to DEFAULT_WIN_RATE for unknown types.
    """
    try:
        from ..models.individual_level_backtest import IndividualLevelBacktest
        from sqlalchemy import func as sa_func

        results = (
            db.query(
                IndividualLevelBacktest.level_type,
                IndividualLevelBacktest.level_source_timeframe,
                sa_func.avg(IndividualLevelBacktest.win_rate),
                sa_func.sum(IndividualLevelBacktest.total_trades),
            )
            .filter(
                IndividualLevelBacktest.status == 'completed',
                IndividualLevelBacktest.total_trades >= 10,
                IndividualLevelBacktest.win_rate.isnot(None),
            )
            .group_by(
                IndividualLevelBacktest.level_type,
                IndividualLevelBacktest.level_source_timeframe,
            )
            .all()
        )
        cache = {}
        for lt, tf, wr, total in results:
            cache[(lt, tf)] = wr / 100.0  # normalize to 0-1
        logger.info("Level scores built from backtest: %d entries", len(cache))
        return cache
    except Exception as exc:
        logger.warning("Could not build level scores: %s. Using defaults.", exc)
        return {}


def get_level_score(level_type: str, timeframe: str,
                    score_cache: dict) -> float:
    """Get the score for a level type + timeframe combo.

    Uses actual backtest win rate. Falls back to DEFAULT_WIN_RATE.
    Score is win_rate * 10 to put it on a 0-10 scale.
    """
    wr = score_cache.get((level_type, timeframe), DEFAULT_WIN_RATE)
    return wr * 10.0  # e.g., 0.77 → 7.7 points


@dataclass
class TradeSetup:
    """A scored trade setup when price touches a level zone."""
    timestamp: datetime
    price: float
    direction: str  # 'LONG' or 'SHORT'
    primary_level_price: float
    primary_level_type: str
    primary_level_tf: str
    stop_loss: float
    take_profit: Optional[float] = None
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)
    confluence_levels: list = field(default_factory=list)
    levels_in_zone: int = 0
    risk_pct: float = 0.0


def score_touch_event(
    candle: dict,
    levels_near: pd.DataFrame,
    score_cache: dict = None,
    next_level_opposite: Optional[float] = None,
    wick_rejection_threshold: float = 0.4,
    zone_width_pct: float = 0.015,
) -> Optional[TradeSetup]:
    """Score a candle that touches a level zone.

    Args:
        candle: dict with open, high, low, close, volume, open_time
        levels_near: DataFrame of naked levels near this candle's price
            (columns: price_level, level_type, timeframe, source)
        score_cache: dict from build_level_scores_from_db() — maps
            (level_type, timeframe) to win_rate. If None, uses DEFAULT_WIN_RATE.
        next_level_opposite: price of next level in opposite direction (for TP)
        wick_rejection_threshold: min wick ratio to count as rejection
        zone_width_pct: zone width for grouping nearby levels

    Returns:
        TradeSetup if a valid setup is found, None otherwise.
    """
    if levels_near.empty:
        return None

    o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
    total_range = h - l
    if total_range == 0:
        return None

    # Determine direction from price action
    # Check each level for touch + close confirmation
    setup = None
    best_score = 0

    for _, level in levels_near.iterrows():
        lp = level['price_level']
        lt = level['level_type']
        tf = level.get('timeframe', 'daily')
        tol = lp * 0.005  # 0.5% touch tolerance

        direction = None
        sl = None

        # LONG: wick pierces level from above, close above level
        if l <= lp + tol and c > lp:
            direction = 'LONG'
            sl = l * 0.999  # SL just below wick low

        # SHORT: wick pierces level from below, close below level
        elif h >= lp - tol and c < lp:
            direction = 'SHORT'
            sl = h * 1.001  # SL just above wick high

        if not direction:
            continue

        # --- SCORING ---
        breakdown = {}

        # 1. Level type quality (from backtest win rates)
        type_score = get_level_score(lt, tf, score_cache or {})
        breakdown['level_type'] = round(type_score, 2)
        breakdown['level_type_detail'] = f'{lt}/{tf}'

        # 3. Wick rejection score (0-3)
        if direction == 'LONG':
            lower_wick = (min(o, c) - l) / total_range
            wick_score = min(lower_wick / wick_rejection_threshold, 1.0) * 3.0
        else:
            upper_wick = (h - max(o, c)) / total_range
            wick_score = min(upper_wick / wick_rejection_threshold, 1.0) * 3.0
        breakdown['wick_rejection'] = round(wick_score, 2)

        # 4. Body position (close near high for LONG, near low for SHORT) (0-2)
        body_pos = (c - l) / total_range
        if direction == 'LONG':
            body_score = body_pos * 2.0  # higher = better for LONG
        else:
            body_score = (1.0 - body_pos) * 2.0  # lower = better for SHORT
        breakdown['body_position'] = round(body_score, 2)

        # 5. Confluence — other levels in the zone (0-5)
        zone_low = lp * (1 - zone_width_pct)
        zone_high = lp * (1 + zone_width_pct)
        zone_levels = levels_near[
            (levels_near['price_level'] >= zone_low) &
            (levels_near['price_level'] <= zone_high)
        ]
        confluence_count = len(zone_levels)
        confluence_types = zone_levels['level_type'].unique().tolist()
        confluence_score = min(confluence_count - 1, 5) * 1.0  # -1 because primary counts
        breakdown['confluence'] = round(confluence_score, 2)

        # 6. Distance precision — how close did wick get to level (0-2)
        if direction == 'LONG':
            touch_dist = abs(l - lp) / lp
        else:
            touch_dist = abs(h - lp) / lp
        precision_score = max(0, 2.0 - touch_dist * 200)  # 0% dist = 2pts, 1% = 0pts
        breakdown['precision'] = round(precision_score, 2)

        # Total score (type_score already incorporates TF via backtest data)
        total = type_score + wick_score + body_score + confluence_score + precision_score
        breakdown['total'] = round(total, 2)

        if total > best_score:
            risk_pct = abs(c - sl) / c
            tp = next_level_opposite if next_level_opposite else None

            setup = TradeSetup(
                timestamp=candle['open_time'],
                price=c,
                direction=direction,
                primary_level_price=lp,
                primary_level_type=lt,
                primary_level_tf=tf,
                stop_loss=sl,
                take_profit=tp,
                score=total,
                score_breakdown=breakdown,
                confluence_levels=confluence_types,
                levels_in_zone=confluence_count,
                risk_pct=risk_pct,
            )
            best_score = total

    return setup


def scan_for_setups(
    db: Session,
    exec_tf: str = '4h',
    symbol: str = 'BTCUSDT',
    min_score: float = 10.0,
    level_filter: str = 'htf_no_igor',
    zone_width_pct: float = 0.015,
) -> list[TradeSetup]:
    """Scan all candles for trade setups using the scoring engine.

    Args:
        exec_tf: Execution timeframe to scan
        min_score: Minimum score to include a setup
        level_filter: Which levels to use ('htf_all', 'htf_no_igor', 'htf_no_fibs')
        zone_width_pct: Zone width for confluence detection

    Returns:
        List of TradeSetup objects sorted by time.
    """
    from ..models import Candle, Level

    # Build score cache from backtest win rates
    score_cache = build_level_scores_from_db(db)

    # Load candles
    candles = (
        db.query(Candle)
        .filter_by(symbol=symbol, timeframe=exec_tf)
        .order_by(Candle.open_time)
        .all()
    )
    if not candles:
        return []

    # Load D/W/M levels
    level_query = db.query(Level).filter(
        Level.timeframe.in_(['daily', 'weekly', 'monthly'])
    )
    if level_filter == 'htf_no_igor':
        level_query = level_query.filter(
            Level.level_type.notin_(['Fib_0.25', 'Fib_0.50', 'Fib_0.75'])
        )
    elif level_filter == 'htf_no_fibs':
        level_query = level_query.filter(~Level.level_type.like('Fib_%'))

    all_levels = pd.DataFrame([{
        'price_level': l.price_level,
        'level_type': l.level_type,
        'timeframe': l.timeframe,
        'source': l.source,
        'created_at': l.created_at,
        'first_touched_at': l.first_touched_at,
    } for l in level_query.all()])

    if all_levels.empty:
        return []

    setups = []
    level_prices = all_levels['price_level'].values

    for candle in candles:
        c_dict = {
            'open_time': candle.open_time,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume,
        }

        # Filter levels: created before this candle AND still naked
        mask = (all_levels['created_at'] <= candle.open_time)
        if 'first_touched_at' in all_levels.columns:
            mask &= (
                all_levels['first_touched_at'].isna() |
                (all_levels['first_touched_at'] > candle.open_time)
            )
        valid_levels = all_levels[mask]

        if valid_levels.empty:
            continue

        # Find levels near this candle (within 2% of close)
        price = candle.close
        near_mask = (
            (valid_levels['price_level'] >= price * 0.98) &
            (valid_levels['price_level'] <= price * 1.02)
        )
        levels_near = valid_levels[near_mask]

        if levels_near.empty:
            continue

        # Find next level in opposite direction for TP
        # (we'll determine direction inside score_touch_event, so pass both)
        above = valid_levels[valid_levels['price_level'] > price * 1.005]
        below = valid_levels[valid_levels['price_level'] < price * 0.995]
        next_above = float(above['price_level'].min()) if not above.empty else None
        next_below = float(below['price_level'].max()) if not below.empty else None

        setup = score_touch_event(
            c_dict, levels_near,
            score_cache=score_cache,
            next_level_opposite=None,  # will be set after direction is known
            zone_width_pct=zone_width_pct,
        )

        if setup and setup.score >= min_score:
            # Set TP based on direction
            if setup.direction == 'LONG' and next_above:
                setup.take_profit = next_above
            elif setup.direction == 'SHORT' and next_below:
                setup.take_profit = next_below
            setups.append(setup)

    logger.info("Scoring scan [%s]: %d setups found (min_score=%.1f) from %d candles",
                exec_tf, len(setups), min_score, len(candles))
    return setups


def backtest_scoring_engine(
    db: Session,
    exec_tf: str = '4h',
    min_score: float = 10.0,
    level_filter: str = 'htf_no_igor',
) -> dict:
    """Backtest the scoring engine: find setups, simulate trades, measure performance.

    Returns summary dict with metrics.
    """
    from ..models import Candle

    setups = scan_for_setups(db, exec_tf=exec_tf, min_score=min_score,
                             level_filter=level_filter)
    if not setups:
        return {'error': 'No setups found'}

    # Load candles for trade simulation
    candles = (
        db.query(Candle)
        .filter_by(symbol='BTCUSDT', timeframe=exec_tf)
        .order_by(Candle.open_time)
        .all()
    )
    candle_map = {c.open_time: c for c in candles}
    candle_list = list(candle_map.values())
    candle_times = [c.open_time for c in candle_list]

    trades = []
    for setup in setups:
        # Find the candle index
        try:
            idx = candle_times.index(setup.timestamp)
        except ValueError:
            continue

        # Simulate: scan forward up to 100 candles
        result = None
        for j in range(idx + 1, min(idx + 101, len(candle_list))):
            fc = candle_list[j]

            if setup.direction == 'LONG':
                if fc.low <= setup.stop_loss:
                    result = {'exit': 'SL_HIT', 'pnl_pct': -(setup.risk_pct)}
                    break
                if setup.take_profit and fc.high >= setup.take_profit:
                    tp_pnl = (setup.take_profit - setup.price) / setup.price
                    result = {'exit': 'TP_HIT', 'pnl_pct': tp_pnl}
                    break
            else:  # SHORT
                if fc.high >= setup.stop_loss:
                    result = {'exit': 'SL_HIT', 'pnl_pct': -(setup.risk_pct)}
                    break
                if setup.take_profit and fc.low <= setup.take_profit:
                    tp_pnl = (setup.price - setup.take_profit) / setup.price
                    result = {'exit': 'TP_HIT', 'pnl_pct': tp_pnl}
                    break

        if result is None:
            result = {'exit': 'TIMEOUT', 'pnl_pct': 0.0}

        trades.append({
            'timestamp': setup.timestamp,
            'direction': setup.direction,
            'score': setup.score,
            'level_type': setup.primary_level_type,
            'level_tf': setup.primary_level_tf,
            'confluence': setup.levels_in_zone,
            **result,
        })

    if not trades:
        return {'error': 'No trades completed'}

    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades['exit'] == 'TP_HIT']
    losses = df_trades[df_trades['exit'] == 'SL_HIT']

    gross_wins = wins['pnl_pct'].sum() if not wins.empty else 0
    gross_losses = abs(losses['pnl_pct'].sum()) if not losses.empty else 0
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    return {
        'exec_tf': exec_tf,
        'min_score': min_score,
        'level_filter': level_filter,
        'total_setups': len(setups),
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'timeouts': len(df_trades[df_trades['exit'] == 'TIMEOUT']),
        'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
        'profit_factor': round(pf, 2),
        'total_pnl_pct': round(df_trades['pnl_pct'].sum() * 100, 2),
        'avg_score': round(df_trades['score'].mean(), 2),
        'avg_winner_score': round(wins['score'].mean(), 2) if not wins.empty else 0,
        'avg_loser_score': round(losses['score'].mean(), 2) if not losses.empty else 0,
        'by_level_type': df_trades.groupby('level_type').agg(
            trades=('exit', 'count'),
            win_rate=('exit', lambda x: round((x == 'TP_HIT').sum() / len(x) * 100, 1)),
            avg_score=('score', 'mean'),
        ).to_dict('index'),
    }
