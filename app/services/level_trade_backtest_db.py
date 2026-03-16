"""Level Trade Backtest — DB-backed.

Backtests each (level_type × timeframe) combination using candles and levels
stored in SQLite.  No CSVs needed.

Entry logic (per candle):
  LONG  — wick pierces level from above (low <= level + tol) AND close > level
  SHORT — wick pierces level from below (high >= level - tol) AND close < level

SL / TP:
  SL   = entry-candle wick extreme  ±  small buffer
  TP   = entry ± RR × risk          (risk = |entry - SL|)

The simulation respects touch order: a level is only considered "naked" until
the first candle that interacts with it (closes on either side while wick hits).
Once touched it is excluded from further entry signals.
"""
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..models import IndividualLevelBacktest, IndividualLevelTrade

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

TOUCH_TOLERANCE_PCT = 0.005   # 0.5 % — wick must come within this of the level
SL_BUFFER_PCT       = 0.001   # 0.1 % beyond the wick extreme
TIMEOUT_CANDLES     = 100     # close at market after N candles if TP/SL not hit
DEFAULT_RR_RATIOS   = [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Data loaders (DB → DataFrame)
# ---------------------------------------------------------------------------

def _query_to_df(session: Session, sql: str, params: dict = None) -> pd.DataFrame:
    """Execute a raw SQL string and return a DataFrame.

    Uses session.execute() — works with both SQLAlchemy 1.x and 2.x without
    the pd.read_sql / text() compatibility issues.
    """
    from sqlalchemy import text
    result = session.execute(text(sql), params or {})
    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=list(result.keys()))


def load_candles_db(session: Session,
                    timeframe: str = '1h',
                    symbol: str = 'BTCUSDT') -> pd.DataFrame:
    """Load all candles for a timeframe from SQLite (fast raw-SQL path)."""
    df = _query_to_df(
        session,
        "SELECT open_time, open, high, low, close, volume "
        "FROM candles "
        "WHERE symbol = :symbol AND timeframe = :tf "
        "ORDER BY open_time ASC",
        {'symbol': symbol, 'tf': timeframe},
    )
    if not df.empty:
        df['open_time'] = pd.to_datetime(df['open_time'])
    return df


def load_levels_db(session: Session,
                   symbol: str = 'BTCUSDT') -> pd.DataFrame:
    """Load all non-invalidated levels from SQLite (fast raw-SQL path)."""
    df = _query_to_df(
        session,
        "SELECT id, price_level, level_type, timeframe, source, created_at "
        "FROM levels "
        "WHERE invalidated_at IS NULL "
        "ORDER BY created_at ASC",
    )
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
    return df


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _simulate_multi_rr(candles: pd.DataFrame,
                        levels: pd.DataFrame,
                        rr_ratios: list[float],
                        tolerance_pct: float = TOUCH_TOLERANCE_PCT,
                        sl_buffer_pct: float = SL_BUFFER_PCT,
                        timeout: int = TIMEOUT_CANDLES,
                        naked_only: bool = True) -> dict[float, list[dict]]:
    """
    Simulate trades for a pre-filtered group of levels over all candles.
    Processes ALL requested RR ratios in a single candle scan (3× faster
    than running separate simulations per RR).

    Entry/exit are shared across RR ratios: same entry candle, same SL,
    different TP.  Touch tracking is shared — a level consumed by any
    strategy is marked consumed for all.

    Returns {rr: [trade_dicts]} without pnl (added by _add_pnl).
    """
    if candles.empty or levels.empty:
        return {rr: [] for rr in rr_ratios}

    # Numpy arrays
    c_times  = candles['open_time'].values
    c_highs  = candles['high'].values.astype(np.float64)
    c_lows   = candles['low'].values.astype(np.float64)
    c_closes = candles['close'].values.astype(np.float64)

    lev_prices  = levels['price_level'].values.astype(np.float64)
    lev_created = pd.to_datetime(levels['created_at']).values
    lev_ids     = levels['id'].values.astype(int)
    n_levels    = len(lev_prices)

    # Shared touch tracker
    level_touched = np.zeros(n_levels, dtype=bool)

    # Per-RR state
    trades:   dict[float, list[dict]] = {rr: [] for rr in rr_ratios}
    positions: dict[float, Optional[dict]] = {rr: None for rr in rr_ratios}

    for i in range(len(candles)):
        ct = c_times[i]
        hi = c_highs[i]
        lo = c_lows[i]
        cl = c_closes[i]

        # ----------------------------------------------------------------
        # Vectorized touch update helper — replaces all Python level loops
        # A level is "touched" when the candle range [lo, hi] overlaps
        # the level's tolerance band [lp*(1-tol), lp*(1+tol)].
        # ----------------------------------------------------------------
        tol = tolerance_pct

        def _mark_touched():
            active = (lev_created < ct) & ~level_touched
            if not active.any():
                return
            hit = active & (lev_prices * (1.0 + tol) >= lo) & (lev_prices * (1.0 - tol) <= hi)
            level_touched[:] |= hit

        # ----------------------------------------------------------------
        # 1. Manage open positions (one per RR)
        # ----------------------------------------------------------------
        any_in_position = False
        for rr in rr_ratios:
            pos = positions[rr]
            if pos is None:
                continue
            any_in_position = True
            pos['candles_held'] += 1
            exited = False

            if pos['direction'] == 'LONG':
                if lo <= pos['sl']:
                    trades[rr].append({**pos, 'exit_time': ct,
                                       'exit_price': pos['sl'], 'exit_reason': 'SL_HIT'})
                    exited = True
                elif hi >= pos['tp']:
                    trades[rr].append({**pos, 'exit_time': ct,
                                       'exit_price': pos['tp'], 'exit_reason': 'TP_HIT'})
                    exited = True
            else:  # SHORT
                if hi >= pos['sl']:
                    trades[rr].append({**pos, 'exit_time': ct,
                                       'exit_price': pos['sl'], 'exit_reason': 'SL_HIT'})
                    exited = True
                elif lo <= pos['tp']:
                    trades[rr].append({**pos, 'exit_time': ct,
                                       'exit_price': pos['tp'], 'exit_reason': 'TP_HIT'})
                    exited = True

            if exited:
                positions[rr] = None
            elif pos['candles_held'] >= timeout:
                trades[rr].append({**pos, 'exit_time': ct,
                                   'exit_price': cl, 'exit_reason': 'TIMEOUT'})
                positions[rr] = None

        # While in position, still mark any levels price crosses
        if any_in_position and naked_only:
            _mark_touched()

        # ----------------------------------------------------------------
        # 2. Look for entry signals (only when ALL RR slots are free)
        # ----------------------------------------------------------------
        if any(positions[rr] is not None for rr in rr_ratios):
            continue

        time_mask = lev_created < ct
        dist      = np.abs(lev_prices - cl) / cl
        dist_mask = dist < 0.15

        if naked_only:
            mask = time_mask & dist_mask & ~level_touched
        else:
            mask = time_mask & dist_mask

        candidates = np.where(mask)[0]
        if candidates.size == 0:
            if naked_only:
                _mark_touched()
            continue

        candidates = candidates[np.argsort(dist[candidates])]

        entered = False
        for j in candidates:
            lp  = lev_prices[j]
            lp_tol = lp * tol

            if lo <= lp + lp_tol and cl > lp:
                direction = 'LONG'
                sl   = lo * (1.0 - sl_buffer_pct)
                risk = cl - sl
            elif hi >= lp - lp_tol and cl < lp:
                direction = 'SHORT'
                sl   = hi * (1.0 + sl_buffer_pct)
                risk = sl - cl
            else:
                continue

            if risk <= 0:
                continue

            level_touched[j] = True
            for rr in rr_ratios:
                tp = (cl + risk * rr) if direction == 'LONG' else (cl - risk * rr)
                positions[rr] = {
                    'entry_time':   ct,
                    'entry_price':  cl,
                    'direction':    direction,
                    'sl':           sl,
                    'tp':           tp,
                    'level_id':     int(lev_ids[j]),
                    'level_price':  lp,
                    'candles_held': 0,
                    'rr':           rr,
                }
            entered = True
            break

        if not entered and naked_only:
            _mark_touched()

    # Close any remaining positions at last candle
    for rr in rr_ratios:
        if positions[rr] is not None:
            trades[rr].append({**positions[rr],
                               'exit_time':   c_times[-1],
                               'exit_price':  float(c_closes[-1]),
                               'exit_reason': 'TIMEOUT'})

    return trades


# ---------------------------------------------------------------------------
# PnL + metrics helpers
# ---------------------------------------------------------------------------

def _add_pnl(t: dict) -> dict:
    if t['direction'] == 'LONG':
        pnl = t['exit_price'] - t['entry_price']
    else:
        pnl = t['entry_price'] - t['exit_price']
    return {**t,
            'pnl':     pnl,
            'pnl_pct': pnl / t['entry_price'] if t['entry_price'] else 0.0}


def _compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0.0, 'profit_factor': 0.0, 'sharpe_ratio': 0.0,
            'max_drawdown': 0.0, 'total_pnl': 0.0,
            'avg_win': 0.0, 'avg_loss': 0.0, 'avg_trade_duration': 0.0,
        }

    pnls = np.array([t['pnl_pct'] for t in trades])
    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    cum         = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    max_dd      = float((running_max - cum).max())
    sharpe      = float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0
    gross_win   = float(wins.sum())   if len(wins)   else 0.0
    gross_loss  = float(abs(losses.sum())) if len(losses) else 0.0

    return {
        'total_trades':      len(trades),
        'winning_trades':    int(len(wins)),
        'losing_trades':     int(len(losses)),
        'win_rate':          float(len(wins) / len(trades) * 100),
        'profit_factor':     float(gross_win / gross_loss) if gross_loss > 0 else (float('inf') if gross_win > 0 else 0.0),
        'sharpe_ratio':      sharpe,
        'max_drawdown':      max_dd * 100,
        'total_pnl':         float(sum(t['pnl'] for t in trades)),
        'avg_win':           float(wins.mean())   if len(wins)   else 0.0,
        'avg_loss':          float(losses.mean()) if len(losses) else 0.0,
        'avg_trade_duration': float(np.mean([t['candles_held'] for t in trades])),
    }


# ---------------------------------------------------------------------------
# Batch runner — all (level_type × timeframe) × rr_ratios
# ---------------------------------------------------------------------------

def run_level_trade_backtest(
    session: Session,
    exec_timeframe: str = '1h',
    rr_ratios: Optional[list[float]] = None,
    tolerance_pct: float = TOUCH_TOLERANCE_PCT,
    timeout: int = TIMEOUT_CANDLES,
    naked_only: bool = True,
    symbol: str = 'BTCUSDT',
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict]:
    """
    Run the full batch backtest from DB data.

    For every (level_type, timeframe) combination found in the Level table
    and every requested RR ratio, simulate trades on the given exec_timeframe
    candles.  Results are persisted to IndividualLevelBacktest / IndividualLevelTrade.

    Args:
        session:        SQLAlchemy session (must be within app context).
        exec_timeframe: Candle timeframe used for trade execution ('1h', '4h').
        rr_ratios:      Risk-reward ratios to test (default [1.0, 2.0, 3.0]).
        tolerance_pct:  Wick-touch tolerance as fraction of level price (default 0.005).
        timeout:        Max candles before forced exit (default 100).
        naked_only:     Only trade on levels that have never been touched.
        symbol:         Trading pair (default 'BTCUSDT').
        progress_cb:    Optional callback(done, total, label) for UI progress.

    Returns:
        List of summary dicts, one per (level_type, timeframe, rr).
    """
    if rr_ratios is None:
        rr_ratios = DEFAULT_RR_RATIOS

    logger.info("Loading %s candles from DB...", exec_timeframe)
    candles = load_candles_db(session, exec_timeframe, symbol)
    if candles.empty:
        logger.error("No candles found for %s %s", symbol, exec_timeframe)
        return []

    logger.info("Loading levels from DB...")
    levels = load_levels_db(session, symbol)
    if levels.empty:
        logger.error("No levels found for %s", symbol)
        return []

    # All distinct (level_type, timeframe) combos
    combos = (
        levels[['level_type', 'timeframe']]
        .drop_duplicates()
        .sort_values(['level_type', 'timeframe'])
        .values.tolist()
    )
    total = len(combos) * len(rr_ratios)  # total records to be saved

    logger.info("Batch: %d combos × %d RR ratios = %d runs (single scan per combo)",
                len(combos), len(rr_ratios), total)

    results: list[dict] = []

    combo_count = 0
    for (lt, tf) in combos:
        combo_count += 1
        group = levels[(levels['level_type'] == lt) & (levels['timeframe'] == tf)].copy()

        # Report progress at start of each combo (covers all its RR ratios)
        done = (combo_count - 1) * len(rr_ratios) + 1
        label = f"{lt} / {tf}"
        if progress_cb:
            progress_cb(done, total, label)

        # Single candle scan → trades for ALL rr_ratios at once
        trades_by_rr = _simulate_multi_rr(
            candles, group,
            rr_ratios=rr_ratios,
            tolerance_pct=tolerance_pct,
            timeout=timeout,
            naked_only=naked_only,
        )

        for rr in rr_ratios:
            raw_trades   = trades_by_rr[rr]
            trades       = [_add_pnl(t) for t in raw_trades]
            metrics      = _compute_metrics(trades)
            strategy_key = f'wick_rr_{rr:.1f}'

            logger.info("  %-35s RR%s → %3d trades  WR=%.0f%%  PF=%.2f",
                        label, f'{rr:.0f}:1', metrics['total_trades'],
                        metrics['win_rate'], metrics['profit_factor'])

            # Persist backtest record
            bt = IndividualLevelBacktest(
                level_type=lt,
                level_source_timeframe=tf,
                trade_execution_timeframe=exec_timeframe,
                strategy_name=strategy_key,
                parameters={
                    'rr':            rr,
                    'tolerance_pct': tolerance_pct,
                    'timeout':       timeout,
                    'naked_only':    naked_only,
                },
                start_date=candles.iloc[0]['open_time'],
                end_date=candles.iloc[-1]['open_time'],
                status='completed',
                finished_at=datetime.now(timezone.utc),
                **metrics,
            )
            session.add(bt)
            session.flush()

            for t in trades:
                tr = IndividualLevelTrade(
                    backtest_id=bt.id,
                    level_id=t.get('level_id'),
                    entry_time=pd.Timestamp(t['entry_time']).to_pydatetime(),
                    entry_price=t['entry_price'],
                    direction=t['direction'],
                    stop_loss=t['sl'],
                    take_profit=t['tp'],
                    exit_time=pd.Timestamp(t['exit_time']).to_pydatetime(),
                    exit_price=t['exit_price'],
                    exit_reason=t['exit_reason'],
                    pnl=t['pnl'],
                    pnl_pct=t['pnl_pct'],
                    candles_held=t['candles_held'],
                    distance_to_level=(
                        abs(t['entry_price'] - t['level_price']) / t['level_price']
                        if t['level_price'] else 0.0
                    ),
                )
                session.add(tr)

            session.commit()

            results.append({
                'level_type':  lt,
                'timeframe':   tf,
                'rr':          rr,
                'backtest_id': bt.id,
                **metrics,
            })

    logger.info("Batch complete — %d backtests saved", len(results))
    return results
