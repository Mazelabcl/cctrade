"""Individual Level Backtest Service.

Tests each level type independently to determine effectiveness.
Generates trade-level data valuable for ML feature engineering.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..models import IndividualLevelBacktest, IndividualLevelTrade

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    entry_time: datetime
    entry_price: float
    direction: str  # "LONG" or "SHORT"
    stop_loss: float
    take_profit: float
    level_id: Optional[int] = None
    level_price: float = 0.0
    entry_volatility: float = 0.0
    volume_ratio: float = 0.0
    distance_to_level: float = 0.0
    zone_confluence: int = 0
    candles_held: int = 0


@dataclass
class ClosedTrade:
    entry_time: datetime
    entry_price: float
    direction: str
    stop_loss: float
    take_profit: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "TP_HIT", "SL_HIT", "TIMEOUT"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    candles_held: int = 0
    level_id: Optional[int] = None
    entry_volatility: float = 0.0
    volume_ratio: float = 0.0
    distance_to_level: float = 0.0
    zone_confluence: int = 0


# ---------------------------------------------------------------------------
# Trading strategies (exit rule factories)
# ---------------------------------------------------------------------------

class TradingStrategy(ABC):
    """Base class for exit-rule strategies."""

    @abstractmethod
    def calculate_sl_tp(self, entry_price: float, direction: str,
                        candle_data: dict, context: dict) -> tuple[float, float]:
        """Return (stop_loss, take_profit) given entry conditions."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class FixedPercentStrategy(TradingStrategy):
    """SL and TP as fixed percentages of entry price."""

    def __init__(self, sl_pct: float = 0.02, tp_pct: float = 0.04, timeout: int = 50):
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.timeout = timeout

    @property
    def name(self) -> str:
        return 'fixed_percent'

    def calculate_sl_tp(self, entry_price, direction, candle_data, context):
        if direction == 'LONG':
            sl = entry_price * (1 - self.sl_pct)
            tp = entry_price * (1 + self.tp_pct)
        else:
            sl = entry_price * (1 + self.sl_pct)
            tp = entry_price * (1 - self.tp_pct)
        return sl, tp


class ATRBasedStrategy(TradingStrategy):
    """SL and TP based on ATR (Average True Range)."""

    def __init__(self, atr_sl_mult: float = 1.5, atr_tp_mult: float = 3.0,
                 atr_period: int = 14, timeout: int = 50):
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.atr_period = atr_period
        self.timeout = timeout

    @property
    def name(self) -> str:
        return 'atr_based'

    def calculate_sl_tp(self, entry_price, direction, candle_data, context):
        atr = context.get('atr', entry_price * 0.02)  # fallback 2%
        if direction == 'LONG':
            sl = entry_price - atr * self.atr_sl_mult
            tp = entry_price + atr * self.atr_tp_mult
        else:
            sl = entry_price + atr * self.atr_sl_mult
            tp = entry_price - atr * self.atr_tp_mult
        return sl, tp


STRATEGIES = {
    'fixed_percent': FixedPercentStrategy,
    'atr_based': ATRBasedStrategy,
}


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------

def compute_atr_series(highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Return ATR array of same length as input (NaN for early values)."""
    n = len(highs)
    atr = np.full(n, np.nan)
    if n < period + 1:
        return atr

    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # Simple MA for initial ATR, then EMA
    atr[period] = tr[1:period + 1].mean()
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def detect_entry_signal(candle: pd.Series, level_price: float,
                        tolerance_pct: float = 0.02) -> Optional[str]:
    """Detect if a candle interacts with a level and confirms direction.

    LONG signal (bounce from support):
      - Candle low touches level: low <= level * (1 + tolerance)
      - Candle closes above level: close > level
      → Level acted as support

    SHORT signal (rejection from resistance):
      - Candle high touches level: high >= level * (1 - tolerance)
      - Candle closes below level: close < level
      → Level acted as resistance

    Returns "LONG", "SHORT", or None.
    """
    low = candle['low']
    high = candle['high']
    close = candle['close']

    tol = level_price * tolerance_pct

    # LONG — price dipped to level (support) and bounced
    if low <= level_price + tol and close > level_price:
        return 'LONG'

    # SHORT — price rose to level (resistance) and rejected
    if high >= level_price - tol and close < level_price:
        return 'SHORT'

    return None


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

def check_exit(pos: OpenPosition, candle: pd.Series) -> Optional[tuple[str, float]]:
    """Check whether an open position should be closed on this candle.

    Checks SL/TP intra-bar using high/low, returns (exit_reason, exit_price).
    For SL/TP hit on the same bar, SL takes priority (conservative).
    """
    high = candle['high']
    low = candle['low']

    if pos.direction == 'LONG':
        # SL check first (conservative)
        if low <= pos.stop_loss:
            return 'SL_HIT', pos.stop_loss
        if high >= pos.take_profit:
            return 'TP_HIT', pos.take_profit
    else:  # SHORT
        if high >= pos.stop_loss:
            return 'SL_HIT', pos.stop_loss
        if low <= pos.take_profit:
            return 'TP_HIT', pos.take_profit

    return None


def close_position(pos: OpenPosition, exit_time: datetime,
                   exit_price: float, exit_reason: str) -> ClosedTrade:
    """Convert an open position into a closed trade with PnL."""
    if pos.direction == 'LONG':
        pnl = exit_price - pos.entry_price
    else:
        pnl = pos.entry_price - exit_price

    pnl_pct = pnl / pos.entry_price if pos.entry_price else 0.0

    return ClosedTrade(
        entry_time=pos.entry_time,
        entry_price=pos.entry_price,
        direction=pos.direction,
        stop_loss=pos.stop_loss,
        take_profit=pos.take_profit,
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl=pnl,
        pnl_pct=pnl_pct,
        candles_held=pos.candles_held,
        level_id=pos.level_id,
        entry_volatility=pos.entry_volatility,
        volume_ratio=pos.volume_ratio,
        distance_to_level=pos.distance_to_level,
        zone_confluence=pos.zone_confluence,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(trades: list[ClosedTrade]) -> dict:
    """Compute aggregate performance metrics from a list of closed trades."""
    if not trades:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0.0, 'profit_factor': 0.0, 'sharpe_ratio': 0.0,
            'max_drawdown': 0.0, 'total_pnl': 0.0, 'avg_win': 0.0,
            'avg_loss': 0.0, 'avg_trade_duration': 0.0,
        }

    pnls = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_win = sum(wins) if wins else 0.0
    total_loss = abs(sum(losses)) if losses else 0.0

    # Sharpe: mean / std of per-trade returns
    arr = np.array(pnls)
    sharpe = float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0

    # Max drawdown on cumulative equity curve
    cum = np.cumsum(arr)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    max_dd = float(drawdowns.max()) if len(drawdowns) else 0.0

    return {
        'total_trades': len(trades),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0.0,
        'profit_factor': total_win / total_loss if total_loss > 0 else float('inf') if total_win > 0 else 0.0,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd * 100,  # as %
        'total_pnl': float(sum(t.pnl for t in trades)),
        'avg_win': float(np.mean([t.pnl for t in trades if t.pnl > 0])) if wins else 0.0,
        'avg_loss': float(np.mean([t.pnl for t in trades if t.pnl <= 0])) if losses else 0.0,
        'avg_trade_duration': float(np.mean([t.candles_held for t in trades])),
    }


# ---------------------------------------------------------------------------
# Level filtering
# ---------------------------------------------------------------------------

def filter_levels_for_backtest(levels_df: pd.DataFrame,
                               level_type: str,
                               source_timeframe: str,
                               naked_only: bool = True) -> pd.DataFrame:
    """Filter levels DataFrame to match the requested type and timeframe.

    Maps user-facing level_type + source_timeframe to the actual CSV values.
    """
    df = levels_df.copy()

    # Normalize created_at: use 'period' for VP levels where created_at is NaT
    if 'period' in df.columns:
        mask = df['created_at'].isna() & df['period'].notna()
        df.loc[mask, 'created_at'] = pd.to_datetime(df.loc[mask, 'period'])

    # Drop levels without a valid created_at
    df = df.dropna(subset=['created_at'])

    # Map source_timeframe codes to CSV timeframe values
    tf_map = {'1h': 'hourly', '4h': 'daily', '1d': 'daily', '1w': 'weekly', '1M': 'monthly'}
    csv_timeframe = tf_map.get(source_timeframe, source_timeframe)

    # Build filter based on level_type
    lt = level_type.upper()

    if lt == 'HTF':
        df = df[df['level_type'].str.contains('HTF', case=False)]
        df = df[df['timeframe'] == csv_timeframe]
    elif lt == 'FRACTAL_HIGH':
        df = df[df['level_type'].str.contains('Fractal_High', case=False)]
        df = df[df['timeframe'] == csv_timeframe]
    elif lt == 'FRACTAL_LOW':
        df = df[df['level_type'].str.contains('Fractal_Low', case=False)]
        df = df[df['timeframe'] == csv_timeframe]
    elif lt.startswith('FIB'):
        # e.g. "Fib_0.618" → match level_type containing "Fib_0.639" or "Fib_0.618"
        # Handle legacy naming: 0.639 in CSV = 0.618 conceptually
        ratio = level_type.split('_')[1] if '_' in level_type else '0.618'
        if ratio == '0.618':
            # CSV uses 0.639 for golden pocket
            df = df[df['level_type'].str.contains('Fib_0.639|Fib_0.618', case=False, regex=True)]
        else:
            df = df[df['level_type'].str.contains(f'Fib_{ratio}', case=False)]
        df = df[df['timeframe'] == csv_timeframe]
    elif lt.startswith('VP'):
        vp_type = level_type.lower()  # e.g. "VP_POC" → "vp_poc"
        # CSV level_types: VP_vah, VP_poc, VP_val
        csv_lt = vp_type.replace('vp_', 'VP_').lower()
        df = df[df['level_type'].str.lower() == csv_lt]
        df = df[df['timeframe'] == csv_timeframe]
    else:
        # Fallback: exact match
        df = df[df['level_type'] == level_type]
        df = df[df['timeframe'] == csv_timeframe]

    df = df.sort_values('created_at').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_individual_level_backtest(
    candles_df: pd.DataFrame,
    levels_df: pd.DataFrame,
    level_type: str,
    source_timeframe: str,
    exec_timeframe: str,
    strategy: TradingStrategy,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    tolerance_pct: float = 0.02,
    timeout: int = 50,
    naked_only: bool = True,
    db: Optional[Session] = None,
) -> tuple[Optional[IndividualLevelBacktest], list[ClosedTrade]]:
    """Run a backtest for a single level type.

    Args:
        candles_df: DataFrame with OHLCV + open_time (sorted by time)
        levels_df: DataFrame with all levels (multi-type, multi-tf)
        level_type: e.g. "HTF", "Fractal_High", "Fib_0.618", "VP_POC"
        source_timeframe: Timeframe the levels come from ("1d", "1w", "1M")
        exec_timeframe: Timeframe of candles we trade on ("1h", "4h", "1d")
        strategy: Exit-rule strategy instance
        start_date: Filter candles from this date
        end_date: Filter candles until this date
        tolerance_pct: % tolerance for level touches (default 2%)
        timeout: Max candles to hold a position
        naked_only: Only trade on levels not yet touched
        db: SQLAlchemy session (optional, to persist results)

    Returns:
        (backtest_record, list_of_trades)
    """
    label = f"{level_type}_{source_timeframe}"
    logger.info("Starting individual level backtest: %s on %s candles (strategy=%s)",
                label, exec_timeframe, strategy.name)

    # ---- Prepare candles ----
    candles = candles_df.copy()
    candles['open_time'] = pd.to_datetime(candles['open_time'])
    candles = candles.sort_values('open_time').reset_index(drop=True)

    if start_date:
        candles = candles[candles['open_time'] >= pd.Timestamp(start_date)]
    if end_date:
        candles = candles[candles['open_time'] <= pd.Timestamp(end_date)]
    candles = candles.reset_index(drop=True)

    if candles.empty:
        logger.warning("No candles in date range for %s", label)
        return None, []

    # ---- Prepare levels ----
    levels = levels_df.copy()
    levels['created_at'] = pd.to_datetime(levels['created_at'], errors='coerce')
    if 'period' in levels.columns:
        mask = levels['created_at'].isna() & levels['period'].notna()
        levels.loc[mask, 'created_at'] = pd.to_datetime(levels.loc[mask, 'period'], errors='coerce')

    filtered = filter_levels_for_backtest(levels, level_type, source_timeframe, naked_only)
    if filtered.empty:
        logger.warning("No levels found for %s", label)
        return None, []

    logger.info("  %d levels of type %s, %d candles to process",
                len(filtered), label, len(candles))

    # ---- Pre-extract numpy arrays for speed ----
    c_open_times = candles['open_time'].values                     # datetime64
    c_highs = candles['high'].values.astype(np.float64)
    c_lows = candles['low'].values.astype(np.float64)
    c_closes = candles['close'].values.astype(np.float64)
    c_volumes = candles['volume'].values.astype(np.float64)

    atr_series = compute_atr_series(c_highs, c_lows, c_closes, period=14)
    vol_ma = pd.Series(c_volumes).rolling(20, min_periods=1).mean().values

    # ---- Pre-extract level arrays (sorted by created_at) ----
    lev_prices = filtered['price_level'].values.astype(np.float64)
    lev_created = filtered['created_at'].values                     # datetime64
    lev_ids = filtered['id'].values if 'id' in filtered.columns else np.zeros(len(filtered), dtype=int)
    n_levels = len(lev_prices)

    # ---- Track which levels have been touched (index → bool) ----
    level_touched = np.zeros(n_levels, dtype=bool)

    # ---- Simulation loop (numpy-accelerated) ----
    closed_trades: list[ClosedTrade] = []
    position: Optional[OpenPosition] = None
    n_candles = len(candles)

    for i in range(n_candles):
        ct = c_open_times[i]
        hi = c_highs[i]
        lo = c_lows[i]
        cl = c_closes[i]

        # ---- If we have an open position, check for exit ----
        if position is not None:
            position.candles_held += 1

            # Inline exit check (avoid function call overhead)
            exited = False
            if position.direction == 'LONG':
                if lo <= position.stop_loss:
                    trade = close_position(position, pd.Timestamp(ct), position.stop_loss, 'SL_HIT')
                    closed_trades.append(trade)
                    exited = True
                elif hi >= position.take_profit:
                    trade = close_position(position, pd.Timestamp(ct), position.take_profit, 'TP_HIT')
                    closed_trades.append(trade)
                    exited = True
            else:  # SHORT
                if hi >= position.stop_loss:
                    trade = close_position(position, pd.Timestamp(ct), position.stop_loss, 'SL_HIT')
                    closed_trades.append(trade)
                    exited = True
                elif lo <= position.take_profit:
                    trade = close_position(position, pd.Timestamp(ct), position.take_profit, 'TP_HIT')
                    closed_trades.append(trade)
                    exited = True

            if exited:
                position = None
            elif position.candles_held >= timeout:
                trade = close_position(position, pd.Timestamp(ct), cl, 'TIMEOUT')
                closed_trades.append(trade)
                position = None

            # Mark levels touched by this candle (for naked tracking)
            if naked_only:
                for j in range(n_levels):
                    if level_touched[j]:
                        continue
                    if lev_created[j] >= ct:
                        break  # levels are sorted by time
                    lp = lev_prices[j]
                    tol = lp * tolerance_pct
                    if lo <= lp + tol and hi >= lp - tol:
                        level_touched[j] = True

            continue

        # ---- No open position: look for entry signals ----
        # Use numpy to find candidate levels: created before this candle, within 10%, not touched
        # Vectorized filtering
        time_mask = lev_created < ct
        dist = np.abs(lev_prices - cl) / cl
        dist_mask = dist < 0.10

        if naked_only:
            mask = time_mask & dist_mask & ~level_touched
        else:
            mask = time_mask & dist_mask

        candidate_idx = np.where(mask)[0]
        if len(candidate_idx) == 0:
            continue

        # Sort candidates by distance (closest first)
        candidate_dists = dist[candidate_idx]
        sort_order = np.argsort(candidate_dists)
        candidate_idx = candidate_idx[sort_order]

        # Check each candidate for entry signal
        entered = False
        for j in candidate_idx:
            lp = lev_prices[j]
            tol = lp * tolerance_pct

            # LONG: price dipped to support and bounced
            if lo <= lp + tol and cl > lp:
                signal = 'LONG'
            # SHORT: price rose to resistance and rejected
            elif hi >= lp - tol and cl < lp:
                signal = 'SHORT'
            else:
                continue

            # Mark as touched
            level_touched[j] = True

            # Calculate SL/TP
            atr_val = float(atr_series[i]) if not np.isnan(atr_series[i]) else cl * 0.02
            sl, tp = strategy.calculate_sl_tp(cl, signal, {}, {'atr': atr_val})

            # Confluence: count nearby levels
            zone_tol = lp * tolerance_pct
            confluence = int(np.sum(
                mask & (lev_prices >= lp - zone_tol) & (lev_prices <= lp + zone_tol)
            ))

            v_ratio = float(c_volumes[i] / vol_ma[i]) if vol_ma[i] > 0 else 1.0

            position = OpenPosition(
                entry_time=pd.Timestamp(ct),
                entry_price=cl,
                direction=signal,
                stop_loss=sl,
                take_profit=tp,
                level_id=int(lev_ids[j]) if lev_ids[j] else None,
                level_price=lp,
                entry_volatility=atr_val,
                volume_ratio=v_ratio,
                distance_to_level=float(dist[j]),
                zone_confluence=confluence,
            )
            entered = True
            break

        # Update touch tracking for untouched levels hit by this candle
        if not entered and naked_only:
            for j in range(n_levels):
                if level_touched[j]:
                    continue
                if lev_created[j] >= ct:
                    break
                lp = lev_prices[j]
                tol = lp * tolerance_pct
                if lo <= lp + tol and hi >= lp - tol:
                    level_touched[j] = True

    # Close any remaining position at last candle close
    if position is not None:
        trade = close_position(position, pd.Timestamp(c_open_times[-1]),
                               float(c_closes[-1]), 'TIMEOUT')
        closed_trades.append(trade)

    # ---- Calculate metrics ----
    metrics = calculate_metrics(closed_trades)

    logger.info("  Backtest complete: %d trades, %.1f%% win rate, PF=%.2f, PnL=$%.2f",
                metrics['total_trades'], metrics['win_rate'],
                metrics['profit_factor'], metrics['total_pnl'])

    # ---- Build parameters dict ----
    params = {'tolerance_pct': tolerance_pct, 'timeout': timeout, 'naked_only': naked_only}
    if isinstance(strategy, FixedPercentStrategy):
        params.update({'sl_pct': strategy.sl_pct, 'tp_pct': strategy.tp_pct})
    elif isinstance(strategy, ATRBasedStrategy):
        params.update({
            'atr_sl_mult': strategy.atr_sl_mult,
            'atr_tp_mult': strategy.atr_tp_mult,
            'atr_period': strategy.atr_period,
        })

    # ---- Persist to DB if session provided ----
    backtest_record = None
    if db is not None:
        backtest_record = IndividualLevelBacktest(
            level_type=level_type,
            level_source_timeframe=source_timeframe,
            trade_execution_timeframe=exec_timeframe,
            strategy_name=strategy.name,
            parameters=params,
            start_date=candles.iloc[0]['open_time'],
            end_date=candles.iloc[-1]['open_time'],
            status='completed',
            created_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            **metrics,
        )
        db.add(backtest_record)
        db.flush()  # Get ID

        for t in closed_trades:
            trade_rec = IndividualLevelTrade(
                backtest_id=backtest_record.id,
                level_id=t.level_id,
                entry_time=t.entry_time,
                entry_price=t.entry_price,
                direction=t.direction,
                stop_loss=t.stop_loss,
                take_profit=t.take_profit,
                exit_time=t.exit_time,
                exit_price=t.exit_price,
                exit_reason=t.exit_reason,
                pnl=t.pnl,
                pnl_pct=t.pnl_pct,
                candles_held=t.candles_held,
                entry_volatility=t.entry_volatility,
                volume_ratio=t.volume_ratio,
                distance_to_level=t.distance_to_level,
                zone_confluence=t.zone_confluence,
            )
            db.add(trade_rec)

        db.commit()
        logger.info("  Persisted backtest #%d with %d trades", backtest_record.id, len(closed_trades))

    return backtest_record, closed_trades


# ---------------------------------------------------------------------------
# Batch runner — test all level types
# ---------------------------------------------------------------------------

# All level types available in the CSV data
ALL_LEVEL_TYPES = [
    'HTF', 'Fractal_High', 'Fractal_Low',
    'Fib_0.50', 'Fib_0.618', 'Fib_0.75', 'Fib_0.786',
    'VP_POC', 'VP_VAH', 'VP_VAL',
]

ALL_SOURCE_TIMEFRAMES = ['1d', '1w', '1M']


def run_batch_backtests(
    candles_df: pd.DataFrame,
    levels_df: pd.DataFrame,
    exec_timeframe: str = '1h',
    strategy: Optional[TradingStrategy] = None,
    level_types: Optional[list[str]] = None,
    source_timeframes: Optional[list[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    tolerance_pct: float = 0.02,
    timeout: int = 50,
    naked_only: bool = True,
    db: Optional[Session] = None,
) -> list[dict]:
    """Run backtests for multiple level type / timeframe combinations.

    Returns a list of summary dicts for each backtest.
    """
    if strategy is None:
        strategy = FixedPercentStrategy(sl_pct=0.02, tp_pct=0.04, timeout=timeout)

    if level_types is None:
        level_types = ALL_LEVEL_TYPES

    if source_timeframes is None:
        source_timeframes = ALL_SOURCE_TIMEFRAMES

    results = []
    total = len(level_types) * len(source_timeframes)
    count = 0

    for lt in level_types:
        for stf in source_timeframes:
            count += 1
            logger.info("=== Batch %d/%d: %s on %s ===", count, total, lt, stf)

            try:
                record, trades = run_individual_level_backtest(
                    candles_df=candles_df,
                    levels_df=levels_df,
                    level_type=lt,
                    source_timeframe=stf,
                    exec_timeframe=exec_timeframe,
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    tolerance_pct=tolerance_pct,
                    timeout=timeout,
                    naked_only=naked_only,
                    db=db,
                )

                metrics = calculate_metrics(trades)
                results.append({
                    'level_type': lt,
                    'source_timeframe': stf,
                    'backtest_id': record.id if record else None,
                    **metrics,
                })

            except Exception as exc:
                logger.error("Backtest failed for %s/%s: %s", lt, stf, exc)
                results.append({
                    'level_type': lt,
                    'source_timeframe': stf,
                    'backtest_id': None,
                    'error': str(exc),
                    'total_trades': 0,
                })

    return results


# ---------------------------------------------------------------------------
# Data loaders (CSV)
# ---------------------------------------------------------------------------

def load_candles_csv(*paths: str) -> pd.DataFrame:
    """Load and concatenate candle CSVs."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df['open_time'] = pd.to_datetime(df['open_time'])
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('open_time').drop_duplicates(subset=['open_time']).reset_index(drop=True)
    return combined


def load_levels_csv(*paths: str) -> pd.DataFrame:
    """Load and concatenate levels CSVs."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        if 'period' in df.columns:
            df['period'] = pd.to_datetime(df['period'], errors='coerce')
            mask = df['created_at'].isna() & df['period'].notna()
            df.loc[mask, 'created_at'] = df.loc[mask, 'period']
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('created_at').reset_index(drop=True)
    return combined
