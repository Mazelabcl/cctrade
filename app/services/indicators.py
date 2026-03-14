"""Indicator computation service — fractal detection, HTF levels, Fibonacci, Volume Profile.

Pure Python + SQLAlchemy, no Flask dependency. Algorithms preserved from legacy/indicators.py.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..models import Candle, Level, PipelineRun

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Level deduplication
# ---------------------------------------------------------------------------

def _level_exists(db: Session, price: float, level_type: str, timeframe: str,
                  source: str, tolerance: float = 0.0001) -> bool:
    """Check if a similar level already exists in DB."""
    low = price * (1 - tolerance)
    high = price * (1 + tolerance)
    return db.query(Level).filter(
        Level.price_level.between(low, high),
        Level.level_type == level_type,
        Level.timeframe == timeframe,
        Level.source == source,
    ).first() is not None


def _add_levels(db: Session, levels: list[dict], skip_duplicates: bool = True) -> int:
    """Add levels to DB, optionally skipping duplicates. Returns count added."""
    added = 0
    for lev in levels:
        if skip_duplicates and _level_exists(
            db, lev['price_level'], lev['level_type'],
            lev['timeframe'], lev['source']
        ):
            continue
        db.add(Level(
            price_level=lev['price_level'],
            level_type=lev['level_type'],
            timeframe=lev['timeframe'],
            source=lev['source'],
            created_at=lev['created_at'],
            metadata_json=lev.get('metadata'),
        ))
        added += 1
    return added


# ---------------------------------------------------------------------------
# Fractal detection
# ---------------------------------------------------------------------------

def detect_fractals_df(df: pd.DataFrame) -> pd.DataFrame:
    """Detect 5-candle swing highs/lows on a DataFrame.

    A bearish fractal (swing high): 2 lower highs on each side.
    A bullish fractal (swing low): 2 higher lows on each side.
    """
    df = df.copy()
    df['bearish_fractal'] = False
    df['bullish_fractal'] = False

    if len(df) < 5:
        return df

    highs = df['high'].values
    lows = df['low'].values

    for i in range(2, len(df) - 2):
        if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2]
                and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]):
            df.iloc[i, df.columns.get_loc('bearish_fractal')] = True

        if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2]
                and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):
            df.iloc[i, df.columns.get_loc('bullish_fractal')] = True

    return df


def run_fractal_detection(db: Session, timeframe: str = '1h',
                          symbol: str = 'BTCUSDT') -> int:
    """Detect fractals on candles and persist the flags.

    Returns the number of candles updated.
    """
    candles = (
        db.query(Candle)
        .filter_by(symbol=symbol, timeframe=timeframe)
        .order_by(Candle.open_time)
        .all()
    )
    if len(candles) < 5:
        return 0

    records = [{'high': c.high, 'low': c.low} for c in candles]
    df = pd.DataFrame(records)
    df = detect_fractals_df(df)

    updated = 0
    for i, candle in enumerate(candles):
        b_frac = bool(df.iloc[i]['bearish_fractal'])
        u_frac = bool(df.iloc[i]['bullish_fractal'])
        if candle.bearish_fractal != b_frac or candle.bullish_fractal != u_frac:
            candle.bearish_fractal = b_frac
            candle.bullish_fractal = u_frac
            updated += 1

    db.commit()
    logger.info("Fractal detection [%s]: %d candles updated", timeframe, updated)
    return updated


# ---------------------------------------------------------------------------
# HTF levels
# ---------------------------------------------------------------------------

def calculate_htf_levels(df: pd.DataFrame, timeframe: str) -> list[dict]:
    """Return HTF levels where candle direction changes.

    A level is created at the open of candle_i+1 when the direction of
    candle_i differs from candle_i+1.
    """
    levels = []
    for i in range(len(df) - 1):
        row_i = df.iloc[i]
        row_next = df.iloc[i + 1]
        dir_i = 'up' if row_i['close'] > row_i['open'] else 'down'
        dir_next = 'up' if row_next['close'] > row_next['open'] else 'down'

        if dir_i != dir_next:
            levels.append({
                'price_level': float(row_next['open']),
                'level_type': 'HTF_level',
                'timeframe': timeframe,
                'source': 'htf',
                'created_at': row_next['open_time'],
            })
    return levels


# ---------------------------------------------------------------------------
# Fibonacci levels
# ---------------------------------------------------------------------------

def calculate_fibonacci_levels(df: pd.DataFrame, timeframe: str) -> list[dict]:
    """Calculate Fibonacci retracement levels between fractals.

    Uses 0.50, 0.618 (golden pocket at 0.639), 0.75, 0.786 ratios.
    Each fractal anchor pulls to future opposite fractals while the anchor
    hasn't been invalidated.
    """
    df = detect_fractals_df(df)
    fib_levels: list[dict] = []

    ratios = [('0.50', 0.50), ('0.618', 0.639), ('0.75', 0.75), ('0.786', 0.786)]

    high_fractals = df[df['bearish_fractal']].copy()
    low_fractals = df[df['bullish_fractal']].copy()

    if high_fractals.empty or low_fractals.empty:
        return fib_levels

    # Low anchors pulling to highs
    for anchor_idx, anchor_row in low_fractals.iterrows():
        anchor_low = anchor_row['low']
        local_high = None

        future_highs = high_fractals[high_fractals.index > anchor_idx]
        for high_idx, high_row in future_highs.iterrows():
            between = df[(df.index > anchor_idx) & (df.index < high_idx)]
            if len(between) and (between['low'] < anchor_low).any():
                break

            if local_high is None or high_row['high'] > local_high:
                local_high = high_row['high']
                range_diff = high_row['high'] - anchor_low

                for label, ratio in ratios:
                    fib_levels.append({
                        'price_level': float(high_row['high'] - ratio * range_diff),
                        'level_type': f'Fib_{label}',
                        'timeframe': timeframe,
                        'source': 'fibonacci',
                        'created_at': high_row['open_time'],
                        'metadata': {
                            'anchor_time': str(anchor_row['open_time']),
                            'direction': 'low_to_high',
                        },
                    })

    # High anchors pulling to lows
    for anchor_idx, anchor_row in high_fractals.iterrows():
        anchor_high = anchor_row['high']
        local_low = None

        future_lows = low_fractals[low_fractals.index > anchor_idx]
        for low_idx, low_row in future_lows.iterrows():
            between = df[(df.index > anchor_idx) & (df.index < low_idx)]
            if len(between) and (between['high'] > anchor_high).any():
                break

            if local_low is None or low_row['low'] < local_low:
                local_low = low_row['low']
                range_diff = anchor_high - low_row['low']

                for label, ratio in ratios:
                    fib_levels.append({
                        'price_level': float(low_row['low'] + ratio * range_diff),
                        'level_type': f'Fib_{label}',
                        'timeframe': timeframe,
                        'source': 'fibonacci',
                        'created_at': low_row['open_time'],
                        'metadata': {
                            'anchor_time': str(anchor_row['open_time']),
                            'direction': 'high_to_low',
                        },
                    })

    return fib_levels


# ---------------------------------------------------------------------------
# Volume Profile
# ---------------------------------------------------------------------------

def calculate_volume_profile(df_1min: pd.DataFrame, bin_size: int = 10) -> dict:
    """Calculate POC, VAH, VAL from 1-minute candle data.

    Uses price bins of `bin_size` dollars for efficiency and precision.
    """
    if df_1min.empty:
        return {}

    volume_by_price: dict[int, float] = {}

    for _, candle in df_1min.iterrows():
        low_bin = int(candle['low'] // bin_size) * bin_size
        high_bin = int(candle['high'] // bin_size) * bin_size
        price_range = range(low_bin, high_bin + bin_size, bin_size)
        size = len(price_range)
        if size == 0:
            continue
        vol_per_level = candle['volume'] / size
        for price in price_range:
            volume_by_price[price] = volume_by_price.get(price, 0) + vol_per_level

    if not volume_by_price:
        return {}

    poc = max(volume_by_price, key=volume_by_price.get)
    total_volume = sum(volume_by_price.values())
    target = total_volume * 0.7
    current = volume_by_price[poc]

    prices = sorted(volume_by_price)
    poc_idx = prices.index(poc)
    above = poc_idx + 1
    below = poc_idx - 1

    while current < target and (above < len(prices) or below >= 0):
        above_vol = volume_by_price[prices[above]] if above < len(prices) else 0
        below_vol = volume_by_price[prices[below]] if below >= 0 else 0
        if above_vol >= below_vol and above < len(prices):
            current += above_vol
            above += 1
        elif below >= 0:
            current += below_vol
            below -= 1
        else:
            break

    vah = prices[min(above, len(prices) - 1)]
    val = prices[max(below, 0)]

    return {'poc': poc, 'vah': vah, 'val': val}


def calculate_volume_profile_levels(df_1min: pd.DataFrame, timeframe: str,
                                    period_group: str, bin_size: int = 10) -> list[dict]:
    """Calculate VP levels (POC/VAH/VAL) grouped by period.

    Args:
        df_1min: 1-minute candle DataFrame with open_time, high, low, volume
        timeframe: Label for the level timeframe (e.g., 'daily', 'weekly', 'monthly')
        period_group: How to group: 'D' (daily), 'W' (weekly), 'ME' (monthly)
        bin_size: Price bin size in dollars
    """
    if df_1min.empty:
        return []

    df = df_1min.copy()
    df['period'] = df['open_time'].dt.to_period(period_group[0])

    levels = []
    for period, group in df.groupby('period'):
        vp = calculate_volume_profile(group, bin_size=bin_size)
        if not vp:
            continue

        period_start = group['open_time'].iloc[0]
        for vp_type in ['poc', 'vah', 'val']:
            levels.append({
                'price_level': float(vp[vp_type]),
                'level_type': f'VP_{vp_type.upper()}',
                'timeframe': timeframe,
                'source': 'volume_profile',
                'created_at': period_start,
                'metadata': {
                    'period': str(period),
                    'vp_type': vp_type,
                },
            })

    return levels


# ---------------------------------------------------------------------------
# Orchestrator — run all indicators for a timeframe
# ---------------------------------------------------------------------------

def run_indicators(db: Session, timeframe: str = '1h',
                   symbol: str = 'BTCUSDT',
                   compute_htf: bool = True,
                   compute_fib: bool = True,
                   compute_vp: bool = False,
                   skip_duplicates: bool = True) -> dict:
    """Run the full indicator pipeline and persist results.

    Returns a summary dict with counts of what was created/updated.
    """
    run = PipelineRun(
        pipeline_type='indicators',
        status='running',
        started_at=datetime.now(timezone.utc),
        metadata_json={'timeframe': timeframe},
    )
    db.add(run)
    db.commit()

    try:
        # 1. Fractal detection
        fractal_count = run_fractal_detection(db, timeframe, symbol)

        # 2. Load candles into a DataFrame for level calculation
        candles = (
            db.query(Candle)
            .filter_by(symbol=symbol, timeframe=timeframe)
            .order_by(Candle.open_time)
            .all()
        )
        df = pd.DataFrame([{
            'open_time': c.open_time,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume,
        } for c in candles])

        new_levels = 0
        tf_label = _tf_label(timeframe)

        # 3. HTF levels
        if compute_htf and not df.empty:
            htf = calculate_htf_levels(df, tf_label)
            new_levels += _add_levels(db, htf, skip_duplicates)

        # 4. Fibonacci levels
        if compute_fib and not df.empty:
            fibs = calculate_fibonacci_levels(df, tf_label)
            new_levels += _add_levels(db, fibs, skip_duplicates)

        db.commit()

        run.status = 'completed'
        run.finished_at = datetime.now(timezone.utc)
        run.rows_processed = fractal_count + new_levels
        run.metadata_json = {
            'timeframe': timeframe,
            'fractals_updated': fractal_count,
            'new_levels': new_levels,
        }
        db.commit()

        logger.info("Indicators [%s]: %d fractals, %d new levels", timeframe, fractal_count, new_levels)
        return {
            'timeframe': timeframe,
            'fractals_updated': fractal_count,
            'new_levels': new_levels,
        }

    except Exception as exc:
        run.status = 'failed'
        run.finished_at = datetime.now(timezone.utc)
        run.error_message = str(exc)
        db.commit()
        raise


def run_indicators_multi(db: Session, symbol: str = 'BTCUSDT',
                         htf_timeframes: list[str] = None,
                         fractal_timeframes: list[str] = None,
                         fib_timeframes: list[str] = None,
                         vp_timeframes: list[str] = None) -> dict:
    """Run indicators across multiple timeframes per level type.

    Returns summary dict with per-type per-TF counts.
    """
    summary = {'fractals': {}, 'htf': {}, 'fibonacci': {}, 'vp': {}}

    # Collect all unique timeframes that need fractal detection
    all_tfs = set()
    if htf_timeframes:
        all_tfs.update(htf_timeframes)
    if fractal_timeframes:
        all_tfs.update(fractal_timeframes)
    if fib_timeframes:
        all_tfs.update(fib_timeframes)

    # Run fractal detection on all needed timeframes first
    for tf in sorted(all_tfs):
        count = run_fractal_detection(db, tf, symbol)
        summary['fractals'][tf] = count
        logger.info("Fractals [%s]: %d updated", tf, count)

    # Fractal levels (create Level records from detected fractals)
    for tf in (fractal_timeframes or []):
        candles = _load_candle_df(db, symbol, tf)
        if candles.empty:
            continue
        detected = detect_fractals_df(candles)
        tf_label = _tf_label(tf)
        fractal_levels = []
        for idx, row in detected.iterrows():
            if row['bearish_fractal']:
                fractal_levels.append({
                    'price_level': float(row['high']),
                    'level_type': 'Fractal_resistance',
                    'timeframe': tf_label,
                    'source': 'fractal',
                    'created_at': row['open_time'],
                })
            if row['bullish_fractal']:
                fractal_levels.append({
                    'price_level': float(row['low']),
                    'level_type': 'Fractal_support',
                    'timeframe': tf_label,
                    'source': 'fractal',
                    'created_at': row['open_time'],
                })
        added = _add_levels(db, fractal_levels, skip_duplicates=True)
        summary['fractals'][tf] = {'flags': summary['fractals'].get(tf, 0), 'levels': added}
        logger.info("Fractal levels [%s]: %d new levels", tf, added)

    # HTF levels
    for tf in (htf_timeframes or []):
        candles = _load_candle_df(db, symbol, tf)
        if candles.empty:
            continue
        htf = calculate_htf_levels(candles, _tf_label(tf))
        added = _add_levels(db, htf, skip_duplicates=True)
        summary['htf'][tf] = added
        logger.info("HTF [%s]: %d new levels", tf, added)

    # Fibonacci levels
    for tf in (fib_timeframes or []):
        candles = _load_candle_df(db, symbol, tf)
        if candles.empty:
            continue
        fibs = calculate_fibonacci_levels(candles, _tf_label(tf))
        added = _add_levels(db, fibs, skip_duplicates=True)
        summary['fibonacci'][tf] = added
        logger.info("Fibonacci [%s]: %d new levels", tf, added)

    db.commit()

    # VP is handled separately (needs 1-min data from Binance)
    # summary['vp'] populated by caller

    return summary


def _load_candle_df(db: Session, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load candles for a symbol/timeframe into a DataFrame."""
    candles = (
        db.query(Candle)
        .filter_by(symbol=symbol, timeframe=timeframe)
        .order_by(Candle.open_time)
        .all()
    )
    if not candles:
        return pd.DataFrame()
    return pd.DataFrame([{
        'open_time': c.open_time,
        'open': c.open,
        'high': c.high,
        'low': c.low,
        'close': c.close,
        'volume': c.volume,
    } for c in candles])


def _tf_label(timeframe: str) -> str:
    """Map candle timeframe codes to level timeframe labels."""
    mapping = {
        '1h': 'hourly',
        '4h': '4hourly',
        '12h': 'daily',
        '1d': 'daily',
        '1w': 'weekly',
        '1M': 'monthly',
    }
    return mapping.get(timeframe, timeframe)
