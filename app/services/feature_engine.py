"""Unified feature engineering pipeline.

Consolidates legacy create_ml_features.py, candle_ratios.py, volume_ratios.py,
time_blocks.py, and fractal_timing.py into one service.
"""
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..models import Candle, Level, Feature, PipelineRun

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candle ratios (from legacy/candle_ratios.py)
# ---------------------------------------------------------------------------

def _candle_ratios(o, h, l, c):
    total = h - l
    if total == 0:
        return 0.0, 0.0, 1.0, 0.5
    body_high = max(o, c)
    body_low = min(o, c)
    return (
        (h - body_high) / total,   # upper_wick_ratio
        (body_low - l) / total,    # lower_wick_ratio
        (body_high - body_low) / total,  # body_total_ratio
        (c - l) / total,           # body_position_ratio
    )


# ---------------------------------------------------------------------------
# Volume ratios (from legacy/volume_ratios.py)
# ---------------------------------------------------------------------------

def _volume_ratios(current_vol, vol_history):
    short_ratio = 0.0
    long_ratio = 0.0
    if len(vol_history) >= 6:
        ma_short = vol_history[-6:].mean()
        if ma_short > 0:
            short_ratio = current_vol / ma_short
    if len(vol_history) >= 168:
        ma_long = vol_history[-168:].mean()
        if ma_long > 0:
            long_ratio = current_vol / ma_long
    return short_ratio, long_ratio


# ---------------------------------------------------------------------------
# UTC time blocks (from legacy/time_blocks.py)
# ---------------------------------------------------------------------------

def _utc_block(dt):
    if hasattr(dt, 'hour'):
        return dt.hour // 4
    return 0


# ---------------------------------------------------------------------------
# Momentum / volatility indicators
# ---------------------------------------------------------------------------

def _compute_rsi(closes, period=14):
    """Compute RSI from a numpy array of close prices ending at current bar."""
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_macd(closes, fast=12, slow=26, signal=9):
    """Compute MACD line, signal line, and histogram."""
    if len(closes) < slow + signal:
        return None, None, None
    s = pd.Series(closes)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - macd_signal
    return float(macd_line.iloc[-1]), float(macd_signal.iloc[-1]), float(hist.iloc[-1])


def _compute_bollinger_width(closes, period=20):
    """Compute Bollinger Band width (upper - lower) / middle."""
    if len(closes) < period:
        return None
    window = closes[-period:]
    mid = window.mean()
    if mid == 0:
        return None
    std = window.std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return float((upper - lower) / mid)


def _compute_atr(highs, lows, closes, period=14):
    """Compute Average True Range from numpy arrays."""
    if len(highs) < period + 1:
        return None
    trs = []
    for j in range(len(highs) - period, len(highs)):
        tr = max(
            highs[j] - lows[j],
            abs(highs[j] - closes[j - 1]),
            abs(lows[j] - closes[j - 1]),
        )
        trs.append(tr)
    return float(np.mean(trs))


def _compute_momentum(closes, period=12):
    """Price momentum: (close - close[n periods ago]) / close[n periods ago]."""
    if len(closes) < period + 1:
        return None
    prev = closes[-(period + 1)]
    if prev == 0:
        return None
    return float((closes[-1] - prev) / prev)


# ---------------------------------------------------------------------------
# Zone detection helpers
# ---------------------------------------------------------------------------

def _find_nearest_levels(price, levels_df, zone_width=0.015):
    """Find nearest support/resistance zones around a price."""
    if levels_df.empty:
        return None, None

    below = levels_df[levels_df['price_level'] <= price].copy()
    above = levels_df[levels_df['price_level'] > price].copy()

    support = None
    if not below.empty:
        nearest_sup = below.loc[below['price_level'].idxmax()]
        sup_price = nearest_sup['price_level']
        zone_start = sup_price * (1 - zone_width)
        zone_end = sup_price * (1 + zone_width)
        zone_levels = below[
            (below['price_level'] >= zone_start) & (below['price_level'] <= zone_end)
        ]
        support = {
            'distance_pct': (price - sup_price) / price if price else 0,
            'zone_start': zone_start,
            'zone_end': zone_end,
            'daily_count': len(zone_levels[zone_levels['timeframe'] == 'daily']),
            'weekly_count': len(zone_levels[zone_levels['timeframe'] == 'weekly']),
            'monthly_count': len(zone_levels[zone_levels['timeframe'] == 'monthly']),
            'fib618_count': len(zone_levels[zone_levels['level_type'].str.contains('0.618', na=False)]),
            'naked_count': len(zone_levels[
                (zone_levels.get('support_touches', pd.Series(dtype=int)) == 0) &
                (zone_levels.get('resistance_touches', pd.Series(dtype=int)) == 0)
            ]) if 'support_touches' in zone_levels.columns else len(zone_levels),
        }

    resistance = None
    if not above.empty:
        nearest_res = above.loc[above['price_level'].idxmin()]
        res_price = nearest_res['price_level']
        zone_start = res_price * (1 - zone_width)
        zone_end = res_price * (1 + zone_width)
        zone_levels = above[
            (above['price_level'] >= zone_start) & (above['price_level'] <= zone_end)
        ]
        resistance = {
            'distance_pct': (res_price - price) / price if price else 0,
            'zone_start': zone_start,
            'zone_end': zone_end,
            'daily_count': len(zone_levels[zone_levels['timeframe'] == 'daily']),
            'weekly_count': len(zone_levels[zone_levels['timeframe'] == 'weekly']),
            'monthly_count': len(zone_levels[zone_levels['timeframe'] == 'monthly']),
            'fib618_count': len(zone_levels[zone_levels['level_type'].str.contains('0.618', na=False)]),
            'naked_count': len(zone_levels[
                (zone_levels.get('support_touches', pd.Series(dtype=int)) == 0) &
                (zone_levels.get('resistance_touches', pd.Series(dtype=int)) == 0)
            ]) if 'support_touches' in zone_levels.columns else len(zone_levels),
        }

    return support, resistance


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_features(db: Session, timeframe: str = '1h',
                     symbol: str = 'BTCUSDT') -> int:
    """Compute ML features for all candles and store in features table.

    Returns number of features computed.
    """
    run = PipelineRun(
        pipeline_type='features',
        status='running',
        started_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()

    try:
        candles = (
            db.query(Candle)
            .filter_by(symbol=symbol, timeframe=timeframe)
            .order_by(Candle.open_time)
            .all()
        )

        if len(candles) < 3:
            run.status = 'completed'
            run.finished_at = datetime.now(timezone.utc)
            run.rows_processed = 0
            db.commit()
            return 0

        # Load levels into a DataFrame for zone detection
        levels = db.query(Level).filter(Level.invalidated_at.is_(None)).all()
        levels_df = pd.DataFrame([{
            'price_level': l.price_level,
            'level_type': l.level_type,
            'timeframe': l.timeframe,
            'source': l.source,
            'created_at': l.created_at,
            'support_touches': l.support_touches,
            'resistance_touches': l.resistance_touches,
        } for l in levels]) if levels else pd.DataFrame()

        # Price / volume history arrays for indicator calculation
        volumes = np.array([c.volume for c in candles])
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])

        # Fractal timing state
        candles_since_up = 0
        candles_since_down = 0

        computed = 0
        for i in range(2, len(candles)):
            candle = candles[i]
            n1 = candles[i - 1]
            n2 = candles[i - 2]

            # Skip if already computed
            existing = db.query(Feature.id).filter_by(candle_id=candle.id).first()
            if existing:
                # Still update timing state
                if candle.bullish_fractal:
                    candles_since_up = 0
                else:
                    candles_since_up += 1
                if candle.bearish_fractal:
                    candles_since_down = 0
                else:
                    candles_since_down += 1
                continue

            # Candle ratios (on N-1)
            uwr, lwr, btr, bpr = _candle_ratios(n1.open, n1.high, n1.low, n1.close)

            # Volume ratios (on N-1)
            vsr, vlr = _volume_ratios(n1.volume, volumes[:i])

            # UTC block
            utc = _utc_block(candle.open_time)

            # Fractal timing
            if candle.bullish_fractal:
                candles_since_up = 0
            else:
                candles_since_up += 1
            if candle.bearish_fractal:
                candles_since_down = 0
            else:
                candles_since_down += 1

            # Momentum / volatility indicators (computed up to N-1)
            rsi = _compute_rsi(closes[:i], period=14)
            macd_l, macd_s, macd_h = _compute_macd(closes[:i])
            bw = _compute_bollinger_width(closes[:i], period=20)
            atr = _compute_atr(highs[:i], lows[:i], closes[:i], period=14)
            mom = _compute_momentum(closes[:i], period=12)

            # Zone detection from N-2 perspective
            sup, res = None, None
            if not levels_df.empty:
                valid = levels_df[levels_df['created_at'] <= n2.open_time]
                sup, res = _find_nearest_levels(n2.close, valid)

            feature = Feature(
                candle_id=candle.id,
                computed_at=datetime.now(timezone.utc),
                upper_wick_ratio=uwr,
                lower_wick_ratio=lwr,
                body_total_ratio=btr,
                body_position_ratio=bpr,
                volume_short_ratio=vsr,
                volume_long_ratio=vlr,
                utc_block=utc,
                candles_since_last_up=candles_since_up,
                candles_since_last_down=candles_since_down,
                rsi_14=rsi,
                macd_line=macd_l,
                macd_signal=macd_s,
                macd_histogram=macd_h,
                bollinger_width=bw,
                atr_14=atr,
                momentum_12=mom,
            )

            if sup:
                feature.support_distance_pct = sup['distance_pct']
                feature.support_zone_start = sup['zone_start']
                feature.support_zone_end = sup['zone_end']
                feature.support_daily_count = sup['daily_count']
                feature.support_weekly_count = sup['weekly_count']
                feature.support_monthly_count = sup['monthly_count']
                feature.support_fib618_count = sup['fib618_count']
                feature.support_naked_count = sup['naked_count']

            if res:
                feature.resistance_distance_pct = res['distance_pct']
                feature.resistance_zone_start = res['zone_start']
                feature.resistance_zone_end = res['zone_end']
                feature.resistance_daily_count = res['daily_count']
                feature.resistance_weekly_count = res['weekly_count']
                feature.resistance_monthly_count = res['monthly_count']
                feature.resistance_fib618_count = res['fib618_count']
                feature.resistance_naked_count = res['naked_count']

            db.add(feature)
            computed += 1

            if computed % 1000 == 0:
                db.commit()
                logger.info("Computed %d features...", computed)

        db.commit()

        run.status = 'completed'
        run.finished_at = datetime.now(timezone.utc)
        run.rows_processed = computed
        db.commit()

        logger.info("Feature computation complete: %d features", computed)
        return computed

    except Exception as exc:
        run.status = 'failed'
        run.finished_at = datetime.now(timezone.utc)
        run.error_message = str(exc)
        db.commit()
        raise
