"""Unified feature engineering pipeline.

Consolidates legacy create_ml_features.py, candle_ratios.py, volume_ratios.py,
time_blocks.py, and fractal_timing.py into one service.
"""
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import func as sa_func
from sqlalchemy.orm import Session

from ..models import Candle, Level, Feature, PipelineRun
from ..models.individual_level_backtest import IndividualLevelBacktest

logger = logging.getLogger(__name__)

DEFAULT_WIN_RATE = 0.5


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
# Volatility / Momentum
# ---------------------------------------------------------------------------

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

def _find_nearest_distances(price, levels_df):
    """Find distance to nearest support/resistance levels around a price.

    Returns (support_distance_pct, resistance_distance_pct).
    """
    if levels_df.empty:
        return None, None

    sup_dist = None
    res_dist = None

    below = levels_df[levels_df['price_level'] <= price]
    if not below.empty:
        nearest_sup = below['price_level'].max()
        sup_dist = (price - nearest_sup) / price if price else 0

    above = levels_df[levels_df['price_level'] > price]
    if not above.empty:
        nearest_res = above['price_level'].min()
        res_dist = (nearest_res - price) / price if price else 0

    return sup_dist, res_dist


def _compute_zone_features(price, levels_df, win_rate_cache, zone_width=0.015):
    """Compute confluence_score and liquidity_consumed for zones around price.

    Support zone: levels between price*(1-zone_width) and price
    Resistance zone: levels between price and price*(1+zone_width)
    """
    result = {
        'support_confluence_score': 0.0,
        'resistance_confluence_score': 0.0,
        'support_liquidity_consumed': 0.0,
        'resistance_liquidity_consumed': 0.0,
    }
    if levels_df.empty:
        return result

    lower_bound = price * (1 - zone_width)
    upper_bound = price * (1 + zone_width)

    # Support zone: levels below price within zone_width
    support_levels = levels_df[
        (levels_df['price_level'] >= lower_bound) &
        (levels_df['price_level'] <= price)
    ]
    if not support_levels.empty:
        scores = []
        touched_count = 0
        for _, lv in support_levels.iterrows():
            key = (lv['level_type'], lv['timeframe'])
            wr = win_rate_cache.get(key, DEFAULT_WIN_RATE)
            scores.append(wr)
            if lv.get('support_touches', 0) > 0 or lv.get('resistance_touches', 0) > 0:
                touched_count += 1
        result['support_confluence_score'] = sum(scores)
        result['support_liquidity_consumed'] = touched_count / len(support_levels)

    # Resistance zone: levels above price within zone_width
    resistance_levels = levels_df[
        (levels_df['price_level'] > price) &
        (levels_df['price_level'] <= upper_bound)
    ]
    if not resistance_levels.empty:
        scores = []
        touched_count = 0
        for _, lv in resistance_levels.iterrows():
            key = (lv['level_type'], lv['timeframe'])
            wr = win_rate_cache.get(key, DEFAULT_WIN_RATE)
            scores.append(wr)
            if lv.get('support_touches', 0) > 0 or lv.get('resistance_touches', 0) > 0:
                touched_count += 1
        result['resistance_confluence_score'] = sum(scores)
        result['resistance_liquidity_consumed'] = touched_count / len(resistance_levels)

    return result


# ---------------------------------------------------------------------------
# Win-rate cache builder
# ---------------------------------------------------------------------------

def _build_win_rate_cache(db):
    """Build (level_type, source_tf) -> avg win_rate dict from backtest results."""
    try:
        results = (
            db.query(
                IndividualLevelBacktest.level_type,
                IndividualLevelBacktest.level_source_timeframe,
                sa_func.avg(IndividualLevelBacktest.win_rate),
            )
            .filter(
                IndividualLevelBacktest.status == 'completed',
                IndividualLevelBacktest.strategy_name.like('wick_rr_%'),
                IndividualLevelBacktest.win_rate.isnot(None),
            )
            .group_by(
                IndividualLevelBacktest.level_type,
                IndividualLevelBacktest.level_source_timeframe,
            )
            .all()
        )
        cache = {
            (lt, tf): wr / 100.0  # normalize to 0-1
            for lt, tf, wr in results
        }
        logger.info("Win-rate cache built: %d entries", len(cache))
        return cache
    except Exception as exc:
        logger.warning("Could not build win-rate cache: %s. Using defaults.", exc)
        return {}


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

        # Build win-rate cache from backtest results
        win_rate_cache = _build_win_rate_cache(db)

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

            # Volatility / momentum (computed up to N-1)
            atr = _compute_atr(highs[:i], lows[:i], closes[:i], period=14)
            mom = _compute_momentum(closes[:i], period=12)

            # Zone detection from N-2 perspective
            sup_dist, res_dist = None, None
            zone_feats = {
                'support_confluence_score': 0.0,
                'resistance_confluence_score': 0.0,
                'support_liquidity_consumed': 0.0,
                'resistance_liquidity_consumed': 0.0,
            }
            if not levels_df.empty:
                valid = levels_df[levels_df['created_at'] <= n2.open_time]
                sup_dist, res_dist = _find_nearest_distances(n2.close, valid)
                zone_feats = _compute_zone_features(n2.close, valid, win_rate_cache)

            # Target: was N-1 a fractal?
            target_bullish = 1 if n1.bullish_fractal else 0
            target_bearish = 1 if n1.bearish_fractal else 0

            feature = Feature(
                candle_id=candle.id,
                computed_at=datetime.now(timezone.utc),
                target_bullish=target_bullish,
                target_bearish=target_bearish,
                upper_wick_ratio=uwr,
                lower_wick_ratio=lwr,
                body_total_ratio=btr,
                body_position_ratio=bpr,
                volume_short_ratio=vsr,
                volume_long_ratio=vlr,
                utc_block=utc,
                candles_since_last_up=candles_since_up,
                candles_since_last_down=candles_since_down,
                support_distance_pct=sup_dist,
                resistance_distance_pct=res_dist,
                atr_14=atr,
                momentum_12=mom,
                support_confluence_score=zone_feats['support_confluence_score'],
                resistance_confluence_score=zone_feats['resistance_confluence_score'],
                support_liquidity_consumed=zone_feats['support_liquidity_consumed'],
                resistance_liquidity_consumed=zone_feats['resistance_liquidity_consumed'],
            )

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
