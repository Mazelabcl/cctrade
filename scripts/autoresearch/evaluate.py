"""Evaluate a configuration by running a fast backtest.

Returns metrics dict in ~5 seconds. Used by the AutoResearch agent loop.
"""
import sys
import os
import time
import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def compute_atr(highs, lows, closes, period=14):
    n = len(highs)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def find_swing_lows(lows, lookback=5):
    n = len(lows)
    swing = np.copy(lows)
    for i in range(lookback, n):
        swing[i] = np.min(lows[max(0, i-lookback):i+1])
    return swing


def find_swing_highs(highs, lookback=5):
    n = len(highs)
    swing = np.copy(highs)
    for i in range(lookback, n):
        swing[i] = np.max(highs[max(0, i-lookback):i+1])
    return swing


def evaluate(config: dict, session=None, _cache={}) -> dict:
    """Run a fast backtest with the given configuration.

    Args:
        config: dict with exit strategy parameters
        session: SQLAlchemy session (optional, creates one if not provided)
        _cache: internal cache for candle data (reused across calls)

    Returns:
        dict with metrics: total_r, profit_factor, win_rate, avg_r, median_r,
                          total_trades, max_consecutive_losses, sharpe_r
    """
    from sqlalchemy import text

    exec_tf = config.get('execution', {}).get('timeframe', '4h')
    level_types = config.get('levels', {}).get('types', ['Fractal_support', 'Fractal_resistance'])
    strategy = config.get('exit', {}).get('strategy', 'swing_trail')
    exit_cfg = config.get('exit', {})

    # Cache candle data for speed
    cache_key = exec_tf
    if cache_key not in _cache:
        from app.services.level_trade_backtest_db import load_candles_db
        candles = load_candles_db(session, timeframe=exec_tf)
        _cache[cache_key] = candles

    candles = _cache[cache_key]
    if candles.empty:
        return {'error': f'No candles for {exec_tf}', 'total_r': 0}

    c_times = candles['open_time'].values
    c_highs = candles['high'].values.astype(np.float64)
    c_lows = candles['low'].values.astype(np.float64)
    c_closes = candles['close'].values.astype(np.float64)

    # Precompute indicators
    swing_lb = exit_cfg.get('swing_lookback', 5)
    atr = compute_atr(c_highs, c_lows, c_closes, period=14)
    swing_lows = find_swing_lows(c_lows, lookback=swing_lb)
    swing_highs = find_swing_highs(c_highs, lookback=swing_lb)

    time_to_idx = {}
    for idx, t in enumerate(c_times):
        time_to_idx[pd.Timestamp(t)] = idx

    # Load trade entries
    sql = """
        SELECT t.entry_time, t.entry_price, t.stop_loss, t.direction,
               b.level_type
        FROM individual_level_trades t
        JOIN individual_level_backtests b ON t.backtest_id = b.id
        WHERE b.trade_execution_timeframe = :tf
        AND b.strategy_name = 'wick_rr_1.0'
        AND b.status = 'completed'
        AND t.exit_reason IN ('TP_HIT', 'SL_HIT')
    """
    rows = session.execute(text(sql), {'tf': exec_tf}).fetchall()
    rows = [r for r in rows if r[4] in level_types]

    if not rows:
        return {'error': 'No trades found', 'total_r': 0}

    # Parameters
    be_rr = exit_cfg.get('breakeven_at_rr', 1.0)
    atr_mult = exit_cfg.get('atr_multiplier', 2.0)
    partial_pct = exit_cfg.get('partial_pct', 0.5)
    partial_rr = exit_cfg.get('partial_rr', 2.0)
    timeout = exit_cfg.get('timeout_candles', 500)

    results = []
    for row in rows:
        entry_time = pd.Timestamp(row[0])
        entry_price = float(row[1])
        stop_loss = float(row[2])
        direction = row[3]
        risk = abs(entry_price - stop_loss)
        if risk == 0:
            continue

        idx = time_to_idx.get(entry_time)
        if idx is None or idx >= len(candles) - 1:
            continue

        pnl_r = _simulate_exit(
            strategy, direction, entry_price, stop_loss, risk, idx,
            c_highs, c_lows, c_closes, atr, swing_lows, swing_highs,
            timeout, be_rr, atr_mult, partial_pct, partial_rr,
        )
        results.append(pnl_r)

    if not results:
        return {'error': 'No valid trades', 'total_r': 0}

    results = np.array(results)
    wins = results[results > 0]
    losses = results[results <= 0]
    gross_win = float(wins.sum()) if len(wins) > 0 else 0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0
    pf = gross_win / gross_loss if gross_loss > 0 else 99.99

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for r in results:
        if r <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    # Sharpe of R values
    sharpe = float(results.mean() / results.std()) if results.std() > 0 else 0

    return {
        'total_r': round(float(results.sum()), 1),
        'profit_factor': round(pf, 2),
        'win_rate': round(len(wins) / len(results) * 100, 1),
        'avg_r': round(float(results.mean()), 2),
        'median_r': round(float(np.median(results)), 2),
        'total_trades': len(results),
        'max_consecutive_losses': max_consec,
        'sharpe_r': round(sharpe, 3),
    }


def _simulate_exit(strategy, direction, entry, sl, risk, idx,
                   highs, lows, closes, atr, sw_lows, sw_highs,
                   timeout, be_rr, atr_mult, partial_pct, partial_rr):
    """Simulate a single trade with the given exit strategy. Returns pnl_r."""
    n = len(highs)
    end = min(idx + timeout, n)

    if strategy == 'swing_trail':
        sl_cur = sl
        for j in range(idx + 1, end):
            if direction == 'LONG':
                if lows[j] <= sl_cur:
                    return round((sl_cur - entry) / risk, 2)
                sl_cur = max(sl_cur, sw_lows[j])
            else:
                if highs[j] >= sl_cur:
                    return round((entry - sl_cur) / risk, 2)
                sl_cur = min(sl_cur, sw_highs[j])

    elif strategy == 'breakeven_trail':
        sl_cur = sl
        reached = False
        for j in range(idx + 1, end):
            if direction == 'LONG':
                if lows[j] <= sl_cur:
                    return round((sl_cur - entry) / risk, 2)
                if highs[j] >= entry + be_rr * risk:
                    reached = True
                if reached:
                    sl_cur = max(sl_cur, max(entry, sw_lows[j]))
            else:
                if highs[j] >= sl_cur:
                    return round((entry - sl_cur) / risk, 2)
                if lows[j] <= entry - be_rr * risk:
                    reached = True
                if reached:
                    sl_cur = min(sl_cur, min(entry, sw_highs[j]))

    elif strategy == 'atr_trail':
        sl_cur = sl
        for j in range(idx + 1, end):
            cur_atr = atr[j] if atr[j] > 0 else risk
            if direction == 'LONG':
                if lows[j] <= sl_cur:
                    return round((sl_cur - entry) / risk, 2)
                sl_cur = max(sl_cur, highs[j] - atr_mult * cur_atr)
            else:
                if highs[j] >= sl_cur:
                    return round((entry - sl_cur) / risk, 2)
                sl_cur = min(sl_cur, lows[j] + atr_mult * cur_atr)

    elif strategy == 'partial':
        sl_cur = sl
        partial_taken = False
        partial_pnl = 0.0
        for j in range(idx + 1, end):
            if direction == 'LONG':
                if lows[j] <= sl_cur:
                    rem = (sl_cur - entry) / risk
                    if partial_taken:
                        return round(partial_pnl + rem * (1 - partial_pct), 2)
                    return round(rem, 2)
                if not partial_taken and highs[j] >= entry + partial_rr * risk:
                    partial_taken = True
                    partial_pnl = partial_rr * partial_pct
                    sl_cur = max(sl_cur, entry)
                elif partial_taken:
                    sl_cur = max(sl_cur, sw_lows[j])
            else:
                if highs[j] >= sl_cur:
                    rem = (entry - sl_cur) / risk
                    if partial_taken:
                        return round(partial_pnl + rem * (1 - partial_pct), 2)
                    return round(rem, 2)
                if not partial_taken and lows[j] <= entry - partial_rr * risk:
                    partial_taken = True
                    partial_pnl = partial_rr * partial_pct
                    sl_cur = min(sl_cur, entry)
                elif partial_taken:
                    sl_cur = min(sl_cur, sw_highs[j])

    else:
        # Default: fixed 1:1
        tp = entry + risk if direction == 'LONG' else entry - risk
        for j in range(idx + 1, end):
            if direction == 'LONG':
                if lows[j] <= sl:
                    return -1.0
                if highs[j] >= tp:
                    return 1.0
            else:
                if highs[j] >= sl:
                    return -1.0
                if lows[j] <= tp:
                    return 1.0

    # Timeout
    last = closes[min(end - 1, n - 1)]
    pnl = (last - entry) / risk if direction == 'LONG' else (entry - last) / risk
    return round(pnl, 2)


if __name__ == '__main__':
    import yaml
    from app import create_app
    from app.extensions import db

    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    app = create_app()
    with app.app_context():
        t0 = time.time()
        metrics = evaluate(config, db.session)
        elapsed = time.time() - t0
        print(f"Evaluation completed in {elapsed:.1f}s", flush=True)
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v}", flush=True)
