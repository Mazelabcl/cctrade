#!/usr/bin/env python
"""Fractal Swing System — complete backtest with equity curve and position sizing.

This is the first TRADEABLE system derived from our research:
- Entry: Only Fractal_support + Fractal_resistance (daily/weekly/monthly)
- Exit: Swing trail stop (best strategy per MFE/trail stop analysis)
- Position sizing: Fixed risk per trade (default 1% of equity)
- Equity curve: Tracks actual $ growth from starting capital

Usage:
    python scripts/fractal_swing_system.py --tf 4h
    python scripts/fractal_swing_system.py --tf 4h --risk-pct 0.02 --capital 10000
    python scripts/fractal_swing_system.py --tf 1h --output scripts/fractal_system_1h.json
"""
import sys
import os
import time
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


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


def run_fractal_system(session, exec_tf='4h', initial_capital=10000.0,
                       risk_pct=0.01, max_lookforward=500):
    """Run the complete Fractal Swing System backtest.

    Args:
        exec_tf: Execution timeframe
        initial_capital: Starting capital in $
        risk_pct: Risk per trade as fraction of current equity (0.01 = 1%)
        max_lookforward: Max candles to hold a trade

    Returns:
        dict with equity_curve, trades, metrics
    """
    from app.services.level_trade_backtest_db import load_candles_db
    from sqlalchemy import text

    # Load candles
    candles = load_candles_db(session, timeframe=exec_tf)
    if candles.empty:
        return {'error': f'No candles for {exec_tf}'}

    c_times = candles['open_time'].values
    c_opens = candles['open'].values.astype(np.float64)
    c_highs = candles['high'].values.astype(np.float64)
    c_lows = candles['low'].values.astype(np.float64)
    c_closes = candles['close'].values.astype(np.float64)

    # Precompute swing levels
    swing_lows = find_swing_lows(c_lows, lookback=5)
    swing_highs = find_swing_highs(c_highs, lookback=5)

    time_to_idx = {}
    for idx, t in enumerate(c_times):
        time_to_idx[pd.Timestamp(t)] = idx

    # Load ONLY Fractal trades
    sql = """
        SELECT t.id, t.entry_time, t.entry_price, t.stop_loss, t.direction,
               t.exit_reason, t.pnl_pct, t.take_profit,
               b.level_type, b.level_source_timeframe
        FROM individual_level_trades t
        JOIN individual_level_backtests b ON t.backtest_id = b.id
        WHERE b.trade_execution_timeframe = :tf
        AND b.strategy_name = 'wick_rr_1.0'
        AND b.status = 'completed'
        AND t.exit_reason IN ('TP_HIT', 'SL_HIT')
        AND b.level_type IN ('Fractal_support', 'Fractal_resistance')
        ORDER BY t.entry_time
    """
    rows = session.execute(text(sql), {'tf': exec_tf}).fetchall()
    print(f"Fractal Swing System: {len(rows)} trade entries ({exec_tf})", flush=True)

    # --- Simulate with position sizing ---
    equity = initial_capital
    peak_equity = initial_capital
    max_drawdown_pct = 0.0

    trades = []
    equity_curve = [{'time': str(c_times[0])[:10], 'equity': equity}]

    wins = 0
    losses = 0
    total_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    consecutive_wins = 0
    consecutive_losses = 0
    max_consec_wins = 0
    max_consec_losses = 0

    t0 = time.time()
    for trade_idx, row in enumerate(rows):
        entry_time = pd.Timestamp(row[1])
        entry_price = float(row[2])
        stop_loss = float(row[3])
        direction = row[4]
        level_type = row[8]
        level_tf = row[9]

        risk = abs(entry_price - stop_loss)
        if risk == 0:
            continue

        idx = time_to_idx.get(entry_time)
        if idx is None or idx >= len(candles) - 2:
            continue

        # Position sizing: risk X% of current equity
        risk_dollars = equity * risk_pct
        position_size = risk_dollars / risk  # units of BTC
        position_value = position_size * entry_price

        # --- Swing Trail Exit ---
        current_sl = stop_loss
        exit_price = None
        exit_reason = 'TIMEOUT'
        exit_idx = None

        for j in range(idx + 1, min(idx + max_lookforward, len(candles))):
            if direction == 'LONG':
                # Check SL hit
                if c_lows[j] <= current_sl:
                    exit_price = current_sl
                    exit_reason = 'TRAIL_SL'
                    exit_idx = j
                    break
                # Update trail: swing low (but never lower than current SL)
                if j > idx + 3:  # give 3 candles before trailing
                    new_sl = swing_lows[j] * 0.999
                    if new_sl > current_sl:
                        current_sl = new_sl
            else:  # SHORT
                if c_highs[j] >= current_sl:
                    exit_price = current_sl
                    exit_reason = 'TRAIL_SL'
                    exit_idx = j
                    break
                if j > idx + 3:
                    new_sl = swing_highs[j] * 1.001
                    if new_sl < current_sl:
                        current_sl = new_sl

        if exit_price is None:
            # Timeout: close at last close
            exit_price = c_closes[min(idx + max_lookforward - 1, len(candles) - 1)]
            exit_idx = min(idx + max_lookforward - 1, len(candles) - 1)

        # Calculate PnL
        if direction == 'LONG':
            pnl_per_unit = exit_price - entry_price
        else:
            pnl_per_unit = entry_price - exit_price

        pnl_dollars = pnl_per_unit * position_size
        pnl_r = pnl_per_unit / risk
        pnl_pct = pnl_dollars / equity * 100

        # Update equity
        equity += pnl_dollars
        total_pnl += pnl_dollars

        if pnl_dollars > 0:
            wins += 1
            gross_profit += pnl_dollars
            consecutive_wins += 1
            consecutive_losses = 0
            max_consec_wins = max(max_consec_wins, consecutive_wins)
        else:
            losses += 1
            gross_loss += abs(pnl_dollars)
            consecutive_losses += 1
            consecutive_wins = 0
            max_consec_losses = max(max_consec_losses, consecutive_losses)

        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd_pct = (peak_equity - equity) / peak_equity * 100
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct

        candles_held = exit_idx - idx if exit_idx else 0
        exit_time = str(c_times[exit_idx])[:19] if exit_idx else None

        trades.append({
            'entry_time': str(entry_time)[:19],
            'exit_time': exit_time,
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'stop_loss': round(stop_loss, 2),
            'direction': direction,
            'level_type': level_type,
            'level_tf': level_tf,
            'pnl_r': round(pnl_r, 2),
            'pnl_dollars': round(pnl_dollars, 2),
            'pnl_pct': round(pnl_pct, 2),
            'equity_after': round(equity, 2),
            'position_size_btc': round(position_size, 6),
            'risk_dollars': round(risk_dollars, 2),
            'candles_held': candles_held,
            'exit_reason': exit_reason,
        })

        equity_curve.append({
            'time': str(entry_time)[:10],
            'equity': round(equity, 2),
            'trade_idx': len(trades),
        })

        if (trade_idx + 1) % 100 == 0:
            print(f"  {trade_idx+1}/{len(rows)} trades, equity=${equity:,.0f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    total_trades = wins + losses

    # Metrics
    win_rate = wins / total_trades * 100 if total_trades else 0
    avg_win = gross_profit / wins if wins else 0
    avg_loss = gross_loss / losses if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.99
    avg_r = sum(t['pnl_r'] for t in trades) / len(trades) if trades else 0
    total_r = sum(t['pnl_r'] for t in trades)
    total_return_pct = (equity - initial_capital) / initial_capital * 100

    # Sharpe-like ratio (using R values)
    if trades:
        rs = [t['pnl_r'] for t in trades]
        sharpe = np.mean(rs) / np.std(rs) if np.std(rs) > 0 else 0
    else:
        sharpe = 0

    # By level type + source TF
    by_combo = {}
    for t in trades:
        key = f"{t['level_type']} ({t['level_tf']})"
        if key not in by_combo:
            by_combo[key] = {'trades': 0, 'wins': 0, 'total_r': 0, 'pnl': 0}
        by_combo[key]['trades'] += 1
        if t['pnl_r'] > 0:
            by_combo[key]['wins'] += 1
        by_combo[key]['total_r'] += t['pnl_r']
        by_combo[key]['pnl'] += t['pnl_dollars']

    metrics = {
        'exec_tf': exec_tf,
        'initial_capital': initial_capital,
        'final_equity': round(equity, 2),
        'total_return_pct': round(total_return_pct, 1),
        'risk_per_trade_pct': risk_pct * 100,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'avg_r': round(avg_r, 2),
        'total_r': round(total_r, 0),
        'avg_win_dollars': round(avg_win, 2),
        'avg_loss_dollars': round(avg_loss, 2),
        'max_drawdown_pct': round(max_drawdown_pct, 1),
        'max_consecutive_wins': max_consec_wins,
        'max_consecutive_losses': max_consec_losses,
        'sharpe_r': round(sharpe, 2),
        'elapsed_sec': round(elapsed, 1),
        'by_combo': {k: {
            'trades': v['trades'],
            'wr': round(v['wins'] / v['trades'] * 100, 1),
            'total_r': round(v['total_r'], 1),
            'total_pnl': round(v['pnl'], 2),
        } for k, v in sorted(by_combo.items(), key=lambda x: x[1]['total_r'], reverse=True)},
    }

    return {
        'metrics': metrics,
        'equity_curve': equity_curve,
        'trades': trades,
    }


def print_summary(result):
    m = result['metrics']
    print(flush=True)
    print("=" * 70, flush=True)
    print("FRACTAL SWING SYSTEM — RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"  Timeframe:       {m['exec_tf']}", flush=True)
    print(f"  Capital:         ${m['initial_capital']:,.0f} -> ${m['final_equity']:,.0f}", flush=True)
    print(f"  Total Return:    {m['total_return_pct']:+.1f}%", flush=True)
    print(f"  Risk/Trade:      {m['risk_per_trade_pct']}%", flush=True)
    print(f"  Trades:          {m['total_trades']} ({m['wins']}W / {m['losses']}L)", flush=True)
    print(f"  Win Rate:        {m['win_rate']}%", flush=True)
    print(f"  Profit Factor:   {m['profit_factor']}", flush=True)
    print(f"  Avg R:           {m['avg_r']}R", flush=True)
    print(f"  Total R:         {m['total_r']}R", flush=True)
    print(f"  Max Drawdown:    {m['max_drawdown_pct']}%", flush=True)
    print(f"  Sharpe (R):      {m['sharpe_r']}", flush=True)
    print(f"  Max Consec Wins: {m['max_consecutive_wins']}", flush=True)
    print(f"  Max Consec Loss: {m['max_consecutive_losses']}", flush=True)
    print(flush=True)
    print("  --- By Level Type ---", flush=True)
    for combo, d in m['by_combo'].items():
        print(f"    {combo:35s}  {d['trades']:3d}t  WR={d['wr']:5.1f}%  "
              f"Total={d['total_r']:+7.1f}R  PnL=${d['total_pnl']:+,.0f}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', default='4h')
    parser.add_argument('--capital', type=float, default=10000)
    parser.add_argument('--risk-pct', type=float, default=0.01)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        result = run_fractal_system(
            db.session,
            exec_tf=args.tf,
            initial_capital=args.capital,
            risk_pct=args.risk_pct,
        )

        print_summary(result)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nSaved to {args.output}", flush=True)
