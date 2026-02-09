#!/usr/bin/env python
"""CLI script to run a backtest for a trained ML model.

Usage:
    python scripts/run_backtest.py --model-id 1
    python scripts/run_backtest.py --model-id 1 --cash 50000 --risk 0.03
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.services.backtest.runner import run_backtest


def main():
    parser = argparse.ArgumentParser(description='Run backtest for a trained model')
    parser.add_argument('--model-id', type=int, required=True, help='MLModel ID to backtest')
    parser.add_argument('--cash', type=float, default=100_000.0, help='Initial cash (default: 100000)')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate (default: 0.001)')
    parser.add_argument('--risk', type=float, default=0.02, help='Risk per trade (default: 0.02)')
    parser.add_argument('--confidence', type=float, default=0.55, help='Confidence threshold (default: 0.55)')
    parser.add_argument('--proximity', type=float, default=0.02, help='Level proximity pct (default: 0.02)')
    parser.add_argument('--sl-mult', type=float, default=1.5, help='ATR stop-loss multiplier (default: 1.5)')
    parser.add_argument('--tp-mult', type=float, default=3.0, help='ATR take-profit multiplier (default: 3.0)')
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        print(f"Running backtest for model {args.model_id}...")
        print(f"  Cash: ${args.cash:,.0f}")
        print(f"  Commission: {args.commission*100:.2f}%")
        print(f"  Risk/trade: {args.risk*100:.1f}%")
        print(f"  Confidence threshold: {args.confidence*100:.0f}%")
        print(f"  Level proximity: {args.proximity*100:.0f}%")
        print(f"  SL mult: {args.sl_mult}x ATR")
        print(f"  TP mult: {args.tp_mult}x ATR")
        print()

        result = run_backtest(
            db.session,
            model_id=args.model_id,
            initial_cash=args.cash,
            commission=args.commission,
            risk_per_trade=args.risk,
            confidence_threshold=args.confidence,
            level_proximity_pct=args.proximity,
            atr_sl_mult=args.sl_mult,
            atr_tp_mult=args.tp_mult,
        )

        print("=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"  Final Value:   ${result.final_value:,.2f}")
        print(f"  Total Return:  {result.total_return*100:.2f}%")
        print(f"  Sharpe Ratio:  {result.sharpe_ratio or 0:.3f}")
        print(f"  Max Drawdown:  {result.max_drawdown*100:.2f}%" if result.max_drawdown else "  Max Drawdown:  N/A")
        print(f"  Total Trades:  {result.total_trades}")
        print(f"  Win Rate:      {result.win_rate*100:.1f}%")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Avg Win:       ${result.avg_win:,.2f}")
        print(f"  Avg Loss:      ${result.avg_loss:,.2f}")
        if result.sqn is not None:
            print(f"  SQN:           {result.sqn:.2f}")
        print(f"  Result ID:     {result.id}")
        print()

        if result.trade_log:
            print(f"Trade Log ({len(result.trade_log)} trades):")
            for i, t in enumerate(result.trade_log[:20], 1):
                print(f"  {i:3d}. {t['direction']:5s} entry={t['entry_price']:.0f} "
                      f"pnl={t['pnlcomm']:+.2f} conf={t['confidence']:.2f}")
            if len(result.trade_log) > 20:
                print(f"  ... and {len(result.trade_log)-20} more")


if __name__ == '__main__':
    main()
