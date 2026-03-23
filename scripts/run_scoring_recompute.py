"""Re-run scoring engine backtest with current data."""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.services.scoring_engine import backtest_scoring_engine

app = create_app()
with app.app_context():
    results = []
    for exec_tf in ['4h', '1h']:
        for min_score in [7.0, 8.0, 9.0, 10.0]:
            label = f"tf={exec_tf} min_score={min_score}"
            print(f"Testing: {label}...", flush=True)
            t0 = time.time()
            try:
                r = backtest_scoring_engine(
                    db.session,
                    exec_tf=exec_tf,
                    min_score=min_score,
                    level_filter='htf_no_igor',
                )
                elapsed = time.time() - t0
                r['elapsed_sec'] = round(elapsed, 1)
                results.append(r)
                if 'error' not in r:
                    print(f"  -> {r['total_trades']} trades | "
                          f"WR={r['win_rate']:.1f}% | PF={r['profit_factor']:.2f} | "
                          f"PnL={r['total_pnl_pct']:.1f}% | "
                          f"avg_score={r['avg_score']:.1f} ({elapsed:.0f}s)", flush=True)
                else:
                    print(f"  -> {r['error']}", flush=True)
            except Exception as e:
                print(f"  -> ERROR: {e}", flush=True)
                results.append({'exec_tf': exec_tf, 'min_score': min_score, 'error': str(e)})

    # Save
    with open('scripts/scoring_results_recomputed.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(flush=True)
    print("=" * 70, flush=True)
    print("SCORING ENGINE RESULTS (recomputed)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'TF':4s} {'Min':>5s} {'Trades':>7s} {'WR%':>6s} {'PF':>6s} {'PnL%':>8s} {'AvgScr':>7s}", flush=True)
    print("-" * 50, flush=True)
    for r in results:
        if 'error' in r:
            continue
        print(f"{r['exec_tf']:4s} {r['min_score']:>5.0f} {r['total_trades']:>7d} "
              f"{r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} "
              f"{r['total_pnl_pct']:>7.1f}% {r['avg_score']:>7.1f}", flush=True)
    print("Done!", flush=True)
