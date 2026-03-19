"""Run the full scoring pipeline: Previous Session levels + VWAP + Scoring Engine backtest.

Usage:
    python scripts/run_scoring_pipeline.py
"""
import sys
import os
import json
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    from app import create_app
    from app.extensions import db
    from app.services.indicators import (
        run_previous_session_levels,
        run_vwap_and_session_vp,
    )
    from app.services.scoring_engine import backtest_scoring_engine
    import sqlite3

    app = create_app()
    with app.app_context():
        # ── Step 1: Generate Previous Session levels ──
        logger.info("=" * 60)
        logger.info("STEP 1: Previous Session levels (D/W/M)")
        logger.info("=" * 60)
        start = time.time()
        ps_result = run_previous_session_levels(db.session, timeframes=['1d', '1w', '1M'])
        logger.info("Previous Session done in %.0fs: %s", time.time() - start, ps_result)

        # ── Step 2: VWAP + Session VP levels ──
        logger.info("=" * 60)
        logger.info("STEP 2: VWAP + Previous Session VP (D/W/M from 1min)")
        logger.info("=" * 60)
        start = time.time()
        vwap_result = run_vwap_and_session_vp(db.session)
        logger.info("VWAP done in %.0fs: %s", time.time() - start, vwap_result)

        # ── Step 3: Level count summary ──
        conn = sqlite3.connect('instance/tradebot.db')
        logger.info("=" * 60)
        logger.info("LEVEL SUMMARY")
        logger.info("=" * 60)
        rows = conn.execute('''
            SELECT level_type, timeframe, count(*)
            FROM levels
            WHERE source IN ('previous_session', 'vwap')
            GROUP BY level_type, timeframe
            ORDER BY count(*) DESC
        ''').fetchall()
        for lt, tf, cnt in rows:
            logger.info("  %s / %s: %d levels", lt, tf, cnt)
        conn.close()

        # ── Step 4: Scoring Engine backtest ──
        logger.info("=" * 60)
        logger.info("STEP 3: Scoring Engine Backtest")
        logger.info("=" * 60)

        results = []
        for exec_tf in ['4h', '1h']:
            for min_score in [8.0, 10.0, 12.0, 15.0]:
                for level_filter in ['htf_no_igor', 'htf_all']:
                    logger.info("  Testing: tf=%s min_score=%.0f filter=%s",
                                exec_tf, min_score, level_filter)
                    start = time.time()
                    try:
                        r = backtest_scoring_engine(
                            db.session,
                            exec_tf=exec_tf,
                            min_score=min_score,
                            level_filter=level_filter,
                        )
                        elapsed = time.time() - start
                        r['elapsed_sec'] = round(elapsed, 1)
                        results.append(r)

                        if 'error' not in r:
                            logger.info(
                                "    → %d trades | WR=%.1f%% | PF=%.2f | PnL=%.2f%% | avg_score=%.1f (%.0fs)",
                                r['total_trades'], r['win_rate'], r['profit_factor'],
                                r['total_pnl_pct'], r['avg_score'], elapsed,
                            )
                        else:
                            logger.info("    → %s", r['error'])
                    except Exception as exc:
                        logger.error("    → ERROR: %s", exc)
                        results.append({
                            'exec_tf': exec_tf, 'min_score': min_score,
                            'level_filter': level_filter, 'error': str(exc),
                        })

        # ── Step 5: Save results ──
        out_path = 'scripts/scoring_results.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", out_path)

        # ── Print summary table ──
        logger.info("=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info("%-4s %-8s %-14s %6s %6s %6s %8s %6s",
                     "TF", "MinScore", "Filter", "Trades", "WR%", "PF", "PnL%", "AvgScr")
        logger.info("-" * 60)
        for r in results:
            if 'error' in r:
                continue
            logger.info("%-4s %-8s %-14s %6d %5.1f%% %6.2f %7.2f%% %6.1f",
                         r['exec_tf'], r['min_score'], r['level_filter'],
                         r['total_trades'], r['win_rate'], r['profit_factor'],
                         r['total_pnl_pct'], r['avg_score'])


if __name__ == '__main__':
    main()
