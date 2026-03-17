"""Multi-timeframe ML experiment.

Fetches all timeframes from Binance (2017-present), runs fractal detection,
computes features, trains RF models, and compares results across execution TFs.

Usage:
    python scripts/tf_experiment.py                # full run
    python scripts/tf_experiment.py --skip-fetch    # skip Binance fetch (data already loaded)
    python scripts/tf_experiment.py --skip-indicators  # skip fetch + fractal detection
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app
from app.extensions import db as flask_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# All timeframes to test as execution TFs
EXECUTION_TFS = ['1h', '4h', '6h', '8h', '12h', '1d']

# All timeframes to fetch from Binance
FETCH_TFS = ['1h', '4h', '6h', '8h', '12h', '1d', '1w', '1M']

# Date range
START_DATE = '17 Aug 2017'
END_DATE = '31 Dec 2025'


def step1_fetch(session):
    """Fetch all timeframes from Binance."""
    from app.models import Setting
    from app.services.data_fetcher import fetch_candles

    # Get API keys from DB
    api_key = session.query(Setting).filter_by(key='binance_api_key').first()
    api_secret = session.query(Setting).filter_by(key='binance_api_secret').first()

    if not api_key or not api_secret:
        logger.error("No Binance API keys in DB. Set them in /settings/")
        return False

    for tf in FETCH_TFS:
        logger.info("=== Fetching %s from %s ===", tf, START_DATE)
        start = time.time()
        try:
            n = fetch_candles(
                session, interval=tf,
                start_str=START_DATE, end_str=END_DATE,
                api_key=api_key.value, api_secret=api_secret.value,
            )
            logger.info("  %s: %d new candles in %.0fs", tf, n, time.time() - start)
        except Exception as e:
            logger.error("  %s FAILED: %s", tf, e)

    # Print summary
    from sqlalchemy import text
    rows = session.execute(text(
        "SELECT timeframe, count(*), min(open_time), max(open_time) "
        "FROM candles GROUP BY timeframe ORDER BY count(*) DESC"
    )).fetchall()
    print("\n=== Candle Summary ===")
    for tf, cnt, mn, mx in rows:
        print(f"  {tf:>4}: {cnt:>6,} candles | {str(mn)[:10]} to {str(mx)[:10]}")
    return True


def step2_fractals(session):
    """Run fractal detection on all timeframes."""
    from app.services.indicators import run_fractal_detection

    for tf in FETCH_TFS:
        logger.info("=== Fractal detection: %s ===", tf)
        start = time.time()
        try:
            result = run_fractal_detection(session, timeframe=tf)
            logger.info("  %s: done in %.0fs — %s", tf, time.time() - start, result)
        except Exception as e:
            logger.error("  %s FAILED: %s", tf, e)

    # Print fractal summary
    from sqlalchemy import text
    rows = session.execute(text(
        "SELECT timeframe, count(*), "
        "SUM(CASE WHEN bullish_fractal=1 THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN bearish_fractal=1 THEN 1 ELSE 0 END) "
        "FROM candles GROUP BY timeframe ORDER BY count(*) DESC"
    )).fetchall()
    print("\n=== Fractal Summary ===")
    for tf, cnt, bull, bear in rows:
        print(f"  {tf:>4}: {cnt:>6,} candles | bull={bull or 0:>5} bear={bear or 0:>5}")


def step3_indicators(session):
    """Run full indicator pipeline on D/W/M to generate levels from backfilled data."""
    from app.services.indicators import run_indicators_multi

    logger.info("=== Running indicator pipeline (D/W/M levels) ===")
    start = time.time()
    try:
        result = run_indicators_multi(session, symbol='BTCUSDT')
        logger.info("  Done in %.0fs — %s", time.time() - start, result)
    except Exception as e:
        logger.error("  FAILED: %s", e)


def step4_experiment(session):
    """Run ML experiment across all execution timeframes."""
    import joblib
    from app.services.feature_engine import compute_features
    from app.services.ml_trainer import train_model
    from sqlalchemy import text

    results = []

    for tf in EXECUTION_TFS:
        logger.info("=== Experiment: %s ===", tf)

        # Check candle count
        row = session.execute(text(
            "SELECT count(*) FROM candles WHERE timeframe=:tf"
        ), {'tf': tf}).fetchone()
        n_candles = row[0]
        if n_candles < 100:
            logger.warning("  %s: only %d candles, skipping", tf, n_candles)
            continue

        # Clear features for this TF's candles (recompute fresh)
        session.execute(text(
            "DELETE FROM features WHERE candle_id IN "
            "(SELECT id FROM candles WHERE timeframe=:tf)"
        ), {'tf': tf})
        session.commit()

        # Compute features
        start = time.time()
        n_feat = compute_features(session, timeframe=tf)
        feat_time = time.time() - start
        logger.info("  Features: %d in %.0fs", n_feat, feat_time)

        if n_feat < 50:
            logger.warning("  %s: only %d features, skipping training", tf, n_feat)
            continue

        # Train bullish model
        try:
            m_bull = train_model(session, algorithm='random_forest',
                                target_column='target_bullish', timeframe=tf)
            bull_acc = m_bull.accuracy
            bull_f1 = m_bull.f1_macro

            # Get feature importance
            bundle = joblib.load(m_bull.file_path)
            model = bundle['model']
            features = bundle['features']
            imps = dict(zip(features, model.feature_importances_))
            level_feats = {k: v for k, v in imps.items()
                          if any(x in k for x in ['distance', 'confluence', 'liquidity'])}
            level_weight_bull = sum(level_feats.values())
        except Exception as e:
            logger.error("  Bullish training failed: %s", e)
            bull_acc = bull_f1 = level_weight_bull = 0
            imps = {}

        # Train bearish model
        try:
            m_bear = train_model(session, algorithm='random_forest',
                                target_column='target_bearish', timeframe=tf)
            bear_acc = m_bear.accuracy
            bear_f1 = m_bear.f1_macro

            bundle2 = joblib.load(m_bear.file_path)
            level_feats2 = {k: v for k, v in
                           zip(bundle2['features'], bundle2['model'].feature_importances_)
                           if any(x in k for x in ['distance', 'confluence', 'liquidity'])}
            level_weight_bear = sum(level_feats2.values())
        except Exception as e:
            logger.error("  Bearish training failed: %s", e)
            bear_acc = bear_f1 = level_weight_bear = 0

        result = {
            'timeframe': tf,
            'candles': n_candles,
            'features': n_feat,
            'bull_acc': round(bull_acc, 4),
            'bull_f1': round(bull_f1, 4),
            'bear_acc': round(bear_acc, 4),
            'bear_f1': round(bear_f1, 4),
            'level_weight_bull': round(level_weight_bull, 4),
            'level_weight_bear': round(level_weight_bear, 4),
            'top_features': sorted(imps.items(), key=lambda x: -x[1])[:5],
        }
        results.append(result)
        logger.info("  %s: Bull(acc=%.3f f1=%.3f) Bear(acc=%.3f f1=%.3f) LevelW=%.1f%%/%.1f%%",
                     tf, bull_acc, bull_f1, bear_acc, bear_f1,
                     level_weight_bull * 100, level_weight_bear * 100)

    return results


def print_results(results):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("MULTI-TIMEFRAME ML EXPERIMENT RESULTS")
    print("=" * 100)
    print(f"{'TF':>4} | {'Candles':>8} | {'Feats':>6} | {'Bull Acc':>8} | {'Bull F1':>7} | "
          f"{'Bear Acc':>8} | {'Bear F1':>7} | {'Lvl Bull':>8} | {'Lvl Bear':>8}")
    print("-" * 100)

    for r in results:
        print(f"{r['timeframe']:>4} | {r['candles']:>8,} | {r['features']:>6,} | "
              f"{r['bull_acc']:>7.1%} | {r['bull_f1']:>6.1%} | "
              f"{r['bear_acc']:>7.1%} | {r['bear_f1']:>6.1%} | "
              f"{r['level_weight_bull']:>7.1%} | {r['level_weight_bear']:>7.1%}")

    print("=" * 100)

    # Find best TF
    if results:
        best_f1 = max(results, key=lambda r: (r['bull_f1'] + r['bear_f1']) / 2)
        best_levels = max(results, key=lambda r: (r['level_weight_bull'] + r['level_weight_bear']) / 2)
        print(f"\nBest F1 score:        {best_f1['timeframe']} "
              f"(avg F1 = {(best_f1['bull_f1'] + best_f1['bear_f1']) / 2:.1%})")
        print(f"Best level relevance: {best_levels['timeframe']} "
              f"(avg level weight = {(best_levels['level_weight_bull'] + best_levels['level_weight_bear']) / 2:.1%})")

    # Top features per TF
    print("\nTop 5 features per timeframe:")
    for r in results:
        top = r.get('top_features', [])
        print(f"  {r['timeframe']:>4}: ", end="")
        for fname, imp in top:
            print(f"{fname}({imp:.0%}) ", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description='Multi-timeframe ML experiment')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip Binance fetch')
    parser.add_argument('--skip-indicators', action='store_true', help='Skip fetch + indicators')
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        session = flask_db.session

        if not args.skip_fetch and not args.skip_indicators:
            step1_fetch(session)
            step2_fractals(session)
            step3_indicators(session)
        elif not args.skip_indicators:
            step2_fractals(session)
            step3_indicators(session)

        results = step4_experiment(session)
        print_results(results)

        # Save results to JSON
        out_path = Path(__file__).parent / 'tf_experiment_results.json'
        with open(out_path, 'w') as f:
            # Convert tuples to lists for JSON
            for r in results:
                r['top_features'] = [[k, v] for k, v in r.get('top_features', [])]
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results,
            }, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
