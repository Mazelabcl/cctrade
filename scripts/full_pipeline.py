"""Full pipeline: recalculate VP from 1min, recompute features, run backtests, train models.

Usage:
    python scripts/full_pipeline.py [--skip-vp] [--skip-backtest]
"""
import sys
import os
import time
import json
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-vp', action='store_true', help='Skip VP recalculation')
    parser.add_argument('--skip-backtest', action='store_true', help='Skip backtests')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip 1min fetch check')
    args = parser.parse_args()

    from app import create_app
    from app.extensions import db as flask_db
    from app.models import Candle, Level

    app = create_app()

    with app.app_context():
        db = flask_db.session

        # ================================================================
        # STEP 0: Verify 1-min data exists
        # ================================================================
        count_1m = db.query(Candle).filter_by(timeframe='1m').count()
        logger.info("1-min candles in DB: %d", count_1m)
        if count_1m == 0 and not args.skip_vp:
            logger.error("No 1-min data! Run fetch first or use --skip-vp")
            return

        # ================================================================
        # STEP 1: Recalculate VP levels from 1-min data
        # ================================================================
        if not args.skip_vp:
            logger.info("=" * 60)
            logger.info("STEP 1: Recalculating VP levels from 1-min data")
            logger.info("=" * 60)

            import pandas as pd
            from app.services.indicators import calculate_volume_profile_levels

            # Delete old VP levels
            old_vp = db.query(Level).filter(Level.source == 'volume_profile').count()
            db.query(Level).filter(Level.source == 'volume_profile').delete()
            db.commit()
            logger.info("Deleted %d old VP levels", old_vp)

            # Load 1-min data in chunks to avoid memory issues
            logger.info("Loading 1-min candles...")
            t0 = time.time()

            from sqlalchemy import text
            rows = db.execute(text(
                "SELECT open_time, high, low, volume FROM candles "
                "WHERE timeframe='1m' AND symbol='BTCUSDT' ORDER BY open_time"
            )).fetchall()
            df_1min = pd.DataFrame(rows, columns=['open_time', 'high', 'low', 'volume'])
            df_1min['open_time'] = pd.to_datetime(df_1min['open_time'])
            logger.info("Loaded %d 1-min candles in %.0fs", len(df_1min), time.time() - t0)

            # Calculate VP for daily, weekly, monthly
            for tf_label, period_group in [('daily', 'D'), ('weekly', 'W'), ('monthly', 'ME')]:
                logger.info("Calculating VP for %s...", tf_label)
                t1 = time.time()
                vp_levels = calculate_volume_profile_levels(df_1min, tf_label, period_group)
                logger.info("  Got %d VP levels in %.0fs", len(vp_levels), time.time() - t1)

                # Insert into DB
                for lv in vp_levels:
                    level = Level(
                        price_level=lv['price_level'],
                        level_type=lv['level_type'],
                        timeframe=lv['timeframe'],
                        source=lv['source'],
                        created_at=lv['created_at'],
                    )
                    db.add(level)
                db.commit()
                logger.info("  Inserted %d %s VP levels", len(vp_levels), tf_label)

            # Summary
            new_vp = db.query(Level).filter(Level.source == 'volume_profile').count()
            logger.info("VP recalculation complete: %d total VP levels", new_vp)
        else:
            logger.info("Skipping VP recalculation (--skip-vp)")

        # ================================================================
        # STEP 2: Clear ALL features (need full recompute with new levels)
        # ================================================================
        logger.info("=" * 60)
        logger.info("STEP 2: Clearing all features for recomputation")
        logger.info("=" * 60)

        from sqlalchemy import text
        db.execute(text("DELETE FROM features"))
        db.commit()
        logger.info("All features cleared")

        # ================================================================
        # STEP 3: Recompute features for each test TF with D/W/M naked levels
        # ================================================================
        logger.info("=" * 60)
        logger.info("STEP 3: Computing features (D/W/M naked levels, no Igor fibs)")
        logger.info("=" * 60)

        from app.services.feature_engine import compute_features

        test_tfs = ['1h', '4h', '6h', '8h', '12h']
        for tf in test_tfs:
            logger.info("Computing features for %s...", tf)
            t1 = time.time()
            n = compute_features(db, timeframe=tf, level_filter='htf_no_igor')
            logger.info("  %s: %d features in %.0fs", tf, n, time.time() - t1)

        # ================================================================
        # STEP 4: Run backtests per level on each test TF
        # ================================================================
        if not args.skip_backtest:
            logger.info("=" * 60)
            logger.info("STEP 4: Running level trade backtests on each test TF")
            logger.info("=" * 60)

            from app.services.level_trade_backtest_db import run_level_trade_backtest

            for tf in test_tfs:
                logger.info("Backtest on %s...", tf)
                t1 = time.time()
                try:
                    results = run_level_trade_backtest(
                        db, exec_timeframe=tf,
                        rr_ratios=[1.0, 2.0, 3.0],
                        naked_only=True, timeout_candles=100,
                    )
                    logger.info("  %s: %d results in %.0fs", tf, len(results), time.time() - t1)
                except Exception as e:
                    logger.error("  %s backtest failed: %s", tf, e)
        else:
            logger.info("Skipping backtests (--skip-backtest)")

        # ================================================================
        # STEP 5: Train RF models and compare
        # ================================================================
        logger.info("=" * 60)
        logger.info("STEP 5: Training RF models per test TF")
        logger.info("=" * 60)

        from app.services.ml_trainer import train_model
        import joblib

        results = []
        for tf in test_tfs:
            for target in ['target_bullish', 'target_bearish']:
                logger.info("Training RF %s on %s...", target, tf)
                try:
                    m = train_model(db, algorithm='random_forest',
                                    target_column=target, timeframe=tf)

                    # Get feature importance
                    bundle = joblib.load(m.file_path)
                    model = bundle['model']
                    features = bundle['features']
                    imps = sorted(zip(features, model.feature_importances_),
                                  key=lambda x: -x[1])

                    level_feats = [f for f in imps if any(
                        x in f[0] for x in ['distance', 'confluence', 'liquidity'])]
                    level_weight = sum(i for _, i in level_feats)

                    result = {
                        'timeframe': tf,
                        'target': target,
                        'accuracy': round(m.accuracy, 4),
                        'f1_macro': round(m.f1_macro, 4),
                        'train_rows': m.train_rows,
                        'level_weight': round(level_weight, 4),
                        'top5': [(f, round(i, 4)) for f, i in imps[:5]],
                    }
                    results.append(result)
                    logger.info("  %s %s: acc=%.3f f1=%.3f levels=%.1f%%",
                                tf, target, m.accuracy, m.f1_macro, level_weight * 100)
                except Exception as e:
                    logger.error("  %s %s training failed: %s", tf, target, e)

        # ================================================================
        # STEP 6: Print comparison table and save results
        # ================================================================
        logger.info("=" * 60)
        logger.info("RESULTS COMPARISON")
        logger.info("=" * 60)

        print(f"\n{'TF':>4} | {'Target':>16} | {'Acc':>6} | {'F1':>6} | {'Levels%':>7} | {'Rows':>6} | Top Feature")
        print("-" * 85)
        for r in results:
            top = r['top5'][0][0] if r['top5'] else '?'
            print(f"{r['timeframe']:>4} | {r['target']:>16} | {r['accuracy']:>6.3f} | "
                  f"{r['f1_macro']:>6.3f} | {r['level_weight']*100:>6.1f}% | "
                  f"{r['train_rows']:>6} | {top}")

        # Save JSON
        out_path = 'scripts/full_pipeline_results.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", out_path)


if __name__ == '__main__':
    main()
