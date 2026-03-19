"""Proximity experiment: test level filtering + proximity thresholds on ML.

Tests two level filter modes (no Igor fibs vs no fibs at all) combined with
proximity thresholds, all on 4h execution timeframe.

Usage:
    python scripts/proximity_experiment.py
"""
import json
import os
import sys
import time

import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.services.feature_engine import compute_features
from app.services.ml_trainer import train_model

EXEC_TF = '4h'

# Test matrix: (level_filter, proximity_threshold)
TEST_MATRIX = [
    ('htf_no_igor', None),
    ('htf_no_igor', 0.02),
    ('htf_no_igor', 0.01),
    ('htf_no_fibs', None),
    ('htf_no_fibs', 0.02),
    ('htf_no_fibs', 0.01),
    ('htf_no_fibs', 0.005),
]

LEVEL_FEATURES = [
    'support_distance_pct', 'resistance_distance_pct',
    'support_confluence_score', 'resistance_confluence_score',
    'support_liquidity_consumed', 'resistance_liquidity_consumed',
]


def run_experiment():
    app = create_app()
    results = []

    with app.app_context():
        last_filter = None

        for level_filter, prox_thresh in TEST_MATRIX:
            label = f"{level_filter} prox={prox_thresh or 'none'}"
            print(f"\n{'='*60}")
            print(f"Testing: {label}")
            print(f"{'='*60}")

            # Recompute features only when level_filter changes
            if level_filter != last_filter:
                print(f"  Clearing {EXEC_TF} features...")
                from app.models import Feature, Candle
                subq = db.session.query(Candle.id).filter_by(timeframe=EXEC_TF)
                db.session.query(Feature).filter(Feature.candle_id.in_(subq)).delete(synchronize_session='fetch')
                db.session.commit()

                print(f"  Computing features (level_filter={level_filter})...")
                start = time.time()
                n = compute_features(db.session, timeframe=EXEC_TF, level_filter=level_filter)
                print(f"  Computed {n} features in {time.time()-start:.0f}s")
                last_filter = level_filter

            # Train bullish + bearish
            combo_result = {
                'level_filter': level_filter,
                'proximity_threshold': prox_thresh,
                'exec_tf': EXEC_TF,
            }

            for target in ['target_bullish', 'target_bearish']:
                tgt_label = 'bull' if 'bullish' in target else 'bear'
                print(f"  Training RF {tgt_label} (prox={prox_thresh})...")

                model = train_model(
                    db.session,
                    algorithm='random_forest',
                    target_column=target,
                    timeframe=EXEC_TF,
                    proximity_threshold=prox_thresh,
                    name=f"prox_{level_filter}_{tgt_label}",
                )

                # Extract feature importance
                bundle = joblib.load(model.file_path)
                rf = bundle['model']
                features = bundle['features']
                imps = dict(zip(features, rf.feature_importances_))

                level_weight = sum(imps.get(f, 0) for f in LEVEL_FEATURES)
                top5 = sorted(imps.items(), key=lambda x: -x[1])[:5]

                combo_result[f'{tgt_label}_acc'] = model.accuracy
                combo_result[f'{tgt_label}_f1'] = model.f1_macro
                combo_result[f'{tgt_label}_level_weight'] = level_weight
                combo_result[f'{tgt_label}_top5'] = [(f, round(v, 4)) for f, v in top5]
                combo_result[f'{tgt_label}_train_rows'] = model.train_rows

                print(f"    Acc={model.accuracy:.3f} F1={model.f1_macro:.3f} LevelWt={level_weight:.3f} Rows={model.train_rows}")

            results.append(combo_result)

    # Print summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Filter':<20} {'Prox':<8} {'Rows':<7} {'AccB':<7} {'F1B':<7} {'AccR':<7} {'F1R':<7} {'LvlB':<7} {'LvlR':<7}")
    print("-" * 80)
    for r in results:
        prox = f"{r['proximity_threshold']*100:.1f}%" if r['proximity_threshold'] else "none"
        print(f"{r['level_filter']:<20} {prox:<8} {r.get('bull_train_rows',0):<7} "
              f"{r.get('bull_acc',0):.3f}  {r.get('bull_f1',0):.3f}  "
              f"{r.get('bear_acc',0):.3f}  {r.get('bear_f1',0):.3f}  "
              f"{r.get('bull_level_weight',0):.3f}  {r.get('bear_level_weight',0):.3f}")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proximity_experiment_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    run_experiment()
