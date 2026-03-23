"""Full recompute: features + ML models for 1h and 4h with fixed naked/mobile levels.

Steps:
1. Delete ALL existing features (force recompute)
2. Compute features for 4h
3. Compute features for 1h
4. Train RF bullish + bearish for 4h
5. Train RF bullish + bearish for 1h
6. Print comparison summary
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.services.feature_engine import compute_features
from app.services.ml_trainer import train_model

app = create_app()

with app.app_context():
    # Step 0: Record existing model metrics for comparison
    from app.models import MLModel, Feature
    from sqlalchemy import text

    print("=" * 70)
    print("FULL RECOMPUTE PIPELINE")
    print("=" * 70)

    # Show existing models
    existing = MLModel.query.order_by(MLModel.created_at.desc()).limit(10).all()
    print("\n--- Existing ML Models (for comparison) ---")
    for m in existing:
        print(f"  {m.name or 'unnamed'}: acc={m.accuracy:.3f} prec={m.precision_macro:.3f} "
              f"rec={m.recall_macro:.3f} f1={m.f1_macro:.3f}")

    # Step 1: Delete all features
    print("\n[1/6] Deleting all existing features...")
    t0 = time.time()
    count = db.session.execute(text("DELETE FROM features")).rowcount
    db.session.commit()
    print(f"  Deleted {count} features in {time.time()-t0:.1f}s")

    # Step 2: Compute features for 4h
    print("\n[2/6] Computing features for 4h...")
    t0 = time.time()
    n4h = compute_features(db.session, timeframe='4h', level_filter='htf_no_igor')
    print(f"  Computed {n4h} features for 4h in {time.time()-t0:.1f}s")

    # Step 3: Compute features for 1h
    print("\n[3/6] Computing features for 1h...")
    t0 = time.time()
    n1h = compute_features(db.session, timeframe='1h', level_filter='htf_no_igor')
    print(f"  Computed {n1h} features for 1h in {time.time()-t0:.1f}s")

    # Step 4: Train RF for 4h
    print("\n[4/6] Training RF bullish 4h...")
    t0 = time.time()
    m4h_bull = train_model(db.session, algorithm='random_forest',
                           target_column='target_bullish',
                           name='RF_bullish_4h_recomputed',
                           timeframe='4h')
    print(f"  4h Bullish: acc={m4h_bull.accuracy:.3f} prec={m4h_bull.precision_macro:.3f} "
          f"rec={m4h_bull.recall_macro:.3f} f1={m4h_bull.f1_macro:.3f} ({time.time()-t0:.1f}s)")

    print("\n[5/6] Training RF bearish 4h...")
    t0 = time.time()
    m4h_bear = train_model(db.session, algorithm='random_forest',
                           target_column='target_bearish',
                           name='RF_bearish_4h_recomputed',
                           timeframe='4h')
    print(f"  4h Bearish: acc={m4h_bear.accuracy:.3f} prec={m4h_bear.precision_macro:.3f} "
          f"rec={m4h_bear.recall_macro:.3f} f1={m4h_bear.f1_macro:.3f} ({time.time()-t0:.1f}s)")

    # Step 5: Train RF for 1h
    print("\n[6/6] Training RF bullish + bearish 1h...")
    t0 = time.time()
    m1h_bull = train_model(db.session, algorithm='random_forest',
                           target_column='target_bullish',
                           name='RF_bullish_1h_recomputed',
                           timeframe='1h')
    print(f"  1h Bullish: acc={m1h_bull.accuracy:.3f} prec={m1h_bull.precision_macro:.3f} "
          f"rec={m1h_bull.recall_macro:.3f} f1={m1h_bull.f1_macro:.3f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    m1h_bear = train_model(db.session, algorithm='random_forest',
                           target_column='target_bearish',
                           name='RF_bearish_1h_recomputed',
                           timeframe='1h')
    print(f"  1h Bearish: acc={m1h_bear.accuracy:.3f} prec={m1h_bear.precision_macro:.3f} "
          f"rec={m1h_bear.recall_macro:.3f} f1={m1h_bear.f1_macro:.3f} ({time.time()-t0:.1f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — New Models (with fixed naked/mobile levels)")
    print("=" * 70)
    print(f"{'Model':<30s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    print("-" * 60)
    for label, m in [
        ('4h Bullish', m4h_bull), ('4h Bearish', m4h_bear),
        ('1h Bullish', m1h_bull), ('1h Bearish', m1h_bear),
    ]:
        print(f"{label:<30s} {m.accuracy:6.3f} {m.precision_macro:6.3f} "
              f"{m.recall_macro:6.3f} {m.f1_macro:6.3f}")

    # Feature stats
    print("\n--- Feature Stats ---")
    for tf in ['1h', '4h']:
        stats = db.session.execute(text(f"""
            SELECT COUNT(*),
                   AVG(f.support_distance_pct),
                   AVG(f.resistance_distance_pct),
                   AVG(f.support_confluence_score)
            FROM features f
            JOIN candles c ON f.candle_id = c.id
            WHERE c.timeframe = '{tf}'
        """)).fetchone()
        print(f"  {tf}: {stats[0]} features, avg sup_dist={stats[1]*100:.3f}%, "
              f"avg res_dist={stats[2]*100:.3f}%, avg confluence={stats[3]:.2f}")

    print("\nDone!")
