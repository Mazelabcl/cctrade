#!/usr/bin/env python
"""Ensure a timeframe has ALL data computed: features, ML, backtests, MFE.

This is the "single source of truth" script. Run it for any TF and it
guarantees parity. If something already exists, it skips it (unless --force).

Usage:
    python scripts/ensure_tf_complete.py --tf 1h
    python scripts/ensure_tf_complete.py --tf 4h
    python scripts/ensure_tf_complete.py --tf 4h --clean-dupes
    python scripts/ensure_tf_complete.py --tf 1h --force  # recompute everything
    python scripts/ensure_tf_complete.py --audit           # just show status

Steps:
  1. Verify candles exist
  2. Compute features (skip if already exist, unless --force)
  3. Train RF models (skip if already exist, unless --force)
  4. Run level backtests for ALL combos (skip existing)
  5. Run MFE analysis
  6. Print audit summary
"""
import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def audit_all(session):
    """Print full audit of what exists across all timeframes."""
    from sqlalchemy import text

    print("=" * 80)
    print("FULL DATA AUDIT")
    print("=" * 80)

    # Candles
    print("\n--- CANDLES ---")
    rows = session.execute(text(
        "SELECT timeframe, COUNT(*), MIN(open_time), MAX(open_time) "
        "FROM candles GROUP BY timeframe ORDER BY COUNT(*) DESC"
    )).fetchall()
    for tf, cnt, mn, mx in rows:
        print(f"  {tf:6s}  {cnt:>9,d}  [{mn[:10]} to {mx[:10]}]")

    # Features
    print("\n--- FEATURES ---")
    rows = session.execute(text(
        "SELECT c.timeframe, COUNT(f.id) FROM features f "
        "JOIN candles c ON f.candle_id = c.id GROUP BY c.timeframe"
    )).fetchall()
    feat_by_tf = {r[0]: r[1] for r in rows}
    for tf in ['1h', '4h', '6h', '8h', '12h', '1d']:
        cnt = feat_by_tf.get(tf, 0)
        status = 'OK' if cnt > 0 else 'MISSING'
        print(f"  {tf:6s}  {cnt:>9,d}  [{status}]")

    # Backtests per exec_tf (unique combos only)
    print("\n--- BACKTESTS (wick_rr_1.0, unique combos) ---")
    rows = session.execute(text(
        "SELECT trade_execution_timeframe, "
        "COUNT(DISTINCT level_type || '|' || level_source_timeframe) as unique_combos, "
        "COUNT(*) as total_rows, "
        "SUM(total_trades) as total_trades "
        "FROM individual_level_backtests "
        "WHERE status = 'completed' AND strategy_name = 'wick_rr_1.0' "
        "GROUP BY trade_execution_timeframe ORDER BY trade_execution_timeframe"
    )).fetchall()
    for tf, combos, total, trades in rows:
        dupes = total - combos
        dupe_warn = f" [!{dupes} DUPES]" if dupes > 0 else ""
        print(f"  {tf:6s}  {combos:>3d} combos  {trades:>8,d} trades{dupe_warn}")

    # ML Models
    print("\n--- ML MODELS ---")
    rows = session.execute(text(
        "SELECT name, accuracy, created_at FROM ml_models ORDER BY created_at DESC"
    )).fetchall()
    for name, acc, dt in rows:
        print(f"  {name:40s}  acc={acc:.3f}  {dt[:10]}")

    # MFE files
    print("\n--- MFE FILES ---")
    for tf in ['1h', '4h']:
        found = False
        for fname in [f'scripts/mfe_results_{tf}.json', 'scripts/mfe_results.json']:
            if os.path.exists(fname):
                sz = os.path.getsize(fname) / 1e6
                with open(fname) as f:
                    data = json.load(f)
                print(f"  {tf}: {len(data):,d} trades ({sz:.1f} MB)")
                found = True
                break
        if not found:
            print(f"  {tf}: MISSING")

    print()


def clean_duplicates(session, exec_tf):
    """Remove duplicate backtests (keep the one with most trades)."""
    from sqlalchemy import text

    print(f"\n--- Cleaning duplicate backtests for {exec_tf} ---")
    dupes = session.execute(text("""
        SELECT level_type, level_source_timeframe, strategy_name,
               COUNT(*) as cnt
        FROM individual_level_backtests
        WHERE trade_execution_timeframe = :tf AND status = 'completed'
        GROUP BY level_type, level_source_timeframe, strategy_name
        HAVING COUNT(*) > 1
    """), {'tf': exec_tf}).fetchall()

    if not dupes:
        print("  No duplicates found.")
        return 0

    total_removed = 0
    for lt, stf, strat, cnt in dupes:
        # Get all IDs for this combo, ordered by total_trades desc
        ids_rows = session.execute(text(
            "SELECT id, total_trades FROM individual_level_backtests "
            "WHERE level_type = :lt AND level_source_timeframe = :stf "
            "AND strategy_name = :strat AND trade_execution_timeframe = :tf "
            "AND status = 'completed' "
            "ORDER BY total_trades DESC"
        ), {'lt': lt, 'stf': stf, 'strat': strat, 'tf': exec_tf}).fetchall()

        keep_id = ids_rows[0][0]
        for rid, _ in ids_rows[1:]:
            session.execute(text(
                "DELETE FROM individual_level_trades WHERE backtest_id = :id"
            ), {'id': rid})
            session.execute(text(
                "DELETE FROM individual_level_backtests WHERE id = :id"
            ), {'id': rid})
            total_removed += 1

    session.commit()
    print(f"  Removed {total_removed} duplicate backtest records.")
    return total_removed


def ensure_complete(session, exec_tf, force=False, do_clean_dupes=False):
    """Ensure all components exist for a given timeframe."""
    from sqlalchemy import text

    print("=" * 80)
    print(f"ENSURING COMPLETE DATA FOR: {exec_tf}")
    print("=" * 80)

    # 0. Verify candles
    candle_cnt = session.execute(text(
        "SELECT COUNT(*) FROM candles WHERE timeframe = :tf"
    ), {'tf': exec_tf}).fetchone()[0]
    if candle_cnt == 0:
        print(f"\nERROR: No candles for {exec_tf}. Fetch data first.")
        return
    print(f"\nCandles: {candle_cnt:,d}", flush=True)

    # 1. Clean duplicates if requested
    if do_clean_dupes:
        clean_duplicates(session, exec_tf)

    # 2. Features
    feat_cnt = session.execute(text(
        "SELECT COUNT(*) FROM features f "
        "JOIN candles c ON f.candle_id = c.id WHERE c.timeframe = :tf"
    ), {'tf': exec_tf}).fetchone()[0]

    if feat_cnt > 0 and not force:
        print(f"\n[1/4] Features: {feat_cnt:,d} exist. SKIPPING.", flush=True)
    else:
        from app.services.feature_engine import compute_features
        if feat_cnt > 0:
            print(f"\n[1/4] Features: FORCE recompute (deleting {feat_cnt:,d})...", flush=True)
        else:
            print(f"\n[1/4] Features: MISSING. Computing...", flush=True)
        t0 = time.time()
        n = compute_features(session, timeframe=exec_tf, level_filter='htf_no_igor')
        print(f"  Computed {n:,d} features in {time.time()-t0:.0f}s", flush=True)

    # 3. ML Models
    existing_models = session.execute(text(
        "SELECT name FROM ml_models WHERE name LIKE :pat"
    ), {'pat': f'RF_%_{exec_tf}%'}).fetchall()
    model_names = {r[0] for r in existing_models}

    needed = [f'RF_bullish_{exec_tf}', f'RF_bearish_{exec_tf}']
    missing = [n for n in needed if n not in model_names]

    if not missing and not force:
        print(f"\n[2/4] ML Models: {len(model_names)} exist. SKIPPING.", flush=True)
    else:
        from app.services.ml_trainer import train_model
        targets = [('target_bullish', 'bullish'), ('target_bearish', 'bearish')]
        for target_col, label in targets:
            name = f'RF_{label}_{exec_tf}'
            if name in model_names and not force:
                print(f"\n[2/4] ML {label}: exists. SKIPPING.", flush=True)
                continue
            print(f"\n[2/4] Training RF {label} for {exec_tf}...", flush=True)
            t0 = time.time()
            m = train_model(session, algorithm='random_forest',
                            target_column=target_col, name=name, timeframe=exec_tf)
            print(f"  acc={m.accuracy:.3f} f1={m.f1_macro:.3f} ({time.time()-t0:.1f}s)",
                  flush=True)

    # 4. Backtests — run ALL combos, skip existing
    from app.services.level_trade_backtest_db import run_level_trade_backtest

    existing_bt = session.execute(text(
        "SELECT COUNT(DISTINCT level_type || '|' || level_source_timeframe) "
        "FROM individual_level_backtests "
        "WHERE trade_execution_timeframe = :tf "
        "AND status = 'completed' AND strategy_name = 'wick_rr_1.0'"
    ), {'tf': exec_tf}).fetchone()[0]

    # Count total possible combos
    total_possible = session.execute(text(
        "SELECT COUNT(DISTINCT level_type || '|' || timeframe) "
        "FROM levels WHERE timeframe IN ('daily','weekly','monthly')"
    )).fetchone()[0]

    if existing_bt >= total_possible and not force:
        print(f"\n[3/4] Backtests: {existing_bt}/{total_possible} combos exist. SKIPPING.",
              flush=True)
    else:
        print(f"\n[3/4] Backtests: {existing_bt}/{total_possible} combos exist. "
              f"Running missing...", flush=True)
        t0 = time.time()

        def _progress(done, total, label):
            if done % 10 == 0 or done == total:
                print(f"  {done}/{total}: {label}", flush=True)

        results = run_level_trade_backtest(
            session,
            exec_timeframe=exec_tf,
            rr_ratios=[1.0],
            tolerance_pct=0.005,
            timeout=100,
            naked_only=True,
            progress_cb=_progress,
        )
        elapsed = time.time() - t0
        new_bt = session.execute(text(
            "SELECT COUNT(DISTINCT level_type || '|' || level_source_timeframe) "
            "FROM individual_level_backtests "
            "WHERE trade_execution_timeframe = :tf "
            "AND status = 'completed' AND strategy_name = 'wick_rr_1.0'"
        ), {'tf': exec_tf}).fetchone()[0]
        print(f"  Now: {new_bt} unique combos ({elapsed:.0f}s)", flush=True)

    # 5. MFE
    mfe_path = f'scripts/mfe_results_{exec_tf}.json'
    if os.path.exists(mfe_path) and not force:
        with open(mfe_path) as f:
            mfe_cnt = len(json.load(f))
        print(f"\n[4/4] MFE: {mfe_cnt:,d} trades exist. SKIPPING.", flush=True)
    else:
        from scripts.mfe_analysis import run_mfe_analysis
        print(f"\n[4/4] Running MFE analysis for {exec_tf}...", flush=True)
        t0 = time.time()
        mfe_results = run_mfe_analysis(session, exec_tf=exec_tf)
        with open(mfe_path, 'w') as f:
            json.dump(mfe_results, f, default=str)
        print(f"  {len(mfe_results):,d} trades in {time.time()-t0:.0f}s -> {mfe_path}",
              flush=True)

    # Final summary
    print(f"\n{'='*80}")
    print(f"COMPLETE for {exec_tf}!")
    feat_final = session.execute(text(
        "SELECT COUNT(*) FROM features f "
        "JOIN candles c ON f.candle_id = c.id WHERE c.timeframe = :tf"
    ), {'tf': exec_tf}).fetchone()[0]
    bt_final = session.execute(text(
        "SELECT COUNT(DISTINCT level_type || '|' || level_source_timeframe) "
        "FROM individual_level_backtests "
        "WHERE trade_execution_timeframe = :tf "
        "AND status = 'completed' AND strategy_name = 'wick_rr_1.0'"
    ), {'tf': exec_tf}).fetchone()[0]
    print(f"  Features:   {feat_final:>9,d}")
    print(f"  BT Combos:  {bt_final:>9,d}")
    print(f"  MFE:        {'YES' if os.path.exists(mfe_path) else 'MISSING'}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensure TF has complete data')
    parser.add_argument('--tf', type=str, help='Execution timeframe (1h, 4h, etc.)')
    parser.add_argument('--force', action='store_true', help='Recompute everything')
    parser.add_argument('--clean-dupes', action='store_true', help='Remove duplicates first')
    parser.add_argument('--audit', action='store_true', help='Just show current state')
    args = parser.parse_args()

    from app import create_app
    from app.extensions import db

    app = create_app()
    with app.app_context():
        if args.audit:
            audit_all(db.session)
            sys.exit(0)

        if not args.tf:
            print("ERROR: --tf is required (e.g. --tf 1h)")
            print("       or use --audit to see current state")
            sys.exit(1)

        ensure_complete(db.session, args.tf,
                        force=args.force,
                        do_clean_dupes=args.clean_dupes)
