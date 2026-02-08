#!/usr/bin/env python3
"""Import existing CSV datasets into SQLite database.

Usage:
    python scripts/import_csv.py                    # Import all datasets
    python scripts/import_csv.py --period 2021_01_01-2024_12_31  # Import specific period
    python scripts/import_csv.py --dry-run           # Show what would be imported
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app
from app.extensions import db
from app.models import Candle, Level, PipelineRun


DATASETS_DIR = Path(__file__).resolve().parent.parent / 'datasets'
MANIFEST_PATH = DATASETS_DIR / 'data_manifest.json'


def load_manifest():
    if not MANIFEST_PATH.exists():
        # Fall back to root-level manifest
        root_manifest = Path(__file__).resolve().parent.parent / 'data_manifest.json'
        if root_manifest.exists():
            with open(root_manifest) as f:
                return json.load(f)
        print("No data_manifest.json found.")
        return {'datasets': []}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def find_csv_files():
    """Discover CSV files from manifest or directory scan."""
    manifest = load_manifest()
    datasets = []

    for entry in manifest.get('datasets', []):
        if entry.get('status') == 'sample':
            continue  # Skip sample datasets

        period = entry['period']
        ml_path = entry['files'].get('ml_dataset', {}).get('path')
        levels_path = entry['files'].get('levels_dataset', {}).get('path')

        # Resolve paths relative to project root
        project_root = Path(__file__).resolve().parent.parent
        if ml_path:
            ml_full = project_root / ml_path
            if not ml_full.exists():
                ml_full = DATASETS_DIR / os.path.basename(ml_path)
        if levels_path:
            levels_full = project_root / levels_path
            if not levels_full.exists():
                levels_full = DATASETS_DIR / os.path.basename(levels_path)

        datasets.append({
            'period': period,
            'ml_file': str(ml_full) if ml_path and ml_full.exists() else None,
            'levels_file': str(levels_full) if levels_path and levels_full.exists() else None,
            'rows': entry.get('rows', 0),
        })

    return datasets


def import_candles(ml_file, period, app):
    """Import candle data from ml_dataset CSV."""
    print(f"  Reading {os.path.basename(ml_file)}...")
    df = pd.read_csv(ml_file)
    df['open_time'] = pd.to_datetime(df['open_time'])

    # Check for existing data in this range
    min_time = df['open_time'].min()
    max_time = df['open_time'].max()

    existing = Candle.query.filter(
        Candle.timeframe == '1h',
        Candle.open_time >= min_time,
        Candle.open_time <= max_time,
    ).count()

    if existing > 0:
        print(f"  Skipping candles: {existing} already exist for this range")
        return 0

    candles = []
    for _, row in df.iterrows():
        candle = Candle(
            symbol='BTCUSDT',
            timeframe='1h',
            open_time=row['open_time'],
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume']),
            quote_volume=float(row.get('quote_asset_volume', 0)) if 'quote_asset_volume' in row.index else None,
            num_trades=int(row.get('number_of_trades', 0)) if 'number_of_trades' in row.index else None,
            bearish_fractal=bool(row.get('bearish_fractal', False)),
            bullish_fractal=bool(row.get('bullish_fractal', False)),
        )
        candles.append(candle)

    db.session.bulk_save_objects(candles)
    db.session.commit()
    print(f"  Imported {len(candles)} candles")
    return len(candles)


def import_levels(levels_file, period, app):
    """Import levels data from levels_dataset CSV."""
    print(f"  Reading {os.path.basename(levels_file)}...")
    df = pd.read_csv(levels_file)

    if df.empty:
        print("  No levels to import")
        return 0

    # Parse created_at
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
    else:
        df['created_at'] = datetime.now(timezone.utc)

    # Check for existing levels in this range
    min_time = df['created_at'].min()
    max_time = df['created_at'].max()

    existing = Level.query.filter(
        Level.created_at >= min_time,
        Level.created_at <= max_time,
    ).count()

    if existing > 0:
        print(f"  Skipping levels: {existing} already exist for this range")
        return 0

    levels = []
    for _, row in df.iterrows():
        # Determine source from level_type
        level_type = str(row.get('level_type', ''))
        if 'Fractal' in level_type:
            source = 'fractal'
        elif 'Fib' in level_type:
            source = 'fibonacci'
        elif 'VP' in level_type or 'vp' in level_type:
            source = 'volume_profile'
        elif 'HTF' in level_type:
            source = 'htf'
        else:
            source = row.get('source', 'unknown')

        # Build metadata from extra columns
        metadata = {}
        for col in ['anchor_time', 'completion_time', 'direction', 'period']:
            if col in row.index and pd.notna(row.get(col)):
                metadata[col] = str(row[col])

        level = Level(
            price_level=float(row['price_level']),
            level_type=level_type,
            timeframe=str(row.get('timeframe', 'daily')),
            source=source,
            created_at=row['created_at'] if pd.notna(row.get('created_at')) else datetime.now(timezone.utc),
            support_touches=int(row.get('support_touches', 0)) if 'support_touches' in row.index and pd.notna(row.get('support_touches')) else 0,
            resistance_touches=int(row.get('resistance_touches', 0)) if 'resistance_touches' in row.index and pd.notna(row.get('resistance_touches')) else 0,
            metadata_json=metadata if metadata else None,
        )
        levels.append(level)

    db.session.bulk_save_objects(levels)
    db.session.commit()
    print(f"  Imported {len(levels)} levels")
    return len(levels)


def main():
    parser = argparse.ArgumentParser(description='Import CSV datasets into SQLite')
    parser.add_argument('--period', type=str, help='Import specific period (e.g., 2021_01_01-2024_12_31)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be imported')
    args = parser.parse_args()

    app = create_app('development')

    datasets = find_csv_files()
    if not datasets:
        print("No datasets found to import.")
        return

    if args.period:
        datasets = [d for d in datasets if d['period'] == args.period]
        if not datasets:
            print(f"Period '{args.period}' not found.")
            return

    print(f"Found {len(datasets)} dataset(s) to import:\n")
    for ds in datasets:
        print(f"  {ds['period']}: ~{ds['rows']} rows")
        if ds['ml_file']:
            print(f"    ML:     {os.path.basename(ds['ml_file'])}")
        if ds['levels_file']:
            print(f"    Levels: {os.path.basename(ds['levels_file'])}")
    print()

    if args.dry_run:
        print("Dry run - no data imported.")
        return

    with app.app_context():
        total_candles = 0
        total_levels = 0

        for ds in datasets:
            print(f"\nImporting period: {ds['period']}")

            # Log pipeline run
            run = PipelineRun(
                pipeline_type='csv_import',
                status='running',
                started_at=datetime.now(timezone.utc),
                metadata_json={'period': ds['period']},
            )
            db.session.add(run)
            db.session.commit()

            try:
                candle_count = 0
                level_count = 0

                if ds['ml_file']:
                    candle_count = import_candles(ds['ml_file'], ds['period'], app)
                    total_candles += candle_count

                if ds['levels_file']:
                    level_count = import_levels(ds['levels_file'], ds['period'], app)
                    total_levels += level_count

                run.status = 'completed'
                run.finished_at = datetime.now(timezone.utc)
                run.rows_processed = candle_count + level_count
                db.session.commit()

            except Exception as e:
                run.status = 'failed'
                run.finished_at = datetime.now(timezone.utc)
                run.error_message = str(e)
                db.session.commit()
                print(f"  ERROR: {e}")

        print(f"\nImport complete: {total_candles} candles, {total_levels} levels")
        print(f"Database: {app.config['SQLALCHEMY_DATABASE_URI']}")


if __name__ == '__main__':
    main()
