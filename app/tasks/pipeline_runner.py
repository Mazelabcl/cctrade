"""Orchestrates the full automated pipeline and foundation pipeline.

Runs: indicators -> features -> predictions -> accuracy backfill.
Each step is independent and logs its own PipelineRun.
"""
import logging
from datetime import datetime, timezone

from flask import Flask

logger = logging.getLogger(__name__)


def run_full_pipeline(app: Flask):
    """Execute the full pipeline sequence."""
    from ..extensions import db
    from ..services.indicators import run_indicators
    from ..services.feature_engine import compute_features
    from ..services.ml_predictor import predict
    from ..views.api import publish_sse
    from .accuracy_tracker import backfill_actuals

    publish_sse('pipeline', {'status': 'running', 'type': 'auto_pipeline'})

    try:
        # Step 1: Run indicators (fractal detection + levels)
        logger.info("Auto pipeline: running indicators...")
        indicator_result = run_indicators(db.session)
        logger.info("Auto pipeline: indicators done — %s", indicator_result)

        # Step 2: Compute features
        logger.info("Auto pipeline: computing features...")
        feature_count = compute_features(db.session)
        logger.info("Auto pipeline: %d features computed", feature_count)

        # Step 3: Run predictions with active model (if one exists)
        logger.info("Auto pipeline: running predictions...")
        try:
            predictions = predict(db.session)
            logger.info("Auto pipeline: %d predictions generated", len(predictions))
        except ValueError as e:
            logger.info("Auto pipeline: skipping predictions — %s", e)
            predictions = []

        # Step 4: Backfill actual classes for past predictions
        backfilled = backfill_actuals(db.session)
        if backfilled:
            logger.info("Auto pipeline: backfilled %d prediction actuals", backfilled)

        publish_sse('pipeline', {
            'status': 'completed', 'type': 'auto_pipeline',
            'features_computed': feature_count,
            'predictions': len(predictions),
            'actuals_backfilled': backfilled,
        })

    except Exception as e:
        logger.error("Auto pipeline failed: %s", e)
        publish_sse('pipeline', {
            'status': 'failed', 'type': 'auto_pipeline',
            'error': str(e),
        })


def run_foundation_pipeline(app: Flask):
    """Run the complete foundation pipeline using configured settings.

    Steps:
    1. Read config from DB settings
    2. Fetch candles for all configured timeframes
    3. Run level detection per type per configured TF (HTF, Fractal, Fib)
    4. Volume Profile (fetches 1-min data from Binance once, computes for all TFs)
    5. Touch tracking (runs on ALL levels including VP)
    6. Report progress in real-time via in-memory tracker
    """
    from ..extensions import db
    from ..models.setting import get_setting
    from ..models import PipelineRun
    from ..services.data_fetcher import fetch_candles
    from ..services.indicators import run_indicators_multi
    from ..services.level_tracker import run_touch_tracking
    from ..services import progress

    progress.start()

    run = PipelineRun(
        pipeline_type='foundation',
        status='running',
        started_at=datetime.now(timezone.utc),
    )
    db.session.add(run)
    db.session.commit()

    try:
        # Read config
        api_key = get_setting('binance_api_key', '')
        api_secret = get_setting('binance_api_secret', '')
        start_date = get_setting('data_start_date', '2020-01-01')
        end_date = get_setting('data_end_date', '2025-12-31')
        fetch_tfs = get_setting('fetch_timeframes', '1d,1w,1M').split(',')
        htf_tfs = get_setting('htf_timeframes', '1d,1w,1M').split(',')
        fractal_tfs = get_setting('fractal_timeframes', '1d,1w,1M').split(',')
        fib_tfs = get_setting('fibonacci_timeframes', '1d,1w').split(',')
        vp_tfs = get_setting('vp_timeframes', '1d,1w,1M').split(',')

        # Filter empty strings
        fetch_tfs = [t for t in fetch_tfs if t]
        htf_tfs = [t for t in htf_tfs if t]
        fractal_tfs = [t for t in fractal_tfs if t]
        fib_tfs = [t for t in fib_tfs if t]
        vp_tfs = [t for t in vp_tfs if t]

        progress.update('config', f'Fetch: {fetch_tfs}, HTF: {htf_tfs}, Frac: {fractal_tfs}, Fib: {fib_tfs}, VP: {vp_tfs}')

        # ── Step 1: Fetch candles ────────────────────────────────
        if api_key and api_secret:
            for tf in fetch_tfs:
                progress.update('fetch', f'Downloading {tf} candles from Binance...')
                try:
                    count = fetch_candles(
                        db.session, symbol='BTCUSDT', interval=tf,
                        start_str=start_date, end_str=end_date,
                        api_key=api_key, api_secret=api_secret,
                    )
                    progress.update('fetch', f'{tf}: {count} new candles')
                except Exception as e:
                    progress.update('fetch', f'{tf}: ERROR - {e}')
                    logger.error("Foundation fetch [%s] failed: %s", tf, e)
        else:
            progress.update('fetch', 'Skipped - no API keys configured')

        # ── Step 2: Run indicators ───────────────────────────────
        progress.update('indicators', 'Running fractal detection + level calculation...')

        indicator_summary = run_indicators_multi(
            db.session, symbol='BTCUSDT',
            htf_timeframes=htf_tfs,
            fractal_timeframes=fractal_tfs,
            fib_timeframes=fib_tfs,
            vp_timeframes=vp_tfs,
        )

        # Report per-type results
        for level_type, tf_counts in indicator_summary.items():
            if tf_counts:
                details = ', '.join(f'{tf}={cnt}' for tf, cnt in tf_counts.items())
                progress.update('indicators', f'{level_type}: {details}')

        # ── Step 3: Volume Profile ─────────────────────────────
        # VP runs BEFORE touch tracking so VP levels also get touches counted
        if vp_tfs and api_key and api_secret:
            progress.update('volume_profile', f'Computing VP for {vp_tfs} (fetching 1-min data)...')
            try:
                vp_summary = _run_volume_profile(db.session, api_key, api_secret,
                                                  start_date, end_date, vp_tfs, progress)
                for tf, result in vp_summary.items():
                    progress.update('volume_profile', f'{tf}: {result} levels')
            except Exception as e:
                progress.update('volume_profile', f'ERROR: {e}')
                logger.error("Foundation VP failed: %s", e)

        # ── Step 4: Touch tracking ───────────────────────────────
        # Runs AFTER all level types are created (HTF, Fractal, Fib, VP)
        from ..models import Level as LevelModel
        progress.update('touch_tracking', 'Resetting touch counts for all levels...')
        db.session.query(LevelModel).update({
            LevelModel.support_touches: 0,
            LevelModel.resistance_touches: 0,
            LevelModel.invalidated_at: None,
            LevelModel.first_touched_at: None,
        })
        db.session.commit()

        # Use finest available data (daily > weekly > monthly) to test ALL levels
        # High threshold during foundation: we want to COUNT touches, not invalidate
        touch_tf = '1d' if '1d' in fetch_tfs else ('1w' if '1w' in fetch_tfs else '1M')
        progress.update('touch_tracking', f'Processing {touch_tf} candles for level touches...')
        tt_result = run_touch_tracking(
            db.session, timeframe=touch_tf, symbol='BTCUSDT',
        )
        progress.update('touch_tracking', f'{touch_tf}: {tt_result["total_touches"]} touches')

        # ── Done ─────────────────────────────────────────────────
        run.status = 'completed'
        run.finished_at = datetime.now(timezone.utc)
        run.metadata_json = _serialize_summary(indicator_summary)
        db.session.commit()

        progress.finish()
        logger.info("Foundation pipeline completed")

    except Exception as e:
        run.status = 'failed'
        run.finished_at = datetime.now(timezone.utc)
        run.error_message = str(e)
        db.session.commit()

        progress.finish(error=str(e))
        logger.error("Foundation pipeline failed: %s", e)
        raise


def _run_volume_profile(db, api_key, api_secret, start_date, end_date, vp_tfs, progress_tracker=None):
    """Fetch 1-min data year-by-year and compute VP levels for all configured TFs."""
    from ..services.indicators import calculate_volume_profile_levels, _add_levels, _tf_label

    if not api_key or not api_secret:
        return {'skipped': 'no API keys'}

    try:
        from binance.client import Client
    except ImportError:
        return {'error': 'python-binance not installed'}

    import pandas as pd
    from datetime import datetime as dt

    period_map = {
        '1d': 'D',
        '1w': 'W',
        '1M': 'ME',
    }

    # Validate TFs upfront
    valid_tfs = [(tf, period_map[tf]) for tf in vp_tfs if tf in period_map]
    if not valid_tfs:
        return {'skipped': 'no valid VP timeframes'}

    # Build year boundaries for chunked fetching
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    years = list(range(start_year, end_year + 1))

    if progress_tracker:
        progress_tracker.update('volume_profile',
            f'Fetching 1-min data from Binance ({len(years)} years: {start_year}-{end_year})...')

    client = Client(api_key, api_secret)
    all_chunks = []
    total_candles = 0

    # Fetch year by year so the user sees progress
    for i, year in enumerate(years):
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'

        # Clip to actual date range
        if year_start < start_date:
            year_start = start_date
        if year_end > end_date:
            year_end = end_date

        if progress_tracker:
            progress_tracker.update('volume_profile',
                f'Downloading {year} ({i+1}/{len(years)})... {total_candles:,} candles so far')

        try:
            klines = client.get_historical_klines(
                'BTCUSDT', '1m', year_start, year_end,
            )
            if klines:
                all_chunks.extend(klines)
                total_candles += len(klines)
                logger.info("VP: %s — %d candles (%d total)", year, len(klines), total_candles)
        except Exception as e:
            logger.error("VP: %s fetch failed: %s", year, e)
            if progress_tracker:
                progress_tracker.update('volume_profile', f'{year}: ERROR — {e}')

    if not all_chunks:
        return {tf: 0 for tf, _ in valid_tfs}

    if progress_tracker:
        progress_tracker.update('volume_profile',
            f'Downloaded {total_candles:,} candles, building DataFrame...')

    df_1min = pd.DataFrame(all_chunks, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore',
    ])
    df_1min['open_time'] = pd.to_datetime(df_1min['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_1min[col] = df_1min[col].astype(float)

    logger.info("VP: total %d 1-min candles loaded", len(df_1min))

    # Process each VP timeframe using the same 1-min data
    vp_summary = {}
    for tf, period_group in valid_tfs:
        try:
            if progress_tracker:
                progress_tracker.update('volume_profile',
                    f'Computing VP {tf} ({period_group} periods) from {total_candles:,} candles...')

            levels = calculate_volume_profile_levels(
                df_1min, _tf_label(tf), period_group,
            )
            added = _add_levels(db, levels, skip_duplicates=True)
            db.commit()
            vp_summary[tf] = added
            logger.info("VP [%s]: %d levels added", tf, added)
        except Exception as e:
            logger.error("VP [%s] failed: %s", tf, e)
            vp_summary[tf] = f'error: {e}'

    return vp_summary


def _serialize_summary(obj):
    """Convert summary dict to JSON-safe format."""
    if isinstance(obj, dict):
        return {str(k): _serialize_summary(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_summary(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
