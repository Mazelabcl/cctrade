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


# ---------------------------------------------------------------------------
# Foundation config reader (shared by all sub-functions)
# ---------------------------------------------------------------------------

def _read_foundation_config():
    """Read foundation pipeline config from DB settings."""
    from ..models.setting import get_setting

    api_key = get_setting('binance_api_key', '')
    api_secret = get_setting('binance_api_secret', '')
    start_date = get_setting('data_start_date', '2020-01-01')
    end_date = get_setting('data_end_date', '2025-12-31')
    fetch_tfs = [t for t in get_setting('fetch_timeframes', '1d,1w,1M').split(',') if t]
    htf_tfs = [t for t in get_setting('htf_timeframes', '1d,1w,1M').split(',') if t]
    fractal_tfs = [t for t in get_setting('fractal_timeframes', '1d,1w,1M').split(',') if t]
    fib_tfs = [t for t in get_setting('fibonacci_timeframes', '1d,1w').split(',') if t]
    vp_tfs = [t for t in get_setting('vp_timeframes', '1d,1w,1M').split(',') if t]

    return {
        'api_key': api_key, 'api_secret': api_secret,
        'start_date': start_date, 'end_date': end_date,
        'fetch_tfs': fetch_tfs, 'htf_tfs': htf_tfs,
        'fractal_tfs': fractal_tfs, 'fib_tfs': fib_tfs, 'vp_tfs': vp_tfs,
    }


# ---------------------------------------------------------------------------
# Foundation sub-functions (individually callable)
# ---------------------------------------------------------------------------

def run_foundation_fetch(app: Flask):
    """Step 1: Fetch candles from Binance for configured timeframes."""
    from ..extensions import db
    from ..services.data_fetcher import fetch_candles
    from ..services import progress

    progress.start()

    try:
        with app.app_context():
            cfg = _read_foundation_config()
            progress.update('config', f'Fetch TFs: {cfg["fetch_tfs"]}, Range: {cfg["start_date"]} to {cfg["end_date"]}')

            result = {}
            if cfg['api_key'] and cfg['api_secret']:
                for tf in cfg['fetch_tfs']:
                    progress.update('fetch', f'Downloading {tf} candles from Binance...')
                    try:
                        count = fetch_candles(
                            db.session, symbol='BTCUSDT', interval=tf,
                            start_str=cfg['start_date'], end_str=cfg['end_date'],
                            api_key=cfg['api_key'], api_secret=cfg['api_secret'],
                        )
                        result[tf] = count
                        progress.update('fetch', f'{tf}: {count} new candles')
                    except Exception as e:
                        result[tf] = f'ERROR: {e}'
                        progress.update('fetch', f'{tf}: ERROR - {e}')
                        logger.error("Foundation fetch [%s] failed: %s", tf, e)
            else:
                progress.update('fetch', 'Skipped - no API keys configured')

            progress.set_result(result)
            progress.finish()
            logger.info("Foundation fetch completed: %s", result)

    except Exception as e:
        progress.finish(error=str(e))
        logger.error("Foundation fetch failed: %s", e)
        raise


def run_foundation_levels(app: Flask):
    """Step 2: Run level detection (fractal, HTF, Fibonacci, VP)."""
    from ..extensions import db
    from ..services.indicators import run_indicators_multi
    from ..services import progress

    progress.start()

    try:
        with app.app_context():
            cfg = _read_foundation_config()
            progress.update('config', f'HTF: {cfg["htf_tfs"]}, Frac: {cfg["fractal_tfs"]}, Fib: {cfg["fib_tfs"]}, VP: {cfg["vp_tfs"]}')

            progress.update('indicators', 'Running fractal detection + level calculation...')
            indicator_summary = run_indicators_multi(
                db.session, symbol='BTCUSDT',
                htf_timeframes=cfg['htf_tfs'],
                fractal_timeframes=cfg['fractal_tfs'],
                fib_timeframes=cfg['fib_tfs'],
                vp_timeframes=cfg['vp_tfs'],
            )

            # Report per-type results
            for level_type, tf_counts in indicator_summary.items():
                if tf_counts:
                    details = ', '.join(f'{tf}={cnt}' for tf, cnt in tf_counts.items())
                    progress.update('indicators', f'{level_type}: {details}')

            # Volume Profile (needs 1-min data from Binance)
            if cfg['vp_tfs'] and cfg['api_key'] and cfg['api_secret']:
                progress.update('volume_profile', f'Computing VP for {cfg["vp_tfs"]} (fetching 1-min data)...')
                try:
                    vp_summary = _run_volume_profile(
                        db.session, cfg['api_key'], cfg['api_secret'],
                        cfg['start_date'], cfg['end_date'], cfg['vp_tfs'], progress,
                    )
                    indicator_summary['vp'] = vp_summary
                    for tf, vp_result in vp_summary.items():
                        progress.update('volume_profile', f'{tf}: {vp_result} levels')
                except Exception as e:
                    progress.update('volume_profile', f'ERROR: {e}')
                    logger.error("Foundation VP failed: %s", e)

            progress.set_result(_serialize_summary(indicator_summary))
            progress.finish()
            logger.info("Foundation levels completed")

    except Exception as e:
        progress.finish(error=str(e))
        logger.error("Foundation levels failed: %s", e)
        raise


def run_foundation_touches(app: Flask):
    """Step 3: Reset and re-run touch tracking on all levels."""
    from ..extensions import db
    from ..models import Level as LevelModel
    from ..services.level_tracker import run_touch_tracking
    from ..services import progress

    progress.start()

    try:
        with app.app_context():
            cfg = _read_foundation_config()

            progress.update('touch_tracking', 'Resetting touch counts for all levels...')
            db.session.query(LevelModel).update({
                LevelModel.support_touches: 0,
                LevelModel.resistance_touches: 0,
                LevelModel.invalidated_at: None,
                LevelModel.first_touched_at: None,
            })
            db.session.commit()

            touch_tf = '1d' if '1d' in cfg['fetch_tfs'] else ('1w' if '1w' in cfg['fetch_tfs'] else '1M')
            progress.update('touch_tracking', f'Processing {touch_tf} candles for level touches...')
            tt_result = run_touch_tracking(
                db.session, timeframe=touch_tf, symbol='BTCUSDT',
            )

            total = tt_result['total_touches']
            # Count naked vs touched
            naked_count = db.session.query(LevelModel).filter(
                LevelModel.support_touches == 0,
                LevelModel.invalidated_at.is_(None),
            ).count()
            touched_count = db.session.query(LevelModel).filter(
                LevelModel.support_touches > 0,
            ).count()

            result = {
                'total_touches': total,
                'naked': naked_count,
                'touched': touched_count,
                'timeframe_used': touch_tf,
            }
            progress.update('touch_tracking', f'{touch_tf}: {total} touches ({naked_count} naked, {touched_count} touched)')
            progress.set_result(result)
            progress.finish()
            logger.info("Foundation touches completed: %s", result)

    except Exception as e:
        progress.finish(error=str(e))
        logger.error("Foundation touches failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# Full foundation pipeline (wrapper calling all 3 steps)
# ---------------------------------------------------------------------------

def run_foundation_pipeline(app: Flask):
    """Run the complete foundation pipeline using configured settings.

    Steps:
    1. Fetch candles for all configured timeframes
    2. Run level detection per type per configured TF (HTF, Fractal, Fib, VP)
    3. Touch tracking (runs on ALL levels including VP)
    """
    from ..extensions import db
    from ..models.setting import get_setting
    from ..models import PipelineRun, Level as LevelModel
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
        cfg = _read_foundation_config()
        progress.update('config', f'Fetch: {cfg["fetch_tfs"]}, HTF: {cfg["htf_tfs"]}, Frac: {cfg["fractal_tfs"]}, Fib: {cfg["fib_tfs"]}, VP: {cfg["vp_tfs"]}')

        # ── Step 1: Fetch candles ────────────────────────────────
        if cfg['api_key'] and cfg['api_secret']:
            for tf in cfg['fetch_tfs']:
                progress.update('fetch', f'Downloading {tf} candles from Binance...')
                try:
                    count = fetch_candles(
                        db.session, symbol='BTCUSDT', interval=tf,
                        start_str=cfg['start_date'], end_str=cfg['end_date'],
                        api_key=cfg['api_key'], api_secret=cfg['api_secret'],
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
            htf_timeframes=cfg['htf_tfs'],
            fractal_timeframes=cfg['fractal_tfs'],
            fib_timeframes=cfg['fib_tfs'],
            vp_timeframes=cfg['vp_tfs'],
        )
        for level_type, tf_counts in indicator_summary.items():
            if tf_counts:
                details = ', '.join(f'{tf}={cnt}' for tf, cnt in tf_counts.items())
                progress.update('indicators', f'{level_type}: {details}')

        # ── Step 3: Volume Profile ─────────────────────────────
        if cfg['vp_tfs'] and cfg['api_key'] and cfg['api_secret']:
            progress.update('volume_profile', f'Computing VP for {cfg["vp_tfs"]} (fetching 1-min data)...')
            try:
                vp_summary = _run_volume_profile(db.session, cfg['api_key'], cfg['api_secret'],
                                                  cfg['start_date'], cfg['end_date'], cfg['vp_tfs'], progress)
                for tf, result in vp_summary.items():
                    progress.update('volume_profile', f'{tf}: {result} levels')
            except Exception as e:
                progress.update('volume_profile', f'ERROR: {e}')
                logger.error("Foundation VP failed: %s", e)

        # ── Step 4: Touch tracking ───────────────────────────────
        progress.update('touch_tracking', 'Resetting touch counts for all levels...')
        db.session.query(LevelModel).update({
            LevelModel.support_touches: 0,
            LevelModel.resistance_touches: 0,
            LevelModel.invalidated_at: None,
            LevelModel.first_touched_at: None,
        })
        db.session.commit()

        touch_tf = '1d' if '1d' in cfg['fetch_tfs'] else ('1w' if '1w' in cfg['fetch_tfs'] else '1M')
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
