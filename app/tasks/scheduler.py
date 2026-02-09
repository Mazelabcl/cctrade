"""APScheduler configuration for automated pipeline execution.

Schedules hourly pipeline: fetch -> indicators -> features -> predict.
Manages dynamic live data sync job.
"""
import logging

from flask import Flask

logger = logging.getLogger(__name__)

LIVE_SYNC_JOB_ID = 'live_data_sync'


def init_scheduler(app: Flask):
    """Register scheduled jobs with the app's APScheduler instance."""
    from ..extensions import scheduler

    if app.config.get('TESTING'):
        logger.info("Scheduler disabled (TESTING)")
        return

    @scheduler.task('interval', id='hourly_pipeline', hours=1,
                    misfire_grace_time=300)
    def hourly_pipeline():
        """Full pipeline: indicators -> features -> predictions."""
        from .pipeline_runner import run_full_pipeline
        with app.app_context():
            run_full_pipeline(app)

    scheduler.start()
    logger.info("Scheduler started with hourly pipeline job")

    # Start live sync if it was enabled in DB settings
    with app.app_context():
        _maybe_start_live_sync(app)


def _maybe_start_live_sync(app: Flask):
    """Check DB settings and start live sync if enabled."""
    try:
        from ..models.setting import get_setting
        enabled = get_setting('live_sync_enabled', 'false') == 'true'
        if enabled:
            interval = int(get_setting('live_sync_interval_minutes', '5'))
            start_live_sync(app, interval)
    except Exception as e:
        logger.warning("Could not restore live sync state: %s", e)


def start_live_sync(app: Flask, interval_minutes: int = 5):
    """Start or restart the live data sync job."""
    from ..extensions import scheduler

    # Remove existing job if present
    stop_live_sync()

    def _sync_job():
        with app.app_context():
            from .data_sync import sync_candle_data
            from ..models.setting import get_setting
            from ..views.api import publish_sse
            try:
                publish_sse('pipeline', {'status': 'running', 'type': 'live_sync'})
                count = sync_candle_data()
                publish_sse('pipeline', {'status': 'completed', 'type': 'live_sync',
                                         'new_candles': count})
                logger.info("Live sync fetched %d candles", count)

                # Optionally run full pipeline after fetch
                run_pipeline = get_setting('run_full_pipeline_on_sync', 'false') == 'true'
                if run_pipeline and count > 0:
                    logger.info("Running full pipeline after live sync...")
                    from .pipeline_runner import run_full_pipeline
                    run_full_pipeline(app)
            except Exception as e:
                publish_sse('pipeline', {'status': 'failed', 'type': 'live_sync',
                                         'error': str(e)})
                logger.error("Live sync failed: %s", e)

    scheduler.add_job(
        id=LIVE_SYNC_JOB_ID,
        func=_sync_job,
        trigger='interval',
        minutes=interval_minutes,
        misfire_grace_time=60,
        replace_existing=True,
    )
    logger.info("Live sync started (every %d minutes)", interval_minutes)


def stop_live_sync():
    """Stop the live data sync job if running."""
    from ..extensions import scheduler
    try:
        scheduler.remove_job(LIVE_SYNC_JOB_ID)
        logger.info("Live sync stopped")
    except Exception:
        pass  # Job didn't exist


def is_live_sync_active() -> bool:
    """Check if the live sync job is currently scheduled."""
    from ..extensions import scheduler
    try:
        job = scheduler.get_job(LIVE_SYNC_JOB_ID)
        return job is not None
    except Exception:
        return False
