"""APScheduler configuration for automated pipeline execution.

Schedules hourly pipeline: fetch → indicators → features → predict.
"""
import logging

from flask import Flask

logger = logging.getLogger(__name__)


def init_scheduler(app: Flask):
    """Register scheduled jobs with the app's APScheduler instance."""
    from ..extensions import scheduler

    if app.config.get('TESTING') or not app.config.get('SCHEDULER_ENABLED', False):
        logger.info("Scheduler disabled (TESTING or SCHEDULER_ENABLED=False)")
        return

    @scheduler.task('interval', id='hourly_pipeline', hours=1,
                    misfire_grace_time=300)
    def hourly_pipeline():
        """Full pipeline: indicators → features → predictions."""
        from .pipeline_runner import run_full_pipeline
        with app.app_context():
            run_full_pipeline(app)

    scheduler.start()
    logger.info("Scheduler started with hourly pipeline job")
