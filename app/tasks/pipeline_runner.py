"""Orchestrates the full automated pipeline.

Runs: indicators → features → predictions → accuracy backfill.
Each step is independent and logs its own PipelineRun.
"""
import logging
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
