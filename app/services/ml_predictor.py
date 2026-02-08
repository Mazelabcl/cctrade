"""Prediction interface for trained models.

Loads a trained model from disk and makes predictions on new candles.
"""
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..models import Candle, Feature, MLModel, Prediction

logger = logging.getLogger(__name__)


def predict(db: Session, model_id: int = None,
            candle_ids: list = None) -> list[dict]:
    """Run predictions using a trained model.

    If model_id is None, uses the active model.
    If candle_ids is None, predicts on all candles with features but no predictions.
    """
    # Get model
    if model_id:
        model_rec = db.query(MLModel).get(model_id)
    else:
        model_rec = db.query(MLModel).filter_by(is_active=True).first()

    if not model_rec:
        raise ValueError("No model found. Train a model first.")

    # Load model from disk
    bundle = joblib.load(model_rec.file_path)
    model = bundle['model']
    scaler = bundle['scaler']
    feature_names = bundle['features']

    # Get features to predict on
    if candle_ids:
        features = db.query(Feature).filter(Feature.candle_id.in_(candle_ids)).all()
    else:
        # Find candles with features but no predictions for this model
        existing_preds = (
            db.query(Prediction.candle_id)
            .filter_by(model_id=model_rec.id)
            .subquery()
        )
        features = (
            db.query(Feature)
            .filter(~Feature.candle_id.in_(existing_preds))
            .order_by(Feature.candle_id)
            .limit(10000)
            .all()
        )

    if not features:
        return []

    # Build feature matrix
    rows = []
    feat_ids = []
    for f in features:
        row = {}
        for col in feature_names:
            val = getattr(f, col, None)
            row[col] = val if val is not None else 0.0
        rows.append(row)
        feat_ids.append(f.candle_id)

    X = pd.DataFrame(rows).fillna(0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

    # Predict
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None

    results = []
    for idx, candle_id in enumerate(feat_ids):
        pred = Prediction(
            model_id=model_rec.id,
            candle_id=candle_id,
            predicted_class=int(y_pred[idx]),
            created_at=datetime.now(timezone.utc),
        )
        if y_proba is not None:
            probs = y_proba[idx]
            pred.prob_no_fractal = float(probs[0]) if len(probs) > 0 else None
            pred.prob_bullish = float(probs[1]) if len(probs) > 1 else None
            pred.prob_bearish = float(probs[2]) if len(probs) > 2 else None
            pred.confidence = float(max(probs))

        db.add(pred)
        results.append(pred.to_dict())

    db.commit()
    logger.info("Generated %d predictions with model %s v%d",
                len(results), model_rec.name, model_rec.version)
    return results
