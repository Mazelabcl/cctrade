"""ML model training service.

Wraps legacy/ml_models/model_trainer.py with SQLAlchemy persistence.
Supports Random Forest, Logistic Regression, XGBoost, LightGBM.
"""
import logging
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sqlalchemy.orm import Session

from ..models import Candle, Feature, MLModel, PipelineRun
from .target_builder import create_fractal_targets

logger = logging.getLogger(__name__)

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Feature columns used for training
FEATURE_COLUMNS = [
    'support_distance_pct', 'support_daily_count', 'support_weekly_count',
    'support_monthly_count', 'support_fib618_count', 'support_naked_count',
    'resistance_distance_pct', 'resistance_daily_count', 'resistance_weekly_count',
    'resistance_monthly_count', 'resistance_fib618_count', 'resistance_naked_count',
    'upper_wick_ratio', 'lower_wick_ratio', 'body_total_ratio', 'body_position_ratio',
    'volume_short_ratio', 'volume_long_ratio',
    'utc_block', 'candles_since_last_up', 'candles_since_last_down',
    'total_support_touches', 'total_resistance_touches',
]


def _build_dataset(db: Session, prediction_horizon: str = 'day') -> tuple:
    """Build feature matrix and target vector from database."""
    targets_df = create_fractal_targets(db, prediction_horizon=prediction_horizon)
    if targets_df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    candle_ids = targets_df['candle_id'].tolist()
    features = (
        db.query(Feature)
        .filter(Feature.candle_id.in_(candle_ids))
        .all()
    )

    if not features:
        return pd.DataFrame(), pd.Series(dtype=int)

    feat_map = {f.candle_id: f for f in features}

    rows = []
    y_vals = []
    for _, row in targets_df.iterrows():
        feat = feat_map.get(row['candle_id'])
        if feat is None:
            continue
        feat_dict = {}
        for col in FEATURE_COLUMNS:
            val = getattr(feat, col, None)
            feat_dict[col] = val if val is not None else 0.0
        rows.append(feat_dict)
        y_vals.append(row['fractal_direction'])

    X = pd.DataFrame(rows)
    y = pd.Series(y_vals, name='fractal_direction')
    return X, y


def _create_model(algorithm: str, random_state: int = 42):
    """Create a model instance by algorithm name."""
    if algorithm == 'random_forest':
        return RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            class_weight='balanced', random_state=random_state, n_jobs=-1,
        )
    elif algorithm == 'logistic_regression':
        return LogisticRegression(
            max_iter=1000, class_weight='balanced',
            random_state=random_state, multi_class='multinomial',
        )
    elif algorithm == 'xgboost' and XGBOOST_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=random_state, n_jobs=-1, eval_metric='mlogloss',
        )
    elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            class_weight='balanced', random_state=random_state, n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown or unavailable algorithm: {algorithm}")


def train_model(
    db: Session,
    algorithm: str = 'random_forest',
    prediction_horizon: str = 'day',
    name: str = None,
    model_dir: str = 'instance/models',
) -> MLModel:
    """Train a model and persist to DB + disk.

    Returns the MLModel record.
    """
    run = PipelineRun(
        pipeline_type='training',
        status='running',
        started_at=datetime.now(timezone.utc),
        metadata_json={'algorithm': algorithm, 'horizon': prediction_horizon},
    )
    db.add(run)
    db.commit()

    start_time = time.time()

    try:
        X, y = _build_dataset(db, prediction_horizon)
        if X.empty:
            raise ValueError("No training data available. Compute features first.")

        # Fill NaN
        X = X.fillna(0)

        # Time-series split: 70% train, 15% val, 15% test
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

        # Train
        model = _create_model(algorithm)
        model.fit(X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        roc = None
        if y_proba is not None:
            try:
                classes = sorted(y.unique())
                y_test_bin = label_binarize(y_test, classes=classes)
                if y_test_bin.shape[1] > 1:
                    roc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
            except Exception:
                pass

        # Save model file
        os.makedirs(model_dir, exist_ok=True)
        version = (
            db.query(db.func.max(MLModel.version))
            .filter_by(algorithm=algorithm, prediction_horizon=prediction_horizon)
            .scalar() or 0
        ) + 1
        model_name = name or f"{algorithm}_{prediction_horizon}"
        file_name = f"{model_name}_v{version}.joblib"
        file_path = os.path.join(model_dir, file_name)

        joblib.dump({'model': model, 'scaler': scaler, 'features': list(X.columns)}, file_path)

        duration = time.time() - start_time

        # Persist metadata
        ml_model = MLModel(
            name=model_name,
            algorithm=algorithm,
            version=version,
            prediction_horizon=prediction_horizon,
            file_path=file_path,
            feature_names=list(X.columns),
            accuracy=acc,
            precision_macro=prec,
            recall_macro=rec,
            f1_macro=f1,
            roc_auc=roc,
            train_rows=len(X_train),
            train_period=f"{X.index[0]}-{X.index[-1]}",
            training_duration_sec=duration,
            created_at=datetime.now(timezone.utc),
        )
        db.add(ml_model)

        run.status = 'completed'
        run.finished_at = datetime.now(timezone.utc)
        run.rows_processed = len(X)
        db.commit()

        logger.info(
            "Model trained: %s v%d — acc=%.3f, f1=%.3f, roc=%.3f",
            model_name, version, acc, f1, roc or 0,
        )
        return ml_model

    except Exception as exc:
        run.status = 'failed'
        run.finished_at = datetime.now(timezone.utc)
        run.error_message = str(exc)
        db.commit()
        raise
