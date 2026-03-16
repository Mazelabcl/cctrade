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
from sqlalchemy import func as sa_func
from sqlalchemy.orm import Session

from ..models import Candle, Feature, MLModel, PipelineRun

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

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Feature columns used for training
FEATURE_COLUMNS = [
    'upper_wick_ratio', 'lower_wick_ratio', 'body_total_ratio', 'body_position_ratio',
    'volume_short_ratio', 'volume_long_ratio',
    'utc_block', 'candles_since_last_up', 'candles_since_last_down',
    'support_distance_pct', 'resistance_distance_pct',
    'atr_14', 'momentum_12',
    'support_confluence_score', 'resistance_confluence_score',
    'support_liquidity_consumed', 'resistance_liquidity_consumed',
]


def _build_dataset(db: Session, target_column: str = 'target_bullish') -> tuple:
    """Build feature matrix and target vector from database.

    target_column: 'target_bullish' or 'target_bearish'
    """
    features = (
        db.query(Feature)
        .filter(getattr(Feature, target_column).isnot(None))
        .order_by(Feature.candle_id)
        .all()
    )

    if not features:
        return pd.DataFrame(), pd.Series(dtype=int)

    rows = []
    y_vals = []
    for f in features:
        feat_dict = {}
        for col in FEATURE_COLUMNS:
            val = getattr(f, col, None)
            feat_dict[col] = val if val is not None else 0.0
        rows.append(feat_dict)
        y_vals.append(getattr(f, target_column))

    X = pd.DataFrame(rows)
    y = pd.Series(y_vals, name=target_column)
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
    target_column: str = 'target_bullish',
    name: str = None,
    model_dir: str = 'instance/models',
) -> MLModel:
    """Train a model and persist to DB + disk.

    target_column: 'target_bullish' or 'target_bearish'
    Returns the MLModel record.
    """
    run = PipelineRun(
        pipeline_type='training',
        status='running',
        started_at=datetime.now(timezone.utc),
        metadata_json={'algorithm': algorithm, 'target': target_column},
    )
    db.add(run)
    db.commit()

    start_time = time.time()

    try:
        X, y = _build_dataset(db, target_column)
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
            db.query(sa_func.max(MLModel.version))
            .filter_by(algorithm=algorithm, prediction_horizon=target_column)
            .scalar() or 0
        ) + 1
        model_name = name or f"{algorithm}_{target_column}"
        file_name = f"{model_name}_v{version}.joblib"
        file_path = os.path.join(model_dir, file_name)

        joblib.dump({'model': model, 'scaler': scaler, 'features': list(X.columns)}, file_path)

        duration = time.time() - start_time

        # Persist metadata
        ml_model = MLModel(
            name=model_name,
            algorithm=algorithm,
            version=version,
            prediction_horizon=target_column,
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


def train_enhanced_model(
    db: Session,
    target_column: str = 'target_bullish',
    name: str = None,
    model_dir: str = 'instance/models',
    n_trials: int = 50,
    use_smote: bool = True,
) -> MLModel:
    """Train an enhanced LightGBM model with Optuna hyperparameter tuning.

    Uses TimeSeriesSplit cross-validation, optional SMOTE oversampling for
    minority fractal classes, and a held-out test set for final evaluation.

    target_column: 'target_bullish' or 'target_bearish'
    Returns the MLModel record.
    """
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is required for train_enhanced_model")
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("optuna is required for train_enhanced_model")

    algorithm = 'lightgbm_enhanced'

    run = PipelineRun(
        pipeline_type='training',
        status='running',
        started_at=datetime.now(timezone.utc),
        metadata_json={'algorithm': algorithm, 'target': target_column,
                       'n_trials': n_trials, 'smote': use_smote},
    )
    db.add(run)
    db.commit()

    start_time = time.time()

    try:
        X, y = _build_dataset(db, target_column)
        if X.empty:
            raise ValueError("No training data available. Compute features first.")

        X = X.fillna(0)

        # Hold out last 15% as final test set
        n = len(X)
        split_idx = int(n * 0.85)
        X_dev, y_dev = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_dev_scaled = pd.DataFrame(scaler.fit_transform(X_dev), columns=X.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

        # Apply SMOTE to dev set if requested
        X_train_final = X_dev_scaled
        y_train_final = y_dev
        if use_smote and SMOTE_AVAILABLE:
            try:
                sm = SMOTE(random_state=42)
                X_train_final, y_train_final = sm.fit_resample(X_dev_scaled, y_dev)
            except ValueError:
                logger.warning("SMOTE failed (possibly too few minority samples), using original data")

        # Optuna objective
        tscv = TimeSeriesSplit(n_splits=5)

        def _objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            scores = []
            for train_idx, val_idx in tscv.split(X_dev_scaled):
                X_tr = X_dev_scaled.iloc[train_idx]
                y_tr = y_dev.iloc[train_idx]
                X_va = X_dev_scaled.iloc[val_idx]
                y_va = y_dev.iloc[val_idx]

                if use_smote and SMOTE_AVAILABLE:
                    try:
                        sm = SMOTE(random_state=42)
                        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                    except ValueError:
                        pass

                mdl = lgb.LGBMClassifier(
                    **params, class_weight='balanced',
                    random_state=42, n_jobs=-1, verbose=-1,
                )
                mdl.fit(X_tr, y_tr)
                y_pred = mdl.predict(X_va)
                scores.append(f1_score(y_va, y_pred, average='macro', zero_division=0))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        logger.info("Optuna best params: %s (f1=%.4f)", best_params, study.best_value)

        # Train final model with best params on full dev set
        model = lgb.LGBMClassifier(
            **best_params, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train_final, y_train_final)

        # Evaluate on held-out test set
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        roc = None
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
            db.query(sa_func.max(MLModel.version))
            .filter_by(algorithm=algorithm, prediction_horizon=target_column)
            .scalar() or 0
        ) + 1
        model_name = name or f"{algorithm}_{target_column}"
        file_name = f"{model_name}_v{version}.joblib"
        file_path = os.path.join(model_dir, file_name)

        joblib.dump({
            'model': model, 'scaler': scaler, 'features': list(X.columns),
            'best_params': best_params,
        }, file_path)

        duration = time.time() - start_time

        ml_model = MLModel(
            name=model_name,
            algorithm=algorithm,
            version=version,
            prediction_horizon=target_column,
            file_path=file_path,
            feature_names=list(X.columns),
            accuracy=acc,
            precision_macro=prec,
            recall_macro=rec,
            f1_macro=f1,
            roc_auc=roc,
            train_rows=len(X_dev),
            train_period=f"{X.index[0]}-{X.index[-1]}",
            hyperparameters=best_params,
            training_duration_sec=duration,
            created_at=datetime.now(timezone.utc),
        )
        db.add(ml_model)

        run.status = 'completed'
        run.finished_at = datetime.now(timezone.utc)
        run.rows_processed = len(X)
        db.commit()

        logger.info(
            "Enhanced model trained: %s v%d — acc=%.3f, f1=%.3f, roc=%.3f (%.1fs)",
            model_name, version, acc, f1, roc or 0, duration,
        )
        return ml_model

    except Exception as exc:
        run.status = 'failed'
        run.finished_at = datetime.now(timezone.utc)
        run.error_message = str(exc)
        db.commit()
        raise
