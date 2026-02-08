"""
Model Trainer for Bitcoin Fractal Prediction

This module provides comprehensive training capabilities for multiple ML algorithms
with time-aware validation and hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Advanced ML algorithms
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

class ModelTrainer:
    """
    Comprehensive model training class with support for multiple algorithms.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = None
        
        # Available algorithms
        self.available_algorithms = {
            'random_forest': self._create_random_forest,
            'logistic_regression': self._create_logistic_regression,
        }
        
        if XGBOOST_AVAILABLE:
            self.available_algorithms['xgboost'] = self._create_xgboost
        
        if LIGHTGBM_AVAILABLE:
            self.available_algorithms['lightgbm'] = self._create_lightgbm
    
    def prepare_data(
        self, 
        data: pd.DataFrame, 
        target_column: str = 'fractal_direction',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        scale_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data with time-aware splits and optional scaling.
        
        Args:
            data: Complete dataset with features and target
            target_column: Name of target variable column
            train_ratio: Proportion for training
            val_ratio: Proportion for validation (remainder goes to test)
            scale_features: Whether to scale numerical features
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        
        # Separate features and target
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Remove non-feature columns
        feature_columns = data.columns.tolist()
        exclude_columns = [target_column, 'timestamp'] + [col for col in feature_columns if col.startswith('timestamp')]
        
        X = data.drop(columns=[col for col in exclude_columns if col in data.columns])
        y = data[target_column]
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts().sort_index()}")
        
        # Time-aware split (critical for time series)
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X.iloc[:train_end].copy()
        X_val = X.iloc[train_end:val_end].copy()
        X_test = X.iloc[val_end:].copy()
        
        y_train = y.iloc[:train_end].copy()
        y_val = y.iloc[train_end:val_end].copy()
        y_test = y.iloc[val_end:].copy()
        
        print(f"\nTime-aware data splits:")
        print(f"  Training: {len(X_train):,} samples ({train_ratio*100:.1f}%)")
        print(f"  Validation: {len(X_val):,} samples ({val_ratio*100:.1f}%)")
        print(f"  Test: {len(X_test):,} samples ({(1-train_ratio-val_ratio)*100:.1f}%)")
        
        # Optional feature scaling
        if scale_features:
            # Only scale numerical columns
            numerical_columns = X.select_dtypes(include=[np.number]).columns
            
            self.scaler = StandardScaler()
            X_train[numerical_columns] = self.scaler.fit_transform(X_train[numerical_columns])
            X_val[numerical_columns] = self.scaler.transform(X_val[numerical_columns])
            X_test[numerical_columns] = self.scaler.transform(X_test[numerical_columns])
            
            print(f"  Scaled {len(numerical_columns)} numerical features")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _create_random_forest(self, **kwargs) -> RandomForestClassifier:
        """Create Random Forest classifier with default parameters."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': self.random_state,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        default_params.update(kwargs)
        return RandomForestClassifier(**default_params)
    
    def _create_logistic_regression(self, **kwargs) -> LogisticRegression:
        """Create Logistic Regression classifier with default parameters.""" 
        default_params = {
            'random_state': self.random_state,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'multi_class': 'ovr'
        }
        default_params.update(kwargs)
        return LogisticRegression(**default_params)
    
    def _create_xgboost(self, **kwargs) -> xgb.XGBClassifier:
        """Create XGBoost classifier with default parameters."""
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost is not available. Install with: pip install xgboost")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'enable_categorical': False
        }
        default_params.update(kwargs)
        return xgb.XGBClassifier(**default_params)
    
    def _create_lightgbm(self, **kwargs) -> lgb.LGBMClassifier:
        """Create LightGBM classifier with default parameters."""
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM is not available. Install with: pip install lightgbm")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'class_weight': 'balanced',
            'verbosity': -1
        }
        default_params.update(kwargs)
        return lgb.LGBMClassifier(**default_params)
    
    def train_model(
        self,
        algorithm: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        model_params: Dict = None,
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """
        Train a single model with optional validation.
        
        Args:
            algorithm: Algorithm name ('xgboost', 'random_forest', 'lightgbm', 'logistic_regression')
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            model_params: Custom model parameters
            cv_folds: Cross-validation folds for training
            
        Returns:
            Dictionary with trained model and metrics
        """
        
        if algorithm not in self.available_algorithms:
            raise ValueError(f"Algorithm '{algorithm}' not available. Choose from: {list(self.available_algorithms.keys())}")
        
        print(f"\nTraining {algorithm.upper()} model...")
        
        # Create model
        model_params = model_params or {}
        model = self.available_algorithms[algorithm](**model_params)
        
        # Train model
        start_time = datetime.now()
        
        if algorithm in ['xgboost', 'lightgbm'] and X_val is not None:
            # Use validation set for early stopping
            try:
                eval_set = [(X_val, y_val)]
                if algorithm == 'xgboost':
                    model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        verbose=False
                    )
                else:  # lightgbm
                    model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        callbacks=[lgb.early_stopping(10)]
                    )
            except Exception as e:
                print(f"Early stopping failed: {e}. Using standard training.")
                model.fit(X_train, y_train)
        else:
            # Standard training
            model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Cross-validation score
        if cv_folds > 1:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                cv_model = self.available_algorithms[algorithm](**model_params)
                cv_model.fit(X_cv_train, y_cv_train)
                cv_score = cv_model.score(X_cv_val, y_cv_val)
                cv_scores.append(cv_score)
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
        else:
            cv_mean = cv_std = None
        
        # Calculate metrics
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val) if X_val is not None else None
        
        result = {
            'model': model,
            'algorithm': algorithm,
            'train_score': train_score,
            'val_score': val_score,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'training_time': training_time,
            'model_params': model_params,
            'feature_names': X_train.columns.tolist()
        }
        
        # Store result
        self.models[algorithm] = model
        self.results[algorithm] = result
        
        print(f"  Training completed in {training_time:.2f} seconds")
        print(f"  Training accuracy: {train_score:.4f}")
        if val_score:
            print(f"  Validation accuracy: {val_score:.4f}")
        if cv_mean:
            print(f"  CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return result
    
    def compare_models(
        self,
        algorithms: List[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        cv_folds: int = 3
    ) -> pd.DataFrame:
        """
        Train and compare multiple models.
        
        Args:
            algorithms: List of algorithm names to compare
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            cv_folds: Cross-validation folds
            
        Returns:
            DataFrame with comparison results
        """
        
        print(f"\nComparing {len(algorithms)} algorithms...")
        
        comparison_results = []
        
        for algorithm in algorithms:
            try:
                result = self.train_model(
                    algorithm=algorithm,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    cv_folds=cv_folds
                )
                
                comparison_results.append({
                    'algorithm': algorithm,
                    'train_accuracy': result['train_score'],
                    'val_accuracy': result['val_score'],
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std'],
                    'training_time': result['training_time']
                })
                
            except Exception as e:
                print(f"  Error training {algorithm}: {e}")
                comparison_results.append({
                    'algorithm': algorithm,
                    'train_accuracy': None,
                    'val_accuracy': None,
                    'cv_mean': None,
                    'cv_std': None,
                    'training_time': None
                })
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Sort by validation accuracy (or CV mean if no validation)
        if 'val_accuracy' in comparison_df.columns and comparison_df['val_accuracy'].notna().any():
            comparison_df = comparison_df.sort_values('val_accuracy', ascending=False)
        elif 'cv_mean' in comparison_df.columns and comparison_df['cv_mean'].notna().any():
            comparison_df = comparison_df.sort_values('cv_mean', ascending=False)
        
        print("\nModel Comparison Results:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return comparison_df
    
    def optimize_hyperparameters(
        self,
        algorithm: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        optimization_method: str = 'grid',
        n_trials: int = 100,
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific algorithm.
        
        Args:
            algorithm: Algorithm to optimize
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            optimization_method: 'grid', 'random', or 'optuna'
            n_trials: Number of trials for random/optuna search
            cv_folds: Cross-validation folds
            
        Returns:
            Dictionary with best parameters and model
        """
        
        if algorithm not in self.available_algorithms:
            raise ValueError(f"Algorithm '{algorithm}' not available")
        
        print(f"\nOptimizing hyperparameters for {algorithm.upper()}...")
        
        # Define search spaces
        param_grids = self._get_param_grids()
        
        if algorithm not in param_grids:
            print(f"No parameter grid defined for {algorithm}. Using default parameters.")
            return self.train_model(algorithm, X_train, y_train, X_val, y_val)
        
        param_grid = param_grids[algorithm]
        
        # Create base model
        base_model = self.available_algorithms[algorithm]()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Choose optimization method
        if optimization_method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        elif optimization_method == 'random':
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_trials,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
        elif optimization_method == 'optuna' and OPTUNA_AVAILABLE:
            return self._optimize_with_optuna(algorithm, X_train, y_train, X_val, y_val, n_trials, cv_folds)
        else:
            raise ValueError(f"Optimization method '{optimization_method}' not available")
        
        # Perform search
        start_time = datetime.now()
        search.fit(X_train, y_train)
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Best CV score: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        
        # Validate on validation set if available
        val_score = None
        if X_val is not None:
            val_score = search.best_estimator_.score(X_val, y_val)
            print(f"Validation score: {val_score:.4f}")
        
        result = {
            'model': search.best_estimator_,
            'algorithm': algorithm,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'val_score': val_score,
            'optimization_time': optimization_time,
            'optimization_method': optimization_method,
            'feature_names': X_train.columns.tolist()
        }
        
        # Store optimized model
        self.models[f"{algorithm}_optimized"] = search.best_estimator_
        self.results[f"{algorithm}_optimized"] = result
        
        return result
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """Get parameter grids for hyperparameter optimization."""
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if XGBOOST_AVAILABLE:
            param_grids['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        if LIGHTGBM_AVAILABLE:
            param_grids['lightgbm'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        return param_grids
    
    def _optimize_with_optuna(
        self,
        algorithm: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int,
        cv_folds: int
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            # Define parameter suggestions based on algorithm
            if algorithm == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
            elif algorithm == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
            elif algorithm == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                }
            else:
                params = {}
            
            # Create and train model
            model = self.available_algorithms[algorithm](**params)
            
            # Use cross-validation for evaluation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(X_cv_train, y_cv_train)
                score = model.score(X_cv_val, y_cv_val)
                scores.append(score)
            
            return np.mean(scores)
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best Optuna score: {study.best_value:.4f}")
        print(f"Best Optuna parameters: {study.best_params}")
        
        # Train final model with best parameters
        final_model = self.available_algorithms[algorithm](**study.best_params)
        final_model.fit(X_train, y_train)
        
        val_score = final_model.score(X_val, y_val) if X_val is not None else None
        
        result = {
            'model': final_model,
            'algorithm': algorithm,
            'best_params': study.best_params,
            'best_cv_score': study.best_value,
            'val_score': val_score,
            'optimization_method': 'optuna',
            'n_trials': n_trials,
            'feature_names': X_train.columns.tolist()
        }
        
        return result
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithms."""
        return list(self.available_algorithms.keys())
    
    def get_model(self, algorithm: str):
        """Get trained model by algorithm name."""
        return self.models.get(algorithm)
    
    def get_results(self, algorithm: str = None) -> Dict:
        """Get training results for specific algorithm or all algorithms."""
        if algorithm:
            return self.results.get(algorithm)
        return self.results


if __name__ == "__main__":
    # Example usage
    print("Model Trainer Example")
    print("Available algorithms:", ModelTrainer().get_available_algorithms())