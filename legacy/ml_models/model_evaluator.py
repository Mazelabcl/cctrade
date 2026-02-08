"""
Model Evaluator for Bitcoin Fractal Prediction

This module provides comprehensive evaluation capabilities including metrics calculation,
visualization, feature importance analysis, and business-specific evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Core evaluation libraries
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, log_loss,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import TimeSeriesSplit

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Plotting libraries not available. Install with: pip install matplotlib seaborn")

# Advanced evaluation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

class ModelEvaluator:
    """
    Comprehensive model evaluation class with metrics, visualization, and business analysis.
    """
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['no_fractal', 'bullish_fractal', 'bearish_fractal']
        self.evaluation_results = {}
        
    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame = None,
        y_train: pd.Series = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            X_train: Training features (for comparison)
            y_train: Training target (for comparison)
            model_name: Name for the model
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        
        print(f"\nEvaluating {model_name.upper()} model...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        # Macro averages
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        
        # Multi-class ROC AUC
        try:
            if y_pred_proba is not None and len(np.unique(y_test)) > 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            else:
                roc_auc = None
        except:
            roc_auc = None
        
        # Log loss
        try:
            if y_pred_proba is not None:
                logloss = log_loss(y_test, y_pred_proba)
            else:
                logloss = None
        except:
            logloss = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(precision):
                class_metrics[class_name] = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1_score': f1[i],
                    'support': support[i]
                }
        
        # Training metrics (if provided)
        train_metrics = None
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_metrics = {'accuracy': train_accuracy}
        
        # Business metrics
        business_metrics = self._calculate_business_metrics(y_test, y_pred, y_pred_proba)
        
        # Feature importance (if available)
        feature_importance = self._get_feature_importance(model, X_test.columns)
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'roc_auc_macro': roc_auc,
            'log_loss': logloss,
            'confusion_matrix': cm,
            'class_metrics': class_metrics,
            'train_metrics': train_metrics,
            'business_metrics': business_metrics,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'test_size': len(y_test)
        }
        
        self.evaluation_results[model_name] = results
        
        # Print summary
        self._print_evaluation_summary(results)
        
        return results
    
    def _calculate_business_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate business-specific metrics for fractal prediction.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary with business metrics
        """
        
        business_metrics = {}
        
        # Fractal detection rate (recall for fractal classes)
        fractal_mask = (y_true == 1) | (y_true == 2)  # bullish or bearish fractals
        fractal_predicted_mask = (y_pred == 1) | (y_pred == 2)
        
        if fractal_mask.sum() > 0:
            fractal_detection_rate = (fractal_mask & fractal_predicted_mask).sum() / fractal_mask.sum()
            business_metrics['fractal_detection_rate'] = fractal_detection_rate
        
        # False signal rate (predicted fractal but actually no fractal)
        no_fractal_mask = y_true == 0
        if fractal_predicted_mask.sum() > 0:
            false_signal_rate = (no_fractal_mask & fractal_predicted_mask).sum() / fractal_predicted_mask.sum()
            business_metrics['false_signal_rate'] = false_signal_rate
        
        # Fractal precision (when we predict fractal, how often are we right?)
        if fractal_predicted_mask.sum() > 0:
            fractal_precision = (fractal_mask & fractal_predicted_mask).sum() / fractal_predicted_mask.sum()
            business_metrics['fractal_precision'] = fractal_precision
        
        # Direction accuracy (when there is a fractal, do we get the direction right?)
        actual_fractals = y_true[fractal_mask]
        predicted_fractals = y_pred[fractal_mask]
        if len(actual_fractals) > 0:
            direction_accuracy = (actual_fractals == predicted_fractals).mean()
            business_metrics['direction_accuracy'] = direction_accuracy
        
        # Confidence analysis (if probabilities available)
        if y_pred_proba is not None:
            # Average confidence for correct predictions
            correct_mask = y_true == y_pred
            if correct_mask.sum() > 0:
                avg_confidence_correct = np.mean(np.max(y_pred_proba[correct_mask], axis=1))
                business_metrics['avg_confidence_correct'] = avg_confidence_correct
            
            # Average confidence for fractal predictions
            if fractal_predicted_mask.sum() > 0:
                fractal_confidences = np.max(y_pred_proba[fractal_predicted_mask], axis=1)
                business_metrics['avg_confidence_fractals'] = np.mean(fractal_confidences)
        
        return business_metrics
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature importance scores
        """
        
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (RF, XGBoost, LightGBM)
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
                
            elif hasattr(model, 'coef_'):
                # Linear models (Logistic Regression)
                if len(model.coef_.shape) == 1:
                    # Binary classification
                    importances = np.abs(model.coef_)
                else:
                    # Multi-class: use mean of absolute coefficients
                    importances = np.mean(np.abs(model.coef_), axis=0)
                importance_dict = dict(zip(feature_names, importances))
                
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """Print evaluation summary to console."""
        
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY: {results['model_name'].upper()}")
        print(f"{'='*60}")
        
        print(f"Overall Metrics:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision (macro): {results['precision_macro']:.4f}")
        print(f"  Recall (macro): {results['recall_macro']:.4f}")
        print(f"  F1-Score (macro): {results['f1_macro']:.4f}")
        
        if results['roc_auc_macro']:
            print(f"  ROC AUC (macro): {results['roc_auc_macro']:.4f}")
        if results['log_loss']:
            print(f"  Log Loss: {results['log_loss']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for class_name, metrics in results['class_metrics'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1_score']:.4f}")
            print(f"    Support: {metrics['support']}")
        
        print(f"\nBusiness Metrics:")
        bm = results['business_metrics']
        for metric_name, value in bm.items():
            print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")
        
        if results['train_metrics']:
            print(f"\nTraining Comparison:")
            print(f"  Train Accuracy: {results['train_metrics']['accuracy']:.4f}")
            print(f"  Test Accuracy: {results['accuracy']:.4f}")
            overfitting = results['train_metrics']['accuracy'] - results['accuracy']
            print(f"  Overfitting: {overfitting:.4f}")
        
        print(f"\nTop 10 Important Features:")
        for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:10]):
            print(f"  {i+1:2d}. {feature}: {importance:.4f}")
    
    def compare_models(self, model_names: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple evaluated models.
        
        Args:
            model_names: List of model names to compare (None for all)
            
        Returns:
            DataFrame with comparison results
        """
        
        if not self.evaluation_results:
            print("No evaluation results available. Evaluate some models first.")
            return pd.DataFrame()
        
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        comparison_data = []
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                print(f"Warning: {model_name} not found in evaluation results")
                continue
                
            results = self.evaluation_results[model_name]
            
            row = {
                'model': model_name,
                'accuracy': results['accuracy'],
                'precision_macro': results['precision_macro'],
                'recall_macro': results['recall_macro'],
                'f1_macro': results['f1_macro'],
                'roc_auc_macro': results.get('roc_auc_macro'),
                'log_loss': results.get('log_loss'),
                'test_size': results['test_size']
            }
            
            # Add business metrics
            bm = results['business_metrics']
            row.update({f"biz_{k}": v for k, v in bm.items()})
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by F1 macro score
        if 'f1_macro' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('f1_macro', ascending=False)
        
        print("\nModel Comparison Results:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return comparison_df
    
    def plot_confusion_matrix(
        self, 
        model_name: str, 
        save_path: str = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot confusion matrix for a specific model.
        
        Args:
            model_name: Name of the model to plot
            save_path: Path to save the plot (optional)
            figsize: Figure size tuple
        """
        
        if not PLOTTING_AVAILABLE:
            print("Plotting not available. Install matplotlib and seaborn.")
            return
        
        if model_name not in self.evaluation_results:
            print(f"Model {model_name} not found in evaluation results")
            return
        
        results = self.evaluation_results[model_name]
        cm = results['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'Confusion Matrix - {model_name.upper()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        model_name: str,
        top_n: int = 20,
        save_path: str = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot feature importance for a specific model.
        
        Args:
            model_name: Name of the model to plot
            top_n: Number of top features to plot
            save_path: Path to save the plot (optional)
            figsize: Figure size tuple
        """
        
        if not PLOTTING_AVAILABLE:
            print("Plotting not available. Install matplotlib and seaborn.")
            return
        
        if model_name not in self.evaluation_results:
            print(f"Model {model_name} not found in evaluation results")
            return
        
        results = self.evaluation_results[model_name]
        importance = results['feature_importance']
        
        if not importance:
            print(f"No feature importance available for {model_name}")
            return
        
        # Get top N features
        top_features = list(importance.items())[:top_n]
        features, importances = zip(*top_features)
        
        plt.figure(figsize=figsize)
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, importances, color='skyblue', alpha=0.8)
        plt.yticks(y_pos, features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name.upper()}')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(
        self,
        metric: str = 'f1_macro',
        save_path: str = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot comparison of multiple models.
        
        Args:
            metric: Metric to compare ('accuracy', 'f1_macro', 'precision_macro', etc.)
            save_path: Path to save the plot (optional)
            figsize: Figure size tuple
        """
        
        if not PLOTTING_AVAILABLE:
            print("Plotting not available. Install matplotlib and seaborn.")
            return
        
        comparison_df = self.compare_models()
        if comparison_df.empty:
            return
        
        if metric not in comparison_df.columns:
            print(f"Metric '{metric}' not available. Available metrics: {list(comparison_df.columns)}")
            return
        
        plt.figure(figsize=figsize)
        
        # Create bar plot
        bars = plt.bar(comparison_df['model'], comparison_df[metric], color='lightcoral', alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, comparison_df[metric]):
            if pd.notna(value):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Model')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def cross_validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation.
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Target
            cv_folds: Number of CV folds
            model_name: Name for the model
            
        Returns:
            Cross-validation results
        """
        
        print(f"\nPerforming {cv_folds}-fold time-series cross-validation for {model_name}...")
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"  Fold {fold + 1}/{cv_folds}...")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone and train model for this fold
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Evaluate fold
            fold_results_dict = self.evaluate_model(
                fold_model, X_val_fold, y_val_fold, 
                model_name=f"{model_name}_fold_{fold+1}"
            )
            
            cv_scores.append(fold_results_dict['accuracy'])
            fold_results.append(fold_results_dict)
        
        # Calculate CV statistics
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        print(f"\nCross-Validation Results for {model_name}:")
        print(f"  Mean Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"  Individual Folds: {[f'{score:.4f}' for score in cv_scores]}")
        
        cv_results = {
            'model_name': model_name,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_scores': cv_scores,
            'fold_results': fold_results,
            'n_folds': cv_folds
        }
        
        return cv_results
    
    def analyze_prediction_confidence(
        self,
        model_name: str,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Analyze prediction confidence and its relationship to accuracy.
        
        Args:
            model_name: Name of the model to analyze
            confidence_threshold: Threshold for high confidence predictions
            
        Returns:
            Confidence analysis results
        """
        
        if model_name not in self.evaluation_results:
            print(f"Model {model_name} not found in evaluation results")
            return {}
        
        results = self.evaluation_results[model_name]
        y_pred_proba = results['prediction_probabilities']
        
        if y_pred_proba is None:
            print(f"No prediction probabilities available for {model_name}")
            return {}
        
        y_true = results.get('y_true')  # Would need to store this in evaluate_model
        y_pred = results['predictions']
        
        # Calculate confidence (max probability for each prediction)
        confidences = np.max(y_pred_proba, axis=1)
        
        # High confidence predictions
        high_conf_mask = confidences >= confidence_threshold
        high_conf_count = high_conf_mask.sum()
        
        analysis = {
            'total_predictions': len(confidences),
            'high_confidence_count': high_conf_count,
            'high_confidence_rate': high_conf_count / len(confidences),
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
        
        print(f"\nConfidence Analysis for {model_name}:")
        print(f"  Total Predictions: {analysis['total_predictions']:,}")
        print(f"  High Confidence (≥{confidence_threshold}): {analysis['high_confidence_count']:,} ({analysis['high_confidence_rate']:.1%})")
        print(f"  Average Confidence: {analysis['avg_confidence']:.4f}")
        print(f"  Confidence Range: {analysis['min_confidence']:.4f} - {analysis['max_confidence']:.4f}")
        
        return analysis
    
    def export_evaluation_results(self, save_path: str, model_names: List[str] = None):
        """
        Export evaluation results to CSV file.
        
        Args:
            save_path: Path to save the results
            model_names: List of model names to export (None for all)
        """
        
        comparison_df = self.compare_models(model_names)
        
        if not comparison_df.empty:
            comparison_df.to_csv(save_path, index=False)
            print(f"Evaluation results exported to {save_path}")
        else:
            print("No evaluation results to export")
    
    def get_evaluation_summary(self, model_name: str) -> str:
        """
        Get a text summary of evaluation results.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Formatted text summary
        """
        
        if model_name not in self.evaluation_results:
            return f"No evaluation results found for {model_name}"
        
        results = self.evaluation_results[model_name]
        
        summary = f"""
Model Evaluation Summary: {model_name.upper()}
{'='*50}

Overall Performance:
- Accuracy: {results['accuracy']:.4f}
- Precision (macro): {results['precision_macro']:.4f}  
- Recall (macro): {results['recall_macro']:.4f}
- F1-Score (macro): {results['f1_macro']:.4f}

Business Metrics:
"""
        
        for metric, value in results['business_metrics'].items():
            summary += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        summary += f"\nTop 5 Most Important Features:\n"
        for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:5]):
            summary += f"{i+1}. {feature}: {importance:.4f}\n"
        
        return summary


if __name__ == "__main__":
    # Example usage
    print("Model Evaluator Example")
    evaluator = ModelEvaluator()
    print(f"Available class names: {evaluator.class_names}")