"""
Prediction Interface for Bitcoin Fractal Prediction Models

This module provides a command-line interface for making predictions using
trained ML models with support for real-time and batch prediction.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.model_manager import ModelManager
from ml_models.model_evaluator import ModelEvaluator
import config

class FractalPredictor:
    """
    Prediction interface for fractal prediction models.
    """
    
    def __init__(self, model_id: str):
        self.model_manager = ModelManager()
        self.model_id = model_id
        self.model = None
        self.metadata = None
        self.class_names = ['no_fractal', 'bullish_fractal', 'bearish_fractal']
        
        self._load_model()
    
    def _load_model(self):
        """Load model and metadata."""
        self.model = self.model_manager.load_model(self.model_id)
        if self.model is None:
            raise ValueError(f"Could not load model: {self.model_id}")
        
        self.metadata = self.model_manager.get_model_info(self.model_id)
        print(f"Loaded model: {self.model_id}")
        print(f"Algorithm: {self.metadata.get('algorithm', 'unknown')}")
        print(f"Created: {self.metadata.get('created_at', 'unknown')}")
    
    def predict(
        self, 
        features: pd.DataFrame,
        include_probabilities: bool = True,
        confidence_threshold: float = 0.7
    ) -> Dict:
        """
        Make predictions on feature data.
        
        Args:
            features: DataFrame with features (same format as training)
            include_probabilities: Whether to include prediction probabilities
            confidence_threshold: Threshold for high confidence predictions
            
        Returns:
            Dictionary with predictions and metadata
        """
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Validate features
        expected_features = self.metadata.get('feature_names', [])
        if expected_features:
            missing_features = [f for f in expected_features if f not in features.columns]
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
            
            # Reorder columns to match training
            available_features = [f for f in expected_features if f in features.columns]
            features = features[available_features]
        
        # Make predictions
        predictions = self.model.predict(features)
        
        results = {
            'predictions': predictions,
            'prediction_labels': [self.class_names[p] for p in predictions],
            'timestamp': datetime.now().isoformat(),
            'model_id': self.model_id,
            'num_predictions': len(predictions)
        }
        
        # Add probabilities if available and requested
        if include_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
            results['probabilities'] = probabilities
            results['confidence'] = np.max(probabilities, axis=1)
            
            # High confidence predictions
            high_conf_mask = results['confidence'] >= confidence_threshold
            results['high_confidence_predictions'] = high_conf_mask
            results['high_confidence_count'] = int(high_conf_mask.sum())
            results['avg_confidence'] = float(np.mean(results['confidence']))
        
        # Prediction summary
        prediction_counts = pd.Series(predictions).value_counts().sort_index()
        results['prediction_summary'] = {
            self.class_names[i]: int(prediction_counts.get(i, 0)) 
            for i in range(len(self.class_names))
        }
        
        return results
    
    def predict_from_dataset(
        self,
        dataset_path: str,
        output_path: str = None,
        **kwargs
    ) -> Dict:
        """
        Make predictions on a dataset file.
        
        Args:
            dataset_path: Path to dataset CSV file
            output_path: Path to save predictions (optional)
            **kwargs: Additional arguments for predict()
            
        Returns:
            Prediction results
        """
        
        print(f"Loading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Remove target column if present
        if 'fractal_direction' in df.columns:
            df_features = df.drop(columns=['fractal_direction'])
            has_target = True
        else:
            df_features = df.copy()
            has_target = False
        
        # Remove non-feature columns
        exclude_columns = ['timestamp'] + [col for col in df_features.columns if col.startswith('timestamp')]
        df_features = df_features.drop(columns=[col for col in exclude_columns if col in df_features.columns])
        
        print(f"Making predictions on {len(df_features):,} samples with {len(df_features.columns)} features...")
        
        # Make predictions
        results = self.predict(df_features, **kwargs)
        
        # Create output DataFrame
        output_df = df.copy()
        output_df['predicted_fractal_direction'] = results['predictions']
        output_df['predicted_label'] = results['prediction_labels']
        
        if 'probabilities' in results:
            # Add probability columns
            prob_df = pd.DataFrame(
                results['probabilities'], 
                columns=[f'prob_{class_name}' for class_name in self.class_names]
            )
            output_df = pd.concat([output_df, prob_df], axis=1)
            
            output_df['prediction_confidence'] = results['confidence']
            
            if 'high_confidence_predictions' in results:
                output_df['high_confidence'] = results['high_confidence_predictions']
        
        # Add accuracy if target is available
        if has_target:
            accuracy = (df['fractal_direction'] == results['predictions']).mean()
            results['accuracy'] = float(accuracy)
            output_df['correct_prediction'] = df['fractal_direction'] == results['predictions']
            print(f"Prediction accuracy: {accuracy:.4f}")
        
        # Save output if requested
        if output_path:
            output_df.to_csv(output_path, index=False)
            print(f"Predictions saved to: {output_path}")
        
        results['output_dataframe'] = output_df
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return self.metadata
    
    def print_prediction_summary(self, results: Dict):
        """Print a formatted summary of prediction results."""
        
        print(f"\n{'='*60}")
        print("PREDICTION SUMMARY")
        print(f"{'='*60}")
        
        print(f"Model: {results['model_id']}")
        print(f"Predictions: {results['num_predictions']:,}")
        print(f"Timestamp: {results['timestamp']}")
        
        print(f"\nPrediction Distribution:")
        for label, count in results['prediction_summary'].items():
            percentage = (count / results['num_predictions']) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        if 'avg_confidence' in results:
            print(f"\nConfidence Analysis:")
            print(f"  Average Confidence: {results['avg_confidence']:.4f}")
            print(f"  High Confidence Predictions: {results['high_confidence_count']:,}")
            high_conf_pct = (results['high_confidence_count'] / results['num_predictions']) * 100
            print(f"  High Confidence Rate: {high_conf_pct:.1f}%")
        
        if 'accuracy' in results:
            print(f"\nPrediction Accuracy: {results['accuracy']:.4f}")


def load_latest_data(max_samples: int = 1000) -> pd.DataFrame:
    """
    Load latest available data for prediction.
    
    Args:
        max_samples: Maximum number of samples to load
        
    Returns:
        DataFrame with latest data
    """
    
    # Look for the most recent ML dataset
    features_dir = "features"
    if os.path.exists(features_dir):
        ml_files = [
            f for f in os.listdir(features_dir) 
            if f.startswith("ml_ready_dataset") and f.endswith(".csv")
        ]
        
        if ml_files:
            # Use the most recent file (by name)
            latest_file = sorted(ml_files)[-1]
            file_path = os.path.join(features_dir, latest_file)
            
            print(f"Loading latest data from: {file_path}")
            df = pd.read_csv(file_path)
            
            if max_samples and len(df) > max_samples:
                # Take the most recent samples
                df = df.tail(max_samples)
                print(f"Using most recent {max_samples:,} samples")
            
            return df
    
    raise FileNotFoundError("No ML datasets found. Run data generation first.")

def get_best_model() -> str:
    """
    Get the best available model based on F1 score.
    
    Returns:
        Model ID of the best model
    """
    
    manager = ModelManager()
    best_model_id, best_metadata = manager.get_best_model(metric='f1_macro')
    
    if best_model_id:
        print(f"Using best model: {best_model_id}")
        return best_model_id
    else:
        raise ValueError("No trained models found. Train a model first.")

def main():
    """Main CLI function."""
    
    parser = argparse.ArgumentParser(
        description="Make predictions using trained fractal prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on latest data using best model
  python predict.py --latest
  
  # Predict on specific dataset
  python predict.py --dataset features/ml_ready_dataset_full_2025.csv --model fractal_predictor_xgboost_v1
  
  # Predict with confidence analysis
  python predict.py --dataset features/ml_ready_dataset_full_2025.csv --model fractal_predictor_xgboost_v1 --confidence
  
  # Save predictions to file
  python predict.py --dataset features/ml_ready_dataset_full_2025.csv --output predictions_2025.csv
        """
    )
    
    # Input options
    parser.add_argument('--dataset', type=str,
                       help='Path to dataset for prediction')
    parser.add_argument('--latest', action='store_true',
                       help='Use latest available data')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum samples to use with --latest')
    
    # Model selection
    parser.add_argument('--model', type=str,
                       help='Model ID to use (uses best model if not specified)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    
    # Prediction options
    parser.add_argument('--confidence', action='store_true',
                       help='Include confidence analysis')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Threshold for high confidence predictions')
    parser.add_argument('--no-probabilities', action='store_true',
                       help='Exclude prediction probabilities')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Path to save predictions CSV file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Utility commands
    if args.list_models:
        manager = ModelManager()
        models_df = manager.list_models()
        if not models_df.empty:
            print("Available models:")
            print(models_df[['model_id', 'algorithm', 'accuracy', 'f1_macro']].to_string(index=False))
        else:
            print("No trained models found")
        return 0
    
    # Validate input options
    if not args.dataset and not args.latest:
        print("Error: Must specify --dataset or --latest")
        return 1
    
    try:
        # Determine model to use
        if args.model:
            model_id = args.model
        else:
            model_id = get_best_model()
        
        # Load predictor
        predictor = FractalPredictor(model_id)
        
        # Load data
        if args.latest:
            dataset_path = None
            # Use in-memory prediction for latest data
            df = load_latest_data(args.max_samples)
            
            # Remove target column if present
            if 'fractal_direction' in df.columns:
                df_features = df.drop(columns=['fractal_direction'])
                has_target = True
            else:
                df_features = df.copy()
                has_target = False
            
            # Remove non-feature columns
            exclude_columns = ['timestamp'] + [col for col in df_features.columns if col.startswith('timestamp')]
            df_features = df_features.drop(columns=[col for col in exclude_columns if col in df_features.columns])
            
            # Make predictions
            results = predictor.predict(
                df_features,
                include_probabilities=not args.no_probabilities,
                confidence_threshold=args.confidence_threshold
            )
            
            # Add accuracy if target available
            if has_target:
                accuracy = (df['fractal_direction'] == results['predictions']).mean()
                results['accuracy'] = float(accuracy)
            
        else:
            dataset_path = args.dataset
            if not os.path.exists(dataset_path):
                print(f"Error: Dataset file not found: {dataset_path}")
                return 1
            
            # Predict from dataset
            results = predictor.predict_from_dataset(
                dataset_path=dataset_path,
                output_path=args.output,
                include_probabilities=not args.no_probabilities,
                confidence_threshold=args.confidence_threshold
            )
        
        # Print results
        if not args.quiet:
            predictor.print_prediction_summary(results)
            
            if args.confidence and 'confidence' in results:
                print(f"\nTop 10 Most Confident Predictions:")
                if 'output_dataframe' in results:
                    df_output = results['output_dataframe']
                    if 'prediction_confidence' in df_output.columns:
                        top_confident = df_output.nlargest(10, 'prediction_confidence')[
                            ['timestamp', 'predicted_label', 'prediction_confidence']
                        ]
                        print(top_confident.to_string(index=False))
        
        # Save output if requested and not already saved
        if args.output and args.latest:
            # Create output DataFrame for latest data
            output_df = df.copy()
            output_df['predicted_fractal_direction'] = results['predictions']
            output_df['predicted_label'] = results['prediction_labels']
            
            if 'probabilities' in results:
                prob_df = pd.DataFrame(
                    results['probabilities'], 
                    columns=[f'prob_{class_name}' for class_name in predictor.class_names]
                )
                output_df = pd.concat([output_df, prob_df], axis=1)
                output_df['prediction_confidence'] = results['confidence']
            
            output_df.to_csv(args.output, index=False)
            print(f"Predictions saved to: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())