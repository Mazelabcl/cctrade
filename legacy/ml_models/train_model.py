"""
CLI Training Interface for Bitcoin Fractal Prediction Models

This module provides a command-line interface for training ML models with
comprehensive configuration options and automated workflows.
"""

import argparse
import sys
import os
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.model_trainer import ModelTrainer
from ml_models.model_evaluator import ModelEvaluator
from ml_models.model_manager import ModelManager
import config

def load_ml_dataset(file_path: str) -> pd.DataFrame:
    """
    Load ML-ready dataset from CSV file.
    
    Args:
        file_path: Path to the ML dataset CSV file
        
    Returns:
        Loaded DataFrame
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Validate required columns
    if 'fractal_direction' not in df.columns:
        raise ValueError("Dataset must contain 'fractal_direction' target column")
    
    return df

def get_available_datasets() -> List[str]:
    """
    Get list of available ML datasets.
    
    Returns:
        List of dataset file paths
    """
    
    features_dir = "features"
    if not os.path.exists(features_dir):
        return []
    
    ml_datasets = [
        os.path.join(features_dir, f) 
        for f in os.listdir(features_dir) 
        if f.startswith("ml_ready_dataset") and f.endswith(".csv")
    ]
    
    return ml_datasets

def train_single_model(
    dataset_path: str,
    algorithm: str,
    model_name: str = None,
    optimize_hyperparams: bool = False,
    optimization_method: str = 'grid',
    n_trials: int = 100,
    cv_folds: int = 3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    scale_features: bool = True,
    save_model: bool = True,
    notes: str = None
) -> Dict:
    """
    Train a single model with specified configuration.
    
    Args:
        dataset_path: Path to ML dataset
        algorithm: Algorithm to use
        model_name: Name for the model
        optimize_hyperparams: Whether to optimize hyperparameters
        optimization_method: Method for optimization ('grid', 'random', 'optuna')
        n_trials: Number of trials for random/optuna optimization
        cv_folds: Cross-validation folds
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        scale_features: Whether to scale features
        save_model: Whether to save the trained model
        notes: Notes about the model
        
    Returns:
        Dictionary with training and evaluation results
    """
    
    print(f"\n{'='*60}")
    print(f"TRAINING {algorithm.upper()} MODEL")
    print(f"{'='*60}")
    
    # Load dataset
    df = load_ml_dataset(dataset_path)
    
    # Initialize components
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    manager = ModelManager() if save_model else None
    
    # Check algorithm availability
    available_algorithms = trainer.get_available_algorithms()
    if algorithm not in available_algorithms:
        raise ValueError(f"Algorithm '{algorithm}' not available. Choose from: {available_algorithms}")
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        df, 
        target_column='fractal_direction',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        scale_features=scale_features
    )
    
    # Train model
    if optimize_hyperparams:
        print(f"\nOptimizing hyperparameters using {optimization_method}...")
        training_results = trainer.optimize_hyperparameters(
            algorithm=algorithm,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            optimization_method=optimization_method,
            n_trials=n_trials,
            cv_folds=cv_folds
        )
    else:
        print(f"\nTraining with default parameters...")
        training_results = trainer.train_model(
            algorithm=algorithm,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cv_folds=cv_folds
        )
    
    # Evaluate model
    print(f"\nEvaluating model on test set...")
    evaluation_results = evaluator.evaluate_model(
        model=training_results['model'],
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        model_name=algorithm
    )
    
    # Save model if requested
    model_id = None
    if save_model and manager:
        model_name = model_name or f"fractal_predictor_{algorithm}"
        model_id = manager.save_model(
            model=training_results['model'],
            model_name=model_name,
            algorithm=algorithm,
            training_results=training_results,
            evaluation_results=evaluation_results,
            feature_names=training_results['feature_names'],
            model_params=training_results.get('best_params', training_results.get('model_params', {})),
            notes=notes
        )
    
    # Compile results
    results = {
        'algorithm': algorithm,
        'model_id': model_id,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'dataset_info': {
            'dataset_path': dataset_path,
            'total_samples': len(df),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'num_features': len(X_train.columns)
        }
    }
    
    return results

def compare_multiple_models(
    dataset_path: str,
    algorithms: List[str],
    optimize_all: bool = False,
    save_best_only: bool = True,
    **kwargs
) -> Dict:
    """
    Train and compare multiple models.
    
    Args:
        dataset_path: Path to ML dataset
        algorithms: List of algorithms to compare
        optimize_all: Whether to optimize hyperparameters for all models
        save_best_only: Whether to save only the best performing model
        **kwargs: Additional arguments for training
        
    Returns:
        Dictionary with comparison results
    """
    
    print(f"\n{'='*60}")
    print(f"COMPARING {len(algorithms)} ALGORITHMS")
    print(f"{'='*60}")
    
    results = {}
    all_results = []
    
    for algorithm in algorithms:
        try:
            print(f"\n{'-'*40}")
            print(f"Training {algorithm}...")
            print(f"{'-'*40}")
            
            result = train_single_model(
                dataset_path=dataset_path,
                algorithm=algorithm,
                optimize_hyperparams=optimize_all,
                save_model=False,  # Save later based on performance
                **kwargs
            )
            
            results[algorithm] = result
            
            # Collect for comparison
            all_results.append({
                'algorithm': algorithm,
                'accuracy': result['evaluation_results']['accuracy'],
                'f1_macro': result['evaluation_results']['f1_macro'],
                'fractal_detection_rate': result['evaluation_results']['business_metrics'].get('fractal_detection_rate', 0),
                'training_time': result['training_results'].get('training_time', 0)
            })
            
        except Exception as e:
            print(f"Error training {algorithm}: {e}")
            results[algorithm] = {'error': str(e)}
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.sort_values('f1_macro', ascending=False)
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Save best model if requested
    if save_best_only and not comparison_df.empty:
        best_algorithm = comparison_df.iloc[0]['algorithm']
        best_result = results[best_algorithm]
        
        if 'error' not in best_result:
            manager = ModelManager()
            model_id = manager.save_model(
                model=best_result['training_results']['model'],
                model_name=f"best_fractal_predictor",
                algorithm=best_algorithm,
                training_results=best_result['training_results'],
                evaluation_results=best_result['evaluation_results'],
                feature_names=best_result['training_results']['feature_names'],
                model_params=best_result['training_results'].get('best_params', best_result['training_results'].get('model_params', {})),
                notes=f"Best model from comparison of {len(algorithms)} algorithms"
            )
            
            print(f"\nBest model saved: {model_id}")
    
    return {
        'individual_results': results,
        'comparison': comparison_df,
        'best_algorithm': comparison_df.iloc[0]['algorithm'] if not comparison_df.empty else None
    }

def main():
    """Main CLI function."""
    
    parser = argparse.ArgumentParser(
        description="Train ML models for Bitcoin fractal prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single XGBoost model
  python train_model.py --dataset features/ml_ready_dataset_full_2025.csv --algorithm xgboost
  
  # Train with hyperparameter optimization
  python train_model.py --dataset features/ml_ready_dataset_full_2025.csv --algorithm xgboost --optimize
  
  # Compare multiple algorithms
  python train_model.py --dataset features/ml_ready_dataset_full_2025.csv --compare --algorithms xgboost random_forest lightgbm
  
  # Quick training with Optuna optimization
  python train_model.py --dataset features/ml_ready_dataset_full_2025.csv --algorithm xgboost --optimize --method optuna --trials 50
        """
    )
    
    # Dataset options
    parser.add_argument('--dataset', type=str,
                       help='Path to ML-ready dataset CSV file')
    
    # Model selection
    parser.add_argument('--algorithm', type=str, choices=['xgboost', 'random_forest', 'lightgbm', 'logistic_regression'],
                       help='Algorithm to train (required if not using --compare)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple algorithms')
    parser.add_argument('--algorithms', nargs='+', 
                       choices=['xgboost', 'random_forest', 'lightgbm', 'logistic_regression'],
                       default=['xgboost', 'random_forest', 'lightgbm'],
                       help='Algorithms to compare (used with --compare)')
    
    # Training options
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize hyperparameters')
    parser.add_argument('--method', choices=['grid', 'random', 'optuna'], default='grid',
                       help='Hyperparameter optimization method')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials for random/optuna optimization')
    parser.add_argument('--cv-folds', type=int, default=3,
                       help='Cross-validation folds')
    
    # Data split options
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation data ratio')
    parser.add_argument('--no-scaling', action='store_true',
                       help='Disable feature scaling')
    
    # Model management
    parser.add_argument('--model-name', type=str,
                       help='Name for the saved model')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the trained model')
    parser.add_argument('--notes', type=str,
                       help='Notes about the model')
    
    # Utility options
    parser.add_argument('--list-datasets', action='store_true',
                       help='List available datasets and exit')
    parser.add_argument('--list-models', action='store_true',
                       help='List saved models and exit')
    
    args = parser.parse_args()
    
    # Utility commands
    if args.list_datasets:
        datasets = get_available_datasets()
        print("Available ML datasets:")
        for dataset in datasets:
            print(f"  {dataset}")
        return 0
    
    if args.list_models:
        manager = ModelManager()
        models_df = manager.list_models()
        if not models_df.empty:
            print("Saved models:")
            print(models_df.to_string(index=False))
        else:
            print("No saved models found")
        return 0
    
    # Validate arguments
    if not args.compare and not args.algorithm and not args.list_datasets and not args.list_models:
        print("Error: Must specify --algorithm or use --compare")
        return 1
    
    if args.dataset and not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        return 1
    
    try:
        # Common training arguments
        training_kwargs = {
            'optimize_hyperparams': args.optimize,
            'optimization_method': args.method,
            'n_trials': args.trials,
            'cv_folds': args.cv_folds,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'scale_features': not args.no_scaling,
            'save_model': not args.no_save,
            'notes': args.notes
        }
        
        if args.compare:
            # Compare multiple models
            results = compare_multiple_models(
                dataset_path=args.dataset,
                algorithms=args.algorithms,
                optimize_all=args.optimize,
                save_best_only=True,
                **training_kwargs
            )
            
            print(f"\nComparison completed successfully!")
            if results['best_algorithm']:
                print(f"Best algorithm: {results['best_algorithm']}")
        
        else:
            # Train single model
            result = train_single_model(
                dataset_path=args.dataset,
                algorithm=args.algorithm,
                model_name=args.model_name,
                **training_kwargs
            )
            
            print(f"\nTraining completed successfully!")
            if result['model_id']:
                print(f"Model saved: {result['model_id']}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())