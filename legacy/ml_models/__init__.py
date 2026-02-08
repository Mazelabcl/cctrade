"""
ML Models Package for Bitcoin Fractal Prediction

This package provides comprehensive machine learning capabilities for training,
evaluating, and deploying models to predict Bitcoin price fractals.

Modules:
- model_trainer: Core training logic with multiple algorithms
- model_evaluator: Evaluation metrics and validation  
- model_manager: Model persistence and versioning
- train_model: CLI interface for training
- predict: Prediction interface
- validate_model: Backtesting and validation
- utils: Shared utilities
"""

__version__ = "1.0.0"
__author__ = "Bitcoin Fractal Prediction System"

# Import key classes for easy access
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator  
from .model_manager import ModelManager

__all__ = [
    'ModelTrainer',
    'ModelEvaluator', 
    'ModelManager'
]