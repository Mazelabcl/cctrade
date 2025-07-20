"""
Model Manager for Bitcoin Fractal Prediction

This module provides model persistence, versioning, and management capabilities
for the ML pipeline.
"""

import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import shutil
import warnings
warnings.filterwarnings('ignore')

class ModelManager:
    """
    Comprehensive model management class for saving, loading, and organizing ML models.
    """
    
    def __init__(self, models_dir: str = "ml_models/models"):
        self.models_dir = models_dir
        self.metadata_file = os.path.join(models_dir, "model_registry.json")
        
        # Ensure models directory exists
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Load or create model registry
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load model registry or create empty one."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading model registry: {e}. Creating new one.")
        
        return {
            "models": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_registry(self):
        """Save model registry to file."""
        self.registry["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving model registry: {e}")
    
    def save_model(
        self,
        model,
        model_name: str,
        algorithm: str,
        training_results: Dict = None,
        evaluation_results: Dict = None,
        feature_names: List[str] = None,
        model_params: Dict = None,
        notes: str = None,
        version: str = None
    ) -> str:
        """
        Save a trained model with comprehensive metadata.
        
        Args:
            model: Trained ML model
            model_name: Name for the model
            algorithm: Algorithm used ('xgboost', 'random_forest', etc.)
            training_results: Results from training
            evaluation_results: Results from evaluation
            feature_names: List of feature names used
            model_params: Model parameters used
            notes: Additional notes about the model
            version: Model version (auto-generated if None)
            
        Returns:
            Model ID for future reference
        """
        
        # Generate version if not provided
        if version is None:
            existing_versions = [
                v["version"] for v in self.registry["models"].values() 
                if v["model_name"] == model_name
            ]
            if existing_versions:
                version_nums = [int(v.split('v')[1]) for v in existing_versions if v.startswith('v')]
                next_version = max(version_nums) + 1 if version_nums else 1
            else:
                next_version = 1
            version = f"v{next_version}"
        
        # Create model ID
        model_id = f"{model_name}_{algorithm}_{version}"
        
        # File paths
        model_file = os.path.join(self.models_dir, f"{model_id}.pkl")
        
        # Save model
        try:
            # Try joblib first (better for sklearn models)
            joblib.dump(model, model_file)
            serialization_method = "joblib"
        except Exception as e:
            try:
                # Fallback to pickle
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                serialization_method = "pickle"
            except Exception as e2:
                print(f"Error saving model with joblib: {e}")
                print(f"Error saving model with pickle: {e2}")
                return None
        
        # Create metadata
        metadata = {
            "model_id": model_id,
            "model_name": model_name,
            "algorithm": algorithm,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "file_path": model_file,
            "file_size": os.path.getsize(model_file),
            "serialization_method": serialization_method,
            "feature_names": feature_names or [],
            "model_params": model_params or {},
            "notes": notes or ""
        }
        
        # Add training results
        if training_results:
            metadata["training"] = {
                "train_score": training_results.get("train_score"),
                "val_score": training_results.get("val_score"),
                "cv_mean": training_results.get("cv_mean"),
                "cv_std": training_results.get("cv_std"),
                "training_time": training_results.get("training_time")
            }
        
        # Add evaluation results
        if evaluation_results:
            metadata["evaluation"] = {
                "accuracy": evaluation_results.get("accuracy"),
                "precision_macro": evaluation_results.get("precision_macro"),
                "recall_macro": evaluation_results.get("recall_macro"),
                "f1_macro": evaluation_results.get("f1_macro"),
                "roc_auc_macro": evaluation_results.get("roc_auc_macro"),
                "business_metrics": evaluation_results.get("business_metrics", {}),
                "test_size": evaluation_results.get("test_size")
            }
        
        # Add to registry
        self.registry["models"][model_id] = metadata
        self._save_registry()
        
        print(f"Model saved successfully:")
        print(f"  Model ID: {model_id}")
        print(f"  File: {model_file}")
        print(f"  Size: {metadata['file_size']:,} bytes")
        
        return model_id
    
    def load_model(self, model_id: str):
        """
        Load a saved model.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Loaded model object
        """
        
        if model_id not in self.registry["models"]:
            print(f"Model {model_id} not found in registry")
            return None
        
        metadata = self.registry["models"][model_id]
        model_file = metadata["file_path"]
        
        if not os.path.exists(model_file):
            print(f"Model file {model_file} not found")
            return None
        
        try:
            # Try method used for saving
            if metadata.get("serialization_method") == "joblib":
                model = joblib.load(model_file)
            else:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            
            print(f"Model {model_id} loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            return None
    
    def list_models(self, algorithm: str = None, model_name: str = None) -> pd.DataFrame:
        """
        List available models with filtering options.
        
        Args:
            algorithm: Filter by algorithm
            model_name: Filter by model name
            
        Returns:
            DataFrame with model information
        """
        
        if not self.registry["models"]:
            print("No models found in registry")
            return pd.DataFrame()
        
        model_list = []
        
        for model_id, metadata in self.registry["models"].items():
            # Apply filters
            if algorithm and metadata["algorithm"] != algorithm:
                continue
            if model_name and metadata["model_name"] != model_name:
                continue
            
            row = {
                "model_id": model_id,
                "model_name": metadata["model_name"],
                "algorithm": metadata["algorithm"],
                "version": metadata["version"],
                "created_at": metadata["created_at"],
                "file_size_mb": metadata["file_size"] / (1024*1024)
            }
            
            # Add training metrics
            if "training" in metadata:
                training = metadata["training"]
                row.update({
                    "train_score": training.get("train_score"),
                    "val_score": training.get("val_score"),
                    "cv_mean": training.get("cv_mean")
                })
            
            # Add evaluation metrics
            if "evaluation" in metadata:
                evaluation = metadata["evaluation"]
                row.update({
                    "accuracy": evaluation.get("accuracy"),
                    "f1_macro": evaluation.get("f1_macro"),
                    "fractal_detection_rate": evaluation.get("business_metrics", {}).get("fractal_detection_rate")
                })
            
            model_list.append(row)
        
        df = pd.DataFrame(model_list)
        
        # Sort by creation date (newest first)
        if not df.empty and 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at', ascending=False)
        
        return df
    
    def get_model_info(self, model_id: str) -> Dict:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary with detailed model information
        """
        
        if model_id not in self.registry["models"]:
            print(f"Model {model_id} not found in registry")
            return {}
        
        return self.registry["models"][model_id]
    
    def delete_model(self, model_id: str, confirm: bool = False) -> bool:
        """
        Delete a model from registry and filesystem.
        
        Args:
            model_id: ID of the model to delete
            confirm: Confirmation flag to prevent accidental deletion
            
        Returns:
            True if deletion successful
        """
        
        if not confirm:
            print(f"Please set confirm=True to delete model {model_id}")
            return False
        
        if model_id not in self.registry["models"]:
            print(f"Model {model_id} not found in registry")
            return False
        
        metadata = self.registry["models"][model_id]
        model_file = metadata["file_path"]
        
        try:
            # Remove file
            if os.path.exists(model_file):
                os.remove(model_file)
            
            # Remove from registry
            del self.registry["models"][model_id]
            self._save_registry()
            
            print(f"Model {model_id} deleted successfully")
            return True
            
        except Exception as e:
            print(f"Error deleting model {model_id}: {e}")
            return False
    
    def compare_models(
        self, 
        model_ids: List[str] = None,
        metric: str = "f1_macro"
    ) -> pd.DataFrame:
        """
        Compare performance of multiple models.
        
        Args:
            model_ids: List of model IDs to compare (None for all)
            metric: Metric to sort by
            
        Returns:
            DataFrame with model comparison
        """
        
        if model_ids is None:
            model_ids = list(self.registry["models"].keys())
        
        comparison_data = []
        
        for model_id in model_ids:
            if model_id not in self.registry["models"]:
                print(f"Warning: Model {model_id} not found")
                continue
            
            metadata = self.registry["models"][model_id]
            
            row = {
                "model_id": model_id,
                "algorithm": metadata["algorithm"],
                "version": metadata["version"]
            }
            
            # Add evaluation metrics
            if "evaluation" in metadata:
                eval_data = metadata["evaluation"]
                row.update({
                    "accuracy": eval_data.get("accuracy"),
                    "precision_macro": eval_data.get("precision_macro"),
                    "recall_macro": eval_data.get("recall_macro"),
                    "f1_macro": eval_data.get("f1_macro"),
                    "roc_auc_macro": eval_data.get("roc_auc_macro")
                })
                
                # Add business metrics
                business_metrics = eval_data.get("business_metrics", {})
                for bm_name, bm_value in business_metrics.items():
                    row[f"biz_{bm_name}"] = bm_value
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by specified metric
        if not comparison_df.empty and metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(metric, ascending=False, na_position='last')
        
        return comparison_df
    
    def get_best_model(
        self, 
        algorithm: str = None,
        metric: str = "f1_macro"
    ) -> Tuple[str, Dict]:
        """
        Get the best performing model based on a metric.
        
        Args:
            algorithm: Filter by algorithm (None for all)
            metric: Metric to use for selection
            
        Returns:
            Tuple of (model_id, metadata)
        """
        
        models_df = self.list_models(algorithm=algorithm)
        
        if models_df.empty:
            return None, {}
        
        if metric not in models_df.columns:
            print(f"Metric '{metric}' not available. Available metrics: {list(models_df.columns)}")
            return None, {}
        
        # Get best model
        best_idx = models_df[metric].idxmax()
        if pd.isna(models_df.loc[best_idx, metric]):
            print(f"No models have valid {metric} scores")
            return None, {}
        
        best_model_id = models_df.loc[best_idx, "model_id"]
        best_metadata = self.get_model_info(best_model_id)
        
        return best_model_id, best_metadata
    
    def export_model_for_production(
        self,
        model_id: str,
        export_dir: str,
        include_metadata: bool = True
    ) -> str:
        """
        Export model for production deployment.
        
        Args:
            model_id: ID of the model to export
            export_dir: Directory to export to
            include_metadata: Whether to include metadata file
            
        Returns:
            Path to exported model
        """
        
        if model_id not in self.registry["models"]:
            print(f"Model {model_id} not found in registry")
            return None
        
        metadata = self.registry["models"][model_id]
        source_file = metadata["file_path"]
        
        if not os.path.exists(source_file):
            print(f"Model file {source_file} not found")
            return None
        
        # Create export directory
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        # Copy model file
        export_model_file = os.path.join(export_dir, f"{model_id}.pkl")
        shutil.copy2(source_file, export_model_file)
        
        # Export metadata if requested
        if include_metadata:
            metadata_file = os.path.join(export_dir, f"{model_id}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        print(f"Model exported for production:")
        print(f"  Model file: {export_model_file}")
        if include_metadata:
            print(f"  Metadata: {metadata_file}")
        
        return export_model_file
    
    def cleanup_old_models(
        self,
        keep_best_n: int = 3,
        algorithm: str = None,
        metric: str = "f1_macro",
        confirm: bool = False
    ) -> int:
        """
        Clean up old models, keeping only the best N models.
        
        Args:
            keep_best_n: Number of best models to keep
            algorithm: Filter by algorithm (None for all algorithms)
            metric: Metric to use for ranking
            confirm: Confirmation flag
            
        Returns:
            Number of models deleted
        """
        
        if not confirm:
            print("Please set confirm=True to delete models")
            return 0
        
        # Get models sorted by performance
        comparison_df = self.compare_models(metric=metric)
        
        if algorithm:
            comparison_df = comparison_df[comparison_df['algorithm'] == algorithm]
        
        if len(comparison_df) <= keep_best_n:
            print(f"Only {len(comparison_df)} models found, no cleanup needed")
            return 0
        
        # Get models to delete (worst performers)
        models_to_delete = comparison_df.iloc[keep_best_n:]['model_id'].tolist()
        
        deleted_count = 0
        for model_id in models_to_delete:
            if self.delete_model(model_id, confirm=True):
                deleted_count += 1
        
        print(f"Cleanup completed: {deleted_count} models deleted, {keep_best_n} best models kept")
        return deleted_count
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the model registry.
        
        Returns:
            Dictionary with registry statistics
        """
        
        models = self.registry["models"]
        
        if not models:
            return {"total_models": 0}
        
        # Count by algorithm
        algorithm_counts = {}
        total_size = 0
        
        for metadata in models.values():
            algorithm = metadata["algorithm"]
            algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
            total_size += metadata["file_size"]
        
        # Find best model
        best_model_id, best_metadata = self.get_best_model()
        
        summary = {
            "total_models": len(models),
            "algorithms": algorithm_counts,
            "total_size_mb": total_size / (1024*1024),
            "registry_created": self.registry["created_at"],
            "last_updated": self.registry["last_updated"],
            "best_model": best_model_id
        }
        
        return summary
    
    def print_registry_summary(self):
        """Print a formatted summary of the model registry."""
        
        summary = self.get_registry_summary()
        
        print("\n" + "="*50)
        print("MODEL REGISTRY SUMMARY")
        print("="*50)
        
        print(f"Total Models: {summary['total_models']}")
        print(f"Total Size: {summary['total_size_mb']:.2f} MB")
        print(f"Registry Created: {summary['registry_created']}")
        print(f"Last Updated: {summary['last_updated']}")
        
        if summary.get('best_model'):
            print(f"Best Model: {summary['best_model']}")
        
        print(f"\nModels by Algorithm:")
        for algorithm, count in summary.get('algorithms', {}).items():
            print(f"  {algorithm}: {count}")


if __name__ == "__main__":
    # Example usage
    print("Model Manager Example")
    manager = ModelManager()
    manager.print_registry_summary()