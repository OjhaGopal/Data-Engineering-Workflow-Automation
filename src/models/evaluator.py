"""Model evaluation utilities."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import mlflow

from src.utils.metrics import calculate_regression_metrics, print_metrics
from src.utils.visualization import plot_predictions, plot_residuals, plot_feature_importance


class ModelEvaluator:
    """Evaluate model performance."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize model evaluator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.evaluation_results = {}
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model",
        log_to_mlflow: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            log_to_mlflow: Whether to log to MLflow
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Evaluating {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_regression_metrics(y_test, y_pred)
        
        # Print metrics
        print_metrics(metrics, model_name)
        
        # Log to MLflow
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"{model_name}_evaluation"):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", value)
        
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred
        }
        
        return metrics
    
    def evaluate_multiple_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate multiple models.
        
        Args:
            models: Dictionary of models
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with comparison results
        """
        self.logger.info(f"Evaluating {len(models)} models")
        
        results = {}
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results[model_name] = metrics
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.sort_values('r2', ascending=False)
        
        self.logger.info("\nModel Comparison:")
        self.logger.info(f"\n{comparison_df.to_string()}")
        
        return comparison_df
    
    def plot_model_performance(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model",
        save_dir: Optional[str] = None
    ) -> None:
        """
        Create performance visualizations.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            save_dir: Directory to save plots
        """
        y_pred = model.predict(X_test)
        
        # Predictions plot
        pred_path = f"{save_dir}/{model_name}_predictions.png" if save_dir else None
        plot_predictions(y_test, y_pred, f"{model_name} - Predictions vs Actual", pred_path)
        
        # Residuals plot
        resid_path = f"{save_dir}/{model_name}_residuals.png" if save_dir else None
        plot_residuals(y_test, y_pred, resid_path)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            imp_path = f"{save_dir}/{model_name}_importance.png" if save_dir else None
            plot_feature_importance(
                X_test.columns.tolist(),
                model.feature_importances_,
                save_path=imp_path
            )
    
    def get_best_model(self, metric: str = 'r2') -> tuple:
        """
        Get the best performing model.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, metrics)
        """
        if not self.evaluation_results:
            raise ValueError("No models have been evaluated yet")
        
        best_model = max(
            self.evaluation_results.items(),
            key=lambda x: x[1]['metrics'][metric]
        )
        
        return best_model[0], best_model[1]['metrics']
