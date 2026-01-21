"""Metrics calculation and evaluation utilities."""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, Any
import pandas as pd


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }
    
    # Adjusted R²
    n = len(y_true)
    p = 1  # Will be updated with actual feature count
    metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*50}")
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():15s}: {value:.4f}")
    print(f"{'='*50}\n")


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple model results.
    
    Args:
        results: Dictionary mapping model names to their metrics
        
    Returns:
        DataFrame with model comparison
    """
    df = pd.DataFrame(results).T
    df = df.sort_values('r2', ascending=False)
    return df
