"""Model training utilities."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

import mlflow
import mlflow.sklearn


class ModelTrainer:
    """Train and evaluate machine learning models."""
    
    def __init__(
        self,
        experiment_name: str = "data-engineering-pipeline",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            experiment_name: MLflow experiment name
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.experiment_name = experiment_name
        self.models = {}
        self.results = {}
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        self.logger.info(f"MLflow experiment: {experiment_name}")
    
    def get_model(self, model_name: str, **params) -> Any:
        """
        Get model instance by name.
        
        Args:
            model_name: Name of the model
            **params: Model parameters
            
        Returns:
            Model instance
        """
        models = {
            'linear_regression': LinearRegression,
            'ridge': Ridge,
            'random_forest': RandomForestRegressor,
        }
        
        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBRegressor
        
        if HAS_LIGHTGBM:
            models['lightgbm'] = lgb.LGBMRegressor
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return models[model_name](**params)
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train a single model.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            params: Model parameters
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuple of (trained model, metrics)
        """
        params = params or {}
        
        self.logger.info(f"Training {model_name} with params: {params}")
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_name", model_name)
            
            # Create and train model
            model = self.get_model(model_name, **params)
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1
            )
            
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Log metrics
            mlflow.log_metric("cv_r2_mean", cv_mean)
            mlflow.log_metric("cv_r2_std", cv_std)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            self.logger.info(f"{model_name} CV R²: {cv_mean:.4f} (+/- {cv_std:.4f})")
            
            metrics = {
                'cv_r2_mean': cv_mean,
                'cv_r2_std': cv_std
            }
            
            self.models[model_name] = model
            
            return model, metrics
    
    def train_multiple_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_configs: Dict[str, Dict[str, Any]],
        cv_folds: int = 5
    ) -> Dict[str, Tuple[Any, Dict[str, float]]]:
        """
        Train multiple models.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_configs: Dictionary mapping model names to their parameters
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of model results
        """
        self.logger.info(f"Training {len(model_configs)} models")
        
        results = {}
        
        for model_name, params in model_configs.items():
            try:
                model, metrics = self.train_model(
                    X_train, y_train,
                    model_name, params,
                    cv_folds
                )
                results[model_name] = (model, metrics)
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
        
        self.results = results
        return results
    
    def save_model(self, model: Any, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            model: Model to save
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, path)
        self.logger.info(f"Saved model to {path}")
    
    @staticmethod
    def load_model(path: str) -> Any:
        """
        Load model from disk.
        
        Args:
            path: Path to model file
            
        Returns:
            Loaded model
        """
        return joblib.load(path)
