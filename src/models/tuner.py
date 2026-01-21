"""Hyperparameter tuning utilities."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import mlflow


class HyperparameterTuner:
    """Tune model hyperparameters."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize hyperparameter tuner.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.best_params = {}
        self.best_scores = {}
    
    def grid_search(
        self,
        model: Any,
        param_grid: Dict[str, list],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5,
        scoring: str = 'r2',
        n_jobs: int = -1
    ) -> tuple:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            model: Model to tune
            param_grid: Parameter grid
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        self.logger.info(f"Starting grid search with {len(param_grid)} parameters")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_scores = grid_search.best_score_
        
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best score: {self.best_scores:.4f}")
        
        # Log to MLflow
        with mlflow.start_run(run_name="grid_search"):
            mlflow.log_params(self.best_params)
            mlflow.log_metric(f"best_{scoring}", self.best_scores)
        
        return grid_search.best_estimator_, self.best_params, self.best_scores
    
    def random_search(
        self,
        model: Any,
        param_distributions: Dict[str, list],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_iter: int = 50,
        cv: int = 5,
        scoring: str = 'r2',
        n_jobs: int = -1,
        random_state: int = 42
    ) -> tuple:
        """
        Perform random search for hyperparameter tuning.
        
        Args:
            model: Model to tune
            param_distributions: Parameter distributions
            X_train: Training features
            y_train: Training target
            n_iter: Number of iterations
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            random_state: Random state
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        self.logger.info(f"Starting random search with {n_iter} iterations")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.best_scores = random_search.best_score_
        
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best score: {self.best_scores:.4f}")
        
        # Log to MLflow
        with mlflow.start_run(run_name="random_search"):
            mlflow.log_params(self.best_params)
            mlflow.log_metric(f"best_{scoring}", self.best_scores)
        
        return random_search.best_estimator_, self.best_params, self.best_scores
