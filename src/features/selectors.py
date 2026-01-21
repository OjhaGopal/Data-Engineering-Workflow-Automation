"""Feature selection utilities."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    RFE
)
from sklearn.ensemble import RandomForestRegressor


class FeatureSelector:
    """Select important features."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize feature selector.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.selected_features = []
        self.feature_scores = {}
    
    def select_by_correlation(
        self,
        df: pd.DataFrame,
        target: str,
        threshold: float = 0.1,
        method: str = 'pearson'
    ) -> List[str]:
        """
        Select features based on correlation with target.
        
        Args:
            df: Input DataFrame
            target: Target column name
            threshold: Minimum absolute correlation threshold
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Selecting features by correlation (threshold={threshold})")
        
        correlations = df.corr(method=method)[target].abs()
        selected = correlations[correlations >= threshold].index.tolist()
        
        # Remove target itself
        if target in selected:
            selected.remove(target)
        
        self.selected_features = selected
        self.feature_scores = correlations.to_dict()
        
        self.logger.info(f"Selected {len(selected)} features")
        
        return selected
    
    def select_by_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20,
        model: Optional[any] = None
    ) -> List[str]:
        """
        Select features based on model importance.
        
        Args:
            X: Feature DataFrame
            y: Target series
            n_features: Number of features to select
            model: Model to use (default: RandomForestRegressor)
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Selecting top {n_features} features by importance")
        
        if model is None:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        model.fit(X, y)
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
        
        selected = X.columns[indices].tolist()
        self.selected_features = selected
        
        # Store scores
        self.feature_scores = dict(zip(X.columns, importances))
        
        self.logger.info(f"Selected {len(selected)} features")
        
        return selected
    
    def select_by_statistical_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20,
        score_func: str = 'f_regression'
    ) -> List[str]:
        """
        Select features using statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target series
            n_features: Number of features to select
            score_func: Scoring function ('f_regression', 'mutual_info')
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Selecting features using {score_func}")
        
        score_funcs = {
            'f_regression': f_regression,
            'mutual_info': mutual_info_regression
        }
        
        selector = SelectKBest(score_func=score_funcs[score_func], k=n_features)
        selector.fit(X, y)
        
        # Get selected features
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        
        self.selected_features = selected
        self.feature_scores = dict(zip(X.columns, selector.scores_))
        
        self.logger.info(f"Selected {len(selected)} features")
        
        return selected
    
    def select_by_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20,
        model: Optional[any] = None
    ) -> List[str]:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature DataFrame
            y: Target series
            n_features: Number of features to select
            model: Model to use for RFE
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Selecting features using RFE")
        
        if model is None:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        # Get selected features
        mask = rfe.support_
        selected = X.columns[mask].tolist()
        
        self.selected_features = selected
        self.feature_scores = dict(zip(X.columns, rfe.ranking_))
        
        self.logger.info(f"Selected {len(selected)} features")
        
        return selected
    
    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95
    ) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            df: Input DataFrame
            threshold: Correlation threshold for removal
            
        Returns:
            List of features to keep
        """
        self.logger.info(f"Removing features with correlation > {threshold}")
        
        corr_matrix = df.corr().abs()
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        features_to_keep = [col for col in df.columns if col not in to_drop]
        
        self.logger.info(f"Removed {len(to_drop)} correlated features, keeping {len(features_to_keep)}")
        
        return features_to_keep
    
    def get_feature_scores(self) -> dict:
        """
        Get feature scores from last selection.
        
        Returns:
            Dictionary of feature scores
        """
        return self.feature_scores
