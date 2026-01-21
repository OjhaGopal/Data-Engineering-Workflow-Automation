"""Feature engineering utilities."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations


class FeatureEngineer:
    """Create and engineer features."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize feature engineer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.created_features = []
    
    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features from
            degree: Polynomial degree
            include_bias: Whether to include bias column
            
        Returns:
            DataFrame with polynomial features
        """
        df = df.copy()
        
        self.logger.info(f"Creating polynomial features (degree={degree}) for {len(columns)} columns")
        
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(df[columns])
        
        # Get feature names
        feature_names = poly.get_feature_names_out(columns)
        
        # Add new features to DataFrame
        for i, name in enumerate(feature_names):
            if name not in df.columns:
                df[name] = poly_features[:, i]
                self.created_features.append(name)
        
        self.logger.info(f"Created {len(feature_names) - len(columns)} new polynomial features")
        
        return df
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        max_interactions: int = 2
    ) -> pd.DataFrame:
        """
        Create interaction features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create interactions from
            max_interactions: Maximum number of features to interact
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        self.logger.info(f"Creating interaction features for {len(columns)} columns")
        
        for r in range(2, max_interactions + 1):
            for combo in combinations(columns, r):
                feature_name = '_x_'.join(combo)
                df[feature_name] = df[list(combo)].product(axis=1)
                self.created_features.append(feature_name)
        
        self.logger.info(f"Created {len(self.created_features)} interaction features")
        
        return df
    
    def create_ratio_features(
        self,
        df: pd.DataFrame,
        numerator_cols: List[str],
        denominator_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create ratio features.
        
        Args:
            df: Input DataFrame
            numerator_cols: Numerator columns
            denominator_cols: Denominator columns
            
        Returns:
            DataFrame with ratio features
        """
        df = df.copy()
        
        self.logger.info(f"Creating ratio features")
        
        for num_col in numerator_cols:
            for den_col in denominator_cols:
                if num_col != den_col:
                    feature_name = f"{num_col}_div_{den_col}"
                    # Avoid division by zero
                    df[feature_name] = df[num_col] / (df[den_col] + 1e-10)
                    self.created_features.append(feature_name)
        
        self.logger.info(f"Created {len(numerator_cols) * len(denominator_cols)} ratio features")
        
        return df
    
    def create_binned_features(
        self,
        df: pd.DataFrame,
        column: str,
        n_bins: int = 5,
        strategy: str = 'quantile'
    ) -> pd.DataFrame:
        """
        Create binned features.
        
        Args:
            df: Input DataFrame
            column: Column to bin
            n_bins: Number of bins
            strategy: Binning strategy ('uniform', 'quantile')
            
        Returns:
            DataFrame with binned feature
        """
        df = df.copy()
        
        feature_name = f"{column}_binned"
        
        if strategy == 'quantile':
            df[feature_name] = pd.qcut(df[column], q=n_bins, labels=False, duplicates='drop')
        else:
            df[feature_name] = pd.cut(df[column], bins=n_bins, labels=False)
        
        self.created_features.append(feature_name)
        self.logger.info(f"Created binned feature: {feature_name}")
        
        return df
    
    def create_aggregated_features(
        self,
        df: pd.DataFrame,
        group_by: str,
        agg_columns: List[str],
        agg_funcs: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """
        Create aggregated features based on grouping.
        
        Args:
            df: Input DataFrame
            group_by: Column to group by
            agg_columns: Columns to aggregate
            agg_funcs: Aggregation functions
            
        Returns:
            DataFrame with aggregated features
        """
        df = df.copy()
        
        self.logger.info(f"Creating aggregated features grouped by {group_by}")
        
        for col in agg_columns:
            for func in agg_funcs:
                feature_name = f"{col}_{func}_by_{group_by}"
                df[feature_name] = df.groupby(group_by)[col].transform(func)
                self.created_features.append(feature_name)
        
        self.logger.info(f"Created {len(agg_columns) * len(agg_funcs)} aggregated features")
        
        return df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        column: str,
        lags: List[int] = [1, 2, 3]
    ) -> pd.DataFrame:
        """
        Create lag features (useful for time series).
        
        Args:
            df: Input DataFrame
            column: Column to create lags from
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        self.logger.info(f"Creating lag features for {column}")
        
        for lag in lags:
            feature_name = f"{column}_lag_{lag}"
            df[feature_name] = df[column].shift(lag)
            self.created_features.append(feature_name)
        
        self.logger.info(f"Created {len(lags)} lag features")
        
        return df
    
    def get_created_features(self) -> List[str]:
        """
        Get list of created features.
        
        Returns:
            List of feature names
        """
        return self.created_features
