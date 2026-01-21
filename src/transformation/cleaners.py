"""Data cleaning utilities."""

import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict
import logging
from sklearn.impute import SimpleImputer, KNNImputer


class DataCleaner:
    """Clean and preprocess data."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize data cleaner.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.imputers = {}
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'mean',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'mode', 'drop', 'knn')
            columns: Columns to process (None = all numeric columns)
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.logger.info(f"Handling missing values with strategy: {strategy}")
        
        if strategy == 'drop':
            df = df.dropna(subset=columns)
            self.logger.info(f"Dropped rows with missing values. Remaining: {len(df):,}")
        
        elif strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
            df[columns] = imputer.fit_transform(df[columns])
            self.imputers[strategy] = imputer
            self.logger.info(f"Imputed missing values using {strategy}")
        
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df[columns] = imputer.fit_transform(df[columns])
            self.imputers['knn'] = imputer
            self.logger.info("Imputed missing values using KNN")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return df
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Columns to process (None = all numeric)
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.logger.info(f"Removing outliers using {method} method")
        initial_rows = len(df)
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            for col in columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        removed_rows = initial_rows - len(df)
        self.logger.info(f"Removed {removed_rows:,} outlier rows ({removed_rows/initial_rows*100:.2f}%)")
        
        return df
    
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            DataFrame with duplicates removed
        """
        df = df.copy()
        initial_rows = len(df)
        
        df = df.drop_duplicates(subset=subset, keep=keep)
        
        removed_rows = initial_rows - len(df)
        self.logger.info(f"Removed {removed_rows:,} duplicate rows")
        
        return df
    
    def normalize_text(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lowercase: bool = True,
        strip: bool = True
    ) -> pd.DataFrame:
        """
        Normalize text columns.
        
        Args:
            df: Input DataFrame
            columns: Text columns to normalize
            lowercase: Convert to lowercase
            strip: Strip whitespace
            
        Returns:
            DataFrame with normalized text
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if lowercase:
                df[col] = df[col].str.lower()
            
            if strip:
                df[col] = df[col].str.strip()
        
        self.logger.info(f"Normalized {len(columns)} text columns")
        
        return df
    
    def convert_dtypes(
        self,
        df: pd.DataFrame,
        dtype_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Convert column data types.
        
        Args:
            df: Input DataFrame
            dtype_map: Dictionary mapping column names to target dtypes
            
        Returns:
            DataFrame with converted types
        """
        df = df.copy()
        
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                    self.logger.info(f"Converted {col} to {dtype}")
                except Exception as e:
                    self.logger.error(f"Failed to convert {col} to {dtype}: {e}")
        
        return df
