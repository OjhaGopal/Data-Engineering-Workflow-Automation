"""Transformation pipeline for data preprocessing."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
import logging
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class TransformationPipeline:
    """Reusable transformation pipeline."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize transformation pipeline.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.steps = []
        self.fitted_transformers = {}
    
    def add_step(
        self,
        name: str,
        transform_func: Callable,
        **kwargs
    ) -> 'TransformationPipeline':
        """
        Add a transformation step.
        
        Args:
            name: Step name
            transform_func: Transformation function
            **kwargs: Arguments for the transformation function
            
        Returns:
            Self for method chaining
        """
        self.steps.append({
            'name': name,
            'func': transform_func,
            'kwargs': kwargs
        })
        self.logger.info(f"Added step: {name}")
        return self
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        df = df.copy()
        
        self.logger.info(f"Executing {len(self.steps)} transformation steps")
        
        for step in self.steps:
            step_name = step['name']
            self.logger.info(f"Executing step: {step_name}")
            
            try:
                df = step['func'](df, **step['kwargs'])
            except Exception as e:
                self.logger.error(f"Error in step {step_name}: {e}")
                raise
        
        self.logger.info("Pipeline execution completed")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted transformers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        return self.fit_transform(df)
    
    def save(self, path: str) -> None:
        """
        Save pipeline to disk.
        
        Args:
            path: Path to save pipeline
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'steps': self.steps,
            'fitted_transformers': self.fitted_transformers
        }, path)
        
        self.logger.info(f"Saved pipeline to {path}")
    
    @classmethod
    def load(cls, path: str, logger: Optional[logging.Logger] = None) -> 'TransformationPipeline':
        """
        Load pipeline from disk.
        
        Args:
            path: Path to pipeline file
            logger: Logger instance
            
        Returns:
            Loaded pipeline
        """
        data = joblib.load(path)
        
        pipeline = cls(logger=logger)
        pipeline.steps = data['steps']
        pipeline.fitted_transformers = data['fitted_transformers']
        
        if logger:
            logger.info(f"Loaded pipeline from {path}")
        
        return pipeline


class FeatureScaler:
    """Scale features using various methods."""
    
    def __init__(
        self,
        method: str = 'standard',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize feature scaler.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
            logger: Logger instance
        """
        self.method = method
        self.logger = logger or logging.getLogger(__name__)
        
        scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler
        }
        
        if method not in scalers:
            raise ValueError(f"Unknown scaling method: {method}")
        
        self.scaler = scalers[method]()
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit and transform features.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale (None = all numeric)
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.logger.info(f"Scaling {len(columns)} features using {self.method}")
        
        df[columns] = self.scaler.fit_transform(df[columns])
        
        return df
    
    def transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df[columns] = self.scaler.transform(df[columns])
        
        return df


class CategoricalEncoder:
    """Encode categorical variables."""
    
    def __init__(
        self,
        method: str = 'onehot',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize categorical encoder.
        
        Args:
            method: Encoding method ('onehot', 'label')
            logger: Logger instance
        """
        self.method = method
        self.logger = logger or logging.getLogger(__name__)
        self.encoders = {}
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit and transform categorical features.
        
        Args:
            df: Input DataFrame
            columns: Columns to encode (None = all object/category columns)
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.logger.info(f"Encoding {len(columns)} categorical features using {self.method}")
        
        if self.method == 'label':
            for col in columns:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
        
        elif self.method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
        
        return df
    
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Transform categorical features using fitted encoders.
        
        Args:
            df: Input DataFrame
            columns: Columns to encode
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        if self.method == 'label':
            for col in columns:
                if col in self.encoders:
                    df[col] = self.encoders[col].transform(df[col].astype(str))
        
        return df
