"""Main pipeline orchestrator."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import yaml

from src.utils.logger import setup_logger, load_config
from src.ingestion import get_loader, DataValidator, BatchProcessor
from src.transformation import DataCleaner, TransformationPipeline, FeatureScaler, CategoricalEncoder
from src.features import FeatureEngineer, FeatureSelector
from src.models import ModelTrainer, ModelEvaluator, HyperparameterTuner

from sklearn.model_selection import train_test_split


class DataPipeline:
    """End-to-end data engineering pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(
            name="DataPipeline",
            log_file=self.config['logging']['file'],
            level=self.config['logging']['level']
        )
        
        self.logger.info("="*60)
        self.logger.info("Data Engineering Pipeline Initialized")
        self.logger.info("="*60)
        
        # Initialize components
        self.validator = DataValidator(logger=self.logger)
        self.cleaner = DataCleaner(logger=self.logger)
        self.feature_engineer = FeatureEngineer(logger=self.logger)
        self.feature_selector = FeatureSelector(logger=self.logger)
        self.trainer = ModelTrainer(
            experiment_name=self.config['mlflow']['experiment_name'],
            logger=self.logger
        )
        self.evaluator = ModelEvaluator(logger=self.logger)
        
        # Data storage
        self.data = {}
    
    def load_data(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            file_path: Path to data file
            file_type: Type of file ('csv', 'json', 'parquet')
            
        Returns:
            Loaded DataFrame
        """
        self.logger.info(f"Loading data from {file_path}")
        
        loader = get_loader(file_type, logger=self.logger)
        df = loader.load(file_path)
        
        self.data['raw'] = df
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        self.logger.info("Validating data")
        
        # Check for missing values
        self.validator.check_missing_values(df, threshold=0.5)
        
        # Check for duplicates
        self.validator.check_duplicates(df)
        
        return True
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning data")
        
        # Handle missing values
        df = self.cleaner.handle_missing_values(
            df,
            strategy=self.config['transformation']['missing_value_strategy']
        )
        
        # Remove duplicates
        df = self.cleaner.remove_duplicates(df)
        
        # Remove outliers
        df = self.cleaner.remove_outliers(
            df,
            method=self.config['transformation']['outlier_method'],
            threshold=self.config['transformation']['outlier_threshold']
        )
        
        self.data['cleaned'] = df
        return df
    
    def engineer_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Engineer features.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Engineering features")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # Create interaction features if configured
        if self.config['features']['create_interactions'] and len(numeric_cols) > 1:
            df = self.feature_engineer.create_interaction_features(
                df,
                numeric_cols[:5],  # Limit to avoid explosion
                max_interactions=2
            )
        
        # Create polynomial features if configured
        if self.config['features']['polynomial_degree'] > 1 and len(numeric_cols) > 0:
            df = self.feature_engineer.create_polynomial_features(
                df,
                numeric_cols[:3],  # Limit to avoid explosion
                degree=self.config['features']['polynomial_degree']
            )
        
        self.data['engineered'] = df
        return df
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Select important features.
        
        Args:
            X: Feature DataFrame
            y: Target series
            n_features: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        self.logger.info("Selecting features")
        
        n_features = n_features or self.config['features']['n_features']
        
        # Select features by importance
        selected_features = self.feature_selector.select_by_importance(
            X, y, n_features=min(n_features, len(X.columns))
        )
        
        return X[selected_features]
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> tuple:
        """
        Prepare data for modeling.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Preparing data for modeling")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state']
        )
        
        # Scale features if configured
        if self.config['features']['scale_features']:
            scaler = FeatureScaler(
                method=self.config['features']['scaler_type'],
                logger=self.logger
            )
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        self.logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Train multiple models.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        self.logger.info("Training models")
        
        # Define model configurations
        model_configs = {}
        for model_name in self.config['model']['models']:
            model_configs[model_name] = {}
        
        # Train models
        results = self.trainer.train_multiple_models(
            X_train, y_train,
            model_configs,
            cv_folds=self.config['model']['cv_folds']
        )
        
        # Extract models
        models = {name: result[0] for name, result in results.items()}
        
        return models
    
    def evaluate_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate trained models.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with evaluation results
        """
        self.logger.info("Evaluating models")
        
        results = self.evaluator.evaluate_multiple_models(models, X_test, y_test)
        
        return results
    
    def run(
        self,
        data_path: str,
        target_col: str,
        file_type: str = 'csv'
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            data_path: Path to data file
            target_col: Target column name
            file_type: Type of data file
            
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting pipeline execution")
        
        # Load data
        df = self.load_data(data_path, file_type)
        
        # Validate data
        self.validate_data(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df, target_col)
        
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Select features
        X = self.select_features(X, y)
        
        # Add target back
        df = X.copy()
        df[target_col] = y
        
        # Prepare train/test split
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        # Train models
        models = self.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(models, X_test, y_test)
        
        # Get best model
        best_model_name, best_metrics = self.evaluator.get_best_model()
        
        self.logger.info("="*60)
        self.logger.info(f"Pipeline execution completed!")
        self.logger.info(f"Best model: {best_model_name}")
        self.logger.info(f"Best R²: {best_metrics['r2']:.4f}")
        self.logger.info("="*60)
        
        return {
            'models': models,
            'results': results,
            'best_model': best_model_name,
            'best_metrics': best_metrics
        }
