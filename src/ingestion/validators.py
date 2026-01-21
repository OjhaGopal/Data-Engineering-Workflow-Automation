"""Data validation utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging


class DataValidator:
    """Validates data quality and schema."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize validator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.validation_results = {}
    
    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        strict: bool = False
    ) -> bool:
        """
        Validate DataFrame schema.
        
        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            strict: If True, DataFrame must have exactly these columns
            
        Returns:
            True if schema is valid
        """
        actual_columns = set(df.columns)
        expected_columns_set = set(expected_columns)
        
        missing_columns = expected_columns_set - actual_columns
        extra_columns = actual_columns - expected_columns_set
        
        if missing_columns:
            self.logger.error(f"Missing columns: {missing_columns}")
            self.validation_results['missing_columns'] = list(missing_columns)
            return False
        
        if strict and extra_columns:
            self.logger.error(f"Unexpected columns: {extra_columns}")
            self.validation_results['extra_columns'] = list(extra_columns)
            return False
        
        self.logger.info("Schema validation passed")
        return True
    
    def check_missing_values(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
        """
        Check for missing values.
        
        Args:
            df: DataFrame to check
            threshold: Maximum allowed missing value ratio
            
        Returns:
            Dictionary of columns with missing value ratios
        """
        missing_ratios = df.isnull().sum() / len(df)
        problematic_columns = missing_ratios[missing_ratios > threshold].to_dict()
        
        if problematic_columns:
            self.logger.warning(
                f"Columns with >{threshold*100}% missing values: {problematic_columns}"
            )
        
        self.validation_results['missing_values'] = missing_ratios.to_dict()
        return problematic_columns
    
    def check_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
        """
        Check for duplicate rows.
        
        Args:
            df: DataFrame to check
            subset: Columns to consider for duplicates
            
        Returns:
            Number of duplicate rows
        """
        n_duplicates = df.duplicated(subset=subset).sum()
        
        if n_duplicates > 0:
            self.logger.warning(f"Found {n_duplicates:,} duplicate rows")
        else:
            self.logger.info("No duplicate rows found")
        
        self.validation_results['n_duplicates'] = n_duplicates
        return n_duplicates
    
    def check_data_types(
        self,
        df: pd.DataFrame,
        expected_types: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Check if columns have expected data types.
        
        Args:
            df: DataFrame to check
            expected_types: Dictionary mapping column names to expected types
            
        Returns:
            Dictionary of columns with incorrect types
        """
        type_mismatches = {}
        
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                continue
            
            actual_type = str(df[col].dtype)
            if expected_type not in actual_type:
                type_mismatches[col] = f"Expected {expected_type}, got {actual_type}"
                self.logger.warning(f"Type mismatch in {col}: {type_mismatches[col]}")
        
        self.validation_results['type_mismatches'] = type_mismatches
        return type_mismatches
    
    def check_value_ranges(
        self,
        df: pd.DataFrame,
        ranges: Dict[str, tuple]
    ) -> Dict[str, int]:
        """
        Check if numeric columns are within expected ranges.
        
        Args:
            df: DataFrame to check
            ranges: Dictionary mapping column names to (min, max) tuples
            
        Returns:
            Dictionary of columns with out-of-range value counts
        """
        out_of_range = {}
        
        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                continue
            
            out_of_range_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if out_of_range_count > 0:
                out_of_range[col] = out_of_range_count
                self.logger.warning(
                    f"{col}: {out_of_range_count:,} values outside range [{min_val}, {max_val}]"
                )
        
        self.validation_results['out_of_range'] = out_of_range
        return out_of_range
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate validation report.
        
        Returns:
            Dictionary with validation results
        """
        return self.validation_results
