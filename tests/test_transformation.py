"""Unit tests for transformation module."""

import pytest
import pandas as pd
import numpy as np

from src.transformation import DataCleaner, FeatureScaler, CategoricalEncoder


class TestDataCleaner:
    """Test data cleaning functionality."""
    
    def test_handle_missing_values_mean(self):
        """Test missing value imputation with mean."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [10, np.nan, 30, 40]
        })
        
        cleaner = DataCleaner()
        result = cleaner.handle_missing_values(df, strategy='mean')
        
        assert result['col1'].isnull().sum() == 0
        assert result['col2'].isnull().sum() == 0
        assert np.isclose(result['col1'].iloc[2], 2.33, atol=0.1)
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })
        
        cleaner = DataCleaner()
        result = cleaner.remove_duplicates(df)
        
        assert len(result) == 3
    
    def test_remove_outliers_iqr(self):
        """Test outlier removal using IQR."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        
        cleaner = DataCleaner()
        result = cleaner.remove_outliers(df, method='iqr', threshold=1.5)
        
        assert len(result) < len(df)
        assert 100 not in result['col1'].values


class TestFeatureScaler:
    """Test feature scaling functionality."""
    
    def test_standard_scaler(self):
        """Test standard scaling."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        scaler = FeatureScaler(method='standard')
        result = scaler.fit_transform(df)
        
        # Check mean is close to 0 and std is close to 1
        assert np.isclose(result['col1'].mean(), 0, atol=1e-10)
        assert np.isclose(result['col1'].std(), 1, atol=0.1)
    
    def test_minmax_scaler(self):
        """
        Verify that FeatureScaler with method 'minmax' scales a numeric column to the [0, 1] range.
        """
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5]
        })
        
        scaler = FeatureScaler(method='minmax')
        result = scaler.fit_transform(df)
        
        # Check values are between 0 and 1
        assert result['col1'].min() == 0
        assert result['col1'].max() == 1


class TestCategoricalEncoder:
    """Test categorical encoding functionality."""
    
    def test_label_encoder(self):
        """Test label encoding."""
        df = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'a', 'b']
        })
        
        encoder = CategoricalEncoder(method='label')
        result = encoder.fit_transform(df)
        
        # Check all values are numeric
        assert result['col1'].dtype in [np.int32, np.int64]
        assert len(result['col1'].unique()) == 3
    
    def test_onehot_encoder(self):
        """Test one-hot encoding."""
        df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': [1, 2, 3]
        })
        
        encoder = CategoricalEncoder(method='onehot')
        result = encoder.fit_transform(df, columns=['col1'])
        
        # Check new columns were created
        assert len(result.columns) > len(df.columns)
        assert 'col2' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])