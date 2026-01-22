"""Unit tests for data ingestion module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.ingestion import CSVLoader, DataValidator, BatchProcessor


class TestCSVLoader:
    """Test CSV loader functionality."""
    
    def test_load_csv(self, tmp_path):
        """Test loading CSV file."""
        # Create test CSV
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # Load CSV
        loader = CSVLoader()
        loaded_df = loader.load(csv_path)
        
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ['col1', 'col2']
    
    def test_load_in_chunks(self, tmp_path):
        """Test loading CSV in chunks."""
        # Create test CSV
        df = pd.DataFrame({
            'col1': range(100),
            'col2': range(100, 200)
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # Load in chunks
        loader = CSVLoader()
        chunks = list(loader.load_in_chunks(csv_path, chunk_size=25))
        
        assert len(chunks) == 4
        assert all(len(chunk) == 25 for chunk in chunks)


class TestDataValidator:
    """Test data validation functionality."""
    
    def test_validate_schema(self):
        """Test schema validation."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        validator = DataValidator()
        
        # Valid schema
        assert validator.validate_schema(df, ['col1', 'col2'])
        
        # Invalid schema
        assert not validator.validate_schema(df, ['col1', 'col3'])
    
    def test_check_missing_values(self):
        """Test missing value detection."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [1, np.nan, np.nan, 4]
        })
        
        validator = DataValidator()
        missing = validator.check_missing_values(df, threshold=0.3)
        
        assert 'col2' in missing
        assert 'col1' not in missing
    
    def test_check_duplicates(self):
        """Test duplicate detection."""
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })
        
        validator = DataValidator()
        n_duplicates = validator.check_duplicates(df)
        
        assert n_duplicates == 1


class TestBatchProcessor:
    """Test batch processing functionality."""
    
    def test_process_chunks(self):
        """Test chunk processing."""
        # Create test data
        chunks = [
            pd.DataFrame({'col1': [1, 2, 3]}),
            pd.DataFrame({'col1': [4, 5, 6]})
        ]
        
        # Define transform function
        def transform(df):
            df['col2'] = df['col1'] * 2
            return df
        
        processor = BatchProcessor()
        result = processor.process_chunks(iter(chunks), transform)
        
        assert len(result) == 6
        assert 'col2' in result.columns
        assert result['col2'].tolist() == [2, 4, 6, 8, 10, 12]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
