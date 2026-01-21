"""Data loaders for various data sources."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Iterator, Union, List
import logging
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """Base class for data loaders."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize loader.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def load(self, *args, **kwargs) -> pd.DataFrame:
        """Load data and return DataFrame."""
        pass
    
    @abstractmethod
    def load_in_chunks(self, *args, **kwargs) -> Iterator[pd.DataFrame]:
        """Load data in chunks for memory efficiency."""
        pass


class CSVLoader(BaseLoader):
    """Loader for CSV files."""
    
    def load(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load CSV file into DataFrame.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        self.logger.info(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path, **kwargs)
        self.logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df
    
    def load_in_chunks(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 100000,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        Load CSV file in chunks.
        
        Args:
            file_path: Path to CSV file
            chunk_size: Number of rows per chunk
            **kwargs: Additional arguments for pd.read_csv
            
        Yields:
            DataFrame chunks
        """
        self.logger.info(f"Loading CSV file in chunks: {file_path}")
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, **kwargs):
            yield chunk


class JSONLoader(BaseLoader):
    """Loader for JSON files."""
    
    def load(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load JSON file into DataFrame.
        
        Args:
            file_path: Path to JSON file
            **kwargs: Additional arguments for pd.read_json
            
        Returns:
            DataFrame with loaded data
        """
        self.logger.info(f"Loading JSON file: {file_path}")
        df = pd.read_json(file_path, **kwargs)
        self.logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df
    
    def load_in_chunks(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 100000,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        Load JSON file in chunks (for line-delimited JSON).
        
        Args:
            file_path: Path to JSON file
            chunk_size: Number of rows per chunk
            **kwargs: Additional arguments for pd.read_json
            
        Yields:
            DataFrame chunks
        """
        self.logger.info(f"Loading JSON file in chunks: {file_path}")
        for chunk in pd.read_json(file_path, lines=True, chunksize=chunk_size, **kwargs):
            yield chunk


class ParquetLoader(BaseLoader):
    """Loader for Parquet files."""
    
    def load(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load Parquet file into DataFrame.
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional arguments for pd.read_parquet
            
        Returns:
            DataFrame with loaded data
        """
        self.logger.info(f"Loading Parquet file: {file_path}")
        df = pd.read_parquet(file_path, **kwargs)
        self.logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df
    
    def load_in_chunks(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 100000,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        Load Parquet file in chunks.
        
        Args:
            file_path: Path to Parquet file
            chunk_size: Number of rows per chunk
            **kwargs: Additional arguments for pd.read_parquet
            
        Yields:
            DataFrame chunks
        """
        self.logger.info(f"Loading Parquet file in chunks: {file_path}")
        # Read file to get total rows
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)
        
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()


def get_loader(file_type: str, logger: Optional[logging.Logger] = None) -> BaseLoader:
    """
    Factory function to get appropriate loader.
    
    Args:
        file_type: Type of file ('csv', 'json', 'parquet')
        logger: Logger instance
        
    Returns:
        Appropriate loader instance
    """
    loaders = {
        'csv': CSVLoader,
        'json': JSONLoader,
        'parquet': ParquetLoader
    }
    
    loader_class = loaders.get(file_type.lower())
    if loader_class is None:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return loader_class(logger=logger)
