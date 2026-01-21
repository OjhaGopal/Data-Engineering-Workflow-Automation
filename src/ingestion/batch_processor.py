"""Batch processing utilities for large datasets."""

import pandas as pd
import numpy as np
from typing import Callable, Iterator, Optional, List, Any
from pathlib import Path
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


class BatchProcessor:
    """Process large datasets in batches for memory efficiency."""
    
    def __init__(
        self,
        chunk_size: int = 100000,
        n_jobs: int = -1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            chunk_size: Number of rows per batch
            n_jobs: Number of parallel jobs (-1 = all cores)
            logger: Logger instance
        """
        self.chunk_size = chunk_size
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.logger = logger or logging.getLogger(__name__)
    
    def process_chunks(
        self,
        chunks: Iterator[pd.DataFrame],
        transform_func: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Optional[str] = None,
        show_progress: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Process data chunks with a transformation function.
        
        Args:
            chunks: Iterator of DataFrame chunks
            transform_func: Function to apply to each chunk
            output_path: Path to save processed data (None = return DataFrame)
            show_progress: Whether to show progress bar
            
        Returns:
            Concatenated DataFrame if output_path is None
        """
        processed_chunks = []
        
        self.logger.info(f"Processing chunks with {self.n_jobs} workers")
        
        chunk_list = list(chunks)
        iterator = tqdm(chunk_list, desc="Processing chunks") if show_progress else chunk_list
        
        for chunk in iterator:
            try:
                processed_chunk = transform_func(chunk)
                processed_chunks.append(processed_chunk)
            except Exception as e:
                self.logger.error(f"Error processing chunk: {e}")
                raise
        
        # Combine results
        if processed_chunks:
            result = pd.concat(processed_chunks, ignore_index=True)
            self.logger.info(f"Processed {len(result):,} total rows")
            
            if output_path:
                self.save_data(result, output_path)
                return None
            else:
                return result
        else:
            self.logger.warning("No chunks were processed")
            return pd.DataFrame()
    
    def process_parallel(
        self,
        chunks: List[pd.DataFrame],
        transform_func: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Process chunks in parallel.
        
        Args:
            chunks: List of DataFrame chunks
            transform_func: Function to apply to each chunk
            output_path: Path to save processed data
            
        Returns:
            Concatenated DataFrame if output_path is None
        """
        self.logger.info(f"Processing {len(chunks)} chunks in parallel with {self.n_jobs} workers")
        
        processed_chunks = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(transform_func, chunk): i for i, chunk in enumerate(chunks)}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    result = future.result()
                    processed_chunks.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel processing: {e}")
                    raise
        
        # Combine results
        if processed_chunks:
            result = pd.concat(processed_chunks, ignore_index=True)
            self.logger.info(f"Processed {len(result):,} total rows")
            
            if output_path:
                self.save_data(result, output_path)
                return None
            else:
                return result
        else:
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_ext = output_path.suffix.lower()
        
        if file_ext == '.csv':
            df.to_csv(output_path, index=False)
        elif file_ext == '.parquet':
            df.to_parquet(output_path, index=False)
        elif file_ext == '.json':
            df.to_json(output_path, orient='records', lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        self.logger.info(f"Saved data to {output_path}")
    
    def estimate_memory_usage(self, df: pd.DataFrame) -> str:
        """
        Estimate memory usage of DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Human-readable memory usage string
        """
        memory_bytes = df.memory_usage(deep=True).sum()
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if memory_bytes < 1024.0:
                return f"{memory_bytes:.2f} {unit}"
            memory_bytes /= 1024.0
        
        return f"{memory_bytes:.2f} TB"
