"""Main pipeline execution script."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.orchestrator import DataPipeline
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run data engineering pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--file-type', type=str, default='csv', help='File type (csv, json, parquet)')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DataPipeline(config_path=args.config)
    
    # Run pipeline
    results = pipeline.run(
        data_path=args.data,
        target_col=args.target,
        file_type=args.file_type
    )
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*60)
    print(f"Best Model: {results['best_model']}")
    print(f"R² Score: {results['best_metrics']['r2']:.4f}")
    print(f"RMSE: {results['best_metrics']['rmse']:.4f}")
    print(f"MAE: {results['best_metrics']['mae']:.4f}")
    print("="*60)
    
    print("\nModel Comparison:")
    print(results['results'].to_string())
    
    print("\nTo view experiment tracking, run: mlflow ui")


if __name__ == '__main__':
    main()
