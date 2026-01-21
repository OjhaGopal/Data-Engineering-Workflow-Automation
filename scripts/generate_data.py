"""Generate synthetic dataset for demonstration."""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def generate_regression_dataset(
    n_samples: int = 1000000,
    n_features: int = 20,
    n_informative: int = 15,
    noise: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic regression dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of informative features
        noise: Noise level
        random_state: Random state
        
    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(random_state)
    
    print(f"Generating dataset with {n_samples:,} samples and {n_features} features...")
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate coefficients for informative features
    coef = np.zeros(n_features)
    coef[:n_informative] = np.random.randn(n_informative) * 10
    
    # Generate target with some non-linearity
    y = np.dot(X, coef)
    
    # Add polynomial terms for some features
    y += 0.5 * X[:, 0] ** 2
    y += 0.3 * X[:, 1] * X[:, 2]
    
    # Add noise
    y += np.random.randn(n_samples) * noise * np.std(y)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features
    df['category_1'] = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
    df['category_2'] = np.random.choice(['X', 'Y', 'Z'], size=n_samples)
    
    # Add some missing values (5%)
    for col in feature_names[:5]:
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
    
    # Add some duplicates (1%)
    n_duplicates = int(n_samples * 0.01)
    duplicate_indices = np.random.choice(n_samples, size=n_duplicates, replace=False)
    df = pd.concat([df, df.iloc[duplicate_indices]], ignore_index=True)
    
    print(f"Generated dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--samples', type=int, default=1000000, help='Number of samples')
    parser.add_argument('--features', type=int, default=20, help='Number of features')
    parser.add_argument('--output', type=str, default='data/raw/synthetic_data.csv', help='Output path')
    
    args = parser.parse_args()
    
    # Generate dataset
    df = generate_regression_dataset(
        n_samples=args.samples,
        n_features=args.features
    )
    
    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")


if __name__ == '__main__':
    main()
