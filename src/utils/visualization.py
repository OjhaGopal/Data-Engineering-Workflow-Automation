"""Visualization utilities for data analysis and model evaluation."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distributions of numerical columns.
    
    Args:
        df: DataFrame to plot
        columns: List of columns to plot (None = all numeric)
        save_path: Path to save figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    # Hide empty subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame to analyze
        save_path: Path to save figure
        figsize: Figure size
    """
    corr = df.select_dtypes(include=[np.number]).corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance values
        top_n: Number of top features to display
        save_path: Path to save figure
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None
) -> None:
    """
    Plot predicted vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot residual analysis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
