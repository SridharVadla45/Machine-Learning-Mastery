"""
Plotting utilities for consistent visualizations across all modules.

This module provides standardized plotting functions for:
- Line plots
- Scatter plots
- Heatmaps
- Loss curves
- Decision boundaries
- Confusion matrices
- And more
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# BASIC PLOTS
# ============================================================================

def plot_line(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Line Plot",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Create a line plot.
    
    Args:
        x: X-axis values
        y: Y-axis values
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save figure (optional)
    
    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig, ax


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    c: Optional[np.ndarray] = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Scatter Plot",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Create a scatter plot.
    
    Args:
        x: X-axis values
        y: Y-axis values
        c: Colors for points (optional)
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(x, y, c=c, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    if c is not None:
        plt.colorbar(scatter, ax=ax)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig, ax


def plot_histogram(
    data: np.ndarray,
    bins: int = 30,
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    title: str = "Histogram",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """Create a histogram."""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig, ax


# ============================================================================
# MACHINE LEARNING PLOTS
# ============================================================================

def plot_loss_curve(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Loss Curve",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot training (and optionally validation) loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig, ax


def plot_regression_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Regression Results",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot true vs predicted values for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig, ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (optional)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Figure and Axes objects
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig, ax


def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    model,
    title: str = "Decision Boundary",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot decision boundary for 2D classification.
    
    Args:
        X: Feature matrix (must be 2D)
        y: Labels
        model: Model with predict method
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Figure and Axes objects
    """
    assert X.shape[1] == 2, "X must have exactly 2 features for decision boundary plot"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mesh
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=50, 
                        edgecolors='black', linewidth=1, cmap='RdYlBu')
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, ax=ax)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig, ax


# ============================================================================
# MATRIX PLOTS
# ============================================================================

def plot_heatmap(
    matrix: np.ndarray,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Heatmap",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'coolwarm',
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot a heatmap of a matrix.
    
    Args:
        matrix: 2D array to visualize
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        save_path: Path to save figure
    
    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig, ax


# ============================================================================
# MULTIPLE PLOTS
# ============================================================================

def plot_multiple_lines(
    x: np.ndarray,
    y_dict: dict,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Multiple Lines",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot multiple lines on same plot.
    
    Args:
        x: X-axis values
        y_dict: Dictionary mapping labels to y-values
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for label, y in y_dict.items():
        ax.plot(x, y, label=label, linewidth=2)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig, ax


# ============================================================================
# UTILITIES
# ============================================================================

def save_figure(fig: Figure, path: str, dpi: int = 300) -> None:
    """
    Save figure to file.
    
    Args:
        fig: Figure object
        path: Path to save to
        dpi: Resolution in dots per inch
    """
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to: {path}")


def show_all() -> None:
    """Display all open figures."""
    plt.show()


def close_all() -> None:
    """Close all open figures."""
    plt.close('all')


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing plotting utilities...\n")
    
    # Test line plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fig, ax = plot_line(x, y, xlabel="Time", ylabel="Amplitude", 
                        title="Sine Wave")
    
    # Test scatter plot
    x = np.random.randn(100)
    y = np.random.randn(100)
    c = np.random.rand(100)
    fig, ax = plot_scatter(x, y, c=c, title="Random Scatter")
    
    # Test loss curve
    train_losses = [1.0 / (i + 1) for i in range(50)]
    val_losses = [1.2 / (i + 1) for i in range(50)]
    fig, ax = plot_loss_curve(train_losses, val_losses)
    
    print("✅ All plotting utilities working!")
    print("Displaying plots... (close windows to continue)")
    
    plt.show()
