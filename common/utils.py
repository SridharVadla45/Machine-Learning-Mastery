"""
Common utility functions used across all modules.

This module provides helper functions for:
- Data loading and preprocessing
- Common calculations
- File management
- Progress tracking
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Union
import json
import pickle


# ============================================================================
# PATH MANAGEMENT
# ============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_datasets_dir() -> Path:
    """Get the datasets directory, create if doesn't exist."""
    datasets_dir = get_project_root() / "common" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir


def get_module_dir(module_name: str) -> Path:
    """Get specific module directory."""
    return get_project_root() / module_name


# ============================================================================
# DATA LOADING
# ============================================================================

def load_csv(filename: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file from the datasets directory.
    
    Args:
        filename: Name of the CSV file
        **kwargs: Additional arguments for pd.read_csv
    
    Returns:
        DataFrame with the loaded data
    """
    filepath = get_datasets_dir() / filename
    return pd.read_csv(filepath, **kwargs)


def save_csv(df: pd.DataFrame, filename: str, **kwargs) -> None:
    """
    Save a DataFrame to the datasets directory.
    
    Args:
        df: DataFrame to save
        filename: Name for the CSV file
        **kwargs: Additional arguments for df.to_csv
    """
    filepath = get_datasets_dir() / filename
    df.to_csv(filepath, index=False, **kwargs)


def load_numpy(filename: str) -> np.ndarray:
    """Load a NumPy array from file."""
    filepath = get_datasets_dir() / filename
    return np.load(filepath)


def save_numpy(arr: np.ndarray, filename: str) -> None:
    """Save a NumPy array to file."""
    filepath = get_datasets_dir() / filename
    np.save(filepath, arr)


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_linear_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic linear regression data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility
    
    Returns:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = np.random.randn()
    
    y = X @ true_weights + true_bias + noise * np.random.randn(n_samples)
    
    return X, y


def generate_classification_data(
    n_samples: int = 100,
    n_features: int = 2,
    n_classes: int = 2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        random_state: Random seed
    
    Returns:
        X: Feature matrix
        y: Class labels
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    samples_per_class = n_samples // n_classes
    X_list = []
    y_list = []
    
    for i in range(n_classes):
        # Generate cluster center
        center = np.random.randn(n_features) * 3
        # Generate samples around center
        X_class = center + np.random.randn(samples_per_class, n_features)
        X_list.append(X_class)
        y_list.append(np.full(samples_per_class, i))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing (0.0 to 1.0)
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def normalize(X: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, float, float]:
    """
    Normalize data to zero mean and unit variance.
    
    Args:
        X: Input array
        axis: Axis along which to normalize
    
    Returns:
        Normalized array, mean, standard deviation
    """
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    
    X_normalized = (X - mean) / std
    
    return X_normalized, mean.squeeze(), std.squeeze()


def standardize(X: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Alias for normalize."""
    return normalize(X)


# ============================================================================
# METRICS
# ============================================================================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MSE value
    """
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy."""
    return np.mean(y_true == y_pred)


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(model: object, filename: str) -> None:
    """
    Save a model using pickle.
    
    Args:
        model: Model object to save
        filename: Name of file to save to
    """
    filepath = get_datasets_dir() / filename
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename: str) -> object:
    """
    Load a model from pickle file.
    
    Args:
        filename: Name of file to load from
    
    Returns:
        Loaded model object
    """
    filepath = get_datasets_dir() / filename
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ============================================================================
# LOGGING AND PROGRESS
# ============================================================================

def print_section(title: str, char: str = "=") -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        char: Character to use for the line
    """
    print(f"\n{char * 60}")
    print(f"{title:^60}")
    print(f"{char * 60}\n")


def print_metrics(metrics: dict) -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
    """
    print("\nMetrics:")
    print("-" * 40)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:.<30} {value:.6f}")
        else:
            print(f"{name:.<30} {value}")
    print("-" * 40)


# ============================================================================
# ARRAY UTILITIES
# ============================================================================

def ensure_2d(X: np.ndarray) -> np.ndarray:
    """
    Ensure array is 2D.
    
    Args:
        X: Input array
    
    Returns:
        2D array
    """
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def softmax(X: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax values for array.
    
    Args:
        X: Input array
        axis: Axis along which to compute softmax
    
    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    X_shifted = X - np.max(X, axis=axis, keepdims=True)
    exp_X = np.exp(X_shifted)
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)


def sigmoid(X: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid function.
    
    Args:
        X: Input array
    
    Returns:
        Sigmoid values
    """
    return 1 / (1 + np.exp(-X))


def relu(X: np.ndarray) -> np.ndarray:
    """
    Compute ReLU activation.
    
    Args:
        X: Input array
    
    Returns:
        ReLU values
    """
    return np.maximum(0, X)


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def check_gradient(
    f,
    grad_f,
    X: np.ndarray,
    epsilon: float = 1e-7
) -> float:
    """
    Check gradient implementation using finite differences.
    
    Args:
        f: Function that computes the loss
        grad_f: Function that computes the gradient
        X: Point at which to check gradient
        epsilon: Step size for finite differences
    
    Returns:
        Relative error between analytical and numerical gradients
    """
    analytical_grad = grad_f(X)
    numerical_grad = np.zeros_like(X)
    
    it = np.nditer(X, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        
        old_value = X[idx]
        
        X[idx] = old_value + epsilon
        f_plus = f(X)
        
        X[idx] = old_value - epsilon
        f_minus = f(X)
        
        numerical_grad[idx] = (f_plus - f_minus) / (2 * epsilon)
        
        X[idx] = old_value
        it.iternext()
    
    # Compute relative error
    numerator = np.linalg.norm(analytical_grad - numerical_grad)
    denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    
    return numerator / denominator if denominator > 0 else 0


# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    # Test utilities
    print_section("Testing Common Utilities")
    
    # Test data generation
    X, y = generate_linear_data(n_samples=50, random_state=42)
    print(f"Generated linear data: X shape {X.shape}, y shape {y.shape}")
    
    # Test train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Test normalization
    X_norm, mean, std = normalize(X)
    print(f"Normalized data: mean={mean:.6f}, std={std:.6f}")
    
    # Test metrics
    y_pred = y + np.random.randn(len(y)) * 0.1
    metrics = {
        'MSE': mse(y, y_pred),
        'RMSE': rmse(y, y_pred),
        'MAE': mae(y, y_pred),
        'R²': r2_score(y, y_pred)
    }
    print_metrics(metrics)
    
    print("\n✅ All utilities working correctly!\n")
