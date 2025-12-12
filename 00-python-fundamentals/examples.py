#!/usr/bin/env python3
"""
Python Fundamentals for Machine Learning - Examples

This file contains fully working examples demonstrating:
1. NumPy operations
2. Pandas data manipulation
3. Data visualization
4. Vectorization techniques

Run this file to see all examples in action:
    uv run 00-python-fundamentals/examples.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def print_section(title):
    """Helper to print section headers."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}\n")


# ============================================================================
# SECTION 1: NumPy Basics
# ============================================================================

def numpy_basics():
    """Demonstrate basic NumPy array operations."""
    print_section("NumPy Basics")
    
    # Creating arrays
    print("Creating Arrays:")
    print("-" * 40)
    
    # From Python list
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"1D array: {arr1}")
    
    # 2D array
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"\n2D array:\n{arr2}")
    
    # Special arrays
    zeros = np.zeros((2, 3))
    print(f"\nZeros (2×3):\n{zeros}")
    
    ones = np.ones((3, 2))
    print(f"\nOnes (3×2):\n{ones}")
    
    identity = np.eye(3)
    print(f"\nIdentity (3×3):\n{identity}")
    
    # Range and linspace
    range_arr = np.arange(0, 10, 2)
    print(f"\nArange (0 to 10, step 2): {range_arr}")
    
    linspace_arr = np.linspace(0, 1, 5)
    print(f"Linspace (0 to 1, 5 points): {linspace_arr}")
    
    # Random arrays
    random_arr = np.random.randn(3, 3)
    print(f"\nRandom normal (3×3):\n{random_arr}")
    
    # Array attributes
    print("\nArray Attributes:")
    print("-" * 40)
    print(f"Shape: {arr2.shape}")
    print(f"Number of dimensions: {arr2.ndim}")
    print(f"Total elements: {arr2.size}")
    print(f"Data type: {arr2.dtype}")


def numpy_indexing():
    """Demonstrate array indexing and slicing."""
    print_section("NumPy Indexing & Slicing")
    
    # 1D indexing
    arr = np.array([10, 20, 30, 40, 50])
    print(f"Array: {arr}")
    print(f"First element: {arr[0]}")
    print(f"Last element: {arr[-1]}")
    print(f"Slice [1:4]: {arr[1:4]}")
    print(f"Every other element: {arr[::2]}")
    
    # 2D indexing
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"\n2D Array:\n{arr2d}")
    print(f"Element at [0, 1]: {arr2d[0, 1]}")
    print(f"First row: {arr2d[0, :]}")
    print(f"Second column: {arr2d[:, 1]}")
    print(f"Top-left 2×2:\n{arr2d[:2, :2]}")
    
    # Boolean indexing
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"\nOriginal: {values}")
    
    mask = values > 5
    print(f"Mask (values > 5): {mask}")
    print(f"Filtered values: {values[mask]}")
    
    # Advanced: Multiple conditions
    mask = (values > 3) & (values < 8)
    print(f"Values between 3 and 8: {values[mask]}")


def numpy_operations():
    """Demonstrate array operations."""
    print_section("NumPy Operations")
    
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])
    
    print(f"a = {a}")
    print(f"b = {b}")
    print()
    
    # Element-wise operations
    print("Element-wise Operations:")
    print("-" * 40)
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    print(f"a ** 2 = {a ** 2}")
    
    # Aggregations
    print("\nAggregations:")
    print("-" * 40)
    print(f"Sum: {a.sum()}")
    print(f"Mean: {a.mean()}")
    print(f"Standard deviation: {a.std():.4f}")
    print(f"Min: {a.min()}")
    print(f"Max: {a.max()}")
    print(f"Argmax (index of max): {a.argmax()}")
    
    # 2D operations
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    
    print(f"\nMatrix:\n{matrix}")
    print(f"Sum of all elements: {matrix.sum()}")
    print(f"Sum along axis 0 (columns): {matrix.sum(axis=0)}")
    print(f"Sum along axis 1 (rows): {matrix.sum(axis=1)}")
    print(f"Mean of each column: {matrix.mean(axis=0)}")


def numpy_broadcasting():
    """Demonstrate broadcasting."""
    print_section("NumPy Broadcasting")
    
    # Example 1: Scalar and array
    arr = np.array([1, 2, 3])
    print(f"Array: {arr}")
    print(f"Array + 10 = {arr + 10}")
    print(f"Array * 2 = {arr * 2}")
    
    # Example 2: 1D and 2D
    row = np.array([1, 2, 3])
    col = np.array([[10], [20], [30]])
    
    print(f"\nRow vector: {row}")
    print(f"Column vector:\n{col}")
    print(f"Row + Column:\n{row + col}")
    
    # Example 3: Normalization (common in ML)
    data = np.random.randn(5, 3)
    print(f"\nData (5 samples, 3 features):\n{data}")
    
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    
    print(f"Mean of each feature: {mean}")
    print(f"Std of each feature: {std}")
    
    # Normalize using broadcasting
    normalized = (data - mean) / std
    print(f"Normalized data:\n{normalized}")
    print(f"Normalized mean: {normalized.mean(axis=0)}")  # Should be ~0
    print(f"Normalized std: {normalized.std(axis=0)}")    # Should be ~1


def numpy_linear_algebra():
    """Demonstrate linear algebra operations."""
    print_section("NumPy Linear Algebra")
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"Matrix A:\n{A}")
    print(f"\nMatrix B:\n{B}")
    
    # Matrix multiplication
    C = A @ B  # or np.dot(A, B)
    print(f"\nA @ B =\n{C}")
    
    # Transpose
    print(f"\nA transpose:\n{A.T}")
    
    # Determinant
    det = np.linalg.det(A)
    print(f"\nDeterminant of A: {det}")
    
    # Inverse
    A_inv = np.linalg.inv(A)
    print(f"\nInverse of A:\n{A_inv}")
    
    # Verify: A @ A_inv = I
    identity = A @ A_inv
    print(f"\nA @ A_inv (should be identity):\n{identity}")
    
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    
    # Solve linear system: Ax = b
    b = np.array([1, 2])
    x = np.linalg.solve(A, b)
    print(f"\nSolve Ax = b where b = {b}")
    print(f"Solution x = {x}")
    print(f"Verification A @ x = {A @ x}")


# ============================================================================
# SECTION 2: Pandas Basics
# ============================================================================

def pandas_basics():
    """Demonstrate basic Pandas operations."""
    print_section("Pandas Basics")
    
    # Create a Series
    s = pd.Series([1, 3, 5, 7, 9])
    print("Series:")
    print(s)
    
    # Series with custom index
    s = pd.Series([1, 3, 5], index=['a', 'b', 'c'])
    print("\nSeries with custom index:")
    print(s)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'],
        'score': [85, 90, 75, 88, 92]
    })
    
    print("\nDataFrame:")
    print(df)
    
    # Basic info
    print("\nDataFrame Info:")
    print("-" * 40)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    print("\nDescriptive Statistics:")
    print(df.describe())


def pandas_selection():
    """Demonstrate data selection in Pandas."""
    print_section("Pandas Data Selection")
    
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'],
        'score': [85, 90, 75, 88, 92]
    })
    
    print("Original DataFrame:")
    print(df)
    
    # Column selection
    print("\nSelect 'age' column:")
    print(df['age'])
    
    print("\nSelect multiple columns:")
    print(df[['name', 'score']])
    
    # Row selection by position
    print("\nFirst row (iloc):")
    print(df.iloc[0])
    
    print("\nFirst 3 rows:")
    print(df.iloc[:3])
    
    # Boolean indexing
    print("\nPeople older than 30:")
    print(df[df['age'] > 30])
    
    print("\nPeople in NYC:")
    print(df[df['city'] == 'NYC'])
    
    # Multiple conditions
    print("\nPeople in NYC with score > 80:")
    print(df[(df['city'] == 'NYC') & (df['score'] > 80)])


def pandas_manipulation():
    """Demonstrate data manipulation."""
    print_section("Pandas Data Manipulation")
    
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'],
        'score': [85, 90, 75, 88, 92]
    })
    
    print("Original DataFrame:")
    print(df)
    
    # Add new column
    df['age_squared'] = df['age'] ** 2
    print("\nAfter adding 'age_squared' column:")
    print(df)
    
    # Modify column
    df['age'] = df['age'] + 1
    print("\nAfter incrementing ages:")
    print(df[['name', 'age']])
    
    # Drop column
    df = df.drop('age_squared', axis=1)
    print("\nAfter dropping 'age_squared':")
    print(df)
    
    # Sort
    df_sorted = df.sort_values('score', ascending=False)
    print("\nSorted by score (descending):")
    print(df_sorted)
    
    # GroupBy
    print("\nAverage score by city:")
    print(df.groupby('city')['score'].mean())
    
    print("\nMultiple aggregations by city:")
    print(df.groupby('city').agg({
        'score': ['mean', 'min', 'max'],
        'age': 'mean'
    }))


def pandas_missing_data():
    """Demonstrate handling missing data."""
    print_section("Pandas Missing Data")
    
    # Create DataFrame with missing values
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, np.nan, 500]
    })
    
    print("DataFrame with missing values:")
    print(df)
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Drop rows with any missing values
    print("\nAfter dropna():")
    print(df.dropna())
    
    # Fill missing values with 0
    print("\nFill with 0:")
    print(df.fillna(0))
    
    # Fill with mean
    print("\nFill with column mean:")
    print(df.fillna(df.mean()))
    
    # Forward fill
    print("\nForward fill:")
    print(df.fillna(method='ffill'))


# ============================================================================
# SECTION 3: Data Visualization
# ============================================================================

def matplotlib_basics():
    """Demonstrate basic matplotlib plots."""
    print_section("matplotlib Basics")
    
    print("Generating plots... (close windows to continue)\n")
    
    # Line plot
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.xlabel('X')
    plt.ylabel('sin(X)')
    plt.title('Sine Wave')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/tmp/sine_wave.png', dpi=150)
    print("✓ Saved sine wave plot to /tmp/sine_wave.png")
    plt.close()
    
    # Scatter plot
    x = np.random.randn(200)
    y = 2*x + np.random.randn(200)*0.5
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot with Correlation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/scatter.png', dpi=150)
    print("✓ Saved scatter plot to /tmp/scatter.png")
    plt.close()
    
    # Histogram
    data = np.random.randn(1000)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Normal Distribution Histogram')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('/tmp/histogram.png', dpi=150)
    print("✓ Saved histogram to /tmp/histogram.png")
    plt.close()
    
    # Subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Line
    axes[0, 0].plot(x, y)
    axes[0, 0].set_title('Line Plot')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter
    axes[0, 1].scatter(x, y, alpha=0.5)
    axes[0, 1].set_title('Scatter Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Histogram
    axes[1, 0].hist(data, bins=30)
    axes[1, 0].set_title('Histogram')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Box plot
    axes[1, 1].boxplot([data])
    axes[1, 1].set_title('Box Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/subplots.png', dpi=150)
    print("✓ Saved subplots to /tmp/subplots.png")
    plt.close()


# ============================================================================
# SECTION 4: Vectorization
# ============================================================================

def vectorization_demo():
    """Demonstrate the power of vectorization."""
    print_section("Vectorization: Loops vs Vectors")
    
    # Create large array
    n = 1000000
    arr = np.random.randn(n)
    
    print(f"Array size: {n:,} elements")
    print()
    
    # Method 1: Python loop
    print("Method 1: Python loop (squaring elements)")
    start = time.time()
    result_loop = []
    for x in arr:
        result_loop.append(x ** 2)
    loop_time = time.time() - start
    print(f"Time: {loop_time:.4f} seconds")
    
    # Method 2: List comprehension
    print("\nMethod 2: List comprehension")
    start = time.time()
    result_comp = [x ** 2 for x in arr]
    comp_time = time.time() - start
    print(f"Time: {comp_time:.4f} seconds")
    print(f"Speedup: {loop_time / comp_time:.2f}x")
    
    # Method 3: NumPy vectorization
    print("\nMethod 3: NumPy vectorization")
    start = time.time()
    result_vec = arr ** 2
    vec_time = time.time() - start
    print(f"Time: {vec_time:.4f} seconds")
    print(f"Speedup over loop: {loop_time / vec_time:.2f}x")
    print(f"Speedup over comprehension: {comp_time / vec_time:.2f}x")
    
    print("\n" + "="*60)
    print("Vectorization is 50-100x faster! 🚀")
    print("="*60)


def vectorization_examples():
    """More vectorization examples."""
    print_section("Vectorization Examples")
    
    # Example 1: Element-wise operations
    print("Example 1: Computing (x^2 + 2x + 1) for array")
    x = np.array([1, 2, 3, 4, 5])
    print(f"x = {x}")
    
    # Vectorized (fast)
    result = x**2 + 2*x + 1
    print(f"x^2 + 2x + 1 = {result}")
    
    # Example 2: Distance calculations
    print("\nExample 2: Euclidean distances")
    points1 = np.random.randn(1000, 3)
    points2 = np.random.randn(1000, 3)
    
    # Vectorized distance
    start = time.time()
    distances = np.sqrt(np.sum((points1 - points2)**2, axis=1))
    vec_time = time.time() - start
    
    print(f"Computed 1000 3D distances in {vec_time*1000:.2f} ms")
    print(f"Sample distances: {distances[:5]}")
    
    # Example 3: Normalization
    print("\nExample 3: Batch normalization")
    data = np.random.randn(100, 5)
    
    # Normalize each feature (column)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    normalized = (data - mean) / std
    
    print(f"Data shape: {data.shape}")
    print(f"Normalized mean (per feature): {normalized.mean(axis=0)}")
    print(f"Normalized std (per feature): {normalized.std(axis=0)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*60)
    print(" Python Fundamentals for ML - Examples ".center(60, "="))
    print("="*60)
    
    # NumPy
    numpy_basics()
    numpy_indexing()
    numpy_operations()
    numpy_broadcasting()
    numpy_linear_algebra()
    
    # Pandas
    pandas_basics()
    pandas_selection()
    pandas_manipulation()
    pandas_missing_data()
    
    # Visualization
    matplotlib_basics()
    
    # Vectorization
    vectorization_demo()
    vectorization_examples()
    
    print("\n" + "="*60)
    print(" All Examples Complete! ".center(60, "="))
    print("="*60)
    print("\nNext steps:")
    print("1. Review the output above")
    print("2. Check saved plots in /tmp/")
    print("3. Complete exercises.py")
    print("4. Attempt problems in problems.md")
    print("\n✨ Happy Learning! ✨\n")


if __name__ == "__main__":
    main()
