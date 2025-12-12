#!/usr/bin/env python3
"""
Python Fundamentals - Solutions

Complete, detailed solutions to all exercises with explanations.

Run: uv run 00-python-fundamentals/solutions.py
"""

import numpy as np
import pandas as pd
import time


def print_solution(number, title):
    """Print formatted solution header."""
    print(f"\n{'='*60}")
    print(f"Solution {number}: {title}")
    print(f"{'='*60}\n")


# ============================================================================
# NUMPY SOLUTIONS
# ============================================================================

def solution_1():
    """
    Solution to Exercise 1: Array Creation and Slicing
    
    Key Concepts:
    - np.arange() creates sequential values
    - reshape() changes array dimensions
    - Slicing syntax: [start:stop, start:stop] for 2D
    """
    print_solution(1, "NumPy Array Creation and Slicing")
    
    # Step 1: Create array with values 1-25
    matrix = np.arange(1, 26).reshape(5, 5)
    print("Step 1: Create 5x5 matrix")
    print(matrix)
    
    # Step 2: Extract center 2x2
    # Center is rows 1-2 (indices 1:3) and cols 1-2 (indices 1:3)
    center = matrix[1:3, 1:3]
    print("\nStep 2: Extract center 2x2")
    print(center)
    
    print("\nExplanation:")
    print("- np.arange(1, 26) creates [1, 2, ..., 25]")
    print("- reshape(5, 5) makes it a 5×5 matrix")
    print("- [1:3, 1:3] gets rows 1-2 and cols 1-2")
    print("- Remember: Python uses 0-indexing and exclusive end")


def solution_2():
    """
    Solution to Exercise 2: Data Normalization
    
    Key Concepts:
    - Normalization: (x - mean) / std
    - Use arr.mean() and arr.std()
    - Result has mean=0, std=1
    """
    print_solution(2, "Data Normalization")
    
    def normalize(arr):
        """
        Normalize array to have mean=0 and std=1.
        
        Formula: z = (x - μ) / σ
        where μ = mean, σ = standard deviation
        """
        mean = arr.mean()
        std = arr.std()
        return (arr - mean) / std
    
    # Test
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    normalized = normalize(data)
    
    print("Original data:")
    print(f"  Values: {data}")
    print(f"  Mean: {data.mean()}")
    print(f"  Std: {data.std()}")
    
    print("\nNormalized data:")
    print(f"  Values: {normalized}")
    print(f"  Mean: {normalized.mean():.10f} (≈ 0)")
    print(f"  Std: {normalized.std():.10f} (≈ 1)")
    
    print("\nWhy normalize?")
    print("- Puts features on same scale")
    print("- Helps gradient descent converge faster")
    print("- Required by many ML algorithms")


def solution_3():
    """
    Solution to Exercise 3: Broadcasting
    
    Key Concepts:
    - Broadcasting: NumPy automatically expands arrays
    - Bias: added to each sample/row
    - No loops needed!
    """
    print_solution(3, "Broadcasting")
    
    np.random.seed(42)
    X = np.random.randn(3, 4)
    bias = np.array([1, 2, 3, 4])
    
    print("Matrix X (3 samples, 4 features):")
    print(X)
    print("\nBias (one per feature):")
    print(bias)
    
    # Broadcasting automatically expands bias to match X
    result = X + bias
    
    print("\nX + bias:")
    print(result)
    
    print("\nHow broadcasting works:")
    print("  X shape: (3, 4)")
    print("  bias shape: (4,)")
    print("  NumPy expands bias to (3, 4) internally")
    print("  Then performs element-wise addition")
    
    print("\nML Application:")
    print("- This is how bias is added in neural networks!")
    print("- X = inputs, bias = learned bias terms")


def solution_4():
    """
    Solution to Exercise 4: Distance Calculation
    
    Key Concepts:
    - Euclidean distance: √(Σ(a - b)²)
    - Vectorized operation (no loops)
    """
    print_solution(4, "Distance Calculation")
    
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    
    # Step-by-step calculation
    diff = a - b
    print(f"\nStep 1 - Difference: {diff}")
    
    squared = diff ** 2
    print(f"Step 2 - Squared: {squared}")
    
    sum_squared = np.sum(squared)
    print(f"Step 3 - Sum: {sum_squared}")
    
    distance = np.sqrt(sum_squared)
    print(f"Step 4 - Distance: {distance:.4f}")
    
    # All in one line
    distance_oneline = np.sqrt(np.sum((a - b) ** 2))
    print(f"\nOne-liner: {distance_oneline:.4f}")
    
    print("\nML Applications:")
    print("- K-Nearest Neighbors")
    print("- K-Means Clustering")
    print("- Similarity measures")


def solution_5():
    """
    Solution to Exercise 5: Boolean Indexing
    
    Key Concepts:
    - Boolean mask: array of True/False
    - & for AND, | for OR
    - Use parentheses with multiple conditions
    """
    print_solution(5, "Boolean Indexing")
    
    arr = np.array([5, 12, 8, 19, 25, 15, 3, 22, 18, 30])
    
    print(f"Array: {arr}")
    
    # Step 1: Create mask
    mask = (arr >= 10) & (arr <= 20)
    print(f"\nMask (10 ≤ value ≤ 20): {mask}")
    
    # Step 2: Apply mask
    result = arr[mask]
    print(f"Filtered values: {result}")
    
    print("\nBreakdown:")
    print(f"  arr >= 10: {arr >= 10}")
    print(f"  arr <= 20: {arr <= 20}")
    print(f"  Combined (&): {mask}")
    
    print("\nImportant:")
    print("- Use & (not 'and') for arrays")
    print("- Use | for OR")
    print("- Always use parentheses!")


# ============================================================================
# PANDAS SOLUTIONS
# ============================================================================

def solution_6():
    """
    Solution to Exercise 6: DataFrame Basics
    """
    print_solution(6, "Pandas DataFrame Basics")
    
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'score': [85, 90, 75, 88, 92]
    })
    
    print("DataFrame:")
    print(df)
    
    # Computations
    mean_age = df['age'].mean()
    max_score = df['score'].max()
    count_over_30 = len(df[df['age'] > 30])
    
    print(f"\nMean age: {mean_age}")
    print(f"  → df['age'].mean()")
    
    print(f"\nMax score: {max_score}")
    print(f"  → df['score'].max()")
    
    print(f"\nPeople over 30: {count_over_30}")
    print(f"  → len(df[df['age'] > 30])")
    
    print("\nOther useful methods:")
    print(f"  min(): {df['age'].min()}")
    print(f"  median(): {df['age'].median()}")
    print(f"  std(): {df['age'].std():.2f}")


def solution_7():
    """
    Solution to Exercise 7: DataFrame Filtering
    """
    print_solution(7, "DataFrame Filtering")
    
    df = pd.DataFrame({
        'product': ['A', 'B', 'C', 'D', 'E'],
        'price': [10, 25, 15, 30, 20],
        'quantity': [100, 50, 75, 30, 60]
    })
    
    print("DataFrame:")
    print(df)
    
    # Filter: price > 15 AND quantity > 50
    filtered = df[(df['price'] > 15) & (df['quantity'] > 50)]
    
    print("\nFiltered (price > 15 AND quantity > 50):")
    print(filtered)
    
    print("\nStep by step:")
    print("1. df['price'] > 15:")
    print(f"   {df['price'] > 15}")
    print("2. df['quantity'] > 50:")
    print(f"   {df['quantity'] > 50}")
    print("3. Combine with & (AND):")
    print(f"   {(df['price'] > 15) & (df['quantity'] > 50)}")
    print("4. Use mask to filter")


def solution_8():
    """
    Solution to Exercise 8: Adding Computed Columns
    """
    print_solution(8, "Adding Computed Columns")
    
    df = pd.DataFrame({
        'item': ['A', 'B', 'C'],
        'price': [10, 20, 15],
        'quantity': [5, 3, 8]
    })
    
    print("Original DataFrame:")
    print(df)
    
    # Add computed column
    df['total'] = df['price'] * df['quantity']
    
    print("\nAfter adding 'total':")
    print(df)
    
    print("\nExplanation:")
    print("- df['total'] creates new column")
    print("- df['price'] * df['quantity'] is element-wise multiplication")
    print("- Pandas aligns by row index automatically")


def solution_9():
    """
    Solution to Exercise 9: GroupBy Operations
    """
    print_solution(9, "GroupBy Operations")
    
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'value': [10, 20, 30, 40, 50, 60]
    })
    
    print("DataFrame:")
    print(df)
    
    # Group and aggregate
    grouped = df.groupby('category')['value'].mean()
    
    print("\nMean value by category:")
    print(grouped)
    
    print("\nHow it works:")
    print("1. Group by 'category'")
    print("   - Category A: rows 0, 2, 4")
    print("   - Category B: rows 1, 3, 5")
    print("2. Select 'value' column")
    print("   - Category A: [10, 30, 50]")
    print("   - Category B: [20, 40, 60]")
    print("3. Compute mean")
    print("   - Category A: (10 + 30 + 50) / 3 = 30")
    print("   - Category B: (20 + 40 + 60) / 3 = 40")


def solution_10():
    """
    Solution to Exercise 10: Missing Data
    """
    print_solution(10, "Missing Data")
    
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, np.nan, 50]
    })
    
    print("DataFrame with missing values:")
    print(df)
    print(f"\nMissing count:\n{df.isnull().sum()}")
    
    # Fill with mean
    df_filled = df.fillna(df.mean())
    
    print("\nAfter filling with column mean:")
    print(df_filled)
    
    print("\nColumn A mean: (1 + 2 + 4 + 5) / 4 = 3.0")
    print("Column B mean: (10 + 30 + 50) / 3 = 30.0")
    
    print("\nOther strategies:")
    print("- fillna(0): Fill with zero")
    print("- fillna(method='ffill'): Forward fill")
    print("- fillna(method='bfill'): Backward fill")
    print("- dropna(): Remove rows with NaN")


# ============================================================================
# VECTORIZATION SOLUTIONS
# ============================================================================

def solution_11():
    """
    Solution to Exercise 11: Vectorization
    """
    print_solution(11, "Vectorization vs Loops")
    
    arr = np.random.rand(100)
    
    print("Task: Compute x² + 2x + 1 for each element\n")
    
    # Loop version
    def loop_version(arr):
        result = []
        for x in arr:
            result.append(x ** 2 + 2*x + 1)
        return np.array(result)
    
    # Vectorized version
    def vectorized_version(arr):
        return arr ** 2 + 2*arr + 1
    
    print("Loop version:")
    print("  result = []")
    print("  for x in arr:")
    print("      result.append(x**2 + 2*x + 1)")
    
    print("\nVectorized version:")
    print("  result = arr**2 + 2*arr + 1")
    
    # Speed comparison
    n = 100000
    arr_large = np.random.rand(n)
    
    start = time.time()
    _ = loop_version(arr_large)
    loop_time = time.time() - start
    
    start = time.time()
    _ = vectorized_version(arr_large)
    vec_time = time.time() - start
    
    print(f"\nSpeed comparison (n={n:,}):")
    print(f"  Loop: {loop_time:.4f}s")
    print(f"  Vectorized: {vec_time:.4f}s")
    print(f"  Speedup: {loop_time/vec_time:.1f}x faster! 🚀")


def solution_12():
    """
    Solution to Exercise 12: Advanced Vectorization
    """
    print_solution(12, "Pairwise Distances (Vectorized)")
    
    points = np.array([[0, 0],
                       [1, 0],
                       [0, 1]])
    
    print("Points:")
    print(points)
    
    # Vectorized solution using broadcasting
    # points[:, np.newaxis, :] has shape (3, 1, 2)
    # points[np.newaxis, :, :] has shape (1, 3, 2)
    # Broadcasting gives shape (3, 3, 2) for differences
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    print(f"\nDifferences shape: {diff.shape}")
    print("(each point vs every other point)")
    
    # Compute distances
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    
    print("\nPairwise distances:")
    print(distances)
    
    print("\nInterpretation:")
    print(f"  distances[0, 1] = {distances[0, 1]:.4f} (point 0 to point 1)")
    print(f"  distances[0, 2] = {distances[0, 2]:.4f} (point 0 to point 2)")
    print(f"  distances[1, 2] = {distances[1, 2]:.4f} (point 1 to point 2)")
    
    print("\nWhy this works:")
    print("- np.newaxis adds dimension for broadcasting")
    print("- Points expand to (3, 3, 2) for all pairs")
    print("- Compute all differences in one operation")
    print("- Much faster than nested loops!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all solutions."""
    print("\n" + "="*60)
    print(" Python Fundamentals - Detailed Solutions ".center(60))
    print("="*60)
    
    # NumPy solutions
    solution_1()
    solution_2()
    solution_3()
    solution_4()
    solution_5()
    
    # Pandas solutions
    solution_6()
    solution_7()
    solution_8()
    solution_9()
    solution_10()
    
    # Vectorization solutions
    solution_11()
    solution_12()
    
    print("\n" + "="*60)
    print(" All Solutions Complete! ".center(60))
    print("="*60)
    print("\n💡 Key Takeaways:")
    print("1. NumPy is the foundation of ML in Python")
    print("2. Vectorization is 10-100x faster than loops")
    print("3. Pandas makes data manipulation easy")
    print("4. Always think in arrays, not individual elements")
    print("\n✨ Now you're ready for Linear Algebra! ✨\n")


if __name__ == "__main__":
    main()
