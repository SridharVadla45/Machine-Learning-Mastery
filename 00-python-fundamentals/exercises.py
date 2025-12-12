#!/usr/bin/env python3
"""
Python Fundamentals - Exercises

Complete the TODO sections to practice NumPy, Pandas, and vectorization.

Run: uv run 00-python-fundamentals/exercises.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_exercise(number, title):
    """Print exercise header."""
    print(f"\n{'='*60}")
    print(f"Exercise {number}: {title}")
    print(f"{'='*60}\n")


# ============================================================================
# NUMPY EXERCISES
# ============================================================================

def exercise_1():
    """
    Create a 5x5 matrix with values 1-25 and extract the 2x2 submatrix
    from the center.
    
    TODO: Complete this function
    """
    print_exercise(1, "NumPy Array Creation and Slicing")
    
    # TODO: Create 5x5 matrix with values 1-25
    matrix = np.arange(1, 26).reshape(5, 5)
    
    # TODO: Extract center 2x2 submatrix (should be [[7, 8], [12, 13]])
    center = matrix[1:3, 1:3]
    
    print("Original matrix:")
    print(matrix)
    print("\nCenter 2x2 submatrix:")
    print(center)
    
    # Test
    expected = np.array([[7, 8], [12, 13]])
    assert np.array_equal(center, expected), "Incorrect result!"
    print("\n✓ Correct!")


def exercise_2():
    """
    Create a function that normalizes an array to have mean=0 and std=1.
    
    TODO: Complete the normalize function
    """
    print_exercise(2, "Data Normalization")
    
    def normalize(arr):
        # TODO: Implement normalization
        mean = arr.mean()
        std = arr.std()
        return (arr - mean) / std
    
    # Test
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    normalized = normalize(data)
    
    print(f"Original: {data}")
    print(f"Normalized: {normalized}")
    print(f"Mean: {normalized.mean():.10f}")  # Should be ~0
    print(f"Std: {normalized.std():.10f}")    # Should be ~1
    
    assert abs(normalized.mean()) < 1e-10, "Mean should be 0"
    assert abs(normalized.std() - 1.0) < 1e-10, "Std should be 1"
    print("\n✓ Correct!")


def exercise_3():
    """
    Use broadcasting to add bias to each row of a matrix.
    
    TODO: Complete using broadcasting (no loops!)
    """
    print_exercise(3, "Broadcasting")
    
    # Matrix: 3 samples, 4 features
    X = np.random.randn(3, 4)
    # Bias: one value per feature
    bias = np.array([1, 2, 3, 4])
    
    # TODO: Add bias to each row using broadcasting
    result = X + bias
    
    print("Matrix X:")
    print(X)
    print("\nBias:")
    print(bias)
    print("\nX + bias:")
    print(result)
    
    # Verify
    assert result.shape == (3, 4), "Shape should be (3, 4)"
    assert np.allclose(result[0, :], X[0, :] + bias), "Incorrect broadcast"
    print("\n✓ Correct!")


def exercise_4():
    """
    Compute the Euclidean distance between two vectors.
    
    TODO: Implement using NumPy (vectorized, no loops)
    """
    print_exercise(4, "Distance Calculation")
    
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    # TODO: Compute Euclidean distance
    # Formula: sqrt(sum((a - b)^2))
    distance = np.sqrt(np.sum((a - b) ** 2))
    
    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print(f"Distance: {distance:.4f}")
    
    # Expected: sqrt(9 + 9 + 9) = sqrt(27) ≈ 5.196
    assert abs(distance - np.sqrt(27)) < 1e-10, "Incorrect distance"
    print("\n✓ Correct!")


def exercise_5():
    """
    Find all values in an array that are between 10 and 20 (inclusive).
    
    TODO: Use boolean indexing
    """
    print_exercise(5, "Boolean Indexing")
    
    arr = np.array([5, 12, 8, 19, 25, 15, 3, 22, 18, 30])
    
    # TODO: Get values between 10 and 20 (inclusive)
    mask = (arr >= 10) & (arr <= 20)
    result = arr[mask]
    
    print(f"Array: {arr}")
    print(f"Values between 10 and 20: {result}")
    
    expected = np.array([12, 19, 15, 18])
    assert np.array_equal(np.sort(result), np.sort(expected)), "Incorrect filtering"
    print("\n✓ Correct!")


# ============================================================================
# PANDAS EXERCISES
# ============================================================================

def exercise_6():
    """
    Create a DataFrame and compute basic statistics.
    
    TODO: Complete the statistics computation
    """
    print_exercise(6, "Pandas DataFrame Basics")
    
    # Create DataFrame
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'score': [85, 90, 75, 88, 92]
    })
    
    print("DataFrame:")
    print(df)
    
    # TODO: Compute mean age
    mean_age = df['age'].mean()
    
    # TODO: Compute max score
    max_score = df['score'].max()
    
    # TODO: Count number of people with age > 30
    count_over_30 = len(df[df['age'] > 30])
    
    print(f"\nMean age: {mean_age}")
    print(f"Max score: {max_score}")
    print(f"People over 30: {count_over_30}")
    
    assert mean_age == 30.0, "Incorrect mean"
    assert max_score == 92, "Incorrect max"
    assert count_over_30 == 2, "Incorrect count"
    print("\n✓ Correct!")


def exercise_7():
    """
    Filter a DataFrame based on multiple conditions.
    
    TODO: Complete the filtering
    """
    print_exercise(7, "DataFrame Filtering")
    
    df = pd.DataFrame({
        'product': ['A', 'B', 'C', 'D', 'E'],
        'price': [10, 25, 15, 30, 20],
        'quantity': [100, 50, 75, 30, 60]
    })
    
    print("Original DataFrame:")
    print(df)
    
    # TODO: Get products with price > 15 AND quantity > 50
    filtered = df[(df['price'] > 15) & (df['quantity'] > 50)]
    
    print("\nFiltered (price > 15 AND quantity > 50):")
    print(filtered)
    
    assert len(filtered) == 2, "Should have 2 rows"
    assert 'B' in filtered['product'].values, "Should include B"
    print("\n✓ Correct!")


def exercise_8():
    """
    Add a computed column to a DataFrame.
    
    TODO: Add the 'total' column
    """
    print_exercise(8, "Adding Computed Columns")
    
    df = pd.DataFrame({
        'item': ['A', 'B', 'C'],
        'price': [10, 20, 15],
        'quantity': [5, 3, 8]
    })
    
    print("Original DataFrame:")
    print(df)
    
    # TODO: Add 'total' column (price * quantity)
    df['total'] = df['price'] * df['quantity']
    
    print("\nAfter adding 'total' column:")
    print(df)
    
    assert 'total' in df.columns, "Missing 'total' column"
    assert df['total'].tolist() == [50, 60, 120], "Incorrect totals"
    print("\n✓ Correct!")


def exercise_9():
    """
    Group data and compute aggregations.
    
    TODO: Complete the groupby operation
    """
    print_exercise(9, "GroupBy Operations")
    
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'value': [10, 20, 30, 40, 50, 60]
    })
    
    print("DataFrame:")
    print(df)
    
    # TODO: Compute mean value for each category
    grouped = df.groupby('category')['value'].mean()
    
    print("\nMean value by category:")
    print(grouped)
    
    assert grouped['A'] == 30.0, "Incorrect mean for A"
    assert grouped['B'] == 40.0, "Incorrect mean for B"
    print("\n✓ Correct!")


def exercise_10():
    """
    Handle missing data.
    
    TODO: Complete the missing data handling
    """
    print_exercise(10, "Missing Data")
    
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, np.nan, 50]
    })
    
    print("DataFrame with missing values:")
    print(df)
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # TODO: Fill missing values with column mean
    df_filled = df.fillna(df.mean())
    
    print("\nAfter filling with mean:")
    print(df_filled)
    
    assert not df_filled.isnull().any().any(), "Still has missing values"
    print("\n✓ Correct!")


# ============================================================================
# VECTORIZATION EXERCISES
# ============================================================================

def exercise_11():
    """
    Replace a loop with vectorized operation.
    
    TODO: Implement the vectorized version
    """
    print_exercise(11, "Vectorization")
    
    # Given: array of numbers
    arr = np.random.rand(1000)
    
    # Bad: Loop version
    def loop_version(arr):
        result = []
        for x in arr:
            result.append(x ** 2 + 2*x + 1)
        return np.array(result)
    
    # TODO: Vectorized version (should be one line!)
    def vectorized_version(arr):
        return arr ** 2 + 2*arr + 1
    
    # Test correctness
    loop_result = loop_version(arr)
    vec_result = vectorized_version(arr)
    
    assert np.allclose(loop_result, vec_result), "Results don't match"
    print("✓ Vectorized version produces same results")
    
    # Compare speed
    import time
    
    n = 100000
    arr_large = np.random.rand(n)
    
    start = time.time()
    _ = loop_version(arr_large)
    loop_time = time.time() - start
    
    start = time.time()
    _ = vectorized_version(arr_large)
    vec_time = time.time() - start
    
    print(f"\nLoop time: {loop_time:.4f}s")
    print(f"Vectorized time: {vec_time:.4f}s")
    print(f"Speedup: {loop_time/vec_time:.1f}x faster!")
    print("\n✓ Correct!")


def exercise_12():
    """
    Compute pairwise distances between points (vectorized).
    
    TODO: Implement without explicit loops
    """
    print_exercise(12, "Advanced Vectorization")
    
    # 3 points in 2D space
    points = np.array([[0, 0],
                       [1, 0],
                       [0, 1]])
    
    # TODO: Compute all pairwise distances
    # Hint: Use broadcasting and advanced indexing
    
    # Solution using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    
    print("Points:")
    print(points)
    print("\nPairwise distances:")
    print(distances)
    
    # Verify
    assert distances[0, 1] == 1.0, "Distance (0,0) to (1,0) should be 1"
    assert abs(distances[0, 2] - 1.0) < 1e-10, "Distance (0,0) to (0,1) should be 1"
    assert abs(distances[1, 2] - np.sqrt(2)) < 1e-10, "Distance (1,0) to (0,1) should be √2"
    print("\n✓ Correct!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all exercises."""
    print("\n" + "="*60)
    print(" Python Fundamentals - Exercises ".center(60))
    print("="*60)
    
    # NumPy exercises
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    
    # Pandas exercises
    exercise_6()
    exercise_7()
    exercise_8()
    exercise_9()
    exercise_10()
    
    # Vectorization exercises
    exercise_11()
    exercise_12()
    
    print("\n" + "="*60)
    print(" All Exercises Complete! ".center(60))
    print("="*60)
    print("\n✨ Great work! Now try the problems in problems.md ✨\n")


if __name__ == "__main__":
    main()
