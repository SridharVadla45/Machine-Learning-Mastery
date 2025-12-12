# Python Fundamentals for Machine Learning - Theory

## Table of Contents

1. [Introduction](#1-introduction)
2. [NumPy Essentials](#2-numpy-essentials)
3. [Advanced NumPy](#3-advanced-numpy)
4. [Pandas Basics](#4-pandas-basics)
5. [Advanced Pandas](#5-advanced-pandas)
6. [Data Visualization](#6-data-visualization)
7. [Vectorization Mindset](#7-vectorization-mindset)
8. [Performance Optimization](#8-performance-optimization)

---

## 1. Introduction

### Why Python for Machine Learning?

Python has become the de facto language for ML because:

1. **Rich Ecosystem** - NumPy, Pandas, scikit-learn, PyTorch
2. **Readable Syntax** - Easy to express complex algorithms
3. **Interactive Development** - Jupyter notebooks for experimentation
4. **Strong Community** - Massive support and resources
5. **Performance** - C/C++ under the hood for heavy computation

### The Core Libraries

```
NumPy → Arrays and numerical computing
Pandas → Data manipulation and analysis
matplotlib → Visualization
```

These three form the foundation of ALL machine learning work in Python.

---

## 2. NumPy Essentials

### What is NumPy?

NumPy (Numerical Python) provides:
- **N-dimensional arrays** - The core data structure
- **Fast operations** - Implemented in C for speed
- **Broadcasting** - Smart handling of different shapes
- **Linear algebra** - Matrix operations

### NumPy Arrays vs Python Lists

**Python List:**
```python
# Slow, flexible, stores references
my_list = [1, 2, 3, 4, 5]
# Each element is a Python object
```

**NumPy Array:**
```python
# Fast, fixed-type, contiguous memory
my_array = np.array([1, 2, 3, 4, 5])
# All elements are same type, stored efficiently
```

**Speed Comparison:**
- NumPy operations are **10-100x faster** than Python loops
- Critical for ML where we process millions of numbers

### Creating Arrays

**From Lists:**
```python
arr = np.array([1, 2, 3])  # 1D array
arr = np.array([[1, 2], [3, 4]])  # 2D array
```

**Special Arrays:**
```python
np.zeros((3, 4))       # 3x4 array of zeros
np.ones((2, 3))        # 2x3 array of ones
np.eye(3)              # 3x3 identity matrix
np.arange(10)          # [0, 1, 2, ..., 9]
np.linspace(0, 1, 5)   # 5 numbers from 0 to 1
```

**Random Arrays:**
```python
np.random.rand(3, 4)         # Uniform [0, 1)
np.random.randn(3, 4)        # Standard normal
np.random.randint(0, 10, 5)  # Random integers
```

### Array Attributes

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape   # (2, 3) - dimensions
arr.ndim    # 2 - number of dimensions
arr.size    # 6 - total elements
arr.dtype   # int64 - data type
```

### Indexing and Slicing

**Basic Indexing:**
```python
arr = np.array([10, 20, 30, 40, 50])
arr[0]      # 10 (first element)
arr[-1]     # 50 (last element)
arr[1:4]    # [20, 30, 40] (slice)
```

**2D Indexing:**
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
arr[0, 1]   # 2 (row 0, col 1)
arr[:, 1]   # [2, 5] (all rows, col 1)
arr[0, :]   # [1, 2, 3] (row 0, all cols)
```

**Boolean Indexing:**
```python
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 3
arr[mask]   # [4, 5] (elements > 3)
```

### Array Operations

**Element-wise Operations:**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b   # [5, 7, 9]
a * b   # [4, 10, 18]
a ** 2  # [1, 4, 9]
```

**Aggregations:**
```python
arr = np.array([1, 2, 3, 4, 5])

arr.sum()    # 15
arr.mean()   # 3.0
arr.std()    # 1.414...
arr.min()    # 1
arr.max()    # 5
```

---

## 3. Advanced NumPy

### Broadcasting

Broadcasting allows operations on different-shaped arrays.

**Rules:**
1. Arrays with fewer dimensions get 1s prepended to shape
2. Dimensions must be compatible (equal or one is 1)

**Example:**
```python
# Shape (3,) + Shape (3, 1)
a = np.array([1, 2, 3])      # Shape: (3,)
b = np.array([[10],
              [20],
              [30]])         # Shape: (3, 1)

result = a + b
# Broadcasting expands:
# a becomes: [[1, 2, 3],
#             [1, 2, 3],
#             [1, 2, 3]]
# Result:    [[11, 12, 13],
#             [21, 22, 23],
#             [31, 32, 33]]
```

**Why Broadcasting Matters:**
- Avoid explicit loops
- Memory efficient
- Fast computation
- Common in ML (e.g., adding bias to layers)

### Reshaping

```python
arr = np.arange(12)  # [0, 1, 2, ..., 11]

arr.reshape(3, 4)    # 3x4 matrix
arr.reshape(2, 6)    # 2x6 matrix
arr.reshape(4, -1)   # 4x3 (auto-compute -1)

arr.ravel()          # Flatten to 1D
arr.T                # Transpose
```

### Linear Algebra

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
A @ B  # or np.dot(A, B)

# Transpose
A.T

# Inverse
np.linalg.inv(A)

# Determinant
np.linalg.det(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solve linear system Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
```

---

## 4. Pandas Basics

### What is Pandas?

Pandas provides:
- **DataFrame** - 2D table with labeled columns
- **Series** - 1D labeled array
- **Data I/O** - Read/write CSV, Excel, SQL, etc.
- **Data cleaning** - Handle missing values, duplicates
- **Aggregation** - GroupBy, pivot tables

### Series

A 1D labeled array:

```python
import pandas as pd

s = pd.Series([1, 2, 3, 4, 5])
# Index: 0, 1, 2, 3, 4
# Values: 1, 2, 3, 4, 5

s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# Index: a, b, c
# Values: 1, 2, 3
```

### DataFrame

A 2D table with labeled rows and columns:

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

#       name  age     city
# 0    Alice   25      NYC
# 1      Bob   30       LA
# 2  Charlie   35  Chicago
```

### Loading Data

```python
# From CSV
df = pd.read_csv('data.csv')

# From Excel
df = pd.read_excel('data.xlsx')

# From dictionary
data = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data)
```

### Inspecting Data

```python
df.head()        # First 5 rows
df.tail()        # Last 5 rows
df.info()        # Column types, non-null counts
df.describe()    # Statistical summary
df.shape         # (rows, columns)
df.columns       # Column names
```

### Selecting Data

**Column Selection:**
```python
df['age']         # Single column (Series)
df[['name', 'age']]  # Multiple columns (DataFrame)
```

**Row Selection:**
```python
df.iloc[0]        # First row by position
df.iloc[0:2]      # First two rows
df.loc[0]         # Row with index 0
```

**Boolean Indexing:**
```python
df[df['age'] > 30]           # Rows where age > 30
df[df['city'] == 'NYC']      # Rows where city is NYC
```

---

## 5. Advanced Pandas

### Handling Missing Data

```python
# Check for missing values
df.isnull().sum()

# Drop rows with any missing values
df.dropna()

# Fill missing values
df.fillna(0)
df.fillna(df.mean())  # Fill with mean
```

### Adding/Modifying Columns

```python
# Add new column
df['age_squared'] = df['age'] ** 2

# Modify existing column
df['age'] = df['age'] + 1

# Delete column
df.drop('age_squared', axis=1, inplace=True)
```

### GroupBy Operations

```python
# Group by city and compute mean age
df.groupby('city')['age'].mean()

# Multiple aggregations
df.groupby('city').agg({
    'age': ['mean', 'min', 'max'],
    'name': 'count'
})
```

### Merging DataFrames

```python
df1 = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
df2 = pd.DataFrame({'id': [1, 2], 'score': [90, 85]})

# Merge on 'id' column
merged = pd.merge(df1, df2, on='id')
```

### Sorting

```python
df.sort_values('age')                    # Ascending
df.sort_values('age', ascending=False)   # Descending
df.sort_values(['city', 'age'])          # Multi-column
```

---

## 6. Data Visualization

### matplotlib Basics

**Line Plot:**
```python
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('sin(X)')
plt.title('Sine Wave')
plt.show()
```

**Scatter Plot:**
```python
x = np.random.randn(100)
y = np.random.randn(100)

plt.scatter(x, y, alpha=0.5)
plt.show()
```

**Histogram:**
```python
data = np.random.randn(1000)

plt.hist(data, bins=30)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

### seaborn for Statistics

```python
import seaborn as sns

# Load sample data
tips = sns.load_dataset('tips')

# Distribution plot
sns.histplot(tips['total_bill'])

# Scatter with regression line
sns.regplot(x='total_bill', y='tip', data=tips)

# Box plot
sns.boxplot(x='day', y='total_bill', data=tips)

# Heatmap (correlation matrix)
sns.heatmap(tips.corr(), annot=True)
```

---

## 7. Vectorization Mindset

### What is Vectorization?

**Vectorization** = expressing operations on entire arrays, not individual elements.

**Bad (Looping):**
```python
result = []
for i in range(len(arr)):
    result.append(arr[i] * 2)
```

**Good (Vectorized):**
```python
result = arr * 2
```

### Why Vectorization Matters

1. **Speed** - 10-100x faster (compiled C code)
2. **Readability** - More concise, clearer intent
3. **Memory** - More efficient memory usage
4. **ML Ready** - All libraries expect vectorized code

### Examples

**Sum of squares (bad):**
```python
total = 0
for x in arr:
    total += x ** 2
```

**Sum of squares (good):**
```python
total = np.sum(arr ** 2)
```

**Distance calculation (bad):**
```python
distances = []
for i in range(len(points1)):
    dist = np.sqrt((points1[i] - points2[i]) ** 2)
    distances.append(dist)
```

**Distance calculation (good):**
```python
distances = np.sqrt((points1 - points2) ** 2)
```

---

## 8. Performance Optimization

### Timing Code

```python
import time

# Method 1: time.time()
start = time.time()
# ... code ...
end = time.time()
print(f"Time: {end - start} seconds")

# Method 2: timeit (more accurate)
import timeit

def my_function():
    # ... code ...
    pass

time_taken = timeit.timeit(my_function, number=1000)
print(f"Average time: {time_taken / 1000} seconds")
```

### NumPy vs Loops

```python
# Create large array
n = 1000000
arr = np.random.randn(n)

# Loop version
start = time.time()
result = []
for x in arr:
    result.append(x ** 2)
end = time.time()
loop_time = end - start

# Vectorized version
start = time.time()
result = arr ** 2
end = time.time()
vectorized_time = end - start

print(f"Speedup: {loop_time / vectorized_time}x faster")
# Typically 100x+ faster!
```

### Memory Efficiency

```python
# Bad: Creates intermediate arrays
result = (arr + 1) * 2 - 3

# Good for very large arrays: In-place operations
arr += 1
arr *= 2
arr -= 3
```

---

## Summary

### Key Takeaways

1. **NumPy** - Foundation of numerical computing in Python
2. **Pandas** - Data manipulation and analysis
3. **matplotlib** - Visualization
4. **Vectorization** - Think in arrays, not loops
5. **Performance** - NumPy is 100x faster than Python loops

### What's Next?

Now that you have Python fundamentals, you're ready for:
- **Linear Algebra** - The math behind ML
- **Calculus** - Understanding gradients
- **Probability** - Modeling uncertainty

### Practice Strategy

1. Type every example yourself
2. Modify examples and experiment
3. Complete all exercises
4. Solve problems without looking at solutions
5. Build small projects to solidify understanding

---

**You now have the tools. Time to build mastery through practice!** 🚀
