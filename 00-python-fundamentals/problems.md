# Python Fundamentals for Machine Learning - Problems

Practice problems to build your ML engineering intuition.

## Problem Guidelines

- **Easy (⭐)**: Quick concept checks, basic operations, simple coding tasks
- **Medium (⭐⭐)**: Multi-step problems, implementations, data analysis
- **Hard (⭐⭐⭐)**: Engineering challenges, optimization, critical thinking

**Recommendation**: Attempt problems before checking solutions.py!

---

## ⭐ Easy Problems (10)

### Problem 1: Array Shape Manipulation
**Difficulty**: Easy | **Type**: Coding

Create a 1D NumPy array with values from 0 to 11. Reshape it into a 3×4 matrix. Then, extract the last row.

**Expected Output**: `[8, 9, 10, 11]`

---

### Problem 2: Element-wise Operations
**Difficulty**: Easy | **Type**: Numerical

Given two arrays `a = [1, 2, 3]` and `b = [4, 5, 6]`, compute:
1. a + b
2. a * b
3. a ** 2

Write the code and verify your answer.

---

### Problem 3: Mean and Standard Deviation
**Difficulty**: Easy | **Type**: Conceptual

Explain in 2-3 sentences: Why do we normalize data in machine learning? What does it mean for data to have mean=0 and std=1?

---

### Problem 4: Boolean Masking
**Difficulty**: Easy | **Type**: Coding

Create an array `arr = np.array([10, 15, 20, 25, 30, 35, 40])`. Filter out all values that are NOT divisible by 10.

**Expected Output**: `[10, 20, 30, 40]`

---

### Problem 5: DataFrame Creation
**Difficulty**: Easy | **Type**: Coding

Create a Pandas DataFrame with columns `'name'`, `'age'`, `'city'` for 3 people of your choice. Then print only the `'name'` and `'age'` columns.

---

### Problem 6: Aggregation
**Difficulty**: Easy | **Type**: Numerical

Given `arr = np.array([2, 4, 6, 8, 10])`, compute:
- Sum of all elements
- Mean
- Maximum value
- Index of maximum value

---

### Problem 7: Broadcasting Basics
**Difficulty**: Easy | **Type**: Conceptual

Explain what happens when you execute `np.array([1, 2, 3]) + 10`. Why does this work even though the shapes are different?

---

### Problem 8: Missing Data Detection
**Difficulty**: Easy | **Type**: Coding

Create a DataFrame with some `np.nan` values. Write code to:
1. Count missing values per column
2. Drop all rows with missing values

---

### Problem 9: Indexing Practice
**Difficulty**: Easy | **Type**: Coding

Create a 5×5 matrix with values 0-24. Extract:
1. The element at position (2, 3)
2. The entire third column
3. The last row

---

### Problem 10: Quick Math
**Difficulty**: Easy | **Type**: Numerical

Without running code, what is the result of:
```python
np.array([1, 2, 3]).dot(np.array([4, 5, 6]))
```

Show your calculation, then verify with code.

---

## ⭐⭐ Medium Problems (10)

### Problem 11: Custom Standardization Function
**Difficulty**: Medium | **Type**: Implementation

Write a function `standardize(X)` that:
- Takes a 2D NumPy array X (samples × features)
- Standardizes each FEATURE (column) to have mean=0 and std=1
- Returns the standardized array AND the means and stds used

Test it on random data and verify the output statistics.

---

### Problem 12: Euclidean Distance Matrix
**Difficulty**: Medium | **Type**: Implementation

Write a function that computes the pairwise Euclidean distance matrix for n points in d-dimensional space.

Input: `points` - array of shape (n, d)  
Output: `distances` - array of shape (n, n) where distances[i, j] is the distance between point i and point j

Test with 4 points in 2D space. Your solution MUST be vectorized (no loops)!

---

### Problem 13: Data Cleaning Pipeline
**Difficulty**: Medium | **Type**: Applied

You have a CSV dataset with columns: `name`, `age`, `salary`, `department`.

Create a complete data cleaning pipeline that:
1. Loads the data (create sample data if needed)
2. Removes duplicate rows
3. Fills missing salary values with department median
4. Filters out ages below 18 or above 65
5. Sorts by salary (descending)

Show before and after statistics.

---

### Problem 14: Performance Comparison
**Difficulty**: Medium | **Type**: Analysis

Compare the performance of three approaches to computing element-wise square root:

1. Python loop with math.sqrt
2. List comprehension with math.sqrt
3. NumPy np.sqrt

Use an array of 1,000,000 elements. Report timing for each and explain why NumPy is faster.

---

### Problem 15: Correlation Analysis
**Difficulty**: Medium | **Type**: Visualization

Create a dataset with 4 features where:
- Feature 1 and 2 are strongly correlated
- Feature 3 is weakly correlated with Feature 1
- Feature 4 is independent

Generate 1000 samples. Compute and visualize the correlation matrix using seaborn's heatmap.

---

### Problem 16: Rolling Statistics
**Difficulty**: Medium | **Type**: Implementation

Given a 1D time series array, implement a function `rolling_mean(arr, window_size)` that computes the rolling mean.

For example, with `arr = [1, 2, 3, 4, 5]` and `window_size = 3`:
- Result: [NaN, NaN, 2, 3, 4]

(First two values are NaN because there aren't enough previous values)

Bonus: Make it vectorized!

---

### Problem 17: GroupBy Custom Aggregation
**Difficulty**: Medium | **Type**: Applied

Create a DataFrame of sales data with columns: `product`, `region`, `quantity`, `price`.

Compute for each region:
- Total revenue (quantity × price)
- Average price
- Number of transactions
- Best-selling product

Present results in a clear format.

---

### Problem 18: Matrix Operations Chain
**Difficulty**: Medium | **Type**: Numerical

Given matrices:
- A (3×4)
- B (4×2)
- C (2×3)

Compute: `(A @ B @ C).T` and explain the shape at each step.

Create actual matrices and verify your shape calculations.

---

### Problem 19: Outlier Detection
**Difficulty**: Medium | **Type**: Implementation

Implement a function that detects outliers using the IQR (Interquartile Range) method:
- Outliers are values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR

Test on data: `[1, 2, 2, 3, 3, 3, 4, 4, 5, 100]`

The value 100 should be detected as an outlier.

---

### Problem 20: Data Reshaping Challenge
**Difficulty**: Medium | **Type**: Coding

You have a 1D array representing a flattened image: 784 values (28×28 pixels).

Write code to:
1. Reshape to 28×28
2. Transpose it
3. Flatten it back to 1D
4. Verify the result is different from the original

Explain why the result changed.

---

## ⭐⭐⭐ Hard Problems (5)

### Problem 21: Efficient Batch Normalization
**Difficulty**: Hard | **Type**: Engineering

**Scenario**: You're training a neural network and need to normalize batches of data efficiently.

Implement a `BatchNorm` class that:
1. Normalizes each feature in a batch to mean=0, std=1
2. Keeps running statistics (exponential moving average) of mean and std
3. Can switch between training mode (normalize with batch stats) and evaluation mode (normalize with running stats)

**Requirements**:
- Must be vectorized
- Should handle batches of any size
- Must update running stats correctly

**Test**: Create random batches and verify:
- Training mode normalizes correctly
- Running stats converge to true distribution stats
- Evaluation mode uses running stats

**Hint**: Research how batch normalization works in deep learning frameworks.

**Expected Insights**:
- Understanding of training vs evaluation behavior
- Importance of tracking statistics
- Why we use exponential moving averages

---

### Problem 22: Memory-Efficient Large Matrix Operations
**Difficulty**: Hard | **Type**: Optimization

**Scenario**: You need to compute the correlation matrix for 10,000 features, but loading everything into memory at once causes out-of-memory errors.

Implement a memory-efficient solution that:
1. Computes correlation in chunks
2. Uses significantly less memory than `np.corrcoef()`
3. Produces the same result (within numerical precision)

**Requirements**:
- Compare memory usage of your solution vs naive approach
- Time both approaches
- Explain the memory/speed tradeoff

**Hint**: Process subsets of features at a time.

**Expected Insights**:
- How to work with data larger than RAM
- Chunking strategies
- Memory vs computation tradeoffs

---

### Problem 23: Debugging Vectorization Gone Wrong
**Difficulty**: Hard | **Type**: Debugging

**Scenario**: A colleague wrote this "vectorized" code but it gives wrong results:

```python
def compute_distances(X, Y):
    # X: (n, d), Y: (m, d)
    # Should return: (n, m) pairwise distances
    return np.sqrt((X ** 2).sum(axis=1) + (Y ** 2).sum(axis=1))
```

**Tasks**:
1. Identify what's wrong (there are multiple bugs!)
2. Explain why it fails
3. Provide the correct implementation
4. Create test cases that expose the bugs

**Hint**: Think about shapes carefully. Use broadcasting properly.

**Expected Insights**:
- Common broadcasting mistakes
- How to debug shape errors
- Testing strategies for numerical code

---

### Problem 24: Custom GroupBy Engine
**Difficulty**: Hard | **Type**: System Design

**Scenario**: Implement your own simplified version of Pandas `groupby` using only NumPy.

**Requirements**:
Implement a class `CustomGroupBy` that:
1. Takes an array of groups and an array of values
2. Can compute mean, sum, min, max per group
3. Returns results as a dictionary: {group: statistic}

**Example**:
```python
groups = ['A', 'B', 'A', 'B', 'A']
values = [10, 20, 30, 40, 50]

gb = CustomGroupBy(groups, values)
gb.mean()  # {'A': 30.0, 'B': 30.0}
gb.sum()   # {'A': 90, 'B': 60}
```

**Bonus**: Make it work with 2D value arrays (multiple columns).

**Hint**: Use `np.unique` with `return_inverse=True`.

**Expected Insights**:
- How groupby operations work internally
- Efficient aggregation strategies
- Data structure choices impact performance

---

### Problem 25: Optimize This Pipeline
**Difficulty**: Hard | **Type**: Performance Challenge

**Scenario**: This data processing pipeline is too slow for production:

```python
def slow_pipeline(data):
    # data: DataFrame with 1M rows, 100 columns
    result = []
    for idx, row in data.iterrows():
        row_result = []
        for col in data.columns:
            if row[col] > 0:
                row_result.append(np.log(row[col]))
            else:
                row_result.append(0)
        result.append(row_result)
    return pd.DataFrame(result)
```

**Tasks**:
1. Identify all performance issues
2. Rewrite it to be 100x+ faster
3. Prove the speedup with timing
4. Explain each optimization you made

**Requirements**:
- Must produce identical results
- Must be fully vectorized
- Document the speedup achieved

**Hint**: Eliminate ALL loops. Use vectorized conditionals.

**Expected Insights**:
- Why iterrows() is slow
- How to vectorize conditionals
- Impact of pandas vs NumPy operations
- Real-world optimization strategies

---

## Solutions

Detailed solutions to all problems are available in `solutions.py`.

**Remember**: The goal is not just to solve problems, but to build ML engineering intuition!

---

## Progress Tracker

Track your progress:

**Easy Problems**: [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] (0/10)

**Medium Problems**: [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] (0/10)

**Hard Problems**: [ ] [ ] [ ] [ ] [ ] (0/5)

---

**Completed all 25 problems?**  
🎉 Congratulations! You've mastered Python fundamentals for ML!  
→ Next module: `01-linear-algebra`
