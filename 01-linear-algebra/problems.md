# Linear Algebra for Machine Learning - Problems

Master linear algebra through carefully designed problems that build ML engineering intuition.

## Problem Guidelines

- **Easy (⭐)**: Foundation building, concept verification, basic computations
- **Medium (⭐⭐)**: Multi-step reasoning, implementations, applied linear algebra
- **Hard (⭐⭐⭐)**: ML engineer-level thinking, optimization, research-oriented

**Recommendation**: Work through problems with pen and paper first, then verify with code!

---

## ⭐ Easy Problems (10)

### Problem 1: Vector Dot Product Mastery
**Difficulty**: Easy | **Type**: Numerical + Conceptual

Given vectors `a = [2, 3, 4]` and `b = [1, -2, 3]`:

1. Compute the dot product `a · b` by hand
2. Verify using NumPy
3. Compute the angle θ between the vectors (in degrees)
4. Are these vectors orthogonal? Why or why not?

**Expected Output**: 
- Dot product value
- Angle in degrees
- Orthogonality answer with reasoning

---

### Problem 2: Matrix Multiplication Dimensions
**Difficulty**: Easy | **Type**: Conceptual

You have the following matrices:
- A: shape (3, 4)
- B: shape (4, 2)
- C: shape (2, 5)
- D: shape (5, 3)

Which of the following operations are valid? For valid operations, what's the output shape?

1. A @ B
2. B @ A
3. A @ B @ C
4. C @ D @ A
5. (A @ B).T
6. D @ C @ B @ A

**Challenge**: Explain the general rule for matrix multiplication compatibility.

---

### Problem 3: Computing Norms
**Difficulty**: Easy | **Type**: Numerical

For vector `v = [3, -4, 12, 5]`:

1. Compute L₁ norm (Manhattan distance)
2. Compute L₂ norm (Euclidean length)
3. Compute L∞ norm (Maximum)
4. Normalize v to unit length
5. Verify that the normalized vector has length 1

Show all calculations explicitly.

---

### Problem 4: Matrix Transpose Properties
**Difficulty**: Easy | **Type**: Proof + Coding

Create two random 3×4 matrices A and B.

**Prove or verify:**
1. (A^T)^T = A
2. (A + B)^T = A^T + B^T
3. (2A)^T = 2A^T

Then, create a 3×4 matrix A and 4×2 matrix B:

4. Show that (AB)^T = B^T A^T
5. Explain why the order reverses

---

### Problem 5: Orthogonality Check
**Difficulty**: Easy | **Type**: Conceptual + Numerical

Given vectors:
- u = [1, 2, -1]
- v = [2, -1, 0]
- w = [1, 0, 1]

1. Check which pairs (if any) are orthogonal
2. For each orthogonal pair, verify geometrically what this means
3. Find a vector that's orthogonal to both u and v (use cross product or solve system)

---

### Problem 6: Identity and Inverse
**Difficulty**: Easy | **Type**: Numerical

For matrix A = [[4, 7], [2, 6]]:

1. Compute the determinant
2. Is it invertible? Why?
3. Compute A^(-1) using NumPy
4. Verify that A @ A^(-1) = I and A^(-1) @ A = I
5. Compute A^(-1) manually using the 2×2 inverse formula:
   ```
   A^(-1) = (1/det(A)) * [[d, -b], [-c, a]]
   ```

---

### Problem 7: Linear Combinations
**Difficulty**: Easy | **Type**: Conceptual

Given vectors v₁ = [1, 0, 1], v₂ = [0, 1, -1], v₃ = [1, 1, 0]:

1. Express [3, 2, 1] as a linear combination of v₁, v₂, v₃
2. Can [1, 1, 1] be expressed as a linear combination? Find it or prove impossible
3. What is the span of {v₁, v₂, v₃}? (Describe geometrically)

**Format**: Write as: target = c₁v₁ + c₂v₂ + c₃v₃

---

### Problem 8: Eigenvalue Intuition
**Difficulty**: Easy | **Type**: Conceptual

For diagonal matrix D = [[3, 0], [0, 5]]:

1. Find the eigenvalues (should be obvious!)
2. Find the eigenvectors
3. Verify Av = λv for each eigenvalue-eigenvector pair
4. Explain geometrically what this matrix does to vectors
5. Why are diagonal matrices so nice to work with?

---

### Problem 9: Matrix Rank
**Difficulty**: Easy | **Type**: Numerical + Conceptual

For each matrix, determine the rank:

1. A = [[1, 2], [2, 4]]
2. B = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
3. C = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

**Questions**:
- Which are full rank?
- What does rank tell us about the matrix?
- How does rank relate to invertibility?

---

### Problem 10: Projection Basics
**Difficulty**: Easy | **Type**: Numerical

Project vector b = [4, 3] onto vector a = [1, 0] (the x-axis).

1. Compute the projection using the formula: proj_a(b) = (a·b / a·a) × a
2. Draw a sketch showing a, b, and the projection
3. What's the component of b perpendicular to a?
4. Verify that projection and perpendicular component sum to b

---

## ⭐⭐ Medium Problems (10)

### Problem 11: Implement Matrix Multiplication from Scratch
**Difficulty**: Medium | **Type**: Implementation

Write a function `matrix_multiply(A, B)` that:
- Takes two NumPy arrays
- Returns their matrix product
- **WITHOUT using @ or np.dot or np.matmul**
- Must use only loops and element-wise operations

**Requirements**:
1. Check dimension compatibility
2. Raise informative error if incompatible
3. Implement using 3 nested loops
4. Test on at least 3 different size combinations
5. Verify against NumPy's @ operator

**Bonus**: Analyze time complexity. Why is this slow?

---

### Problem 12: Gram-Schmidt Orthogonalization
**Difficulty**: Medium | **Type**: Algorithm Implementation

Implement the Gram-Schmidt process to orthogonalize vectors.

**Given**: Three linearly independent vectors in R³
```python
v1 = np.array([1, 1, 0])
v2 = np.array([1, 0, 1])
v3 = np.array([0, 1, 1])
```

**Task**:
1. Implement Gram-Schmidt to get orthogonal vectors u1, u2, u3
2. Normalize them to get orthonormal vectors e1, e2, e3
3. Verify orthonormality: ei · ej = δij (1 if i=j, 0 otherwise)
4. Show that span{v1,v2,v3} = span{e1,e2,e3}

**Algorithm**:
```
u1 = v1
u2 = v2 - proj_u1(v2)
u3 = v3 - proj_u1(v3) - proj_u2(v3)
```

---

### Problem 13: Eigenvalues by Hand
**Difficulty**: Medium | **Type**: Analytical

For matrix A = [[5, 2], [2, 2]]:

1. Find eigenvalues by solving det(A - λI) = 0
2. For each eigenvalue, find corresponding eigenvector by solving (A - λI)v = 0
3. Verify Av = λv for both pairs
4. Reconstruct A using eigenvalue decomposition: A = QΛQ^T
5. Verify reconstruction is accurate

**Show all work step-by-step!**

---

### Problem 14: SVD Low-Rank Approximation
**Difficulty**: Medium | **Type**: Applied

Create a 10×10 matrix with rank 3:
```python
A = np.random.randn(10, 3) @ np.random.randn(3, 10)
```

1. Compute the SVD: A = UΣV^T
2. What are the singular values? How many are effectively non-zero?
3. Create rank-1 and rank-2 approximations
4. Plot approximation error vs rank
5. Explain:why does the error drop to ~0 at rank 3?

**Visualization**: Create a bar plot of singular values to see the "rank gap"

---

### Problem 15: Solving Linear Systems Multiple Ways
**Difficulty**: Medium | **Type**: Comparative Analysis

Given system: Ax = b where
```python
A = [[2, 1], [1, 3]]
b = [5, 7]
```

Solve using THREE methods:

1. **Matrix Inversion**: x = A^(-1)b
2. **Gaussian Elimination**: Row reduce [A|b]
3. **NumPy solve**: np.linalg.solve()

**Compare**:
- Computational cost (count operations)
- Numerical stability (try with nearly singular matrix)
- When to use each method

**Challenge**: Create a nearly singular matrix and show why inversion fails but solve() works.

---

### Problem 16: PCA Implementation
**Difficulty**: Medium | **Type**: Full Implementation

Implement PCA from scratch (no sklearn):

**Steps**:
1. Generate 2D data with correlation (use np.random.multivariate_normal)
2. Center the data (subtract mean)
3. Compute covariance matrix
4. Eigenvalue decomposition
5. Sort eigenvectors by eigenvalues (descending)
6. Project data onto principal components
7. Visualize: original data + principal component directions

**Deliverables**:
- Working code
- Visualization showing PC1 and PC2
- Percentage variance explained by each PC

---

### Problem 17: Matrix Conditioning
**Difficulty**: Medium | **Type**: Numerical Analysis

Understand why some linear systems are hard to solve:

1. Create a well-conditioned matrix (condition number ~1)
2. Create an ill-conditioned matrix (condition number >10^6)
3. For each, solve Ax = b
4. Add small noise to b: b_noisy = b + 0.01*random
5. Solve again: x_noisy
6. Compare ||x - x_noisy|| for both matrices

**Observations**:
- Why does the ill-conditioned system give wildly different solutions?
- How does this relate to ML? (Hint: regularization!)

**Formula**: condition(A) = σ_max / σ_min

---

### Problem 18: Geometric Transformations
**Difficulty**: Medium | **Type**: Geometric + Implementation

Create transformation matrices for:

1. **Rotation** by 45° counterclockwise
2. **Scaling** by 2x in x-direction, 0.5x in y-direction
3. **Shear** that moves (0,1) to (1,1), keeps (1,0) fixed
4. **Reflection** across the line y = x

**Tasks**:
- Derive each 2×2 matrix analytically
- Apply to unit square: points [(0,0), (1,0), (1,1), (0,1)]
- Visualize all 4 transformations
- Compose them: what does Rotation ∘ Scaling do?

---

### Problem 19: Determinant as Volume
**Difficulty**: Medium | **Type**: Geometric Interpretation

For matrix A = [[2, 1], [0, 3]]:

1. Compute det(A)
2. Draw the parallelogram formed by column vectors
3.Calculate the parallelogram's area geometrically
4. Verify it equals |det(A)|

**3D Extension**:
Create A = [[1,0,0], [0,2,0], [1,1,3]]
1. det(A) = ?
2. What's the volume of the parallelepiped formed by columns?

**Insight**: Determinant = signed volume scaling factor!

---

### Problem 20: Least Squares Regression
**Difficulty**: Medium | **Type**: ML Application

Generate noisy linear data:
```python
x = np.linspace(0, 10, 50)
y = 3*x + 2 + np.random.randn(50)*2
```

**Solve using linear algebra**:
1. Set up the system: X @ β = y where β = [intercept, slope]
2. X should be [ones, x_values] (design matrix)
3. Solve using normal equations: β = (X^T X)^(-1) X^T y
4. This is the same as linear regression!

**Verify**:
- Compare to sklearn LinearRegression
- Plot data, fitted line, and residuals
- Compute R² score

---

## ⭐⭐⭐ Hard Problems (5)

### Problem 21: Power Iteration for Eigenvectors
**Difficulty**: Hard | **Type**: Algorithm + Theory

Implement the power iteration method to find the largest eigenvalue and eigenvector.

**Algorithm**:
```
1. Start with random vector v
2. Repeat:
   v = A @ v
   v = v / ||v||  (normalize)
3. Converge to dominant eigenvector
4. Eigenvalue = v^T @ A @ v
```

**Tasks**:
1. Implement power iteration
2. Test on matrix A = [[4, 1], [2, 3]]
3. Track convergence (plot eigenvalue estimate vs iteration)
4. Explain why it converges to the largest eigenvalue
5. What happens if you start with random vector orthogonal to dominant eigenvector?

**Extensions**:
- Implement inverse iteration for smallest eigenvalue
- How does convergence rate depend on eigenvalue gap?

**Hint**: This is how PageRank is actually computed!

---

### Problem 22: Backpropagation is Matrix Calculus
**Difficulty**: Hard | **Type**: ML Deep Dive

Understand backprop through linear algebra:

**Simple network**: Input → Linear → ReLU → Linear → Output
```python
# Forward
z1 = W1 @ x + b1  # (h × d)
a1 = relu(z1)     # (h × 1)
z2 = W2 @ a1 + b2 # (1 × 1)
loss = (z2 - y)²
```

**Derive gradients analytically**:
1. ∂loss/∂z2 = ?
2. ∂loss/∂W2 = ? (use chain rule)
3. ∂loss/∂a1 = ?
4. ∂loss/∂z1 = ? (handle ReLU derivative)
5. ∂loss/∂W1 = ?

**Implement**:
- Forward pass
- Backward pass computing all gradients
- Verify using numerical gradient checking:
  ```
  (f(θ + ε) - f(θ - ε)) / (2ε)
  ```

**Key insight**: Backprop is just recursive application of chain rule using matrix operations!

---

### Problem 23: Optimal Low-Rank Matrix Approximation
**Difficulty**: Hard | **Type**: Optimization + Proof

**Theorem**: Truncated SVD gives the best rank-k approximation (Eckart-Young).

**Task**: Prove this for rank-1 case.

**Given**: Matrix A (m × n), want best rank-1 approximation: A ≈ u × v^T

**Minimize**: ||A - uv^T||_F² (Frobenius norm)

**Steps**:
1. Expand ||A - uv^T||_F²
2. Take derivatives ∂/∂u and ∂/∂v
3. Set to zero and solve
4. Show solution is u = u₁, v = σ₁v₁ (first singular vectors/value)

**Implement**:
- Create random 5×5 matrix
- Compute rank-1 SVD approximation
- Try random rank-1 matrices and show SVD is better
- Visualize error landscape

**Generalization**: Explain why this extends to rank-k approximations.

---

### Problem 24: Designing a Stable Matrix Solver
**Difficulty**: Hard | **Type**: Numerical Methods Engineering

**Scenario**: You need to solve Ax = b reliably in production ML code.

**Design requirements**:
1. Handle near-singular matrices
2. Warn when solution is unreliable
3. Offer automatic regularization
4. Be computationally efficient

**Implementation**:
```python
def stable_solve(A, b, tol=1e-10):
    """
    Solve Ax = b with numerical stability checks.
    
    Returns: x, metadata
    metadata contains:
    - condition_number
    - method_used
    - warnings
    - is_reliable
    """
    # Your code here
```

**Components**:
- Check condition number (σ_max/σ_min)
- If cond > 10^10, add regularization: (A^T A + λI)^(-1) A^T b
- Choose λ based on singular values
- Use QR or SVD for better stability than normal equations
- Return confidence score

**Test on:**
- Well-conditioned system
- Nearly singular system
- Badly scaled system
- Rectangular system (overdetermined)

**This is production ML engineering!**

---

### Problem 25: Transformer Attention is Linear Algebra
**Difficulty**: Hard | **Type**: ML Research Application

Implement multi-head attention mechanism from transformers using pure linear algebra.

**Given**: Input matrix X (sequence_length × d_model)

**Formulas**:
```
Q = X @ W_Q  (query)
K = X @ W_K  (key)
V = X @ W_V  (value)

Attention(Q,K,V) = softmax(QK^T / √d_k) @ V
```

**Tasks**:

1. **Single-head attention**:
   - Create random X (10 × 64)
   - Random weight matrices W_Q, W_K, W_V
   - Compute attention output
   - Explain what QK^T represents (similarity matrix!)

2. **Multi-head attention**:
   - Split into h=8 heads
   - Compute attention for each head
   - Concatenate and project

3. **Visualization**:
   - Plot attention matrix (QK^T after softmax)
   - This shows which tokens attend to which!
   - Darker = more attention

4. **Analysis**:
   - Why divide by √d_k?
   - What does softmax do?
   - How is this "self-attention"?

5. **Optimization**:
   - Current: O(n² d) where n = sequence length
   - Propose approximation for long sequences
   - Research: How do efficient transformers work?

**This is how GPT works!** Understanding the linear algebra is crucial.

**Extensions**:
- Add mask for causal attention (GPT-style)
- Implement cross-attention (encoder-decoder)
- Analyze complexity and suggest optimizations

---

## Solutions

Complete solutions with step-by-step explanations are in `solutions.py`.

---

## Problem-Solving Strategy

### For Easy Problems:
1. Review relevant theory section
2. Work out by hand first
3. Verify with NumPy
4. Understand geometric meaning

### For Medium Problems:
1. Break into smaller steps
2. Implement incrementally
3. Test with simple cases first
4. Visualize when possible
5. Compare multiple approaches

### For Hard Problems:
1. Read related research/papers
2. Start with simplified version
3. Derive mathematically before coding
4. Validate extensively
5. Analyze edge cases
6. Consider real-world implications

---

## Progress Tracker

**Easy Problems**: [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] (0/10)

**Medium Problems**: [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] (0/10)

**Hard Problems**: [ ] [ ] [ ] [ ] [ ] (0/5)

**Total**: 0/25

---

## Achievement Levels

- **Bronze** (10/25): Good foundation
- **Silver** (17/25): Strong understanding
- **Gold** (22/25): Mastery level
- **Platinum** (25/25): Elite ML Engineer! 🏆

---

**Complete all 25?** You're ready for advanced ML! Move to **Module 02: Calculus**

**Remember**: These problems are designed to make you THINK like an ML engineer, not just code!

