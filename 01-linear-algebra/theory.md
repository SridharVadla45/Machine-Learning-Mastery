# Linear Algebra for Machine Learning - Theory

**The Mathematical Foundation of Modern AI**

---

## Table of Contents

1. [Why Linear Algebra?](#1-why-linear-algebra)
2. [Scalars, Vectors, and Matrices](#2-scalars-vectors-and-matrices)
3. [Vector Operations](#3-vector-operations)
4. [Matrix Operations](#4-matrix-operations)
5. [Matrix Multiplication Deep Dive](#5-matrix-multiplication-deep-dive)
6. [Linear Transformations](#6-linear-transformations)
7. [Systems of Linear Equations](#7-systems-of-linear-equations)
8. [Matrix Rank and Span](#8-matrix-rank-and-span)
9. [Determinants](#9-determinants)
10. [Eigenvalues and Eigenvectors](#10-eigenvalues-and-eigenvectors)
11. [Matrix Decompositions](#11-matrix-decompositions)
12. [Norms and Distances](#12-norms-and-distances)
13. [Orthogonality and Projections](#13-orthogonality-and-projections)
14. [Computational Considerations](#14-computational-considerations)
15. [ML Applications](#15-ml-applications)

---

## 1. Why Linear Algebra?

### The Language of Machine Learning

**Every ML algorithm is built on linear algebra:**

- **Neural Networks**: Matrix multiplications and transformations
- **PCA**: Eigenvalue decomposition
- **SVD**: Collaborative filtering, dimensionality reduction
- **Linear Regression**: Matrix inversion, projections
- **Gradient Descent**: Vector operations
- **Covariance**: Matrix operations
- **Transformers**: Attention is matrix operations

### The Power of Linear Algebra

**Why is linear algebra so powerful for ML?**

1. **Compact Representation**: Express operations on millions of numbers in one line
2. **Geometric Intuition**: Understand transformations visually
3. **Computational Efficiency**: Optimized matrix operations (GPU acceleration)
4. **Generalization**: Same concepts work in any dimension
5. **Theoretical Foundation**: Rigorous mathematical framework

### ML Without Linear Algebra?

**Impossible.** You'd be:
- Writing millions of individual operations
- Missing geometric intuition
- Unable to use GPUs effectively
- Reinventing the wheel constantly

**With Linear Algebra:**
- Express complex operations concisely
- Understand what operations do geometrically
- Leverage optimized libraries (BLAS, cuBLAS)
- Build on established theory

---

## 2. Scalars, Vectors, and Matrices

### Scalars

A **scalar** is a single number.

**Examples:**
```
x = 5
α = 0.001 (learning rate)
λ = 2.5 (regularization parameter)
```

**In ML:**
- Loss values
- Learning rates
- Hyperparameters
- Predictions (regression)

### Vectors

A **vector** is an ordered array of numbers.

**Notation:**
```
      ⎡ 1 ⎤
  v = ⎢ 2 ⎥  (column vector, preferred in ML)
      ⎣ 3 ⎦

  v = [1, 2, 3]  (row vector)
```

**Dimensions:**
- Length n (number of elements)
- Lives in ℝⁿ (n-dimensional real space)

**Geometric Interpretation:**
- Point in n-dimensional space
- Arrow from origin to that point
- Direction and magnitude

**In ML:**
```python
# A single data sample
x = [age, income, height]

# Model parameters
w = [w₁, w₂, w₃]  # weights

# Predictions for a batch
y_pred = [0.8, 0.3, 0.9, 0.1]
```

### Matrices

A **matrix** is a 2D array of numbers.

**Notation:**
```
      ⎡ 1  2  3 ⎤
  A = ⎢ 4  5  6 ⎥  (2×3 matrix)
      ⎣ 7  8  9 ⎦
```

**Dimensions:**
- m × n matrix (m rows, n columns)
- Lives in ℝᵐˣⁿ

**Indexing:**
- Aᵢⱼ = element at row i, column j
- Indices typically start at 1 (math) or 0 (programming)

**In ML:**
```python
# Dataset: 1000 samples, 5 features
X = [[x₁₁, x₁₂, x₁₃, x₁₄, x₁₅],
     [x₂₁, x₂₂, x₂₃, x₂₄, x₂₅],
     ...
     [x₁₀₀₀,₁, ..., x₁₀₀₀,₅]]

# Weight matrix in neural network
W = [[w₁₁, w₁₂, w₁₃],
     [w₂₁, w₂₂, w₂₃]]
```

### Tensors

A **tensor** is a generalization to higher dimensions.

**Dimensions:**
- 0D tensor: scalar
- 1D tensor: vector
- 2D tensor: matrix
- 3D tensor: cube of numbers
- nD tensor: n-dimensional array

**In ML:**
```python
# Batch of images: (batch_size, height, width, channels)
images = np.zeros((32, 224, 224, 3))  # 4D tensor

# Video: (batch, time, height, width, channels)
video = np.zeros((16, 30, 224, 224, 3))  # 5D tensor
```

---

## 3. Vector Operations

### Vector Addition

**Element-wise addition:**

```
⎡ 1 ⎤   ⎡ 4 ⎤   ⎡ 5 ⎤
⎢ 2 ⎥ + ⎢ 5 ⎥ = ⎢ 7 ⎥
⎣ 3 ⎦   ⎣ 6 ⎦   ⎣ 9 ⎦
```

**Properties:**
- Commutative: a + b = b + a
- Associative: (a + b) + c = a + (b + c)
- Identity: a + 0 = a

**Geometric Interpretation:**
- Parallelogram rule
- "Walk" along first vector, then second

**In ML:**
```python
# Adding bias in neural networks
output = Wx + b  # b is added to each element
```

### Scalar Multiplication

**Multiply each element by scalar:**

```
    ⎡ 1 ⎤   ⎡ 2 ⎤
2 × ⎢ 2 ⎥ = ⎢ 4 ⎥
    ⎣ 3 ⎦   ⎣ 6 ⎦
```

**Properties:**
- Distributive: c(a + b) = ca + cb
- Associative: c₁(c₂a) = (c₁c₂)a

**Geometric Interpretation:**
- Scales the vector's magnitude
- Negative scalar flips direction

**In ML:**
```python
# Learning rate scaling gradient
update = -learning_rate * gradient
```

### Dot Product (Inner Product)

**Definition:**
```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ = Σ aᵢbᵢ
```

**Example:**
```
[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

**Properties:**
- Commutative: a · b = b · a
- Distributive: a · (b + c) = a · b + a · c
- Scalar multiplication: (ca) · b = c(a · b)

**Geometric Interpretation:**
```
a · b = ||a|| × ||b|| × cos(θ)
```
Where θ is the angle between vectors.

**Important Cases:**
- θ = 0°: a · b = ||a|| × ||b|| (parallel, same direction)
- θ = 90°: a · b = 0 (perpendicular, orthogonal)
- θ = 180°: a · b = -||a|| × ||b|| (opposite direction)

**In ML:**
```python
# Linear regression prediction
y_pred = w · x + b

# Similarity measure
similarity = vec1 · vec2

# Attention mechanism
attention_score = query · key
```

### Vector Norm (Length)

**L2 Norm (Euclidean length):**
```
||v||₂ = √(v₁² + v₂² + ... + vₙ²) = √(v · v)
```

**L1 Norm (Manhattan distance):**
```
||v||₁ = |v₁| + |v₂| + ... + |vₙ|
```

**L∞ Norm (Maximum norm):**
```
||v||∞ = max(|v₁|, |v₂|, ..., |vₙ|)
```

**Properties:**
- ||v|| ≥ 0
- ||v|| = 0 ⟺ v = 0
- ||cv|| = |c| × ||v||
- Triangle inequality: ||a + b|| ≤ ||a|| + ||b||

**In ML:**
```python
# Regularization
L2_penalty = λ × ||w||₂²
L1_penalty = λ × ||w||₁

# Gradient clipping
if ||gradient|| > threshold:
    gradient = gradient × threshold / ||gradient||
```

### Unit Vector (Normalization)

**Definition:**
```
û = v / ||v||
```

**Properties:**
- ||û|| = 1
- Points in same direction as v
- Essential for many algorithms

**In ML:**
```python
# Normalize embeddings
normalized_embedding = embedding / ||embedding||

# Cosine similarity works on normalized vectors
cosine_sim = normalized_a · normalized_b
```

### Cross Product (3D only)

**Definition (for 3D vectors):**
```
a × b = ⎡ a₂b₃ - a₃b₂ ⎤
        ⎢ a₃b₁ - a₁b₃ ⎥
        ⎣ a₁b₂ - a₂b₁ ⎦
```

**Properties:**
- Result is perpendicular to both a and b
- ||a × b|| = ||a|| × ||b|| × sin(θ)
- Anti-commutative: a × b = -(b × a)

**Less common in ML, but used in:**
- 3D computer vision
- Robotics
- Graphics

---

## 4. Matrix Operations

### Matrix Addition

**Element-wise:**
```
⎡ 1  2 ⎤   ⎡ 5  6 ⎤   ⎡ 6  8  ⎤
⎢ 3  4 ⎥ + ⎢ 7  8 ⎥ = ⎢ 10  12 ⎥
```

**Requirements:**
- Same dimensions (m × n)

**Properties:**
- Same as vector addition

**In ML:**
```python
# Combining gradients
total_gradient = gradient1 + gradient2

# Adding regularization
loss_with_reg = loss + regularization_term
```

### Scalar Multiplication

**Multiply each element:**
```
    ⎡ 1  2 ⎤   ⎡ 2  4 ⎤
2 × ⎢ 3  4 ⎥ = ⎢ 6  8 ⎥
```

**In ML:**
```python
# Scale learning rate
update = -0.01 * gradient
```

### Transpose

**Flip rows and columns:**
```
    ⎡ 1  2  3 ⎤ᵀ   ⎡ 1  4 ⎤
A = ⎢ 4  5  6 ⎥  = ⎢ 2  5 ⎥
                    ⎣ 3  6 ⎦
```

**Notation:** Aᵀ

**Properties:**
- (Aᵀ)ᵀ = A
- (A + B)ᵀ = Aᵀ + Bᵀ
- (AB)ᵀ = BᵀAᵀ (order reverses!)
- (cA)ᵀ = cAᵀ

**Dimensions:**
- If A is m × n, then Aᵀ is n × m

**In ML:**
```python
# Convert between row and column vectors
y = x.T

# Compute covariance matrix
Σ = (X - μ)ᵀ(X - μ) / n

# Backpropagation
grad_W = grad_output.T @ activations
```

### Hadamard Product (Element-wise)

**Element-wise multiplication:**
```
⎡ 1  2 ⎤     ⎡ 5  6 ⎤   ⎡ 5   12 ⎤
⎢ 3  4 ⎥  ⊙  ⎢ 7  8 ⎥ = ⎢ 21  32 ⎥
```

**Notation:** A ⊙ B or A * B (NumPy uses *)

**Requirements:**
- Same dimensions

**In ML:**
```python
# Applying activation functions element-wise
activated = sigmoid(z) ⊙ (1 - sigmoid(z))  # sigmoid derivative

# Attention mechanism
attended = values ⊙ attention_weights
```

---

## 5. Matrix Multiplication Deep Dive

### The Most Important Operation in ML

**Matrix multiplication is everywhere:**
- Forward pass in neural networks
- Backpropagation
- Transformers (attention)
- Convolutions (can be expressed as matrix multiply)

### Definition

**Matrix-vector multiplication:**
```
⎡ 1  2  3 ⎤   ⎡ 1 ⎤   ⎡ 1×1 + 2×2 + 3×3 ⎤   ⎡ 14 ⎤
⎢ 4  5  6 ⎥ × ⎢ 2 ⎥ = ⎢ 4×1 + 5×2 + 6×3 ⎥ = ⎢ 32 ⎥
                ⎣ 3 ⎦
```

**Matrix-matrix multiplication:**
```
      ⎡ 1  2 ⎤
A =   ⎢ 3  4 ⎥  (2×2)
      
      ⎡ 5  6 ⎤
B =   ⎢ 7  8 ⎥  (2×2)

      ⎡ 1×5+2×7  1×6+2×8 ⎤   ⎡ 19  22 ⎤
AB =  ⎢ 3×5+4×7  3×6+4×8 ⎥ = ⎢ 43  50 ⎥
```

**General Rule:**
```
(AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ
```

The element at row i, column j of AB is the dot product of:
- Row i of A
- Column j of B

### Dimension Requirements

**For A (m × n) and B (p × q):**
- Can multiply AB only if **n = p** (inner dimensions match)
- Result AB is **m × q** (outer dimensions)

**Memory tip:**
```
(m × n) × (n × q) = (m × q)
     ↑       ↑
  must match!
```

### Properties

**NOT Commutative:**
- AB ≠ BA (in general)
- Order matters!

**Associative:**
- (AB)C = A(BC)

**Distributive:**
- A(B + C) = AB + AC
- (A + B)C = AC + BC

**Identity:**
- AI = IA = A (where I is identity matrix)

### Geometric Interpretation

**Matrix multiplication is function composition:**

```
y = Bx  (first apply B)
z = Ay  (then apply A)

Combined: z = A(Bx) = (AB)x
```

**Each matrix is a transformation:**
- Rotation
- Scaling
- Shearing
- Reflection
- Projection

### Why Order Matters

**Example:**
```
Rotate 90°, then scale 2x ≠ Scale 2x, then rotate 90°
```

The order of transformations affects the final result!

### Computational Complexity

**Naive algorithm:**
- O(mnq) for (m×n) × (n×q)
- For square matrices (n×n): O(n³)

**Optimized algorithms:**
- Strassen's: O(n^2.807)
- Best theoretical: O(n^2.373)
- Practical: Use BLAS libraries (highly optimized)

**In ML:**
```
# Forward pass in neural network layer
# X: (batch_size, in_features)
# W: (in_features, out_features)
output = X @ W  # (batch_size, out_features)

# Complexity: O(batch_size × in_features × out_features)
# For batch=32, in=1000, out=500:
# 32 × 1000 × 500 = 16 million operations
```

### Block Matrix Multiplication

**Large matrices can be split into blocks:**

```
⎡ A₁₁  A₁₂ ⎤   ⎡ B₁₁  B₁₂ ⎤   ⎡ A₁₁B₁₁+A₁₂B₂₁  A₁₁B₁₂+A₁₂B₂₂ ⎤
⎢ A₂₁  A₂₂ ⎥ × ⎢ B₂₁  B₂₂ ⎥ = ⎢ A₂₁B₁₁+A₂₂B₂₁  A₂₁B₁₂+A₂₂B₂₂ ⎥
```

**Used for:**
- Parallel computation
- GPU optimization
- Memory efficiency

---

## 6. Linear Transformations

### What is a Linear Transformation?

A function T: ℝⁿ → ℝᵐ that preserves:

1. **Addition:** T(u + v) = T(u) + T(v)
2. **Scalar multiplication:** T(cu) = cT(u)

**Every linear transformation can be represented as matrix multiplication!**

```
T(x) = Ax
```

### Examples of Linear Transformations

**2D Rotation (by angle θ):**
```
R(θ) = ⎡ cos(θ)  -sin(θ) ⎤
       ⎢ sin(θ)   cos(θ) ⎥
```

**2D Scaling:**
```
S = ⎡ sₓ   0  ⎤
    ⎢ 0   sᵧ  ⎥
```

**2D Reflection (across x-axis):**
```
F = ⎡ 1   0 ⎤
    ⎢ 0  -1 ⎥
```

**Projection onto x-axis:**
```
P = ⎡ 1  0 ⎤
    ⎢ 0  0 ⎥
```

### Understanding Transformations Geometrically

**Key insight:** Matrix columns show where basis vectors go!

```
A = ⎡ a  b ⎤
    ⎢ c  d ⎥

Transforms:
  ⎡ 1 ⎤      ⎡ a ⎤
  ⎢ 0 ⎥  →   ⎢ c ⎥

  ⎡ 0 ⎤      ⎡ b ⎤
  ⎢ 1 ⎥  →   ⎢ d ⎥
```

### In Machine Learning

**Every layer in a neural network is a linear transformation followed by nonlinearity:**

```python
# Linear transformation
z = Wx + b

# Nonlinear activation
a = σ(z)  # e.g., ReLU, sigmoid
```

**Why nonlinearity?**
- Without it, stacking layers is just matrix multiplication
- Multiple linear transformations = one linear transformation
- Need nonlinearity to learn complex functions!

---

## 7. Systems of Linear Equations

### The Fundamental Problem

**Solve for x:**
```
2x₁ + 3x₂ = 8
x₁ - x₂ = -1
```

**Matrix form:**
```
⎡ 2   3 ⎤   ⎡ x₁ ⎤   ⎡ 8  ⎤
⎢ 1  -1 ⎥ × ⎢ x₂ ⎥ = ⎢ -1 ⎥

Ax = b
```

### Three Possibilities

1. **Unique solution**: Exactly one x satisfies Ax = b
2. **No solution**: No x satisfies Ax = b (inconsistent)
3. **Infinite solutions**: Many x satisfy Ax = b

**Depends on:**
- Rank of A
- Relationship between A and b

### Solving Methods

**Method 1: Matrix Inversion (if A is square and invertible)**
```
Ax = b
A⁻¹Ax = A⁻¹b
x = A⁻¹b
```

**Caution:**
- Only works if A is invertible
- Numerically unstable
- O(n³) complexity
- Don't actually compute A⁻¹!

**Method 2: Gaussian Elimination**
- Convert to row echelon form
- Back substitution
- More stable than inversion
- Still O(n³)

**Method 3: LU Decomposition**
- A = LU (lower × upper triangular)
- Solve Ly = b, then Ux = y
- Faster for multiple b vectors
- O(n³) decomposition, O(n²) per solve

**Method 4: Iterative Methods**
- Conjugate Gradient
- GMRES
- For large, sparse systems
- Can be faster than O(n³)

### In Machine Learning

**Linear regression:**
```
minimize ||Ax - b||²

Solution: x = (AᵀA)⁻¹Aᵀb  (normal equations)
```

**But we use gradient descent instead because:**
- More stable numerically
- Works for non-convex problems
- Scales better to large datasets
- Doesn't require matrix inversion

---

## 8. Matrix Rank and Span

### Rank

**Definition:** 
Maximum number of linearly independent row/column vectors.

**Intuition:**
- Dimensionality of the output space
- How much information the matrix contains

**Properties:**
- rank(A) ≤ min(m, n) for m × n matrix
- rank(A) = rank(Aᵀ)
- rank(AB) ≤ min(rank(A), rank(B))

**Full Rank:**
- rank(A) = min(m, n)
- Columns (or rows) are linearly independent
- Maximum information

**Rank Deficient:**
- rank(A) < min(m, n)
- Columns (or rows) are linearly dependent
- Lost information / redundancy

### Span

**Definition:**
Set of all possible linear combinations of vectors.

```
span{v₁, v₂, ..., vₙ} = {c₁v₁ + c₂v₂ + ... + cₙvₙ | c₁,...,cₙ ∈ ℝ}
```

**Column Space (Range):**
- span of columns of A
- All possible outputs of Ax

**Row Space:**
- span of rows of A
- All possible inputs that give non-zero output

**Null Space (Kernel):**
- {x | Ax = 0}
- Inputs that map to zero

### Rank-Nullity Theorem

```
rank(A) + nullity(A) = n  (number of columns)
```

**Intuition:**
- Dimensions must be accounted for
- Either contribute to range or kernel

### In Machine Learning

**Feature redundancy:**
- Low rank X means redundant features
- PCA removes this redundancy

**Model capacity:**
- Weight matrix rank limits expressiveness
- Low-rank bottleneck in autoencoders

**Gradient flow:**
- Rank-deficient Jacobian → vanishing gradients
- Information bottleneck

---

## 9. Determinants

### Definition

**For 2×2:**
```
det(⎡ a  b ⎤) = ad - bc
    ⎢ c  d ⎥
```

**For 3×3 (expansion by minors):**
```
det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃
```

Where Cᵢⱼ are cofactors.

**For n×n:**
- Recursive definition via cofactors
- O(n!) complexity naively
- O(n³) with LU decomposition

### Geometric Interpretation

**Determinant = signed volume of transformation**

**2D:**
- |det(A)| = area of parallelogram formed by column vectors

**3D:**
- |det(A)| = volume of parallelepiped

**Sign:**
- Positive: preserves orientation
- Negative: flips orientation
- Zero: collapses to lower dimension

### Properties

1. **det(I) = 1**
2. **det(AB) = det(A) × det(B)**
3. **det(Aᵀ) = det(A)**
4. **det(A⁻¹) = 1/det(A)**
5. **det(cA) = cⁿdet(A)** (for n×n matrix)
6. **Swapping rows multiplies det by -1**
7. **Row of zeros → det = 0**
8. **Linearly dependent rows → det = 0**

### Invertibility

**A is invertible ⟺ det(A) ≠ 0**

**Why?**
- det(A) = 0 means transformation collapses dimensionality
- Cannot recover original space
- No unique inverse

### In Machine Learning

**Checking invertibility:**
```python
if np.linalg.det(A) close to 0:
    # Matrix is nearly singular
    # Use regularization or pseudoinverse
```

**Change of variables (probability):**
```
p_y(y) = p_x(x) / |det(J)|
```
Where J is Jacobian matrix.

**Gaussian distribution:**
```
N(μ, Σ) ∝ exp(-½(x-μ)ᵀΣ⁻¹(x-μ)) / √det(Σ)
```

**Numerical stability:**
- Computing det directly is unstable
- Use log-determinant for large matrices
- log det(A) = sum of log eigenvalues

---

## 10. Eigenvalues and Eigenvectors

### The Most Important Concept in ML

**Definition:**

For square matrix A, eigenvalue λ and eigenvector v satisfy:

```
Av = λv
```

**Intuition:**
- v is a special direction
- A just scales v by λ
- Doesn't change direction!

### Finding Eigenvalues

**Characteristic equation:**
```
det(A - λI) = 0
```

**Example:**
```
A = ⎡ 4  2 ⎤
    ⎢ 1  3 ⎥

det(⎡ 4-λ   2  ⎤) = 0
    ⎢  1   3-λ ⎥

(4-λ)(3-λ) - 2 = 0
λ² - 7λ + 10 = 0
(λ - 5)(λ - 2) = 0

λ₁ = 5, λ₂ = 2
```

### Finding Eigenvectors

**For each λ, solve:**
```
(A - λI)v = 0
```

**Properties:**
- n×n matrix has n eigenvalues (counting multiplicities)
- May be complex even if A is real
- Eigenvectors are unique up to scaling

### Special Cases

**Symmetric Matrix (A = Aᵀ):**
- All eigenvalues are real
- Eigenvectors are orthogonal
- Can be diagonalized: A = QΛQᵀ
- Q is orthogonal matrix (QᵀQ = I)

**Positive Definite Matrix:**
- All eigenvalues > 0
- Important for optimization
- Hessian at minimum is positive definite

**Diagonal Matrix:**
- Eigenvalues = diagonal elements
- Standard basis vectors are eigenvectors

### Spectral Theorem

**Any symmetric matrix A can be written as:**
```
A = QΛQᵀ

Where:
- Λ = diagonal matrix of eigenvalues
- Q = orthogonal matrix of eigenvectors
- QᵀQ = I
```

This is **eigenvalue decomposition** or **spectral decomposition**.

### Powers of Matrices

**If A = QΛQᵀ, then:**
```
A² = (QΛQᵀ)(QΛQᵀ) = QΛ²Qᵀ
A³ = QΛ³Qᵀ
Aⁿ = QΛⁿQᵀ
```

**Computing Aⁿ:**
1. Eigendecompose once: O(n³)
2. Raise diagonal Λ to power n: O(n)
3. Reconstruct: O(n²)

Much better than n matrix multiplications!

### In Machine Learning

**Principal Component Analysis (PCA):**
```
1. Compute covariance: Σ = XᵀX/n
2. Eigendecompose: Σ = QΛQᵀ
3. Principal components = eigenvectors
4. Variance explained = eigenvalues
```

**PageRank (Google's algorithm):**
- Largest eigenvector of link matrix
- Stationary distribution of random walk

**Stability Analysis:**
- Eigenvalues of Jacobian determine stability
- |λ| > 1: unstable (exploding gradients)
- |λ| < 1: stable (vanishing gradients)

**Spectral Clustering:**
- Eigenvalues of graph Laplacian
- Community detection

**Variance in data:**
- Eigenvalues show variance in each direction
- Large eigenvalue = high variance direction

---

## 11. Matrix Decompositions

### Why Decompose Matrices?

**Benefits:**
1. **Computational efficiency**: Faster operations
2. **Numerical stability**: More accurate
3. **Insight**: Understand structure
4. **Compression**: Store less data

### LU Decomposition

**A = LU**

Where:
- L = lower triangular
- U = upper triangular

**Used for:**
- Solving linear systems
- Computing determinants
- Matrix inversion

**Complexity:** O(n³)

### QR Decomposition

**A = QR**

Where:
- Q = orthogonal (QᵀQ = I)
- R = upper triangular

**Used for:**
- Least squares problems
- Finding eigenvalues (QR algorithm)
- Orthogonalizing vectors

**Properties:**
- More stable than normal equations
- Gram-Schmidt is one method

### Cholesky Decomposition

**A = LLᵀ**

Where:
- A must be positive definite
- L is lower triangular

**Used for:**
- Solving positive definite systems
- Sampling from multivariate Gaussian
- Faster than general LU (about 2x)

**Complexity:** O(n³/3)

### Singular Value Decomposition (SVD)

**The most important decomposition in ML!**

**A = UΣVᵀ**

Where:
- U: m×m orthogonal (left singular vectors)
- Σ: m×n diagonal (singular values σᵢ ≥ 0)
- V: n×n orthogonal (right singular vectors)

**Properties:**
- Works for ANY matrix (rectangular, singular, etc.)
- Singular values are always real and non-negative
- σᵢ = √λᵢ where λᵢ are eigenvalues of AᵀA

**Relationship to Eigen:**
- AᵀA = VΣ²Vᵀ (V are eigenvectors of AᵀA)
- AAᵀ = UΣ²Uᵀ (U are eigenvectors of AAᵀ)

**Geometric Interpretation:**
1. Rotate by Vᵀ
2. Scale by Σ
3. Rotate by U

**Truncated SVD (Low-rank approximation):**
```
A ≈ Σᵢ₌₁ᵏ σᵢuᵢvᵢᵀ  (keep largest k singular values)
```

**This is the best rank-k approximation of A!**

### SVD Applications in ML

**Dimensionality Reduction:**
```python
# Keep top k singular values
U, s, Vt = np.linalg.svd(X)
X_reduced = U[:, :k] @ np.diag(s[:k])
```

**Recommender Systems:**
- Matrix completion
- Collaborative filtering
- Netflix prize!

**Image Compression:**
- Store only top k singular values/vectors
- Huge compression with little quality loss

**PCA:**
- SVD of centered data = PCA
- Faster and more stable than covariance eigendecomp

**Pseudoinverse:**
```
A⁺ = VΣ⁺Uᵀ
```
Where Σ⁺ inverts non-zero singular values.

**Numerical Rank:**
```
rank(A) = number of singular values > ε
```

**Condition Number:**
```
cond(A) = σ_max / σ_min
```
- Large condition number = ill-conditioned
- Small perturbations cause large changes

---

## 12. Norms and Distances

### Vector Norms

**Lp Norm:**
```
||x||_p = (|x₁|ᵖ + |x₂|ᵖ + ... + |xₙ|ᵖ)^(1/p)
```

**Common cases:**

**L1 (Manhattan):**
```
||x||₁ = |x₁| + |x₂| + ... + |xₙ|
```
- Sum of absolute values
- Encourages sparsity (Lasso regression)

**L2 (Euclidean):**
```
||x||₂ = √(x₁² + x₂² + ... + xₙ²)
```
- Standard distance
- Used in Ridge regression
- Smooth, differentiable

**L∞ (Maximum):**
```
||x||∞ = max(|x₁|, |x₂|, ..., |xₙ|)
```
- Largest component
- Used in minimax problems

**L0 "Norm" (not actually a norm):**
```
||x||₀ = number of non-zero elements
```
- Sparsity measure
- NP-hard to optimize
- Approximated by L1

### Properties of Norms

1. **Non-negativity:** ||x|| ≥ 0
2. **Zero:** ||x|| = 0 ⟺ x = 0
3. **Scaling:** ||αx|| = |α| × ||x||
4. **Triangle inequality:** ||x + y|| ≤ ||x|| + ||y||

### Matrix Norms

**Frobenius Norm:**
```
||A||_F = √(Σᵢⱼ |aᵢⱼ|²)
```
- Like L2 for matrices
- Sum of squared elements

**Operator Norms:**
```
||A||_p = max_{||x||_p=1} ||Ax||_p
```

**Spectral Norm (2-norm):**
```
||A||₂ = σ_max(A)  (largest singular value)
```

**Nuclear Norm:**
```
||A||_* = Σᵢ σᵢ  (sum of singular values)
```
- Convex relaxation of rank
- Used in matrix completion

### Distances

**Euclidean Distance:**
```
d(x, y) = ||x - y||₂
```

**Manhattan Distance:**
```
d(x, y) = ||x - y||₁
```

**Cosine Similarity (not a distance):**
```
sim(x, y) = (x · y) / (||x|| × ||y||)
```
- Measures angle between vectors
- Range: [-1, 1]
- 1 = same direction, -1 = opposite

**Cosine Distance:**
```
d(x, y) = 1 - sim(x, y)
```

**Mahalanobis Distance:**
```
d(x, y) = √((x-y)ᵀΣ⁻¹(x-y))
```
- Accounts for covariance
- Scale-invariant

### In Machine Learning

**Regularization:**
```
L1: minimize loss + λ||w||₁  (Lasso)
L2: minimize loss + λ||w||₂²  (Ridge)
```

**Distance metrics:**
- K-NN: Euclidean or Manhattan
- Word embeddings: Cosine similarity
- Anomaly detection: Mahalanobis

**Optimization:**
- Gradient norm for convergence: ||∇f|| < ε

**Robustness:**
- L∞ norm for adversarial robustness

---

## 13. Orthogonality and Projections

### Orthogonality

**Vectors are orthogonal if:**
```
u · v = 0
```

**Geometric meaning:**
- Perpendicular (90° angle)
- No component in each other's direction

**Orthonormal:**
- Orthogonal AND unit length
- ||u|| = ||v|| = 1
- u · v = 0

**Orthogonal Matrix Q:**
```
QᵀQ = QQᵀ = I
```

**Properties:**
- Columns are orthonormal
- Rows are orthonormal
- Preserves lengths: ||Qx|| = ||x||
- Preserves angles: (Qx) · (Qy) = x · y
- det(Q) = ±1
- Q⁻¹ = Qᵀ (cheap to invert!)

### Projections

**Project vector b onto vector a:**
```
proj_a(b) = (a · b / a · a) × a
```

**Explicitly:**
```
proj_a(b) = (aᵀb / aᵀa) × a
```

**If a is unit vector:**
```
proj_a(b) = (aᵀb) × a
```

**Geometric meaning:**
- Shadow of b on a
- Component of b in direction of a

**Project onto subspace spanned by columns of A:**
```
proj_A(b) = A(AᵀA)⁻¹Aᵀb
```

**Projection matrix:**
```
P = A(AᵀA)⁻¹Aᵀ
```

**Properties:**
- P² = P (idempotent)
- Pᵀ = P (symmetric)
- Pb gives projection of b onto column space of A

### Gram-Schmidt Process

**Orthogonalize vectors {v₁, v₂, ..., vₙ}:**

```
u₁ = v₁
u₂ = v₂ - proj_{u₁}(v₂)
u₃ = v₃ - proj_{u₁}(v₃) - proj_{u₂}(v₃)
...
```

**Then normalize:**
```
eᵢ = uᵢ / ||uᵢ||
```

**Result:** Orthonormal basis {e₁, e₂, ..., eₙ}

### In Machine Learning

**QR Decomposition:**
- Gram-Schmidt gives Q and R

**PCA:**
- Principal components are orthogonal

**Whitening:**
- Transform data to have orthogonal features

**Least Squares:**
```
Ax = b has no exact solution
Solution: x̂ = (AᵀA)⁻¹Aᵀb
Geometric: project b onto column space of A
```

**Residuals:**
```
residual = b - Ax̂  (orthogonal to column space)
```

---

## 14. Computational Considerations

### Why Care About Computation?

**Modern ML deals with:**
- Millions of parameters
- Billions of data points
- Need for real-time inference

**Bad algorithms → Training for years**
**Good algorithms → Training in hours**

### Matrix Multiplication Efficiency

**Naive algorithm: O(n³)**
```python
# Don't do this!
for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i,j] += A[i,k] * B[k,j]
```

**Use optimized libraries:**
```python
# Do this!
C = A @ B  # NumPy uses BLAS
C = torch.mm(A, B)  # PyTorch uses cuBLAS on GPU
```

**Speedup: 100-1000x with good BLAS!**

### Memory Layout

**Row-major (C, NumPy default):**
```
[1, 2, 3, 4, 5, 6]  # for 2×3 matrix
```

**Column-major (Fortran, MATLAB):**
```
[1, 4, 2, 5, 3, 6]  # for 2×3 matrix
```

**Why care?**
- Cache efficiency
- Accessing rows vs columns speed

**Best practice:**
- Access in storage order
- NumPy: iterate rows, not columns

### Avoiding Matrix Inversion

**Never compute A⁻¹ explicitly!**

**Bad:**
```python
x = np.linalg.inv(A) @ b  # O(n³) + numerical issues
```

**Good:**
```python
x = np.linalg.solve(A, b)  # More stable, same complexity
```

**Why?**
- inv(A) may not exist due to numerical errors
- Solving is more stable
- Don't need full inverse

### Sparse Matrices

**If most elements are zero:**
```python
from scipy.sparse import csr_matrix

# Dense: store n²  values
A_dense = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

# Sparse: store only non-zeros
A_sparse = csr_matrix(A_dense)
```

**Benefits:**
- Memory: O(nnz) instead of O(n²)
- Speed: Operations on non-zeros only

**Used in:**
- NLP (TF-IDF matrices)
- Graph algorithms
- Finite element methods

### GPU Acceleration

**GPUs excel at:**
- Matrix multiplication
- Element-wise operations
- Parallel operations

**CPU vs GPU:**
```
Matrix multiply (1000×1000):
CPU: ~100 ms
GPU: ~5 ms
Speedup: 20x
```

**When to use GPU:**
- Large matrices (>1000×1000)
- Batch operations
- Deep learning training

**PyTorch example:**
```python
A = torch.randn(1000, 1000).cuda()
B = torch.randn(1000, 1000).cuda()
C = A @ B  # Runs on GPU
```

### Numerical Stability

**Problems:**

1. **Catastrophic cancellation:**
```python
# Bad: subtracting nearly equal numbers
x = 1.0000001
y = 1.0000000
diff = x - y  # Lost precision!
```

2. **Overflow/Underflow:**
```python
# Exponentials can overflow
exp(1000)  # inf!

# Use log-sum-exp trick:
log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
```

3. **Ill-conditioned matrices:**
```python
# Large condition number → unstable
cond = np.linalg.cond(A)
if cond > 1e10:
    # Add regularization!
    A_reg = A + λ*I
```

### Best Practices

1. **Use library functions**
   - NumPy, SciPy, PyTorch
   - Optimized C/Fortran implementations

2. **Vectorize operations**
   - Avoid Python loops
   - Use broadcast operations

3. **Choose right data types**
   - float32 vs float64
   - Sparse vs dense

4. **Profile your code**
   - Find bottlenecks
   - Optimize what matters

5. **Numerical stability**
   - Avoid inverting matrices
   - Use stable algorithms (SVD > eigendecomp)
   - Add regularization when needed

---

## 15. ML Applications

### Linear Regression

**Model:**
```
y = Xw + b
```

**Matrix form:**
```
      ⎡   —— x₁ᵀ ——   ⎤       ⎡ y₁ ⎤
X =   ⎢   —— x₂ᵀ ——   ⎥   y = ⎢ y₂ ⎥
      ⎢       ⋮         ⎥       ⎢  ⋮ ⎥
      ⎣   —— xₙᵀ ——   ⎦       ⎣ yₙ ⎦
```

**Solution (closed form):**
```
w = (XᵀX)⁻¹Xᵀy
```

**Linear algebra:**
- XᵀX: covariance-like matrix
- (XᵀX)⁻¹Xᵀ: pseudoinverse
- Projection onto column space

### Principal Component Analysis (PCA)

**Goal:** Find directions of maximum variance

**Algorithm:**
```
1. Center data: X̃ = X - mean(X)
2. Compute covariance: Σ = X̃ᵀX̃ / n
3. Eigendecompose: Σ = QΛQᵀ
4. Principal components = columns of Q
5. Variance explained = diagonal of Λ
```

**Dimensionality reduction:**
```
X_reduced = X̃ @ Q[:, :k]  # Keep top k components
```

**Using SVD (better):**
```
X̃ = UΣVᵀ
PC = V  (right singular vectors)
```

### Neural Networks

**Single layer:**
```
z = Wx + b  # Linear transformation
a = σ(z)    # Nonlinear activation
```

**Multi-layer:**
```
a₀ = x
a₁ = σ(W₁a₀ + b₁)
a₂ = σ(W₂a₁ + b₂)
...
```

**Backpropagation = Chain rule + Matrix calculus**

**Gradient of loss w.r.t. weights:**
```
∂L/∂W = ∂L/∂z × ∂z/∂W
```

All matrix operations!

### Transformers (Attention)

**Self-Attention:**
```
Q = XW_Q  (queries)
K = XW_K  (keys)
V = XW_V  (values)

Attention = softmax(QKᵀ/√d) V
```

**Pure matrix operations!**
- QKᵀ: all pairwise dot products
- Softmax row-wise
- Multiply by values

**Multi-head attention:**
- Multiple W_Q, W_K, W_V
- Parallel matrix multiplications
- Concat and project

### Recommender Systems

**Matrix Factorization:**
```
R ≈ UV ᵀ
```
- R: user-item ratings (sparse!)
- U: user embeddings
- V: item embeddings

**SVD for collaborative filtering:**
```
R = UΣVᵀ
Keep top k singular values
Predict missing entries
```

### Eigenfaces

**Face recognition using PCA:**
```
1. Collect face images as columns of X
2. Compute mean face: μ = mean(X, axis=1)
3. Center: X̃ = X - μ
4. SVD: X̃ = UΣVᵀ
5. Eigenfaces = columns of U
6. Project faces: coeffs = Uᵀ(face - μ)
7. Compare coefficients for recognition
```

### Graph Algorithms

**PageRank:**
```
score = (A)score
```
- Largest eigenvector of link matrix
- Power iteration to find it

**Spectral Clustering:**
```
1. Build similarity matrix S
2. Compute Laplacian: L = D - S
3. Find k smallest eigenvectors
4. K-means on eigenvectors
```

---

## Summary

### Key Takeaways

1. **Linear algebra is the language of ML**
   - Every algorithm uses it
   - Understanding it deeply is essential

2. **Geometric intuition matters**
   - Matrices are transformations
   - Eigenvalues show important directions
   - SVD reveals structure

3. **Computational efficiency is critical**
   - O(n³) doesn't scale
   - Use optimized libraries
   - GPU acceleration

4. **Numerical stability is important**
   - Avoid matrix inversion
   - Use stable algorithms
   - Regularize when needed

5. **Applications everywhere**
   - Regression, PCA, neural nets
   - Transformers, recommenders
   - Graph algorithms

### Master These

**Essential Operations:**
- Matrix multiplication
- Transpose
- Dot product
- Norms

**Essential Decompositions:**
- Eigenvalue decomposition
- SVD
- QR decomposition

**Essential Concepts:**
- Linear transformations
- Orthogonality
- Projections
- Rank

### What's Next?

**Practice:**
- Implement algorithms from scratch
- Solve the problems in problems.md
- Visualize transformations

**Apply:**
- Build PCA from scratch
- Implement linear regression
- Understand neural network math

**Deep Dive:**
- Matrix calculus (for backprop)
- Optimization theory
- Numerical linear algebra

---

**You now have the foundation. Time to practice!** 🚀

**Next**: Work through `examples.py` to see these concepts in code!
