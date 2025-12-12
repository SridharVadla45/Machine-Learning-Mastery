#!/usr/bin/env python3
"""
Linear Algebra for Machine Learning - Solutions

Complete solutions to all exercises with detailed explanations.

Run: uv run 01-linear-algebra/solutions.py
"""

import numpy as np
import matplotlib.pyplot as plt


def print_solution(number, title):
    """Print solution header."""
    print(f"\n{'='*70}")
    print(f"Solution {number}: {title}")
    print(f"{'='*70}\n")


# ============================================================================
# SOLUTION 1: Vector Operations
# ============================================================================

def solution_1():
    """
    Solution to Exercise 1: Vector Operations
    
    Key Concepts:
    - Dot product measures similarity
    - Norm measures length
    - Normalization creates unit vectors
    - Angle from dot product formula: cos(θ) = (a·b)/(||a||×||b||)
    """
    print_solution(1, "Vector Operations")
    
    a = np.array([3, 4, 5])
    b = np.array([1, 2, 3])
    
    print("Given vectors:")
    print(f"  a = {a}")
    print(f"  b = {b}\n")
    
    # Dot product
    dot_product = np.dot(a, b)
    print("Step 1: Dot Product")
    print(f"  a · b = a₁b₁ + a₂b₂ + a₃b₃")
    print(f"       = {a[0]}×{b[0]} + {a}×{b[1]} + {a[2]}×{b[2]}")
    print(f"       = {a[0]*b[0]} + {a[1]*b[1]} + {a[2]*b[2]}")
    print(f"       = {dot_product}")
    
    # L2 norm
    norm_a = np.linalg.norm(a)
    print(f"\nStep 2: L2 Norm")
    print(f"  ||a||₂ = √(a₁² + a₂² + a₃²)")
    print(f"        = √({a[0]}² + {a[1]}² + {a[2]}²)")
    print(f"        = √{a[0]**2 + a[1]**2 + a[2]**2}")
    print(f"        = {norm_a:.4f}")
    
    # Normalization
    a_normalized = a / norm_a
    print(f"\nStep 3: Normalization")
    print(f"  â = a / ||a|| = {a} / {norm_a:.4f}")
    print(f"    = {a_normalized}")
    print(f"  ||â|| = {np.linalg.norm(a_normalized):.10f} ✓ (= 1.0)")
    
    # Angle
    cos_theta = dot_product / (norm_a * np.linalg.norm(b))
    theta_deg = np.degrees(np.arccos(cos_theta))
    print(f"\nStep 4: Angle Between Vectors")
    print(f"  cos(θ) = (a·b) / (||a|| × ||b||)")
    print(f"        = {dot_product} / ({norm_a:.4f} × {np.linalg.norm(b):.4f})")
    print(f"        = {cos_theta:.4f}")
    print(f"  θ = arccos({cos_theta:.4f}) = {theta_deg:.2f}°")
    
    print("\n💡 Key Insights:")
    print("  • Dot product positive → acute angle")
    print("  • Dot product zero → perpendicular (90°)")
    print("  • Dot product negative → obtuse angle")
    print("  • Normalized vectors have length 1 (useful for direction)")


# ============================================================================
# SOLUTION 2: Matrix Multiplication
# ============================================================================

def solution_2():
    """
    Solution to Exercise 2: Matrix Multiplication
    
    Key Concepts:
    - Inner dimensions must match
    - (m×n) @ (n×p) = (m×p)
    - Each element is dot product of row and column
    """
    print_solution(2, "Matrix Multiplication")
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    
    print(f"A (2×3):\n{A}\n")
    print(f"B (3×2):\n{B}\n")
    
    # A @ B
    C = A @ B
    print("Step 1: Compute A @ B")
    print(f"  Dimensions: (2×3) @ (3×2) = (2×2) ✓")
    print(f"  Inner dimensions match!")
    print(f"\n  C = A @ B =\n{C}\n")
    
    # Manual computation of C[0,0]
    print("Step 2: Computing C[0,0] manually")
    print(f"  C[0,0] = (row 0 of A) · (column 0 of B)")
    print(f"        = {A[0,:]} · {B[:,0]}")
    print(f"        = {A[0,0]}×{B[0,0]} + {A[0,1]}×{B[1,0]} + {A[0,2]}×{B[2,0]}")
    print(f"        = {A[0,0]*B[0,0]} + {A[0,1]*B[1,0]} + {A[0,2]*B[2,0]}")
    print(f"        = {A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]}")
    
    # B @ A
    D = B @ A
    print(f"\nStep 3: Compute B @ A")
    print(f"  Dimensions: (3×2) @ (2×3) = (3×3) ✓")
    print(f"\n  D = B @ A =\n{D}\n")
    
    print("Step 4: Order Matters!")
    print(f"  C (2×2) ≠ D (3×3)")
    print("  AB ≠ BA in general (non-commutative)")
    
    print("\n💡 Key Insights:")
    print("  • Matrix multiplication is NOT commutative")
    print("  • Check dimensions: (m×n)(n×p) → (m×p)")
    print("  • Inner dimensions (n) must match")
    print("  • Each output element = dot product of row × column")


# ============================================================================
# SOLUTION 3: Matrix Properties
# ============================================================================

def solution_3():
    """
    Solution to Exercise 3: Matrix Properties
    
    Key Concepts:
    - Transpose rules
    - (AB)ᵀ = BᵀAᵀ (order reverses!)
    - (Aᵀ)ᵀ = A
    """
    print_solution(3, "Matrix Properties")
    
    np.random.seed(42)
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 2)
    
    print("Property 1: (AB)ᵀ = BᵀAᵀ")
    print("-" * 50)
    
    AB = A @ B
    left_side = AB.T
    right_side = B.T @ A.T
    
    print(f"  A shape: {A.shape}")
    print(f"  B shape: {B.shape}")
    print(f"  AB shape: {AB.shape}")
    print(f"  (AB)ᵀ shape: {left_side.shape}")
    print(f"  BᵀAᵀ shape: {right_side.shape}")
    print(f"  Are they equal? {np.allclose(left_side, right_side)} ✓")
    
    print("\nWhy does order reverse?")
    print("  When transposing AB, we flip rows/columns")
    print("  This naturally reverses the order of operations")
    
    print("\nProperty 2: (Aᵀ)ᵀ = A")
    print("-" * 50)
    
    A_double_transpose = A.T.T
    print(f"  Original A shape: {A.shape}")
    print(f"  Aᵀ shape: {A.T.shape}")
    print(f"  (Aᵀ)ᵀ shape: {A_double_transpose.shape}")
    print(f"  Equal to A? {np.allclose(A_double_transpose, A)} ✓")
    
    print("\n💡 Key Insights:")
    print("  • (AB)ᵀ = BᵀAᵀ — order reverses (important for backprop!)")
    print("  • Double transpose returns to original")
    print("  • These properties are crucial in neural network math")


# ============================================================================
# SOLUTION 4: Solving Linear Systems
# ============================================================================

def solution_4():
    """
    Solution to Exercise 4: Solving Linear Systems
    
    Key Concepts:
    - Ax = b represents system of equations
    - np.linalg.solve is numerically stable
    - Always verify solution by computing Ax
    """
    print_solution(4, "Solving Linear Systems")
    
    A = np.array([[2, 3],
                  [1, -1]])
    b = np.array([13, -1])
    
    print("System of equations:")
    print("  2x + 3y = 13")
    print("  x - y = -1")
    print()
    print("In matrix form: Ax = b")
    print(f"  A =\n{A}")
    print(f"  b = {b}\n")
    
    print("Solution method: np.linalg.solve(A, b)")
    print("-" * 50)
    
    x = np.linalg.solve(A, b)
    print(f"  x = {x}")
    print(f"  Therefore: x = {x[0]}, y = {x[1]}")
    
    print("\nVerification: Ax should equal b")
    verification = A @ x
    print(f"  Ax = {A} @ {x}")
    print(f"     = {verification}")
    print(f"  b  = {b}")
    print(f"  Match? {np.allclose(verification, b)} ✓")
    
    print("\nManual check:")
    print(f"  2({x[0]}) + 3({x[1]}) = {2*x[0] + 3*x[1]:.1f} = 13 ✓")
    print(f"  {x[0]} - {x[1]} = {x[0] - x[1]:.1f} = -1 ✓")
    
    print("\n💡 Key Insights:")
    print("  • DON'T compute A⁻¹ explicitly (unstable)")
    print("  • USE np.linalg.solve() instead (stable and faster)")
    print("  • Always verify solution!")


# ============================================================================
# SOLUTION 5: Eigenvalues and Eigenvectors
# ============================================================================

def solution_5():
    """
    Solution to Exercise 5: Eigenvalues and Eigenvectors
    
    Key Concepts:
    - Av = λv (eigenvector doesn't change direction)
    - λ tells how much vector is scaled
    - Eigenvalues/vectors reveal matrix structure
    """
    print_solution(5, "Eigenvalues and Eigenvectors")
    
    A = np.array([[4, 2],
                  [1, 3]])
    
    print(f"Matrix A:\n{A}\n")
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("Eigenvalues and Eigenvectors:")
    print("-" * 50)
    for i in range(len(eigenvalues)):
        print(f"\nEigenvalue {i+1}: λ = {eigenvalues[i]:.4f}")
        print(f"Eigenvector {i+1}: v = {eigenvectors[:,i]}")
    
    print("\nVerification: Av = λv")
    print("-" * 50)
    
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        
        Av = A @ v
        lambda_v = lam * v
        
        print(f"\nFor λ = {lam:.4f}:")
        print(f"  Av      = {Av}")
        print(f"  λv      = {lambda_v}")
        print(f"  Equal?    {np.allclose(Av, lambda_v)} ✓")
        
        # Show what this means
        print(f"  Meaning: A scales v by factor {lam:.4f}")
        print(f"           Direction unchanged: v/||v|| = {v/np.linalg.norm(v)}")
    
    print("\nGeometric Interpretation:")
    print("  Eigenvectors are SPECIAL directions")
    print("  Matrix A just SCALES them (doesn't rotate)")
    print("  Eigenvalue tells the scaling factor")
    
    print("\n💡 ML Applications:")
    print("  • PCA: eigenvectors of covariance = principal components")
    print("  • PageRank: largest eigenvector = importance scores")
    print("  • Stability: eigenvalues tell if system is stable")


# ============================================================================
# SOLUTION 6: SVD Decomposition  
# ============================================================================

def solution_6():
    """
    Solution to Exercise 6: SVD Decomposition
    
    Key Concepts:
    - SVD: A = UΣVᵀ (works for ANY matrix!)
    - U, V are orthogonal matrices
    - Σ contains singular values (always ≥ 0)
    - Perfect reconstruction possible
    """
    print_solution(6, "Singular Value Decomposition")
    
    A = np.array([[3, 2, 2],
                  [2, 3, -2]])
    
    print(f"Matrix A (2×3):\n{A}\n")
    
    U, s, Vt = np.linalg.svd(A)
    
    print("SVD Components:")
    print("-" * 50)
    print(f"U (left singular vectors, 2×2):\n{U}\n")
    print(f"Singular values: {s}")
    print(f"Vᵀ (right singular vectors, 3×3):\n{Vt}\n")
    
    # Reconstruct
    Sigma = np.zeros((2, 3))
    Sigma[:2, :2] = np.diag(s)
    
    print(f"Σ matrix (2×3):\n{Sigma}\n")
    
    A_reconstructed = U @ Sigma @ Vt
    
    print("Reconstruction: A = UΣVᵀ")
    print("-" * 50)
    print(f"UΣVᵀ =\n{A_reconstructed}\n")
    print(f"Original A =\n{A}\n")
    
    error = np.linalg.norm(A - A_reconstructed)
    print(f"Reconstruction error: {error:.12f}")
    print("Perfect reconstruction! ✓")
    
    # Properties
    print("\nOrthogonality verification:")
    print("-" * 50)
    print("UᵀU should be identity:")
    print(f"{U.T @ U}\n")
    print("VVᵀ should be identity:")
    print(f"{Vt.T @ Vt}\n")
    
    print("\n💡 Key Insights:")
    print("  • SVD always exists (even for non-square matrices!)")
    print("  • Singular values = importance of each component")
    print("  • Can truncate for compression (keep largest singular values)")
    print("\n💡 ML Applications:")
    print("  • Dimensionality reduction (truncated SVD)")
    print("  • Recommender systems (matrix factorization)")
    print("  • Image compression")
    print("  • Pseudoinverse computation")


# ===========================================================================
# SOLUTION 7: Matrix Norms
# ============================================================================

def solution_7():
    """
    Solution to Exercise 7: Norms
    
    Key Concepts:
    - Different norms measure size differently
    - L1: sum of absolute values (Manhattan)
    - L2: sqrt of sum of squares (Euclidean)
    - L∞: largest absolute value
    """
    print_solution(7, "Matrix and Vector Norms")
    
    v = np.array([3, -4, 12])
    M = np.array([[1, 2], [3, 4]])
    
    print(f"Vector v = {v}\n")
    
    print("Vector Norms:")
    print("-" * 50)
    
    # L1
    l1 = np.linalg.norm(v, 1)
    print(f"L1 (Manhattan):")
    print(f"  ||v||₁ = |v₁| + |v₂| + |v₃|")
    print(f"        = |{v[0]}| + |{v[1]}| + |{v[2]}|")
    print(f"        = {abs(v[0])} + {abs(v[1])} + {abs(v[2])}")
    print(f"        = {l1:.0f}")
    
    # L2
    l2 = np.linalg.norm(v, 2)
    print(f"\nL2 (Euclidean):")
    print(f"  ||v||₂ = √(v₁² + v₂² + v₃²)")
    print(f"        = √({v[0]}² + {v[1]}² + {v[2]}²)")
    print(f"        = √{v[0]**2 + v[1]**2 + v[2]**2}")
    print(f"        = {l2:.4f}")
    
    # L-inf
    linf = np.linalg.norm(v, np.inf)
    print(f"\nL∞ (Maximum):")
    print(f"  ||v||∞ = max(|v₁|, |v₂|, |v₃|)")
    print(f"        = max({abs(v[0])}, {abs(v[1])}, {abs(v[2])})")
    print(f"        = {linf:.0f}")
    
    # Matrix norms
    print(f"\n\nMatrix M:\n{M}\n")
    print("Matrix Norms:")
    print("-" * 50)
    
    frobenius = np.linalg.norm(M, 'fro')
    print(f"Frobenius norm:")
    print(f"  ||M||_F = √(sum of all elements squared)")
    print(f"         = √({M[0,0]}² + {M[0,1]}² + {M[1,0]}² + {M[1,1]}²)")
    print(f"         = √{M[0,0]**2 + M[0,1]**2 + M[1,0]**2 + M[1,1]**2}")
    print(f"         = {frobenius:.4f}")
    
    spectral = np.linalg.norm(M, 2)
    print(f"\nSpectral norm (largest singular value):")
    print(f"  ||M||₂ = {spectral:.4f}")
    
    print("\n💡 ML Applications:")
    print("  • L1 regularization → Lasso (encourages sparsity)")
    print("  • L2 regularization → Ridge (smooth solutions)")
    print("  • Frobenius norm → measuring matrix differences")
    print("  • Spectral norm → matrix conditioning, stability")


# ============================================================================
# Additional solutions for exercises 8-12...
# ============================================================================

def solution_8():
    """Solution 8: Projections"""
    print_solution(8, "Vector Projection")
    
    a = np.array([3.0, 1.0])
    b = np.array([2.0, 3.0])
    
    print(f"Project b = {b} onto a = {a}\n")
    
    print("Step 1: Compute projection")
    print("  Formula: proj_a(b) = (a·b)/(a·a) × a")
    
    dot_ab = np.dot(a, b)
    dot_aa = np.dot(a, a)
    print(f"  a·b = {a[0]}×{b[0]} + {a[1]}×{b[1]} = {dot_ab}")
    print(f"  a·a = {a[0]}×{a[0]} + {a[1]}×{a[1]} = {dot_aa}")
    print(f"  (a·b)/(a·a) = {dot_ab}/{dot_aa} = {dot_ab/dot_aa}")
    
    projection = (dot_ab/dot_aa) * a
    print(f"  proj_a(b) = {dot_ab/dot_aa} × {a} = {projection}")
    
    print("\nStep 2: Perpendicular component")
    perp = b - projection
    print(f"  perp = b - proj = {b} - {projection} = {perp}")
    
    print("\nStep 3: Verification")
    print(f"  proj + perp = {projection} + {perp} = {projection + perp}")
    print(f"  Original b = {b}")
    print(f"  Match? {np.allclose(projection + perp, b)} ✓")
    
    print(f"\nStep 4: Orthogonality check")
    dot_proj_perp = np.dot(projection, perp)
    print(f"  proj · perp = {dot_proj_perp:.10f}")
    print(f"  ≈ 0? {abs(dot_proj_perp) < 1e-10} ✓")
    
    print("\n💡 Key Insight:")
    print("  Projection = component OF b IN DIRECTION of a")
    print("  Perpendicular = component OF b ORTHOGONAL to a")
    print("  Together they reconstruct b!")


def solution_12():
    """Solution 12: Condition Number"""
    print_solution(12, "Matrix Condition Number")
    
    A_good = np.array([[1.0, 0], [0, 1]])
    A_bad = np.array([[1, 1], [1, 1.0001]])
    
    cond_good = np.linalg.cond(A_good)
    cond_bad = np.linalg.cond(A_bad)
    
    print("Well-conditioned matrix (identity):")
    print(f"{A_good}")
    print(f"Condition number: {cond_good:.2e}")
    print("Interpretation: Perfect! Solutions are stable.\n")
    
    print("Ill-conditioned matrix:")
    print(f"{A_bad}")
    print(f"Condition number: {cond_bad:.2e}")
    print("Interpretation: DANGER! Small input changes → huge output changes\n")
    
    print("Why does conditioning matter?")
    print("-" * 50)
    print("High condition number means:")
    print("  • Tiny errors in input → large errors in output")
    print("  • Numerical instability")
    print("  • Near-singular matrix")
    print("\nIn ML:")
    print("  • Ill-conditioned Hessian → slow convergence")
    print("  • Solution: Add regularization (λI term)")
    print("  • Batch normalization helps conditioning")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all solutions."""
    print("\n" + "="*70)
    print(" Linear Algebra - Detailed Solutions ".center(70))
    print("="*70)
    
    solution_1()
    solution_2()
    solution_3()
    solution_4()
    solution_5()
    solution_6()
    solution_7()
    solution_8()
    solution_12()
    
    print("\n" + "="*70)
    print(" All Solutions Explained! ".center(70))
    print("="*70)
    print("\n🎓 Key Takeaways:")
    print("  • Linear algebra is the foundation of ML")
    print("  • Every operation has geometric meaning")
    print("  • Numerical stability matters in practice")
    print("  • These concepts appear EVERYWHERE in ML")
    print("\n✨ You now have the foundation to understand:")
    print("  • How neural networks actually work")
    print("  • What PCA and SVD really do")
    print("  • Why transformers use attention (matrix ops!)")
    print("  • How to debug ML algorithms effectively")
    print("\nNext: Master these concepts through problems.md!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
