#!/usr/bin/env python3
"""
Linear Algebra for Machine Learning - Exercises

Complete the TODO sections to practice linear algebra concepts.
Each exercise builds your understanding step-by-step.

Run: uv run 01-linear-algebra/exercises.py
"""

import numpy as np
import matplotlib.pyplot as plt


def print_exercise(number, title):
    """Print exercise header."""
    print(f"\n{'='*70}")
    print(f"Exercise {number}: {title}")
    print(f"{'='*70}\n")


# ============================================================================
# EXERCISE 1: Vector Operations
# ============================================================================

def exercise_1():
    """
    Practice basic vector operations.
    
    TODO: Complete the missing operations
    """
    print_exercise(1, "Vector Operations")
    
    a = np.array([3, 4, 5])
    b = np.array([1, 2, 3])
    
    print(f"a = {a}")
    print(f"b = {b}\n")
    
    # TODO: Compute dot product
    dot_product = np.dot(a, b)  # YOUR CODE: Use np.dot()
    print(f"a · b = {dot_product}")
    
    # TODO: Compute L2 norm of a
    norm_a = np.linalg.norm(a)  # YOUR CODE: Use np.linalg.norm()
    print(f"||a||₂ = {norm_a:.4f}")
    
    # TODO: Normalize a to unit vector
    a_normalized = a / np.linalg.norm(a)  # YOUR CODE: Divide by norm
    print(f"Normalized a = {a_normalized}")
    print(f"||normalized a|| = {np.linalg.norm(a_normalized):.10f}")
    
    # TODO: Compute angle between a and b in degrees
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)  # YOUR CODE: Convert to degrees
    print(f"Angle between a and b = {theta_deg:.2f}°")
    
    # Test
    assert abs(np.linalg.norm(a_normalized) - 1.0) < 1e-10, "Normalized vector should have length 1"
    print("\n✓ Exercise 1 complete!")


# ============================================================================
# EXERCISE 2: Matrix Multiplication
# ============================================================================

def exercise_2():
    """
    Practice matrix multiplication with different dimensions.
    
    TODO: Perform the matrix multiplications
    """
    print_exercise(2, "Matrix Multiplication")
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    
    print(f"A (2×3):\n{A}\n")
    print(f"B (3×2):\n{B}\n")
    
    # TODO: Multiply A @ B
    C = A @ B  # YOUR CODE: Matrix multiplication
    print(f"C = A @ B (2×2):\n{C}\n")
    
    # TODO: Check if B @ A is possible and compute if yes
    try:
        D = B @ A  # YOUR CODE: Try this multiplication
        print(f"D = B @ A (3×3):\n{D}\n")
    except:
        print("B @ A not possible (dimension mismatch)\n")
    
    # TODO: Compute element [0,0] of C manually
    c_00_manual = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]  # YOUR CODE
    print(f"C[0,0] computed manually = {c_00_manual}")
    print(f"C[0,0] from matrix multiply = {C[0,0]}")
    
    # Test
    assert C.shape == (2, 2), "C should be 2×2"
    assert c_00_manual == C[0,0], "Manual computation should match"
    print("\n✓ Exercise 2 complete!")


# ============================================================================
# EXERCISE 3: Matrix Properties
# ============================================================================

def exercise_3():
    """
    Verify matrix properties.
    
    TODO: Verify the transpose and multiplication properties
    """
    print_exercise(3, "Matrix Properties")
    
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 2)
    
    # TODO: Verify (AB)^T = B^T A^T
    AB = A @ B
    AB_transpose = AB.T  # YOUR CODE
    
    BT_AT = B.T @ A.T  # YOUR CODE: Compute in reverse order
    
    print("Verifying (AB)^T = B^T A^T:")
    print(f"Are they equal? {np.allclose(AB_transpose, BT_AT)}")
    
    # TODO: Verify (A^T)^T = A
    A_double_transpose = A.T.T  # YOUR CODE
    print(f"\nVerifying (A^T)^T = A:")
    print(f"Are they equal? {np.allclose(A_double_transpose, A)}")
    
    # Test
    assert np.allclose(AB_transpose, BT_AT), "Transpose property should hold"
    assert np.allclose(A_double_transpose, A), "Double transpose should equal original"
    print("\n✓ Exercise 3 complete!")


# ============================================================================
# EXERCISE 4: Solving Linear Systems
# ============================================================================

def exercise_4():
    """
    Solve a system of linear equations.
    
    System: 2x + 3y = 13
            x - y = -1
    
    TODO: Solve using np.linalg.solve
    """
    print_exercise(4, "Solving Linear Systems")
    
    # TODO: Set up the system Ax = b
    A = np.array([[2, 3],
                  [1, -1]])  # YOUR CODE: Coefficient matrix
    b = np.array([13, -1])  # YOUR CODE: Right-hand side
    
    print("System:")
    print("  2x + 3y = 13")
    print("  x - y = -1\n")
    
    # TODO: Solve for x using np.linalg.solve
    x = np.linalg.solve(A, b)  # YOUR CODE
    
    print(f"Solution: x = {x[0]}, y = {x[1]}")
    
    # TODO: Verify by computing Ax and comparing to b
    verification = A @ x  # YOUR CODE
    print(f"\nVerification: Ax = {verification}")
    print(f"Should equal b = {b}")
    print(f"Match? {np.allclose(verification, b)}")
    
    # Test
    assert np.allclose(A @ x, b), "Solution should satisfy Ax = b"
    print("\n✓ Exercise 4 complete!")


# ============================================================================
# EXERCISE 5: Eigenvalues and Eigenvectors
# ============================================================================

def exercise_5():
    """
    Compute and understand eigenvalues/eigenvectors.
    
    TODO: Find eigenvalues and verify the eigenvalue equation
    """
    print_exercise(5, "Eigenvalues and Eigenvectors")
    
    A = np.array([[4, 2],
                  [1, 3]])
    
    print(f"Matrix A:\n{A}\n")
    
    # TODO: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)  # YOUR CODE
    
    print("Eigenvalues:", eigenvalues)
    print(f"Eigenvectors:\n{eigenvectors}\n")
    
    # TODO: Verify Av = λv for first eigenvalue/eigenvector
    v1 = eigenvectors[:, 0]  # YOUR CODE: First eigenvector
    lambda1 = eigenvalues[0]  # YOUR CODE: First eigenvalue
    
    Av1 = A @ v1  # YOUR CODE: Multiply A by v1
    lambda_v1 = lambda1 * v1  # YOUR CODE: Multiply λ by v1
    
    print(f"Verification for eigenvalue {lambda1:.4f}:")
    print(f"  Av₁ = {Av1}")
    print(f"  λv₁ = {lambda_v1}")
    print(f"  Equal? {np.allclose(Av1, lambda_v1)}")
    
    # Test
    assert np.allclose(Av1, lambda_v1), "Eigenvalue equation should hold"
    print("\n✓ Exercise 5 complete!")


# ============================================================================
# EXERCISE 6: SVD Decomposition
# ============================================================================

def exercise_6():
    """
    Perform Singular Value Decomposition.
    
    TODO: Decompose a matrix and reconstruct it
    """
    print_exercise(6, "Singular Value Decomposition (SVD)")
    
    A = np.array([[3, 2, 2],
                  [2, 3, -2]])
    
    print(f"Matrix A (2×3):\n{A}\n")
    
    # TODO: Compute SVD
    U, s, Vt = np.linalg.svd(A)  # YOUR CODE
    
    print(f"U (2×2):\n{U}\n")
    print(f"Singular values: {s}\n")
    print(f"V^T (3×3):\n{Vt}\n")
    
    # TODO: Reconstruct A from U, s, V^T
    Sigma = np.zeros((2, 3))
    Sigma[:2, :2] = np.diag(s)  # YOUR CODE: Create Sigma matrix
    
    A_reconstructed = U @ Sigma @ Vt  # YOUR CODE: Reconstruct
    
    print(f"Reconstructed A:\n{A_reconstructed}\n")
    
    # TODO: Compute reconstruction error
    error = np.linalg.norm(A - A_reconstructed)  # YOUR CODE
    print(f"Reconstruction error: {error:.10f}")
    
    # Test
    assert error < 1e-10, "Reconstruction should be nearly perfect"
    print("\n✓ Exercise 6 complete!")


# ============================================================================
# EXERCISE 7: Matrix Norms
# ============================================================================

def exercise_7():
    """
    Compute different types of norms.
    
    TODO: Calculate various norms for vectors and matrices
    """
    print_exercise(7, "Matrix and Vector Norms")
    
    v = np.array([3, -4, 12])
    M = np.array([[1, 2], [3, 4]])
    
    print(f"Vector v = {v}")
    print(f"Matrix M:\n{M}\n")
    
    # TODO: Compute vector norms
    l1_norm = np.linalg.norm(v, 1)  # YOUR CODE: L1 norm
    l2_norm = np.linalg.norm(v, 2)  # YOUR CODE: L2 norm
    linf_norm = np.linalg.norm(v, np.inf)  # YOUR CODE: L-infinity norm
    
    print(f"Vector norms:")
    print(f"  L1 (Manhattan): {l1_norm}")
    print(f"  L2 (Euclidean): {l2_norm:.4f}")
    print(f"  L∞ (Maximum): {linf_norm}")
    
    # TODO: Compute matrix norms
    frobenius = np.linalg.norm(M, 'fro')  # YOUR CODE: Frobenius norm
    spectral = np.linalg.norm(M, 2)  # YOUR CODE: Spectral norm
    
    print(f"\nMatrix norms:")
    print(f"  Frobenius: {frobenius:.4f}")
    print(f"  Spectral (largest singular value): {spectral:.4f}")
    
    # Test
    assert abs(l1_norm - 19) < 1e-10, "L1 norm should be 19"
    assert abs(linf_norm - 12) < 1e-10, "L∞ norm should be 12"
    print("\n✓ Exercise 7 complete!")


# ============================================================================
# EXERCISE 8: Projections
# ============================================================================

def exercise_8():
    """
    Project one vector onto another.
    
    TODO: Compute projection using the projection formula
    """
    print_exercise(8, "Vector Projection")
    
    a = np.array([3.0, 1.0])
    b = np.array([2.0, 3.0])
    
    print(f"a = {a}")
    print(f"b = {b}\n")
    
    # TODO: Compute projection of b onto a
    # Formula: proj_a(b) = (a·b / a·a) * a
    dot_ab = np.dot(a, b)  # YOUR CODE
    dot_aa = np.dot(a, a)  # YOUR CODE
    projection = (dot_ab / dot_aa) * a  # YOUR CODE
    
    print(f"Projection of b onto a: {projection}")
    
    # TODO: Compute perpendicular component
    perpendicular = b - projection  # YOUR CODE
    print(f"Perpendicular component: {perpendicular}")
    
    # TODO: Verify: projection + perpendicular = b
    reconstructed = projection + perpendicular  # YOUR CODE
    print(f"\nVerification:")
    print(f"  projection + perpendicular = {reconstructed}")
    print(f"  original b = {b}")
    print(f"  Equal? {np.allclose(reconstructed, b)}")
    
    # TODO: Verify orthogonality: projection · perpendicular = 0
    dot_proj_perp = np.dot(projection, perpendicular)  # YOUR CODE
    print(f"  projection · perpendicular = {dot_proj_perp:.10f} (should be ~0)")
    
    # Test
    assert np.allclose(reconstructed, b), "Should reconstruct b"
    assert abs(dot_proj_perp) < 1e-10, "Should be orthogonal"
    print("\n✓ Exercise 8 complete!")


# ============================================================================
# EXERCISE 9: Matrix Inverse
# ============================================================================

def exercise_9():
    """
    Compute matrix inverse and verify properties.
    
    TODO: Compute inverse and verify A @ A^(-1) = I
    """
    print_exercise(9, "Matrix Inverse")
    
    A = np.array([[4, 7],
                  [2, 6]])
    
    print(f"Matrix A:\n{A}\n")
    
    # TODO: Compute determinant
    det_A = np.linalg.det(A)  # YOUR CODE
    print(f"det(A) = {det_A:.4f}")
    
    if abs(det_A) > 1e-10:
        # TODO: Compute inverse
        A_inv = np.linalg.inv(A)  # YOUR CODE
        print(f"\nA^(-1):\n{A_inv}\n")
        
        # TODO: Verify A @ A^(-1) = I
        product1 = A @ A_inv  # YOUR CODE
        print(f"A @ A^(-1):\n{product1}\n")
        
        # TODO: Verify A^(-1) @ A = I
        product2 = A_inv @ A  # YOUR CODE
        print(f"A^(-1) @ A:\n{product2}\n")
        
        identity = np.eye(2)
        print(f"Both should equal identity:\n{identity}\n")
        
        # Test
        assert np.allclose(product1, identity), "A @ A^(-1) should be I"
        assert np.allclose(product2, identity), "A^(-1) @ A should be I"
        print("✓ Exercise 9 complete!")
    else:
        print("Matrix is singular (not invertible)")


# ============================================================================
# EXERCISE 10: Gram-Schmidt Orthogonalization
# ============================================================================

def exercise_10():
    """
    Orthogonalize vectors using Gram-Schmidt process.
    
    TODO: Implement Gram-Schmidt algorithm
    """
    print_exercise(10, "Gram-Schmidt Orthogonalization")
    
    # Three linearly independent vectors
    v1 = np.array([1.0, 1.0, 0.0])
    v2 = np.array([1.0, 0.0, 1.0])
    v3 = np.array([0.0, 1.0, 1.0])
    
    print(f"Original vectors:")
    print(f"  v1 = {v1}")
    print(f"  v2 = {v2}")
    print(f"  v3 = {v3}\n")
    
    # TODO: Apply Gram-Schmidt
    # u1 = v1
    u1 = v1.copy()  # YOUR CODE
    
    # u2 = v2 - proj_u1(v2)
    proj_u1_v2 = (np.dot(v2, u1) / np.dot(u1, u1)) * u1  # YOUR CODE
    u2 = v2 - proj_u1_v2  # YOUR CODE
    
    # u3 = v3 - proj_u1(v3) - proj_u2(v3)
    proj_u1_v3 = (np.dot(v3, u1) / np.dot(u1, u1)) * u1  # YOUR CODE
    proj_u2_v3 = (np.dot(v3, u2) / np.dot(u2, u2)) * u2  # YOUR CODE
    u3 = v3 - proj_u1_v3 - proj_u2_v3  # YOUR CODE
    
    print(f"Orthogonal vectors:")
    print(f"  u1 = {u1}")
    print(f"  u2 = {u2}")
    print(f"  u3 = {u3}\n")
    
    # TODO: Normalize to get orthonormal vectors
    e1 = u1 / np.linalg.norm(u1)  # YOUR CODE
    e2 = u2 / np.linalg.norm(u2)  # YOUR CODE
    e3 = u3 / np.linalg.norm(u3)  # YOUR CODE
    
    print(f"Orthonormal vectors:")
    print(f"  e1 = {e1}")
    print(f"  e2 = {e2}")
    print(f"  e3 = {e3}\n")
    
    # TODO: Verify orthogonality
    print("Verification (dot products should be ~0):")
    print(f"  e1 · e2 = {np.dot(e1, e2):.10f}")
    print(f"  e1 · e3 = {np.dot(e1, e3):.10f}")
    print(f"  e2 · e3 = {np.dot(e2, e3):.10f}")
    
    # TODO: Verify normalization
    print("\nVerification (norms should be 1):")
    print(f"  ||e1|| = {np.linalg.norm(e1):.10f}")
    print(f"  ||e2|| = {np.linalg.norm(e2):.10f}")
    print(f"  ||e3|| = {np.linalg.norm(e3):.10f}")
    
    # Test
    assert abs(np.dot(e1, e2)) < 1e-10, "Should be orthogonal"
    assert abs(np.linalg.norm(e1) - 1) < 1e-10, "Should be unit length"
    print("\n✓ Exercise 10 complete!")


# ============================================================================
# EXERCISE 11: PCA Basics
# ============================================================================

def exercise_11():
    """
    Perform basic PCA on 2D data.
    
    TODO: Compute principal components using eigenvalue decomposition
    """
    print_exercise(11, "Principal Component Analysis (PCA)")
    
    # Generate correlated 2D data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[2, 1.5], [1.5, 1]]
    data = np.random.multivariate_normal(mean, cov, 100)
    
    print(f"Data shape: {data.shape}")
    print(f"Data mean: {data.mean(axis=0)}\n")
    
    # TODO: Center the data
    data_centered = data - data.mean(axis=0)  # YOUR CODE
    
    # TODO: Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)  # YOUR CODE
    print(f"Covariance matrix:\n{cov_matrix}\n")
    
    # TODO: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  # YOUR CODE
    
    # TODO: Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]  # YOUR CODE
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("Principal Components:")
    print(f"  PC1 (λ={eigenvalues[0]:.4f}): {eigenvectors[:, 0]}")
    print(f"  PC2 (λ={eigenvalues[1]:.4f}): {eigenvectors[:, 1]}")
    
    # TODO: Compute variance explained
    total_var = eigenvalues.sum()  # YOUR CODE
    var_explained = eigenvalues / total_var * 100  # YOUR CODE
    
    print(f"\nVariance explained:")
    print(f"  PC1: {var_explained[0]:.2f}%")
    print(f"  PC2: {var_explained[1]:.2f}%")
    
    # Test
    assert var_explained.sum() - 100 < 1e-10, "Should sum to 100%"
    print("\n✓ Exercise 11 complete!")


# ============================================================================
# EXERCISE 12: Condition Number
# ============================================================================

def exercise_12():
    """
    Understand matrix conditioning.
    
    TODO: Compute condition numbers and see effect on solutions
    """
    print_exercise(12, "Matrix Condition Number")
    
    # Well-conditioned matrix
    A_good = np.array([[1, 0], [0, 1]])
    
    # Ill-conditioned matrix
    A_bad = np.array([[1, 1], [1, 1.0001]])
    
    # TODO: Compute condition numbers
    cond_good = np.linalg.cond(A_good)  # YOUR CODE
    cond_bad = np.linalg.cond(A_bad)  # YOUR CODE
    
    print(f"Well-conditioned matrix:\n{A_good}")
    print(f"Condition number: {cond_good:.2e}\n")
    
    print(f"Ill-conditioned matrix:\n{A_bad}")
    print(f"Condition number: {cond_bad:.2e}\n")
    
    # TODO: Solve Ax = b for each
    b = np.array([1, 2])
    
    x_good = np.linalg.solve(A_good, b)  # YOUR CODE
    print(f"Solution for well-conditioned: {x_good}")
    
    try:
        x_bad = np.linalg.solve(A_bad, b)  # YOUR CODE
        print(f"Solution for ill-conditioned: {x_bad}")
        
        # Add tiny noise to b
        b_noisy = b + 0.0001 * np.random.randn(2)
        x_bad_noisy = np.linalg.solve(A_bad, b_noisy)
        
        change = np.linalg.norm(x_bad - x_bad_noisy) / np.linalg.norm(x_bad)
        print(f"Relative change from tiny noise: {change:.4f}")
        print("(Large change indicates ill-conditioning)")
    except:
        print("Matrix too ill-conditioned to solve reliably!")
    
    print("\n✓ Exercise 12 complete!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all exercises."""
    print("\n" + "="*70)
    print(" Linear Algebra - Exercises ".center(70))
    print("="*70)
    
    print("\nComplete the TODO sections in each exercise.")
    print("Solutions with full code are in solutions.py\n")
    
    try:
        exercise_1()
        exercise_2()
        exercise_3()
        exercise_4()
        exercise_5()
        exercise_6()
        exercise_7()
        exercise_8()
        exercise_9()
        exercise_10()
        exercise_11()
        exercise_12()
        
        print("\n" + "="*70)
        print(" All Exercises Complete! ".center(70))
        print("="*70)
        print("\n🎉 Excellent work! You've practiced:")
        print("  • Vector operations and norms")
        print("  • Matrix multiplication")
        print("  • Solving linear systems")
        print("  • Eigenvalues and eigenvectors")
        print("  • SVD decomposition")
        print("  • Projections and orthogonalization")
        print("  • PCA basics")
        print("\nNext: Solve problems in problems.md for deeper understanding!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error in exercise: {e}")
        print("Check your TODO implementations!")


if __name__ == "__main__":
    main()
