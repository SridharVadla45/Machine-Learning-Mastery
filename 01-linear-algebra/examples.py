#!/usr/bin/env python3
"""
Linear Algebra for Machine Learning - Examples

Comprehensive working examples demonstrating all key linear algebra concepts for ML.

Run: uv run 01-linear-algebra/examples.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.plotting import plot_line, plot_scatter, save_figure

# Set random seed
np.random.seed(42)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def print_subsection(title):
    """Print formatted subsection header."""
    print(f"\n{'-'*70}")
    print(f"{title}")
    print(f"{'-'*70}")


# ============================================================================
# 1. VECTORS: THE FOUNDATION
# ============================================================================

def vectors_basics():
    """Demonstrate basic vector operations."""
    print_section("1. Vectors - The Foundation of ML")
    
    # Create vectors
    v = np.array([1, 2, 3])
    w = np.array([4, 5, 6])
    
    print("Vectors:")
    print(f"v = {v}")
    print(f"w = {w}")
    
    # Vector addition
    print_subsection("Vector Addition")
    print(f"v + w = {v + w}")
    print("Geometric: Parallelogram rule")
    
    # Scalar multiplication
    print_subsection("Scalar Multiplication")
    print(f"2 * v = {2 * v}")
    print(f"-v = {-v}  (flips direction)")
    
    # Dot product
    print_subsection("Dot Product (Inner Product)")
    dot = np.dot(v, w)
    print(f"v · w = {dot}")
    print(f"Method 1: np.dot(v, w) = {np.dot(v, w)}")
    print(f"Method 2: v @ w = {v @ w}")
    print(f"Method 3: sum(v * w) = {np.sum(v * w)}")
    print(f"All equivalent!")
    
    # Vector norm
    print_subsection("Vector Norms")
    print(f"||v||₂ (Euclidean length) = {np.linalg.norm(v):.4f}")
    print(f"||v||₁ (Manhattan) = {np.linalg.norm(v, 1):.4f}")
    print(f"||v||∞ (Maximum) = {np.linalg.norm(v, np.inf):.4f}")
    
    # Unit vector
    v_unit = v / np.linalg.norm(v)
    print(f"\nUnit vector: v̂ = {v_unit}")
    print(f"||v̂|| = {np.linalg.norm(v_unit):.10f}  (should be 1.0)")
    
    # Angle between vectors
    cos_theta = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    print_subsection("Angle Between Vectors")
    print(f"cos(θ) = {cos_theta:.4f}")
    print(f"θ = {theta_deg:.2f}°")


def vectors_geometric():
    """Visualize vectors geometrically."""
    print_section("2. Geometric Interpretation of Vectors")
    
    # 2D vectors
    v = np.array([3, 2])
    w = np.array([1, 3])
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot 1: Basic vectors
    ax = axes[0, 0]
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', width=0.01, label='v')
    ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='b', width=0.01, label='w')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Vectors in 2D')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Plot 2: Vector addition
    ax = axes[0, 1]
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', width=0.01, label='v')
    ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='b', width=0.01, label='w')
    ax.quiver(0, 0, (v+w)[0], (v+w)[1], angles='xy', scale_units='xy', scale=1, 
              color='g', width=0.01, label='v+w', linestyle='--')
    # Parallelogram
    ax.quiver(v[0], v[1], w[0], w[1], angles='xy', scale_units='xy', scale=1, 
              color='gray', width=0.005, alpha=0.5)
    ax.quiver(w[0], w[1], v[0], v[1], angles='xy', scale_units='xy', scale=1, 
              color='gray', width=0.005, alpha=0.5)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Vector Addition (Parallelogram Rule)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Plot 3: Scalar multiplication
    ax = axes[1, 0]
    for scalar in [0.5, 1, 1.5, 2]:
        scaled = scalar * v
        ax.quiver(0, 0, scaled[0], scaled[1], angles='xy', scale_units='xy', scale=1, 
                  width=0.008, label=f'{scalar}v', alpha=0.7)
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Scalar Multiplication')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Plot 4: Dot product and projection
    ax = axes[1, 1]
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', width=0.01, label='v')
    ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='b', width=0.01, label='w')
    # Projection of w onto v
    proj = (np.dot(w, v) / np.dot(v, v)) * v
    ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, 
              color='purple', width=0.01, label='proj_v(w)', linestyle='--')
    # Perpendicular line
    ax.plot([w[0], proj[0]], [w[1], proj[1]], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Dot Product & Projection')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('/tmp/vectors_geometric.png', dpi=150)
    print("✓ Saved visualization to /tmp/vectors_geometric.png")
    plt.close()


# ============================================================================
# 2. MATRICES: TRANSFORMATIONS
# ============================================================================

def matrix_basics():
    """Demonstrate basic matrix operations."""
    print_section("3. Matrices - Linear Transformations")
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    
    print("Matrices:")
    print(f"A (2×3):\n{A}\n")
    print(f"B (3×2):\n{B}\n")
    
    # Matrix addition
    C = np.array([[1, 1, 1],
                  [2, 2, 2]])
    print_subsection("Matrix Addition")
    print(f"A + C:\n{A + C}\n")
    
    # Scalar multiplication
    print_subsection("Scalar Multiplication")
    print(f"2 * A:\n{2 * A}\n")
    
    # Transpose
    print_subsection("Transpose")
    print(f"Aᵀ (3×2):\n{A.T}\n")
    
    # Matrix multiplication
    print_subsection("Matrix Multiplication")
    print(f"A @ B (2×2):\n{A @ B}\n")
    print("Dimensionality: (2×3) @ (3×2) = (2×2)")
    print("Inner dimensions (3) must match!")
    
    # Element-wise multiplication (Hadamard)
    print_subsection("Element-wise Multiplication (Hadamard)")
    D = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print(f"A ⊙ D:\n{A * D}\n")
    print("Uses * operator in NumPy (not matrix multiply!)")


def matrix_multiplication_deep():
    """Deep dive into matrix multiplication."""
    print_section("4. Matrix Multiplication - The Heart of ML")
    
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    print("Understanding AB element by element:")
    print(f"A:\n{A}\n")
    print(f"B:\n{B}\n")
    
    # Compute manually
    print("(AB)[0,0] = A[0,:] · B[:,0]")
    print(f"         = {A[0,:]} · {B[:,0]}")
    print(f"         = {A[0,0]}×{B[0,0]} + {A[0,1]}×{B[1,0]}")
    print(f"         = {A[0,0]*B[0,0]} + {A[0,1]*B[1,0]}")
    print(f"         = {A[0,0]*B[0,0] + A[0,1]*B[1,0]}")
    
    print(f"\n(AB)[0,1] = A[0,:] · B[:,1]")
    print(f"         = {A[0,:]} · {B[:,1]}")
    print(f"         = {A[0,0]*B[0,1] + A[0,1]*B[1,1]}")
    
    print(f"\nFull result:")
    print(f"AB:\n{A @ B}\n")
    
    # Order matters!
    print_subsection("ORDER MATTERS!")
    print(f"AB:\n{A @ B}\n")
    print(f"BA:\n{B @ A}\n")
    print("AB ≠ BA in general!")
    
    # Matrix-vector multiplication
    print_subsection("Matrix-Vector Multiplication")
    x = np.array([1, 2])
    result = A @ x
    print(f"A @ x = {result}")
    print(f"This is a linear transformation of x!")
    
    # Batch processing
    print_subsection("Batch Processing (ML Perspective)")
    # 3 samples, 2 features
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    # Weight matrix: 2 inputs, 3 outputs
    W = np.array([[0.1, 0.2, 0.3],
                  [0.4, 0.5, 0.6]])
    
    output = X @ W
    print(f"X (3 samples, 2 features):\n{X}\n")
    print(f"W (2 inputs, 3 outputs):\n{W}\n")
    print(f"Output (3 samples, 3 outputs):\n{output}\n")
    print("This is a forward pass in a neural network layer!")


def matrix_transformations():
    """Visualize matrix transformations geometrically."""
    print_section("5. Matrices as Geometric Transformations")
    
    # Create unit square
    square = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0]])
    
    # Different transformations
    transformations = {
        'Identity': np.array([[1, 0], [0, 1]]),
        'Scale (2x)': np.array([[2, 0], [0, 2]]),
        'Rotate 45°': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                 [np.sin(np.pi/4), np.cos(np.pi/4)]]),
        'Shear': np.array([[1, 0.5], [0, 1]]),
        'Reflection (x-axis)': np.array([[1, 0], [0, -1]]),
        'Projection (x-axis)': np.array([[1, 0], [0, 0]])
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (name, T) in enumerate(transformations.items()):
        ax = axes[idx]
        
        # Original square
        ax.plot(square[0], square[1], 'b-', linewidth=2, label='Original', alpha=0.5)
        ax.fill(square[0], square[1], 'blue', alpha=0.1)
        
        # Transformed square
        transformed = T @ square
        ax.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='Transformed')
        ax.fill(transformed[0], transformed[1], 'red', alpha=0.1)
        
        # Basis vectors
        e1 = np.array([[0, 1], [0, 0]])
        e2 = np.array([[0, 0], [0, 1]])
        Te1 = T @ e1
        Te2 = T @ e2
        
        ax.quiver(0, 0, Te1[0,1], Te1[1,1], angles='xy', scale_units='xy', scale=1, 
                  color='red', width=0.01, alpha=0.7)
        ax.quiver(0, 0, Te2[0,1], Te2[1,1], angles='xy', scale_units='xy', scale=1, 
                  color='green', width=0.01, alpha=0.7)
        
        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(-1.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(f'{name}\n{T}', fontsize=10)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('/tmp/matrix_transformations.png', dpi=150)
    print("✓ Saved transformation visualization to /tmp/matrix_transformations.png")
    plt.close()
    
    print("\nKey insight: Matrix columns show where basis vectors go!")
    print("For transformation T = [[a, b], [c, d]]:")
    print("  [1, 0] → [a, c]  (first column)")
    print("  [0, 1] → [b, d]  (second column)")


# ============================================================================
# 3. EIGENVALUES AND EIGENVECTORS
# ============================================================================

def eigenvalues_eigenvectors():
    """Demonstrate eigenvalues and eigenvectors."""
    print_section("6. Eigenvalues & Eigenvectors - The Soul of Data")
    
    # Symmetric matrix (guarantees real eigenvalues)
    A = np.array([[4, 2],
                  [2, 3]])
    
    print(f"Matrix A:\n{A}\n")
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("Eigenvalues:")
    for i, val in enumerate(eigenvalues):
        print(f"  λ₁ = {val:.4f}")
    
    print("\nEigenvectors:")
    for i in range(len(eigenvalues)):
        vec = eigenvectors[:, i]
        print(f"  v₁ = {vec}")
    
    # Verify: Av = λv
    print_subsection("Verification: Av = λv")
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        
        Av = A @ v
        lambda_v = lam * v
        
        print(f"\nEigenvalue {i+1}:")
        print(f"  Av = {Av}")
        print(f"  λv = {lambda_v}")
        print(f"  Equal? {np.allclose(Av, lambda_v)}")
    
    # Eigenvalue decomposition
    print_subsection("Eigenvalue Decomposition: A = QΛQᵀ")
    Q = eigenvectors
    Lambda = np.diag(eigenvalues)
    
    A_reconstructed = Q @ Lambda @ Q.T
    print(f"Original A:\n{A}\n")
    print(f"QΛQᵀ:\n{A_reconstructed}\n")
    print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.10f}")


def eigenvalue_visualization():
    """Visualize what eigenvectors mean."""
    print_section("7. Eigenvectors - Special Directions")
    
    A = np.array([[2, 1],
                  [1, 2]])
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Eigenvectors
    ax = axes[0]
    
    # Draw eigenvectors
    for i in range(2):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                  color=f'C{i}', width=0.015, label=f'v{i+1} (λ={lam:.2f})')
    
    # Draw some random vectors and their transformations
    np.random.seed(42)
    for _ in range(5):
        v = np.random.randn(2)
        v = v / np.linalg.norm(v)  # Normalize
        
        Av = A @ v
        
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                  color='gray', width=0.005, alpha=0.3)
        ax.quiver(0, 0, Av[0], Av[1], angles='xy', scale_units='xy', scale=1,
                  color='black', width=0.005, alpha=0.5)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Eigenvectors don\'t change direction under A')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot 2: Transformation along eigenvector directions
    ax = axes[1]
    
    # Create ellipse of points
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Transform
    ellipse = A @ circle
    
    ax.plot(circle[0], circle[1], 'b-', linewidth=2, label='Original circle', alpha=0.5)
    ax.plot(ellipse[0], ellipse[1], 'r-', linewidth=2, label='Transformed')
    
    # Show eigenvector directions
    for i in range(2):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        ax.quiver(0, 0, lam*v[0], lam*v[1], angles='xy', scale_units='xy', scale=1,
                  color=f'C{i}', width=0.015, label=f'λ{i+1}v{i+1}')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Transformation stretches along eigenvector directions')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('/tmp/eigenvector_visualization.png', dpi=150)
    print("✓ Saved eigenvector visualization to /tmp/eigenvector_visualization.png")
    plt.close()


# ============================================================================
# 4. SINGULAR VALUE DECOMPOSITION (SVD)
# ============================================================================

def svd_demonstration():
    """Demonstrate SVD and its power."""
    print_section("8. SVD - The Swiss Army Knife of Linear Algebra")
    
    # Create a matrix
    A = np.array([[3, 2, 2],
                  [2, 3, -2]])
    
    print(f"Matrix A (2×3):\n{A}\n")
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(A)
    
    print("SVD: A = UΣVᵀ\n")
    print(f"U (2×2):\n{U}\n")
    print(f"Singular values σ:\n{s}\n")
    print(f"Vᵀ (3×3):\n{Vt}\n")
    
    # Reconstruct
    Sigma = np.zeros((2, 3))
    Sigma[:2, :2] = np.diag(s)
    
    A_reconstructed = U @ Sigma @ Vt
    print(f"Reconstructed A:\n{A_reconstructed}\n")
    print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.10f}")
    
    # Properties
    print_subsection("Properties")
    print(f"U is orthogonal: UᵀU =\n{U.T @ U}\n")
    print(f"V is orthogonal: VᵀV =\n{Vt.T @ Vt}\n")
    
    # Low-rank approximation
    print_subsection("Low-Rank Approximation")
    
    # Keep only largest singular value
    k = 1
    A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    
    print(f"Rank-{k} approximation:\n{A_approx}\n")
    print(f"Original A:\n{A}\n")
    print(f"Approximation error: {np.linalg.norm(A - A_approx):.4f}")
    print(f"Compression: stored {k} singular values instead of {A.size} elements")


def svd_image_compression():
    """Demonstrate SVD for image compression."""
    print_section("9. SVD Application: Image Compression")
    
    # Create a simple "image" (checkerboard pattern)
    n = 100
    image = np.zeros((n, n))
    square_size = 10
    for i in range(0, n, square_size*2):
        for j in range(0, n, square_size*2):
            image[i:i+square_size, j:j+square_size] = 1
            image[i+square_size:i+square_size*2, j+square_size:j+square_size*2] = 1
    
    # Add some noise
    image += np.random.randn(n, n) * 0.1
    
    # SVD
    U, s, Vt = np.linalg.svd(image)
    
    # Try different ranks
    ranks = [1, 5, 10, 20, 50]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Original
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'Original\n{n}×{n} = {n*n} values')
    axes[0].axis('off')
    
    # Reconstructions
    for idx, k in enumerate(ranks):
        # Reconstruct
        approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        
        # Plot
        axes[idx+1].imshow(approx, cmap='gray')
        
        # Compute compression ratio
        original_size = n * n
        compressed_size = k * (n + 1 + n)  # U[:,:k] + s[:k] + Vt[:k,:]
        ratio = original_size / compressed_size
        
        # Error
        error = np.linalg.norm(image - approx) / np.linalg.norm(image)
        
        axes[idx+1].set_title(f'Rank {k}\nCompression: {ratio:.1f}x\nError: {error:.3f}')
        axes[idx+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/svd_compression.png', dpi=150)
    print("✓ Saved SVD compression visualization to /tmp/svd_compression.png")
    plt.close()
    
    print("\nSVD allows lossy compression:")
    print("- Keep only top k singular values")
    print("- Reconstruct with less data")
    print("- Trade-off between compression and quality")


# ============================================================================
# 5. PCA APPLICATION
# ============================================================================

def pca_example():
    """Demonstrate PCA using eigenvalue decomposition."""
    print_section("10. PCA - Finding Important Directions in Data")
    
    # Generate correlated 2D data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[3, 1.5],
           [1.5, 1]]
    data = np.random.multivariate_normal(mean, cov, 300)
    
    print(f"Data shape: {data.shape}")
    print(f"Mean: {data.mean(axis=0)}")
    
    # Center the data
    data_centered = data - data.mean(axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    print(f"\nCovariance matrix:\n{cov_matrix}\n")
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"Eigenvalues (variance explained):")
    for i, val in enumerate(eigenvalues):
        percent = 100 * val / eigenvalues.sum()
        print(f"  PC{i+1}: {val:.4f} ({percent:.1f}%)")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Original data with principal components
    ax = axes[0]
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=20)
    
    # Draw principal components
    for i in range(2):
        vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.3, head_length=0.3,
                 fc=f'C{i+1}', ec=f'C{i+1}', linewidth=2,
                 label=f'PC{i+1} (λ={eigenvalues[i]:.2f})')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('PCA: Principal Components')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot 2: Data in PC space (rotated)
    ax = axes[1]
    data_pca = data_centered @ eigenvectors
    ax.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5, s=20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Data in Principal Component Space')
    ax.set_xlabel('PC1 (First Principal Component)')
    ax.set_ylabel('PC2 (Second Principal Component)')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('/tmp/pca_example.png', dpi=150)
    print("\n✓ Saved PCA visualization to /tmp/pca_example.png")
    plt.close()
    
    print("\nPCA Summary:")
    print("1. Center the data")
    print("2. Compute covariance matrix")
    print("3. Find eigenvectors (principal components)")
    print("4. Eigenvalues = variance in each direction")
    print("5. Project data onto principal components")


# ============================================================================
# 6. LINEAR SYSTEMS
# ============================================================================

def linear_systems():
    """Solve systems of linear equations."""
    print_section("11. Solving Linear Systems: Ax = b")
    
    # System: 2x + 3y = 8, x - y = -1
    A = np.array([[2, 3],
                  [1, -1]])
    b = np.array([8, -1])
    
    print("System of equations:")
    print("  2x + 3y = 8")
    print("  x - y = -1")
    print()
    print(f"Matrix form: Ax = b")
    print(f"A =\n{A}\n")
    print(f"b = {b}\n")
    
    # Method 1: Matrix inversion (don't do this in practice!)
    x_inv = np.linalg.inv(A) @ b
    print("Method 1: x = A⁻¹b (DON'T USE IN PRACTICE)")
    print(f"x = {x_inv}\n")
    
    # Method 2: Solve (preferred)
    x_solve = np.linalg.solve(A, b)
    print("Method 2: np.linalg.solve (PREFERRED)")
    print(f"x = {x_solve}\n")
    
    # Verify
    print("Verification: Ax = b?")
    result = A @ x_solve
    print(f"Ax = {result}")
    print(f"b  = {b}")
    print(f"Equal? {np.allclose(result, b)}\n")
    
    # Overdetermined system (more equations than unknowns)
    print_subsection("Overdetermined System (Least Squares)")
    A_over = np.array([[1, 1],
                       [1, 2],
                       [1, 3]])
    b_over = np.array([2, 3, 5])
    
    print(f"A (3×2):\n{A_over}\n")
    print(f"b (3×1): {b_over}\n")
    print("No exact solution! Use least squares.\n")
    
    # Least squares solution
    x_ls = np.linalg.lstsq(A_over, b_over, rcond=None)[0]
    print(f"Least squares solution: x = {x_ls}")
    print(f"Ax = {A_over @ x_ls}")
    print(f"b  = {b_over}")
    print(f"Residual ||Ax - b|| = {np.linalg.norm(A_over @ x_ls - b_over):.6f}")


# ============================================================================
# 7. NORMS AND DISTANCES
# ============================================================================

def norms_distances():
    """Demonstrate different norms and their properties."""
    print_section("12. Norms & Distances - Measuring Size")
    
    v = np.array([3, -4, 12])
    
    print(f"Vector v = {v}\n")
    
    # Different norms
    print("Vector Norms:")
    print(f"  L0 (sparsity): {np.count_nonzero(v)}")
    print(f"  L1 (Manhattan): {np.linalg.norm(v, 1):.4f}")
    print(f"  L2 (Euclidean): {np.linalg.norm(v, 2):.4f}")
    print(f"  L∞ (Maximum): {np.linalg.norm(v, np.inf):.4f}")
    
    # Distance between vectors
    print_subsection("Distances Between Vectors")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    print(f"a = {a}")
    print(f"b = {b}\n")
    
    print("Distances:")
    print(f"  Euclidean: {np.linalg.norm(a - b):.4f}")
    print(f"  Manhattan: {np.linalg.norm(a - b, 1):.4f}")
    
    # Cosine similarity
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"  Cosine similarity: {cos_sim:.4f}")
    print(f"  Cosine distance: {1 - cos_sim:.4f}")
    
    # Matrix norms
    print_subsection("Matrix Norms")
    M = np.array([[1, 2],
                  [3, 4]])
    
    print(f"Matrix M:\n{M}\n")
    print(f"Frobenius norm: {np.linalg.norm(M, 'fro'):.4f}")
    print(f"Spectral norm (largest singular value): {np.linalg.norm(M, 2):.4f}")


# ============================================================================
# 8. PROJECTIONS
# ============================================================================

def projections():
    """Demonstrate vector projections."""
    print_section("13. Projections - Shadows and Least Squares")
    
    # Project b onto a
    a = np.array([3, 1])
    b = np.array([2, 3])
    
    # Projection formula: proj_a(b) = (a·b / a·a) * a
    proj = (np.dot(a, b) / np.dot(a, a)) * a
    
    print(f"Project b onto a:")
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  proj_a(b) = {proj}")
    
    # Component perpendicular to a
    perp = b - proj
    print(f"  perpendicular component = {perp}")
    
    # Verify orthogonality
    print(f"  proj · perp = {np.dot(proj, perp):.10f} (should be ~0)")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Vectors
    ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.015, label='a')
    ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1,
              color='red', width=0.015, label='b')
    ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1,
              color='green', width=0.015, label='proj_a(b)')
    
    # Perpendicular line
    ax.plot([b[0], proj[0]], [b[1], proj[1]], 'k--', linewidth=2, alpha=0.5,
            label='perpendicular')
    
    # Extend line of a
    t = np.linspace(-1, 4, 100)
    line_a = np.outer(a, t)
    ax.plot(line_a[0], line_a[1], 'b--', alpha=0.3, linewidth=1)
    
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Vector Projection')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('/tmp/projection.png', dpi=150)
    print("\n✓ Saved projection visualization to /tmp/projection.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" Linear Algebra for Machine Learning - Examples ".center(70, "="))
    print("="*70)
    
    # Section 1: Vectors
    vectors_basics()
    vectors_geometric()
    
    # Section 2: Matrices
    matrix_basics()
    matrix_multiplication_deep()
    matrix_transformations()
    
    # Section 3: Eigenvalues
    eigenvalues_eigenvectors()
    eigenvalue_visualization()
    
    # Section 4: SVD
    svd_demonstration()
    svd_image_compression()
    
    # Section 5: PCA
    pca_example()
    
    # Section 6: Linear Systems
    linear_systems()
    
    # Section 7: Norms
    norms_distances()
    
    # Section 8: Projections
    projections()
    
    # Final summary
    print("\n" + "="*70)
    print(" Examples Complete! ".center(70, "="))
    print("="*70)
    print("\nVisualizations saved to /tmp/:")
    print("  - vectors_geometric.png")
    print("  - matrix_transformations.png")
    print("  - eigenvector_visualization.png")
    print("  - svd_compression.png")
    print("  - pca_example.png")
    print("  - projection.png")
    print("\nKey Takeaways:")
    print("1. Vectors are the building blocks")
    print("2. Matrices represent transformations")
    print("3. Eigenvalues show important directions")
    print("4. SVD is the ultimate decomposition")
    print("5. PCA finds structure in data")
    print("\nNext: Complete exercises.py to practice!")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
