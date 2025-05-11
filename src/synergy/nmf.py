# synergy/nmf.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import NMF as SklearnNMF
from typing import Tuple, Optional, Dict, Any
import warnings


class NMFSynergy:
    """GPU-accelerated Non-negative Matrix Factorization for muscle synergy extraction."""

    def __init__(self, n_components: int = 3,
                 max_iter: int = 500,
                 tol: float = 1e-4,
                 init: str = 'random',
                 random_state: Optional[int] = None,
                 use_gpu: bool = True):
        """
        Initialize NMF for muscle synergy extraction.

        Args:
            n_components: Number of synergies to extract
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            init: Initialization method ('random', 'svd')
            random_state: Random seed for reproducibility
            use_gpu: Whether to use GPU acceleration if available
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.use_gpu = self.device.type == 'cuda'

        if self.use_gpu:
            print(f"Using GPU acceleration on: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU (GPU not available or disabled)")

        # Initialize components
        self.W_ = None
        self.H_ = None
        self.reconstruction_error_ = None

    def _initialize_matrices(self, X: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize W and H matrices."""
        n_muscles, n_samples = X.shape

        if self.init == 'random':
            if self.random_state is not None:
                np.random.seed(self.random_state)
                torch.manual_seed(self.random_state)

            W = np.random.rand(n_muscles, self.n_components)
            H = np.random.rand(self.n_components, n_samples)

        elif self.init == 'svd':
            # Use SVD for initialization
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            W = np.abs(U[:, :self.n_components])
            H = np.abs(S[:self.n_components, None] * Vt[:self.n_components, :])

        else:
            raise ValueError(f"Unknown initialization method: {self.init}")

        # Convert to torch tensors and move to device
        W = torch.from_numpy(W).float().to(self.device)
        H = torch.from_numpy(H).float().to(self.device)

        # Normalize
        W = W / torch.sum(W, dim=0, keepdim=True)

        return W, H

    def _update_matrices(self, X: torch.Tensor, W: torch.Tensor, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update W and H using multiplicative update rules."""
        eps = 1e-10

        # Update H
        numerator = W.T @ X
        denominator = W.T @ W @ H + eps
        H = H * (numerator / denominator)

        # Update W
        numerator = X @ H.T
        denominator = W @ H @ H.T + eps
        W = W * (numerator / denominator)

        # Normalize W
        W = W / torch.sum(W, dim=0, keepdim=True)

        return W, H

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform NMF decomposition on EMG data.

        Args:
            X: EMG data matrix (muscles x time)

        Returns:
            W: Synergy matrix (muscles x synergies)
            H: Activation matrix (synergies x time)
        """
        # Ensure X is non-negative
        X = np.maximum(X, 0)

        if self.use_gpu:
            return self._fit_transform_gpu(X)
        else:
            return self._fit_transform_cpu(X)

    def _fit_transform_gpu(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated NMF using PyTorch."""
        # Convert to torch tensor and move to GPU
        X_torch = torch.from_numpy(X).float().to(self.device)

        # Initialize matrices
        W, H = self._initialize_matrices(X)

        # Optimization loop
        prev_error = float('inf')
        for iteration in range(self.max_iter):
            # Update matrices
            W, H = self._update_matrices(X_torch, W, H)

            # Calculate reconstruction error
            reconstruction = W @ H
            error = torch.norm(X_torch - reconstruction, 'fro').item()

            # Check convergence
            if abs(prev_error - error) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            prev_error = error

            if iteration % 50 == 0:
                print(f"Iteration {iteration}: Error = {error:.6f}")

        # Store results
        self.W_ = W.cpu().numpy()
        self.H_ = H.cpu().numpy()
        self.reconstruction_error_ = error

        return self.W_, self.H_

    def _fit_transform_cpu(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CPU-based NMF using sklearn."""
        model = SklearnNMF(
            n_components=self.n_components,
            init=self.init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )

        W = model.fit_transform(X)
        H = model.components_

        # Normalize W to match GPU version
        W = W / np.sum(W, axis=0, keepdims=True)

        self.W_ = W
        self.H_ = H
        self.reconstruction_error_ = model.reconstruction_err_

        return W, H

    def calculate_metrics(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics for the decomposition."""
        if self.W_ is None or self.H_ is None:
            raise ValueError("Model must be fitted before calculating metrics")

        # Reconstruction
        X_reconstructed = self.W_ @ self.H_

        # Metrics
        mse = np.mean((X - X_reconstructed) ** 2)
        rmse = np.sqrt(mse)

        # Variance accounted for (VAF)
        total_variance = np.var(X)
        residual_variance = np.var(X - X_reconstructed)
        vaf = 1 - (residual_variance / total_variance)

        # R-squared
        ss_tot = np.sum((X - np.mean(X)) ** 2)
        ss_res = np.sum((X - X_reconstructed) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return {
            'mse': mse,
            'rmse': rmse,
            'vaf': vaf,
            'r2': r2,
            'reconstruction_error': self.reconstruction_error_
        }