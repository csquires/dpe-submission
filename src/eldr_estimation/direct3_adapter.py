"""
Adapter for DirectELDREstimator3 that accepts torch tensors.

The original DirectELDREstimator3 takes numpy arrays. This adapter wraps it
to accept torch tensors, handling the conversion automatically.
"""

import torch
import numpy as np
from typing import Optional

from src.eldr_estimation.direct3 import DirectELDREstimator3
from src.eldr_estimation.base import ELDREstimator


class Direct3Adapter(ELDREstimator):
    """
    Adapter that wraps DirectELDREstimator3 to accept torch tensors.

    This allows seamless integration with experiments that work with torch tensors
    while the underlying implementation uses numpy arrays.
    """

    def __init__(self, input_dim: int, device: str = "cuda", **kwargs):
        """
        Args:
            input_dim: Dimensionality of input samples
            device: Device to use ('cuda', 'cpu', etc.)
            **kwargs: Passed through to DirectELDREstimator3
        """
        super().__init__(input_dim)
        self.device = device
        self.estimator = DirectELDREstimator3(input_dim, device=device, **kwargs)

    def estimate_eldr(
        self,
        samples_pstar: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        mu0: Optional[torch.Tensor] = None,
        Sigma0: Optional[torch.Tensor] = None,
        mu1: Optional[torch.Tensor] = None,
        Sigma1: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Estimate the Expected Log Density Ratio E_{p_*}[log(p_0(x)/p_1(x))].

        Args:
            samples_pstar: Samples from the base distribution p_* (torch tensor)
            samples_p0: Samples from p_0 (torch tensor)
            samples_p1: Samples from p_1 (torch tensor)
            mu0: Optional mean of p_0 (for sanity check logging)
            Sigma0: Optional covariance of p_0 (for sanity check logging)
            mu1: Optional mean of p_1 (for sanity check logging)
            Sigma1: Optional covariance of p_1 (for sanity check logging)

        Returns:
            Scalar ELDR estimate
        """
        # Convert torch tensors to numpy arrays
        samples_pstar_np = samples_pstar.cpu().numpy() if isinstance(samples_pstar, torch.Tensor) else samples_pstar
        samples_p0_np = samples_p0.cpu().numpy() if isinstance(samples_p0, torch.Tensor) else samples_p0
        samples_p1_np = samples_p1.cpu().numpy() if isinstance(samples_p1, torch.Tensor) else samples_p1

        return self.estimator.estimate_eldr(
            samples_pstar_np,
            samples_p0_np,
            samples_p1_np,
            mu0=mu0,
            Sigma0=Sigma0,
            mu1=mu1,
            Sigma1=Sigma1,
        )


def make_direct3_estimator(input_dim: int, device: str = "cuda", **kwargs) -> Direct3Adapter:
    """
    Factory for Direct3Adapter with sensible defaults.

    The Direct3 estimator directly learns the expected time-derivative of log p_t
    along a stochastic interpolant path, then integrates to get ELDR.

    Args:
        input_dim: Dimensionality of input samples
        device: Device to use ('cuda', 'cpu', etc.)
        **kwargs: Override any default hyperparameters

    Returns:
        Configured Direct3Adapter instance

    Default hyperparameters (tuned via HPO in __main__ block):
        k: 16.0 - Interpolant parameter for gamma(t)
        eps_train: 0.03 - Boundary epsilon for training
        eps_eval: 0.03 - Boundary epsilon for integration
        learning_rate: 2e-6 - Learning rate for optimizer
        weight_decay: 1e-4 - Weight decay for regularization
        num_epochs: 1000 - Maximum training epochs
        batch_size: 256 - Training batch size
        hidden_dim: 256 - Hidden layer dimension
        num_layers: 3 - Number of hidden layers
        time_embed_size: 128 - Size of Fourier time embedding
        integration_steps: 100 - Grid points for integration
        patience: 100000 - Steps to wait for convergence
        convergence_threshold: 1e-8 - Convergence threshold
        verbose: False - Suppress training output
    """
    defaults = {
        'k': 16.0,
        'eps_train': 0.03,
        'eps_eval': 0.03,
        'learning_rate': 2e-6,
        'weight_decay': 1e-4,
        'num_epochs': 1000,
        'batch_size': 256,
        'hidden_dim': 256,
        'num_layers': 3,
        'time_embed_size': 128,
        'integration_steps': 100,
        'patience': 100000,
        'convergence_threshold': 1e-8,
        'verbose': False,
    }
    defaults.update(kwargs)
    return Direct3Adapter(input_dim, device=device, **defaults)
