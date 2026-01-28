"""
Adapters for Direct ELDR Estimators (Direct3, Direct4, Direct5).

These adapters provide a unified interface for the direct ELDR estimation methods,
handling tensor conversions where needed and providing sensible HPO-tuned defaults.
"""

import torch
import numpy as np
from typing import Optional

from src.eldr_estimation.direct3 import DirectELDREstimator3
from src.eldr_estimation.direct4 import DirectELDREstimator4
from src.eldr_estimation.direct5 import DirectELDREstimator5
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


class Direct4Adapter(ELDREstimator):
    """
    Thin adapter for DirectELDREstimator4.

    DirectELDREstimator4 already accepts torch tensors natively, so this adapter
    simply provides a consistent interface with the other adapters.
    """

    def __init__(self, input_dim: int, device: str = "cuda", **kwargs):
        """
        Args:
            input_dim: Dimensionality of input samples
            device: Device to use ('cuda', 'cpu', etc.)
            **kwargs: Passed through to DirectELDREstimator4
        """
        super().__init__(input_dim)
        self.device = device
        self.estimator = DirectELDREstimator4(input_dim, device=device, **kwargs)

    def estimate_eldr(
        self,
        samples_pstar: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
    ) -> float:
        """
        Estimate the Expected Log Density Ratio E_{p_*}[log(p_0(x)/p_1(x))].

        Args:
            samples_pstar: Samples from the base distribution p_*
            samples_p0: Samples from p_0
            samples_p1: Samples from p_1

        Returns:
            Scalar ELDR estimate
        """
        return self.estimator.estimate_eldr(samples_pstar, samples_p0, samples_p1)


class Direct5Adapter(ELDREstimator):
    """
    Thin adapter for DirectELDREstimator5.

    DirectELDREstimator5 already accepts torch tensors natively, so this adapter
    simply provides a consistent interface with the other adapters.
    """

    def __init__(self, input_dim: int, device: str = "cuda", **kwargs):
        """
        Args:
            input_dim: Dimensionality of input samples
            device: Device to use ('cuda', 'cpu', etc.)
            **kwargs: Passed through to DirectELDREstimator5
        """
        super().__init__(input_dim)
        self.device = device
        self.estimator = DirectELDREstimator5(input_dim, device=device, **kwargs)

    def estimate_eldr(
        self,
        samples_pstar: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
    ) -> float:
        """
        Estimate the Expected Log Density Ratio E_{p_*}[log(p_0(x)/p_1(x))].

        Args:
            samples_pstar: Samples from the base distribution p_*
            samples_p0: Samples from p_0
            samples_p1: Samples from p_1

        Returns:
            Scalar ELDR estimate
        """
        return self.estimator.estimate_eldr(samples_pstar, samples_p0, samples_p1)


def make_direct3_estimator(input_dim: int, device: str = "cuda", **kwargs) -> Direct3Adapter:
    """
    Factory for Direct3Adapter with HPO-tuned defaults.

    The Direct3 estimator directly learns the expected time-derivative of log p_t
    along a stochastic interpolant path, then integrates to get ELDR.

    Args:
        input_dim: Dimensionality of input samples
        device: Device to use ('cuda', 'cpu', etc.)
        **kwargs: Override any default hyperparameters

    Returns:
        Configured Direct3Adapter instance

    Default hyperparameters (tuned via HPO):
        k: 30.0 - Interpolant parameter for gamma(t)
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
        'k': 30.0,
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


def make_direct4_estimator(input_dim: int, device: str = "cuda", **kwargs) -> Direct4Adapter:
    """
    Factory for Direct4Adapter with HPO-tuned defaults.

    Direct4 uses separate velocity (b) and denoiser (eta) networks with the
    spatial velocity-denoiser approach from spatial_velo_denoiser2.py.

    Args:
        input_dim: Dimensionality of input samples
        device: Device to use ('cuda', 'cpu', etc.)
        **kwargs: Override any default hyperparameters

    Returns:
        Configured Direct4Adapter instance

    Default hyperparameters (tuned via HPO):
        k: 16.0 - Interpolant parameter for gamma(t)
        eps_train: 0.002 - Training epsilon for t sampling bounds and grad clipping
        eps_eval: 0.005 - Integration epsilon for bounds
        lr: 0.005 - Learning rate for optimizer
        loss_weight_exp: 0 - Loss weighting exponent (disabled)
        antithetic: True - Use antithetic sampling for variance reduction
        n_epochs: 600 - Number of training epochs
        batch_size: 512 - Training batch size
        hidden_dim: 256 - Hidden layer dimension
        integration_steps: 3000 - Grid points for integration
        integration_type: '2' - Use trapezoid integration
        verbose: False - Suppress training output
    """
    defaults = {
        'k': 16.0,
        'eps_train': 0.002,
        'eps_eval': 0.005,
        'lr': 0.005,
        'loss_weight_exp': 0,
        'antithetic': True,
        'n_epochs': 600,
        'batch_size': 512,
        'hidden_dim': 256,
        'integration_steps': 3000,
        'integration_type': '2',
        'verbose': False,
    }
    defaults.update(kwargs)
    return Direct4Adapter(input_dim, device=device, **defaults)


def make_direct5_estimator(input_dim: int, device: str = "cuda", **kwargs) -> Direct5Adapter:
    """
    Factory for Direct5Adapter with HPO-tuned defaults.

    Direct5 extends Direct4 with gamma-scaled denoiser training for improved stability.
    The denoiser predicts eta_gamma = eta * gamma instead of eta directly.

    Args:
        input_dim: Dimensionality of input samples
        device: Device to use ('cuda', 'cpu', etc.)
        **kwargs: Override any default hyperparameters

    Returns:
        Configured Direct5Adapter instance

    Default hyperparameters (tuned via HPO):
        k: 20.0 - Interpolant parameter for gamma(t)
        eps_train: 0.05 - Training epsilon for clamping gamma in grad norm clipping
        eps_eval: 0.005 - Integration epsilon for bounds
        lr: 0.002 - Learning rate for optimizer
        loss_weight_exp: 2 - Loss weighting exponent (1/(gamma+eps)^2)
        antithetic: True - Use antithetic sampling for variance reduction
        n_epochs: 600 - Number of training epochs
        batch_size: 512 - Training batch size
        hidden_dim: 256 - Hidden layer dimension
        integration_steps: 3000 - Grid points for integration
        integration_type: '2' - Use trapezoid integration
        verbose: False - Suppress training output
    """
    defaults = {
        'k': 20.0,
        'eps_train': 0.05,
        'eps_eval': 0.005,
        'lr': 0.002,
        'loss_weight_exp': 2,
        'antithetic': True,
        'n_epochs': 600,
        'batch_size': 512,
        'hidden_dim': 256,
        'integration_steps': 3000,
        'integration_type': '2',
        'verbose': False,
    }
    defaults.update(kwargs)
    return Direct5Adapter(input_dim, device=device, **defaults)
