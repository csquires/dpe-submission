"""
EIG estimation plugins using Direct ELDR estimators (Direct3, Direct4, Direct5).

These plugins estimate Expected Information Gain (EIG) = I(theta; y|xi) using
the Direct methods to estimate the ELDR between the joint and product-of-marginals
distributions.
"""

import torch

from src.eig_estimation.base import EIGEstimation
from src.eldr_estimation.direct_adapters import (
    make_direct3_estimator,
    make_direct4_estimator,
    make_direct5_estimator,
    Direct3Adapter,
    Direct4Adapter,
    Direct5Adapter,
)


class EIGDirect3Plugin(EIGEstimation):
    """
    Use Direct3 to estimate EIG = I(theta; y|xi).

    EIG is the mutual information between theta and y, which equals the
    KL divergence between the joint p(theta, y) and the product of marginals
    p(theta) x p(y). This plugin uses the Direct3 method to estimate this
    ELDR directly without first fitting a density ratio estimator.
    """

    def __init__(self, input_dim: int, device: str = "cuda", **kwargs):
        """
        Args:
            input_dim: Dimensionality of (theta, y) concatenated samples
            device: Device to use ('cuda', 'cpu', etc.)
            **kwargs: Passed through to make_direct3_estimator
        """
        self.input_dim = input_dim
        self.device = device
        self.direct3 = make_direct3_estimator(input_dim, device=device, **kwargs)

    def _create_marginal_samples(
        self, samples_theta: torch.Tensor, samples_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Create samples from p(theta) x p(y) by shuffling.

        This breaks the dependence between theta and y by independently
        permuting each, giving samples from the product of marginals.

        Args:
            samples_theta: Samples of theta from joint distribution [n, dim_theta]
            samples_y: Samples of y from joint distribution [n, dim_y]

        Returns:
            Concatenated shuffled samples [n, dim_theta + dim_y]
        """
        shuffled_thetas = samples_theta[torch.randperm(samples_theta.shape[0])]
        shuffled_ys = samples_y[torch.randperm(samples_y.shape[0])]
        return torch.cat([shuffled_thetas, shuffled_ys], dim=1)

    def estimate_eig(
        self, samples_theta: torch.Tensor, samples_y: torch.Tensor
    ) -> float:
        """
        Estimate EIG = I(theta; y|xi).

        EIG = E_{p(theta,y)}[log(p(theta,y) / (p(theta)p(y)))]
            = ELDR where p_* = p_0 = joint, p_1 = product of marginals

        Args:
            samples_theta: Samples of theta from joint distribution [n, dim_theta]
            samples_y: Samples of y from joint distribution [n, dim_y]

        Returns:
            Scalar EIG estimate
        """
        # p0 samples are from joint distribution (theta, y)
        samples_p0 = torch.cat([samples_theta, samples_y], dim=1)

        # p1 samples are from product of marginals (shuffled)
        samples_p1 = self._create_marginal_samples(samples_theta, samples_y)

        # For EIG, we evaluate the expectation under the joint (p_* = p_0)
        return self.direct3.estimate_eldr(samples_p0, samples_p0, samples_p1)


class EIGDirect4Plugin(EIGEstimation):
    """
    Use Direct4 to estimate EIG = I(theta; y|xi).

    EIG is the mutual information between theta and y, which equals the
    KL divergence between the joint p(theta, y) and the product of marginals
    p(theta) x p(y). This plugin uses the Direct4 method (spatial velocity-denoiser
    approach) to estimate this ELDR directly.
    """

    def __init__(self, input_dim: int, device: str = "cuda", **kwargs):
        """
        Args:
            input_dim: Dimensionality of (theta, y) concatenated samples
            device: Device to use ('cuda', 'cpu', etc.)
            **kwargs: Passed through to make_direct4_estimator
        """
        self.input_dim = input_dim
        self.device = device
        self.direct4 = make_direct4_estimator(input_dim, device=device, **kwargs)

    def _create_marginal_samples(
        self, samples_theta: torch.Tensor, samples_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Create samples from p(theta) x p(y) by shuffling.

        Args:
            samples_theta: Samples of theta from joint distribution [n, dim_theta]
            samples_y: Samples of y from joint distribution [n, dim_y]

        Returns:
            Concatenated shuffled samples [n, dim_theta + dim_y]
        """
        shuffled_thetas = samples_theta[torch.randperm(samples_theta.shape[0])]
        shuffled_ys = samples_y[torch.randperm(samples_y.shape[0])]
        return torch.cat([shuffled_thetas, shuffled_ys], dim=1)

    def estimate_eig(
        self, samples_theta: torch.Tensor, samples_y: torch.Tensor
    ) -> float:
        """
        Estimate EIG = I(theta; y|xi).

        Args:
            samples_theta: Samples of theta from joint distribution [n, dim_theta]
            samples_y: Samples of y from joint distribution [n, dim_y]

        Returns:
            Scalar EIG estimate
        """
        samples_p0 = torch.cat([samples_theta, samples_y], dim=1)
        samples_p1 = self._create_marginal_samples(samples_theta, samples_y)
        return self.direct4.estimate_eldr(samples_p0, samples_p0, samples_p1)


class EIGDirect5Plugin(EIGEstimation):
    """
    Use Direct5 to estimate EIG = I(theta; y|xi).

    EIG is the mutual information between theta and y, which equals the
    KL divergence between the joint p(theta, y) and the product of marginals
    p(theta) x p(y). This plugin uses the Direct5 method (gamma-scaled denoiser
    approach) to estimate this ELDR directly.
    """

    def __init__(self, input_dim: int, device: str = "cuda", **kwargs):
        """
        Args:
            input_dim: Dimensionality of (theta, y) concatenated samples
            device: Device to use ('cuda', 'cpu', etc.)
            **kwargs: Passed through to make_direct5_estimator
        """
        self.input_dim = input_dim
        self.device = device
        self.direct5 = make_direct5_estimator(input_dim, device=device, **kwargs)

    def _create_marginal_samples(
        self, samples_theta: torch.Tensor, samples_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Create samples from p(theta) x p(y) by shuffling.

        Args:
            samples_theta: Samples of theta from joint distribution [n, dim_theta]
            samples_y: Samples of y from joint distribution [n, dim_y]

        Returns:
            Concatenated shuffled samples [n, dim_theta + dim_y]
        """
        shuffled_thetas = samples_theta[torch.randperm(samples_theta.shape[0])]
        shuffled_ys = samples_y[torch.randperm(samples_y.shape[0])]
        return torch.cat([shuffled_thetas, shuffled_ys], dim=1)

    def estimate_eig(
        self, samples_theta: torch.Tensor, samples_y: torch.Tensor
    ) -> float:
        """
        Estimate EIG = I(theta; y|xi).

        Args:
            samples_theta: Samples of theta from joint distribution [n, dim_theta]
            samples_y: Samples of y from joint distribution [n, dim_y]

        Returns:
            Scalar EIG estimate
        """
        samples_p0 = torch.cat([samples_theta, samples_y], dim=1)
        samples_p1 = self._create_marginal_samples(samples_theta, samples_y)
        return self.direct5.estimate_eldr(samples_p0, samples_p0, samples_p1)


def make_eig_direct3_plugin(input_dim: int, device: str = "cuda", **kwargs) -> EIGDirect3Plugin:
    """
    Factory for EIGDirect3Plugin with sensible defaults.

    Args:
        input_dim: Dimensionality of (theta, y) concatenated samples
        device: Device to use ('cuda', 'cpu', etc.)
        **kwargs: Override any default hyperparameters for Direct3

    Returns:
        Configured EIGDirect3Plugin instance
    """
    return EIGDirect3Plugin(input_dim, device=device, **kwargs)


def make_eig_direct4_plugin(input_dim: int, device: str = "cuda", **kwargs) -> EIGDirect4Plugin:
    """
    Factory for EIGDirect4Plugin with sensible defaults.

    Args:
        input_dim: Dimensionality of (theta, y) concatenated samples
        device: Device to use ('cuda', 'cpu', etc.)
        **kwargs: Override any default hyperparameters for Direct4

    Returns:
        Configured EIGDirect4Plugin instance
    """
    return EIGDirect4Plugin(input_dim, device=device, **kwargs)


def make_eig_direct5_plugin(input_dim: int, device: str = "cuda", **kwargs) -> EIGDirect5Plugin:
    """
    Factory for EIGDirect5Plugin with sensible defaults.

    Args:
        input_dim: Dimensionality of (theta, y) concatenated samples
        device: Device to use ('cuda', 'cpu', etc.)
        **kwargs: Override any default hyperparameters for Direct5

    Returns:
        Configured EIGDirect5Plugin instance
    """
    return EIGDirect5Plugin(input_dim, device=device, **kwargs)


if __name__ == "__main__":
    # Simple test
    DATA_DIM = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate fake correlated data (to have non-zero MI)
    samples_theta = torch.randn(1000, DATA_DIM).to(DEVICE)
    noise = torch.randn(1000, 1).to(DEVICE) * 0.5
    samples_y = (samples_theta[:, 0:1] + noise)  # y depends on theta[0]

    # Test Direct3 plugin
    print("Testing Direct3 plugin...")
    eig_plugin3 = EIGDirect3Plugin(input_dim=DATA_DIM + 1, device=DEVICE, verbose=True)
    est_eig3 = eig_plugin3.estimate_eig(samples_theta, samples_y)
    print(f"Direct3 Estimated EIG: {est_eig3:.4f}")

    # Test Direct4 plugin
    print("\nTesting Direct4 plugin...")
    eig_plugin4 = EIGDirect4Plugin(input_dim=DATA_DIM + 1, device=DEVICE, verbose=True)
    est_eig4 = eig_plugin4.estimate_eig(samples_theta, samples_y)
    print(f"Direct4 Estimated EIG: {est_eig4:.4f}")

    # Test Direct5 plugin
    print("\nTesting Direct5 plugin...")
    eig_plugin5 = EIGDirect5Plugin(input_dim=DATA_DIM + 1, device=DEVICE, verbose=True)
    est_eig5 = eig_plugin5.estimate_eig(samples_theta, samples_y)
    print(f"Direct5 Estimated EIG: {est_eig5:.4f}")
