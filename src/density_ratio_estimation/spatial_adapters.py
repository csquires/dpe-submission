"""
Factory functions for spatial DRE methods with sensible defaults.

These adapters allow integrating spatial methods into experiments without
modifying the original implementation files.
"""

from src.density_ratio_estimation.spatial_velo_denoiser2 import SpatialVeloDenoiser


def make_spatial_velo_denoiser(input_dim: int, device: str = "cuda", **kwargs) -> SpatialVeloDenoiser:
    """
    Factory for SpatialVeloDenoiser with sensible defaults.

    The SpatialVeloDenoiser uses stochastic interpolants with a denoiser-based
    approach to estimate density ratios. It trains velocity (b) and denoiser (eta)
    networks sequentially, then integrates the time score.

    Args:
        input_dim: Dimensionality of input samples
        device: Device to use ('cuda', 'cpu', etc.)
        **kwargs: Override any default hyperparameters

    Returns:
        Configured SpatialVeloDenoiser instance

    Default hyperparameters (tuned for typical use cases):
        k: 24 - Interpolant parameter controlling gamma curvature
        eps: 9e-4 - Boundary epsilon for time sampling
        n_epochs: 300 - Training epochs per network (b and eta)
        batch_size: 512 - Training batch size
        lr: 9e-3 - Learning rate for Adam optimizer
        integration_steps: 5000 - Grid points for time integration
        integration_type: '2' - Trapezoidal integration
        antithetic: True - Use antithetic sampling for variance reduction
        verbose: False - Suppress training output
    """
    defaults = {
        'k': 24,
        'eps': 9e-4,
        'n_epochs': 300,
        'batch_size': 512,
        'lr': 9e-3,
        'integration_steps': 5000,
        'integration_type': '2',
        'antithetic': True,
        'verbose': False,
    }
    defaults.update(kwargs)
    return SpatialVeloDenoiser(input_dim, device=device, **defaults)
