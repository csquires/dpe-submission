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

    Default hyperparameters (from denoiser2 __main__ grid search):
        Grid ranges tested:
            eps: [2.1e-3, 2.2e-3, 2.3e-3]
            lr: [1.3e-3, 1.4e-3, 1.5e-3]
            k: [20]
            epochs: [300]
            steps: [5000]
            type: ['2']
            antithetic: [True]

        Selected values (midpoints where applicable):
            k: 20 - Interpolant parameter controlling gamma curvature
            eps: 2.2e-3 - Boundary epsilon for time sampling
            n_epochs: 300 - Training epochs per network (b and eta)
            hidden_dim: 256 - Hidden layer dimension for MLP networks
            batch_size: 512 - Training batch size
            lr: 1.3e-3 - Learning rate for Adam optimizer
            n_t: 50 - Number of time points for batch sampling
            integration_steps: 3000 - Grid points for time integration
            integration_type: '2' - Trapezoidal integration
            antithetic: True - Use antithetic sampling for variance reduction
            log_every: 101 - Log frequency (effectively disables logging)
            verbose: False - Suppress training output
    """
    defaults = {
        'k': 20,
        'eps': 2.2e-3,
        'n_epochs': 300,
        'hidden_dim': 256,
        'batch_size': 512,
        'lr': 1.3e-3,
        'n_t': 50,
        'integration_steps': 3000,
        'integration_type': '2',
        'antithetic': True,
        'log_every': 101,
        'verbose': False,
    }
    defaults.update(kwargs)
    return SpatialVeloDenoiser(input_dim, device=device, **defaults)
