import torch
import torch.nn as nn
from torch import Tensor


def sample_flow(
    model: nn.Module,
    n_samples: int,
    latent_dim: int,
    device: str = "cuda",
    steps: int = 100,
) -> Tensor:
    """generate samples by Euler integration of velocity field ODE.

    integrates dz/dt = v(z, t) from t=0 to t=1 with n_samples random
    initializations. model should be in eval mode before calling.

    Args:
        model: nn.Module computing velocity v(z, t). expects forward(z, t)
               with z shape [n_samples, latent_dim] and t shape [n_samples],
               returns velocity shape [n_samples, latent_dim].
        n_samples: number of independent samples to generate
        latent_dim: dimension of latent space
        device: device string ('cuda' or 'cpu') for tensor allocation
        steps: number of Euler integration steps (higher = finer discretization)

    Returns:
        tensor of shape [n_samples, latent_dim] containing sampled latents.
    """
    # initialize from standard Gaussian
    z = torch.randn(n_samples, latent_dim, device=device)  # [n_samples, latent_dim]

    # uniform step size for discretization
    dt = 1.0 / steps

    with torch.no_grad():
        for i in range(steps):
            # compute time for this step: [n_samples]
            t = torch.full((n_samples,), i * dt, device=device)

            # predict velocity: [n_samples, latent_dim]
            v = model(z, t)

            # Euler step: [n_samples, latent_dim]
            z = z + v * dt

    return z
