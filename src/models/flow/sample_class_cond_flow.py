import torch
import torch.nn as nn
from torch import Tensor


def sample_class_cond_flow(
    model: nn.Module,
    y: int | Tensor,
    n: int,
    latent_dim: int,
    device: str = "cuda",
    steps: int = 100,
) -> Tensor:
    """generate samples by Euler integration of class-conditional velocity field ODE.

    integrates dz/dt = v(z, t, y) from t=0 to t=1 with n random initializations,
    conditioning on class label y. model should be in eval mode before calling.

    args:
        model: nn.Module computing velocity v(z, t, y). expects forward(z, t, y)
               with z shape [n, latent_dim], t shape [n], y shape [n] (long),
               returns velocity shape [n, latent_dim].
        y: class label(s): int (broadcast to all samples) or Tensor [n] (per-sample).
        n: number of independent samples to generate.
        latent_dim: dimension of latent space.
        device: device string ('cuda' or 'cpu') for tensor allocation.
        steps: number of Euler integration steps (higher = finer discretization).

    returns:
        tensor of shape [n, latent_dim] containing sampled latents.
    """
    model.eval()

    # normalize class label y
    if isinstance(y, int):
        y_t = torch.full((n,), y, dtype=torch.long, device=device)
    else:
        assert y.shape == (n,) and y.dtype == torch.long
        y_t = y.to(device)

    # initialize from standard Gaussian
    z = torch.randn(n, latent_dim, device=device)  # [n, latent_dim]

    # uniform step size for discretization
    dt = 1.0 / steps

    with torch.no_grad():
        for i in range(steps):
            # compute time for this step: [n]
            t_tensor = torch.full((n,), i * dt, device=device, dtype=torch.float32)

            # predict velocity: [n, latent_dim]
            v = model(z, t_tensor, y_t)

            # Euler step: [n, latent_dim]
            z = z + v * dt

    return z
