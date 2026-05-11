"""closed-form Schroedinger-bridge path and CTSM regression target."""
import torch
from torch import Tensor


def sb_target(
    x0: Tensor,
    x1: Tensor,
    sigma: float,
    tau: Tensor,
    epsilon: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """one-leg SB path and canonical CTSM target.

    path:
        x_tau = (1 - tau) x0 + tau x1 + sigma sqrt(tau (1-tau)) epsilon

    target / weight (both detached; only x_tau carries gradient):
        var      = sigma^2 tau (1 - tau)
        std      = sigma sqrt(tau (1 - tau))
        d_var    = sigma^2 (1 - 2 tau)
        delta    = x1 - x0
        temp     = sqrt(2 ||delta||^2 + 1e-8)
        lambda_t = var / temp
        target   = (d_var (||eps||^2 - D) / 2 + std <delta, eps>) / temp

    Args:
        x0, x1: [B, D] endpoints.
        sigma: noise amplitude > 0.
        tau: [B, 1]; caller clamps to (eps, 1-eps).
        epsilon: [B, D] standard Gaussian noise.

    Returns:
        x_tau [B, D], target [B, 1], lambda_t [B, 1].
    """
    var = sigma ** 2 * tau * (1 - tau)
    std = sigma * torch.sqrt(tau * (1 - tau))
    d_var = sigma ** 2 * (1 - 2 * tau)

    delta = x1 - x0
    delta_sq = torch.sum(delta ** 2, dim=-1, keepdim=True)
    eps_sq = torch.sum(epsilon ** 2, dim=-1, keepdim=True)
    delta_dot_eps = torch.sum(delta * epsilon, dim=-1, keepdim=True)
    dim = epsilon.shape[-1]

    temp = torch.sqrt(2 * delta_sq + 1e-8)
    lam_t = (var / temp).detach()
    target = ((d_var * (eps_sq - dim) / 2 + std * delta_dot_eps) / temp).detach()
    x_tau = (1 - tau) * x0 + tau * x1 + std * epsilon

    return x_tau, target, lam_t
