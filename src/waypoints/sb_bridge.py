import torch
from torch import Tensor


def sb_bridge_target(
    x_start: Tensor,
    x_end: Tensor,
    sigma: float,
    tau: Tensor,
    epsilon: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Single Schrödinger-Bridge leg: compute (x_tau, target, lambda_t).

    Closed-form CTSM regression path and target for score-based DRE training.

    Path parametrization:
        x_tau = (1 - tau) * x_start + tau * x_end + sigma * sqrt(tau * (1 - tau)) * epsilon

    Closed-form regression target (detached) and per-sample weight (detached):
        var = sigma^2 * tau * (1 - tau)
        std = sigma * sqrt(tau * (1 - tau))
        d_var = sigma^2 * (1 - 2 * tau)
        delta = x_end - x_start
        delta_sq = sum(delta^2, dim=-1, keepdim=True)
        eps_sq = sum(epsilon^2, dim=-1, keepdim=True)
        delta_dot_eps = sum(delta * epsilon, dim=-1, keepdim=True)
        dim = epsilon.shape[-1]
        temp = sqrt(2 * delta_sq + 1e-8)
        lambda_t = var / temp
        target = (d_var * (eps_sq - dim) / 2 + std * delta_dot_eps) / temp

    Args:
        x_start: [B, D] bootstrap-sampled endpoint from p0.
        x_end: [B, D] bootstrap-sampled endpoint from p1.
        sigma: positive float; noise amplitude.
        tau: [B, 1] time parameter. Caller responsible for clamping to [eps, 1-eps].
        epsilon: [B, D] standard Gaussian noise ~ N(0, I).

    Returns:
        x_tau: [B, D] sample on the SB path at time tau.
        target: [B, 1] closed-form regression target, DETACHED from autograd graph.
        lambda_t: [B, 1] per-sample weight factor, DETACHED from autograd graph.

    Notes:
        - Numerical stability: 1e-8 guard in temp = sqrt(2 * delta_sq + 1e-8).
        - Both target and lambda_t are DETACHED; caller assumes no gradient flow.
        - Caller invariant: x_start, x_end, epsilon must have requires_grad=False
          (typically tensors freshly indexed from sample buffers). Gradients flow
          only through model(x_tau, tau) at the call site, not through path samples.
        - Caller is responsible for tau in [eps, 1-eps] (see path_1d.py:CtsmPath1D).
        - All tensors returned on same device as input tensors.

    Source: ctsm.py:93-137 (_epsilon_target method), ctsm.py:171 (x_t computation).
    """
    # compute variance, std, d_var [B, 1]
    var = sigma**2 * tau * (1 - tau)
    std = sigma * torch.sqrt(tau * (1 - tau))
    d_var = sigma**2 * (1 - 2 * tau)

    # compute delta and dot products [B, D] and [B, 1]
    delta = x_end - x_start
    delta_sq = torch.sum(delta**2, dim=-1, keepdim=True)
    eps_sq = torch.sum(epsilon**2, dim=-1, keepdim=True)
    delta_dot_eps = torch.sum(delta * epsilon, dim=-1, keepdim=True)

    # extract dimension
    dim = epsilon.shape[-1]

    # numerically stable denominator [B, 1]
    temp = torch.sqrt(2 * delta_sq + 1e-8)

    # compute lambda_t and target (both detached) [B, 1]
    lambda_t = (var / temp).detach()
    target = ((d_var * (eps_sq - dim) / 2 + std * delta_dot_eps) / temp).detach()

    # compute x_tau (NOT detached; needed for forward pass) [B, D]
    x_tau = (1 - tau) * x_start + tau * x_end + std * epsilon

    return x_tau, target, lambda_t
