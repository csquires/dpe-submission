"""
log probability evaluation via backward ODE integration with simultaneous divergence accumulation.

evaluates log p(z) = log N(z_0; 0, I) - integral_0^1 div(v(z_t, t)) dt
where z_0 is obtained by backward ODE integration from z at t=1 to t=0,
and div is the trace of the Jacobian of the velocity field w.r.t. z.
"""

import torch
import torch.nn as nn
from torch import Tensor
import math


def _compute_div(
    model: nn.Module,
    z: Tensor,
    t: Tensor,
) -> Tensor:
    """
    compute trace of jacobian of model w.r.t. z.

    inputs:
      z: [B, D] positions
      t: [B,] time values
    output:
      [B,] trace of Jacobian dv/dz for each sample

    approach: define single-sample jacobian-trace function, apply vmap over batch.
    """

    def model_single(z_single: Tensor, t_single: Tensor) -> Tensor:
        """wrap model call for single sample. z_single: [D], t_single: scalar."""
        z_batch = z_single.unsqueeze(0)  # [1, D]
        t_batch = t_single.unsqueeze(0)  # [1,]
        out = model(z_batch, t_batch)  # [1, D]
        return out.squeeze(0)  # [D]

    def jac_trace(z_single: Tensor, t_single: Tensor) -> Tensor:
        """jacobian trace for single (z, t) pair."""
        jac = torch.func.jacrev(
            lambda z_s: model_single(z_s, t_single)
        )(z_single)  # [D, D]
        return torch.trace(jac)

    # vmap over batch: [B,] output
    return torch.vmap(jac_trace)(z, t)


def _compute_div_chunked(
    model: nn.Module,
    z: Tensor,
    t: Tensor,
    chunk_size: int,
) -> Tensor:
    """
    compute divergence in chunks to prevent OOM from vmap on large batches.

    inputs:
      z: [B, D]
      t: [B,]
      chunk_size: max samples per chunk
    output:
      [B,] divergence values
    """
    B = z.shape[0]

    if B <= chunk_size:
        return _compute_div(model, z, t)

    div_list = []
    num_chunks = (B + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, B)

        z_chunk = z[start:end]  # [C, D]
        t_chunk = t[start:end]  # [C,]
        div_chunk = _compute_div(model, z_chunk, t_chunk)  # [C,]
        div_list.append(div_chunk)

    return torch.cat(div_list, dim=0)  # [B,]


def log_prob(
    model: nn.Module,
    z: Tensor,
    steps: int = 200,
    device: str = "cuda",
    chunk_size: int = 500,
) -> Tensor:
    """
    evaluate log density at query points via backward ODE integration.

    formula: log p(z) = log N(z_0; 0, I) - integral_0^1 div(v(z_t, t)) dt

    inputs:
      model: neural network with signature (z [B,D], t [B,]) -> [B,D]
      z: query points at t=1, shape [B, D]
      steps: number of backward integration steps (default 200)
      device: computation device (default "cuda")
      chunk_size: batch chunk size for divergence vmap (default 500)

    output:
      tensor [B,] of log density values
    """
    B, D = z.shape
    z_t = z.clone().detach().to(device)  # current position, [B, D]
    dt = 1.0 / steps  # uniform timestep

    div_integral = torch.zeros(B, device=device)  # accumulator, [B,]

    # backward ODE loop from t=1 to t=0
    for i in range(steps):
        t_val = 1.0 - i * dt  # current time, descending
        t_tensor = torch.full((B,), t_val, device=device)  # [B,]

        # compute velocity at current position
        with torch.no_grad():
            v = model(z_t, t_tensor)  # [B, D]

            # compute divergence and accumulate
            div = _compute_div_chunked(model, z_t, t_tensor, chunk_size)  # [B,]
            div_integral = div_integral + div * dt

            # backward Euler step
            z_t = z_t - v * dt

    # compute base log density at z_0
    log_2pi = torch.log(torch.tensor(2.0 * math.pi, device=device))
    log_p_base = -0.5 * (z_t.pow(2).sum(dim=-1) + D * log_2pi)  # [B,]

    return log_p_base - div_integral
