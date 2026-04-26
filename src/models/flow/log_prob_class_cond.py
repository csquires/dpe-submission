"""log probability evaluation for class-conditional flows via backward ODE integration with simultaneous divergence accumulation.

evaluates log p(z|y) = log N(z_0; 0, I) - integral_0^1 div_z( v(z_t, t, y) ) dt
where z_0 is obtained by backward ODE integration from z at t=1 to t=0,
and div is the trace of the Jacobian of the velocity field w.r.t. z (y held fixed).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _compute_div_cond(
    model: nn.Module,
    z: Tensor,
    t: Tensor,
    y_onehot: Tensor,
) -> Tensor:
    """compute trace of jacobian of class-conditional velocity field w.r.t. z.

    inputs:
      model: nn.Module exposing forward_from_onehot(z, t, y_onehot)
      z: [B, D] positions
      t: [B,] time values
      y_onehot: [B, K] float one-hot labels (precomputed; F.one_hot does not compose with vmap)
    output:
      [B,] trace of Jacobian dv/dz for each sample
    """

    def jac_trace(z_s: Tensor, t_s: Tensor, c_s: Tensor) -> Tensor:
        def model_single(z_in: Tensor) -> Tensor:
            v = model.forward_from_onehot(z_in.unsqueeze(0), t_s.unsqueeze(0), c_s.unsqueeze(0))
            return v.squeeze(0)
        jac = torch.func.jacrev(model_single)(z_s)
        return torch.trace(jac)

    return torch.vmap(jac_trace)(z, t, y_onehot)


def _compute_div_cond_chunked(
    model: nn.Module,
    z: Tensor,
    t: Tensor,
    y_onehot: Tensor,
    chunk_size: int,
) -> Tensor:
    """compute divergence in chunks to prevent OOM from vmap on large batches."""
    B = z.shape[0]
    if B <= chunk_size:
        return _compute_div_cond(model, z, t, y_onehot)

    div_list = []
    num_chunks = (B + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, B)
        div_list.append(_compute_div_cond(model, z[start:end], t[start:end], y_onehot[start:end]))
    return torch.cat(div_list, dim=0)


def log_prob_class_cond(
    model: nn.Module,
    z: Tensor,
    y: Tensor,
    steps: int = 200,
    device: str = "cuda",
    chunk_size: int = 500,
) -> Tensor:
    """evaluate log density at query points for class-conditional flow via backward ODE integration.

    formula: log p(z|y) = log N(z_0; 0, I) - integral_0^1 div_z( v(z_t, t, y) ) dt

    inputs:
      model: nn.Module with forward_from_onehot(z, t, y_onehot) and num_classes attribute.
      z: query points at t=1, shape [B, D]
      y: class labels, shape [B], dtype torch.long
      steps: number of backward integration steps (default 200)
      device: computation device (default "cuda")
      chunk_size: batch chunk size for divergence vmap (default 500)

    output:
      tensor [B,] of log density values
    """
    B, D = z.shape
    z_t = z.clone().detach().to(device)
    y_d = y.to(device)
    y_onehot = F.one_hot(y_d, num_classes=model.num_classes).to(z_t.dtype)

    dt = 1.0 / steps
    div_integral = torch.zeros(B, device=device)

    for i in range(steps):
        t_val = 1.0 - i * dt
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)

        with torch.no_grad():
            v = model.forward_from_onehot(z_t, t_tensor, y_onehot)
            div = _compute_div_cond_chunked(model, z_t, t_tensor, y_onehot, chunk_size)
            div_integral = div_integral + div * dt
            z_t = z_t - v * dt

    log_2pi = torch.log(torch.tensor(2.0 * math.pi, device=device))
    log_p_base = -0.5 * (z_t.pow(2).sum(dim=-1) + D * log_2pi)
    return log_p_base - div_integral
