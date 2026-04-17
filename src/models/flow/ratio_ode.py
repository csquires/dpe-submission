"""
integrate the ratio ODE backward from t=1 to t=0 to compute log density ratios.

for two conditional densities with velocity fields u_t(x|c0), u_t(x|c1)
and score fields s_t(x|c0), s_t(x|c1), the log density ratio r_t evolves as:

  d/dt log r_t = div[u_t(x|c1) - u_t(x|c0)]
               + [u_t(x|c1) - u_t(x|c0)]^T s_t(x|c1)

the spatial state evolves as:

  dx_t/dt = u_t(x_t|c0)

integration is performed backward in time via fixed-step Euler integration.
the spatial state x_t evolves under the simulation velocity u_t(x|c0);
the log ratio accumulates the divergence and correction terms.

the model should support the s1 setting where b_t = u_t(x|c0),
eliminating the numerator correction term.
"""

import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
import warnings


def ratio_ode(
    model: nn.Module,
    xs: Tensor,
    steps: int = 10000,
    eps: float = 0.01,
    device: str | None = None,
    div_method: str = "exact",
) -> Tensor:
    """
    integrate the ratio ODE backward from t=1 to t=0 to compute log density ratios.

    computes log(p(x|c0) / p(x|c1)) by backward integration of the coupled
    state-space ODE. spatial state evolves under u_t(x|c0); log ratio
    accumulates divergence and correction terms from velocity difference.

    integration boundary condition: log r_0 = 0 (shared prior).
    goal: compute log r_1 = integral_0^1 (div + correction) dt.

    args:
        model: nn.Module with forward(t, x, c) -> (velocity, score) tuple.
               t: [B, 1] time values
               x: [B, D] positions
               c: [B, 1] condition tensor (0.0 for numerator, 1.0 for denominator)
               returns: (velocity [B, D], score [B, D])

        xs: [B, D] query points at t=1 (reference time)

        steps: number of integration steps from t=1-eps to t=eps (default 10000)

        eps: small time offset from boundaries to avoid singularities (default 0.01)

        device: optional device override. if none, infers from xs (default none)

        div_method: divergence estimation method (default "exact")
                    options: "exact" (backprop, slow but accurate)
                             "hutch_gaussian" (hutchinson with gaussian noise)
                             "hutch_rademacher" (hutchinson with rademacher noise)

    returns:
        [B] tensor of log density ratios log(p(x|c0) / p(x|c1)) at query points.
    """
    from src.models.flow.div_estimators import exact_div, hutch_div

    if device is None:
        device = xs.device.type if hasattr(xs.device, 'type') else str(xs.device)

    B, D = xs.shape
    x_t = xs.clone().to(device)
    log_r = torch.zeros(B, device=device, dtype=xs.dtype)

    t_vals = torch.linspace(1.0 - eps, eps, steps, device=device, dtype=xs.dtype)
    dt = t_vals[1] - t_vals[0]

    model.eval()

    # select divergence function once before loop
    if div_method == "exact":
        div_fn = exact_div
    elif div_method == "hutch_gaussian":
        div_fn = partial(hutch_div, noise="gaussian")
    elif div_method == "hutch_rademacher":
        div_fn = partial(hutch_div, noise="rademacher")
    else:
        raise ValueError(
            f"unknown div_method: {div_method}. "
            f"must be 'exact', 'hutch_gaussian', or 'hutch_rademacher'."
        )

    with torch.no_grad():
        for i in range(steps - 1):
            t_i = t_vals[i].item()

            t_batch = torch.full((B, 1), t_i, device=device, dtype=xs.dtype)
            c0 = torch.zeros(B, 1, device=device, dtype=xs.dtype)
            c1 = torch.ones(B, 1, device=device, dtype=xs.dtype)

            v0, s0 = model(t_batch, x_t, c0)
            v1, s1 = model(t_batch, x_t, c1)

            v_diff = v1 - v0

            # single-sample closure for divergence: y [D] -> v_diff [D]
            def vecfield(y: Tensor) -> Tensor:
                y_b = y.unsqueeze(0)
                t_b = t_batch[:1]
                c0_b = c0[:1]
                c1_b = c1[:1]
                v0_s = model(t_b, y_b, c0_b)[0].squeeze(0)
                v1_s = model(t_b, y_b, c1_b)[0].squeeze(0)
                return v1_s - v0_s

            div = div_fn(vecfield, x_t)

            correction = (v_diff * s1).sum(dim=-1)

            d_log_r = div + correction
            log_r = log_r + d_log_r * abs(dt)

            x_t = x_t + v0 * dt

    return log_r


def ratio_ode_s2(
    model: nn.Module,
    xs: Tensor,
    steps: int = 10000,
    eps: float = 0.01,
    device: str | None = None,
    div_method: str = "exact",
    uncond_cond: float = -1.0,
    warn_uncond: bool = True,
) -> Tensor:
    """
    integrate the ratio ODE backward from t=1 to t=0 in the s2 setting.

    the s2 setting uses the unconditional velocity field u_t(x|c_f) as the
    simulation trajectory while all three ODE terms survive:

      d/dt log r_t = div[u_t(x|c1) - u_t(x|c0)]
                   + [u_t(x|c_f) - u_t(x|c0)]^T s_t(x|c0)
                   + [u_t(x|c1) - u_t(x|c_f)]^T s_t(x|c1)

    spatial state evolves under the unconditional field u_t(x|c_f).

    reference: Antipov et al., arXiv:2602.24201, Section 4.2.

    args:
        model: nn.Module with forward(t, x, c) -> (velocity, score) tuple.
        xs: [B, D] query points at t=1
        steps: number of integration steps (default 10000)
        eps: time boundary offset (default 0.01)
        device: optional device override
        div_method: "exact", "hutch_gaussian", or "hutch_rademacher"
        uncond_cond: condition value for unconditional field (default -1.0)
        warn_uncond: emit warning about CFG dropout requirement (default true)

    returns:
        [B] log density ratios log(p(x|c0) / p(x|c1)).

    cost: 3 velocity + 2 score evaluations per step (plus divergence).
    """
    from src.models.flow.div_estimators import exact_div, hutch_div

    if device is None:
        device = xs.device.type if hasattr(xs.device, 'type') else str(xs.device)

    B, D = xs.shape
    x_t = xs.clone().to(device)
    log_r = torch.zeros(B, device=device, dtype=xs.dtype)

    t_vals = torch.linspace(1.0 - eps, eps, steps, device=device, dtype=xs.dtype)
    dt = t_vals[1] - t_vals[0]

    model.eval()

    if div_method == "exact":
        div_fn = exact_div
    elif div_method == "hutch_gaussian":
        div_fn = partial(hutch_div, noise="gaussian")
    elif div_method == "hutch_rademacher":
        div_fn = partial(hutch_div, noise="rademacher")
    else:
        raise ValueError(
            f"unknown div_method: {div_method}. "
            f"must be 'exact', 'hutch_gaussian', or 'hutch_rademacher'."
        )

    if warn_uncond:
        warnings.warn(
            "ratio_ode_s2 expects a model trained with CFG dropout (p_uncond > 0). "
            "results may be unreliable otherwise.",
            stacklevel=2
        )

    with torch.no_grad():
        for i in range(steps - 1):
            t_i = t_vals[i].item()

            t_batch = torch.full((B, 1), t_i, device=device, dtype=xs.dtype)
            c0 = torch.zeros(B, 1, device=device, dtype=xs.dtype)
            c1 = torch.ones(B, 1, device=device, dtype=xs.dtype)
            cf = torch.full((B, 1), uncond_cond, device=device, dtype=xs.dtype)

            v0, s0 = model(t_batch, x_t, c0)
            v1, s1 = model(t_batch, x_t, c1)
            vf, _  = model(t_batch, x_t, cf)

            v_diff = v1 - v0

            def vecfield(y: Tensor) -> Tensor:
                y_b = y.unsqueeze(0)
                t_b = t_batch[:1]
                c0_b = c0[:1]
                c1_b = c1[:1]
                v0_s = model(t_b, y_b, c0_b)[0].squeeze(0)
                v1_s = model(t_b, y_b, c1_b)[0].squeeze(0)
                return v1_s - v0_s

            div = div_fn(vecfield, x_t)

            corr_num = (vf - v0) * s0
            corr_den = (v1 - vf) * s1
            correction = (corr_num + corr_den).sum(dim=-1)

            d_log_r = div + correction
            log_r = log_r + d_log_r * abs(dt)

            x_t = x_t + vf * dt

    return log_r
