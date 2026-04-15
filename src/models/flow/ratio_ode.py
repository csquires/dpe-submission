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


def ratio_ode(
    model: nn.Module,
    xs: Tensor,
    steps: int = 10000,
    eps: float = 0.01,
    device: str | None = None,
    div_method: str = "exact",
    chunk_size: int = 500,
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

        chunk_size: batch size for divergence chunking, memory efficiency (default 500)

    returns:
        [B] tensor of log density ratios log(p(x|c0) / p(x|c1)) at query points.
    """
    # import divergence functions. note: these should be implemented in div_estimators.py
    try:
        from src.models.flow.div_estimators import exact_div, hutch_div, div_chunked
    except ImportError:
        raise ImportError(
            "div_estimators module not found. "
            "ensure src/models/flow/div_estimators.py exists with "
            "exact_div, hutch_div, div_chunked functions."
        )

    # device inference and initialization
    if device is None:
        device = xs.device.type if hasattr(xs.device, 'type') else str(xs.device)

    B, D = xs.shape
    x_t = xs.clone().to(device)  # [B, D] current spatial state, evolves backward
    log_r = torch.zeros(B, device=device, dtype=xs.dtype)  # [B] accumulated ratio

    # time grid from 1-eps down to eps (backward integration)
    t_vals = torch.linspace(1.0 - eps, eps, steps, device=device, dtype=xs.dtype)
    dt = t_vals[1] - t_vals[0]  # will be negative since going backward

    # set model to evaluation mode
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

    # time stepping loop: integrate from t=1-eps down to t=eps
    with torch.no_grad():
        for i in range(steps - 1):
            t_i = t_vals[i].item()

            # prepare batch tensors for both conditions
            t_batch = torch.full((B, 1), t_i, device=device, dtype=xs.dtype)  # [B, 1]
            c0 = torch.zeros(B, 1, device=device, dtype=xs.dtype)  # [B, 1] numerator
            c1 = torch.ones(B, 1, device=device, dtype=xs.dtype)   # [B, 1] denominator

            # evaluate model at both conditions
            v0, s0 = model(t_batch, x_t, c0)  # [B, D], [B, D]
            v1, s1 = model(t_batch, x_t, c1)  # [B, D], [B, D]

            # compute velocity difference
            v_diff = v1 - v0  # [B, D]

            # define single-sample vecfield closure for divergence computation
            def vecfield(y: Tensor) -> Tensor:
                """
                vecfield for divergence estimation.

                single-sample closure: y is [D] (single sample).
                unsqueeze for batch model calls, squeeze output.
                computes v_diff = u_t(y|c1) - u_t(y|c0).
                """
                y_b = y.unsqueeze(0)  # [1, D] unsqueeze for batch
                t_b = t_batch[:1]     # [1, 1]
                c0_b = c0[:1]         # [1, 1]
                c1_b = c1[:1]         # [1, 1]

                # model calls return (velocity, score)
                v0_s = model(t_b, y_b, c0_b)[0].squeeze(0)  # [D]
                v1_s = model(t_b, y_b, c1_b)[0].squeeze(0)  # [D]

                return v1_s - v0_s  # [D]

            # estimate divergence with chunking
            div = div_chunked(div_fn, vecfield, x_t, chunk_size)  # [B]

            # compute correction term: [u_t(x|c1) - u_t(x|c0)]^T s_t(x|c1)
            correction = (v_diff * s1).sum(dim=-1)  # [B] dot product along D

            # accumulate log ratio
            d_log_r = div + correction  # [B]
            log_r = log_r + d_log_r * abs(dt)  # use abs(dt) for backward integration

            # update spatial state via backward Euler
            x_t = x_t + v0 * dt  # [B, D] dt < 0, so moves backward along v0

    return log_r  # [B]
