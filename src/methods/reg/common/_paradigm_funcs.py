"""Paradigm-specific target and time-score functions for VFM and CTSM.

Pure callables (no state, no side effects). Operate on path dataclass objects
and training data tensors. Safe inside torch.vmap for divergence estimation.
"""

import torch
from torch import Tensor
from typing import Callable

from src.waypoints.dataclass_paths import (
    DirectPath1D, TriangularPath1D, TriangularPath2D,
)


# ============================================================================
# VFM Velocity Targets
# ============================================================================

def vfm_velocity_target_1d(
    path: TriangularPath1D,
    x0: Tensor,    # [B, D]
    x1: Tensor,    # [B, D]
    xstar: Tensor, # [B, D]
    tau: Tensor,   # [B, 1]
    z: Tensor,     # [B, D]; Gaussian noise
) -> tuple[Tensor, Tensor]:
    """Return (x_t, v_star) where v_star is detached.

    Closed-form velocity target for VFM's drift-estimation phase.
    x_t remains attached; v_star is detached as the regression label.

    vmap: in_dims=(None, 0, 0, 0, 0, 0), out_dims=(0, 0).
    """
    w = path.weights(tau)  # NamedTuple: (alpha, beta, w_star, d_alpha, d_beta, d_w_star)

    mu = w.alpha * x0 + w.beta * x1 + w.w_star * xstar  # [B, D]
    dmu = w.d_alpha * x0 + w.d_beta * x1 + w.d_w_star * xstar  # [B, D]

    gamma_t = path.gamma(tau)  # [B, 1]
    dgamma_t = path.dgamma_dtau(tau)  # [B, 1]

    x_t = mu + gamma_t * z  # [B, D]
    v_star = dmu + dgamma_t * z  # [B, D]

    return x_t, v_star.detach()


def vfm_velocity_target_direct_1d(
    path: DirectPath1D,
    x0: Tensor,    # [B, D]
    x1: Tensor,    # [B, D]
    tau: Tensor,   # [B, 1]
    z: Tensor,     # [B, D]
) -> tuple[Tensor, Tensor]:
    """Return (x_t, v_star) where v_star is detached. No xstar.

    Direct (two-source) variant. Identical numeric logic to triangular,
    omitting the xstar and w_star terms.

    vmap: in_dims=(None, 0, 0, 0, 0), out_dims=(0, 0).
    """
    w = path.weights(tau)  # NamedTuple: (alpha, beta, d_alpha, d_beta)

    mu = w.alpha * x0 + w.beta * x1  # [B, D]
    dmu = w.d_alpha * x0 + w.d_beta * x1  # [B, D]

    gamma_t = path.gamma(tau)  # [B, 1]
    dgamma_t = path.dgamma_dtau(tau)  # [B, 1]

    x_t = mu + gamma_t * z  # [B, D]
    v_star = dmu + dgamma_t * z  # [B, D]

    return x_t, v_star.detach()


def vfm_velocity_target_2d(
    path: TriangularPath2D,
    x0: Tensor,    # [B, D]
    x1: Tensor,    # [B, D]
    xstar: Tensor, # [B, D]
    t1: Tensor,    # [B, 1]
    t2: Tensor,    # [B, 1]
    z: Tensor,     # [B, D]
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (x_t, v1_star, v2_star) where v1_star, v2_star detached.

    Two-component velocity targets for V3 VFM (stacked 2D geometry).
    Each component is independent.

    vmap: in_dims=(None, 0, 0, 0, 0, 0, 0), out_dims=(0, 0, 0).
    """
    w = path.weights(t1, t2)
    # NamedTuple: (alpha, beta, w_star, d_alpha_dt1, d_beta_dt1, d_w_star_dt1,
    #              d_alpha_dt2, d_beta_dt2, d_w_star_dt2)

    mu = w.alpha * x0 + w.beta * x1 + w.w_star * xstar  # [B, D]
    dmu_dt1 = w.d_alpha_dt1 * x0 + w.d_beta_dt1 * x1 + w.d_w_star_dt1 * xstar  # [B, D]
    dmu_dt2 = w.d_alpha_dt2 * x0 + w.d_beta_dt2 * x1 + w.d_w_star_dt2 * xstar  # [B, D]

    gamma_t = path.gamma(t1, t2)  # [B, 1]
    dgamma_dt1 = path.dgamma_dt1(t1, t2)  # [B, 1]
    dgamma_dt2 = path.dgamma_dt2(t1, t2)  # [B, 1]

    x_t = mu + gamma_t * z  # [B, D]
    v1_star = dmu_dt1 + dgamma_dt1 * z  # [B, D]
    v2_star = dmu_dt2 + dgamma_dt2 * z  # [B, D]

    return x_t, v1_star.detach(), v2_star.detach()


# ============================================================================
# CTSM Regression Targets
# ============================================================================

def ctsm_regression_target_1d(
    path: TriangularPath1D,
    x0: Tensor,     # [B, D]
    x1: Tensor,     # [B, D]
    xstar: Tensor,  # [B, D]
    tau: Tensor,    # [B, 1]
    epsilon: Tensor, # [B, D]; standard Gaussian noise
    sigma: float,   # noise amplitude scale
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (x_t, target, lambda_t) where target and lambda_t detached.

    Port from BarycentricCtsm1D.sample_and_target (triangular_continuous.py:136-175).
    Switches weight reads from inline _barycentric_weights to path.weights(tau).
    Closed-form regression target with chord-norm normalization.

    vmap: in_dims=(None, 0, 0, 0, 0, 0, None), out_dims=(0, 0, 0).
    """
    w = path.weights(tau)
    # NamedTuple: (alpha, beta, w_star, d_alpha, d_beta, d_w_star)
    alpha_t, beta_t, w_star_t = w.alpha, w.beta, w.w_star
    d_alpha_t, d_beta_t, d_w_star_t = w.d_alpha, w.d_beta, w.d_w_star

    # noise variance and its derivative
    g_t = tau * (1 - tau)                       # [B, 1]
    dg_dtau_t = 1 - 2 * tau                     # [B, 1]
    std_t = sigma * torch.sqrt(g_t)             # [B, 1]

    # path mean and drift direction
    mu_tau = alpha_t * x0 + beta_t * x1 + w_star_t * xstar  # [B, D]
    Delta = d_alpha_t * x0 + d_beta_t * x1 + d_w_star_t * xstar  # [B, D]

    # noisy sample
    x_tau = mu_tau + std_t * epsilon            # [B, D]

    # closed-form target computation
    epsilon_sq = (epsilon ** 2).sum(dim=-1, keepdim=True)   # [B, 1]
    delta_dot_epsilon = (Delta * epsilon).sum(dim=-1, keepdim=True)  # [B, 1]
    dim = epsilon.shape[-1]

    # variance schedule
    var_t = sigma ** 2 * g_t                    # [B, 1]
    d_var_t = sigma ** 2 * dg_dtau_t            # [B, 1]

    # chord-norm normalization (endpoint-based, not path-velocity-based)
    delta_endpoint_sq = ((x1 - x0) ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    temp = torch.sqrt(2 * delta_endpoint_sq + 1e-8)                 # [B, 1]

    target = (d_var_t * (epsilon_sq - dim) / 2.0
              + std_t * delta_dot_epsilon) / temp
    lambda_t = var_t / temp

    return x_tau, target.detach(), lambda_t.detach()


def ctsm_regression_target_direct_1d(
    path: DirectPath1D,
    x0: Tensor,     # [B, D]
    x1: Tensor,     # [B, D]
    tau: Tensor,    # [B, 1]
    epsilon: Tensor, # [B, D]
    sigma: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (x_t, target, lambda_t) where target and lambda_t detached. No xstar.

    Direct (two-source) CTSM target. Identical numeric logic to triangular,
    omitting xstar terms.

    vmap: in_dims=(None, 0, 0, 0, 0, None), out_dims=(0, 0, 0).
    """
    w = path.weights(tau)
    # NamedTuple: (alpha, beta, d_alpha, d_beta)
    alpha_t, beta_t = w.alpha, w.beta
    d_alpha_t, d_beta_t = w.d_alpha, w.d_beta

    # noise schedule
    g_t = tau * (1 - tau)                       # [B, 1]
    dg_dtau_t = 1 - 2 * tau                     # [B, 1]
    std_t = sigma * torch.sqrt(g_t)             # [B, 1]

    # path mean and drift direction
    mu_tau = alpha_t * x0 + beta_t * x1  # [B, D]
    Delta = d_alpha_t * x0 + d_beta_t * x1  # [B, D]

    # noisy sample
    x_tau = mu_tau + std_t * epsilon            # [B, D]

    # closed-form target computation
    epsilon_sq = (epsilon ** 2).sum(dim=-1, keepdim=True)   # [B, 1]
    delta_dot_epsilon = (Delta * epsilon).sum(dim=-1, keepdim=True)  # [B, 1]
    dim = epsilon.shape[-1]

    # variance schedule
    var_t = sigma ** 2 * g_t                    # [B, 1]
    d_var_t = sigma ** 2 * dg_dtau_t            # [B, 1]

    # chord-norm normalization
    delta_endpoint_sq = ((x1 - x0) ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    temp = torch.sqrt(2 * delta_endpoint_sq + 1e-8)                 # [B, 1]

    target = (d_var_t * (epsilon_sq - dim) / 2.0
              + std_t * delta_dot_epsilon) / temp
    lambda_t = var_t / temp

    return x_tau, target.detach(), lambda_t.detach()


def ctsm_regression_target_2d(
    path: TriangularPath2D,
    x0: Tensor,     # [B, D]
    x1: Tensor,     # [B, D]
    xstar: Tensor,  # [B, D]
    t1: Tensor,     # [B, 1]
    t2: Tensor,     # [B, 1]
    epsilon: Tensor, # [B, D]
    sigma: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (x_t, target [B, 2], lambda_t [B, 2]) where target, lambda_t detached.

    Port from Stacked2DCtsm.sample_and_target (triangular_continuous_2d.py:131-191).
    Switches weights from inline to path.weights(t1, t2); sigma is parameter.
    Uses Option A2 (uniform weighting).

    vmap: in_dims=(None, 0, 0, 0, 0, 0, 0, None), out_dims=(0, 0, 0).
    """
    w = path.weights(t1, t2)
    # NamedTuple with full derivative components

    # stacked interpolant
    i_1 = (1.0 - t1) * x0 + t1 * x1                 # [B, D]
    mu  = (1.0 - t2) * i_1 + t2 * xstar             # [B, D]

    # noise schedule and std
    g    = path.gamma(t1, t2)                       # [B, 1]
    std  = sigma * torch.sqrt(g)                    # [B, 1]

    # noisy sample on path
    x    = mu + std * epsilon                       # [B, D]

    # partial derivatives of mu w.r.t. t_1, t_2
    dmu_dt1 = (1.0 - t2) * (x1 - x0)                # [B, D]
    dmu_dt2 = xstar - i_1                           # [B, D]

    # partial derivatives of g w.r.t. t_1, t_2
    dg_dt1 = path.dgamma_dt1(t1, t2)                # [B, 1]
    dg_dt2 = path.dgamma_dt2(t1, t2)                # [B, 1]

    # quantities reused in target
    eps_sq        = (epsilon ** 2).sum(dim=-1, keepdim=True)         # [B, 1]
    dim           = epsilon.shape[-1]
    dmu_dt1_dot_e = (dmu_dt1 * epsilon).sum(dim=-1, keepdim=True)    # [B, 1]
    dmu_dt2_dot_e = (dmu_dt2 * epsilon).sum(dim=-1, keepdim=True)    # [B, 1]
    sigma_sq      = sigma ** 2

    # Option A2 (uniform weighting) — currently active
    target_1 = sigma_sq * dg_dt1 * (eps_sq - dim) / 2.0 + std * dmu_dt1_dot_e   # [B, 1]
    target_2 = sigma_sq * dg_dt2 * (eps_sq - dim) / 2.0 + std * dmu_dt2_dot_e   # [B, 1]
    target   = torch.cat([target_1, target_2], dim=-1)                          # [B, 2]
    lam      = sigma_sq * g                                                     # [B, 1]
    lambda_t = lam.expand(-1, 2)                                                # [B, 2]

    return x, target.detach(), lambda_t.detach()


# ============================================================================
# VFM Time-Score Helpers (Inference)
# ============================================================================

def vfm_time_score_1d(
    net_b: Callable,    # returns [B, D] given (x [B, D], tau [B, 1])
    net_eta: Callable,  # same signature
    path: TriangularPath1D | DirectPath1D,
    x: Tensor,         # [B, D]
    tau: Tensor,       # [B, 1]
    div_fn: Callable,  # divergence estimator: (net_b, x, tau) -> [B] or [B, 1]
) -> Tensor:  # [B]
    """Compute negative time-score for VFM 1D inference.

    Combines the velocity divergence and scaled velocity-denoiser interaction.
    Called inside torch.vmap for Hutchinson divergence estimation.

    vmap: in_dims=(None, None, None, 0, 0, None), out_dims=0.
    """
    b = net_b(x, tau)  # [B, D]
    eta = net_eta(x, tau)  # [B, D]

    div_b = div_fn(net_b, x, tau)  # [B] or [B, 1]

    gamma_t = path.gamma(tau)  # [B, 1]

    # normalize by gamma; squeeze to [B]
    numerator = (b * eta).sum(-1)  # [B]
    denominator = gamma_t.squeeze(-1)  # [B]

    # handle both [B] and [B, 1] from div_fn
    if div_b.dim() > 1:
        div_b = div_b.squeeze(-1)  # [B]

    return -div_b + numerator / denominator


def vfm_time_score_2d(
    net_b1: Callable,   # returns [B, D] given (x [B, D], t1 [B, 1], t2 [B, 1])
    net_b2: Callable,   # same signature
    net_eta: Callable,  # same signature
    path: TriangularPath2D,
    x: Tensor,         # [B, D]
    t1: Tensor,        # [B, 1]
    t2: Tensor,        # [B, 1]
    div_fn: Callable,  # divergence estimator: (net, x, t1, t2) -> [B]
) -> tuple[Tensor, Tensor]:  # (s1 [B], s2 [B])
    """Compute negative time-scores for VFM 2D inference.

    Two independent time-score components for V3 VFM. predict_ldr_via_curve
    chains these with the 2D curve derivatives and integrator.

    vmap: in_dims=(None, None, None, None, 0, 0, 0, None), out_dims=(0, 0).
    """
    b1 = net_b1(x, t1, t2)  # [B, D]
    b2 = net_b2(x, t1, t2)  # [B, D]
    eta = net_eta(x, t1, t2)  # [B, D]

    div_b1 = div_fn(net_b1, x, t1, t2)  # [B] or [B, 1]
    div_b2 = div_fn(net_b2, x, t1, t2)  # [B] or [B, 1]

    gamma_dt1 = path.dgamma_dt1(t1, t2)  # [B, 1]
    gamma_dt2 = path.dgamma_dt2(t1, t2)  # [B, 1]

    # normalize by gamma derivatives; squeeze to [B]
    numerator_1 = (b1 * eta).sum(-1)  # [B]
    numerator_2 = (b2 * eta).sum(-1)  # [B]
    denominator_1 = gamma_dt1.squeeze(-1)  # [B]
    denominator_2 = gamma_dt2.squeeze(-1)  # [B]

    # handle both [B] and [B, 1] from div_fn
    if div_b1.dim() > 1:
        div_b1 = div_b1.squeeze(-1)  # [B]
    if div_b2.dim() > 1:
        div_b2 = div_b2.squeeze(-1)  # [B]

    s1 = -div_b1 + numerator_1 / denominator_1
    s2 = -div_b2 + numerator_2 / denominator_2

    return s1, s2
