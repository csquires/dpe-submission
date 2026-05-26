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

    # noise variance and its derivative (derived from path)
    gamma_t = path.gamma(tau)                   # [B, 1]
    dgamma_t = path.dgamma_dtau(tau)            # [B, 1]
    var_t = gamma_t ** 2                        # [B, 1]
    d_var_t = 2 * gamma_t * dgamma_t            # [B, 1]
    std_t = gamma_t                             # [B, 1]

    # path mean and drift direction
    mu_tau = alpha_t * x0 + beta_t * x1 + w_star_t * xstar  # [B, D]
    Delta = d_alpha_t * x0 + d_beta_t * x1 + d_w_star_t * xstar  # [B, D]

    # noisy sample
    x_tau = mu_tau + std_t * epsilon            # [B, D]

    # closed-form target computation
    epsilon_sq = (epsilon ** 2).sum(dim=-1, keepdim=True)   # [B, 1]
    delta_dot_epsilon = (Delta * epsilon).sum(dim=-1, keepdim=True)  # [B, 1]
    dim = epsilon.shape[-1]

    # factor-aware temp from dre-prob-paths reference (see
    # notes/dre_prob_paths_audit.md). factor=2 -> temp == 1 uniformly,
    # removing the close-pair outlier-spike (lambda_t.max/median ~ 532 -> ~1)
    # without changing the bayes-optimal target (per-sample identity
    # target/lambda_t = d_tau log p_tau is invariant under temp choice).
    factor = 2.0
    one_minus_tau = 1.0 - tau
    temp = torch.sqrt(1.0 - 4.0 * tau + 4.0 * tau * tau
                      + 2.0 * factor * tau * one_minus_tau)         # [B, 1]

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

    # noise variance and its derivative (derived from path)
    gamma_t = path.gamma(tau)                   # [B, 1]
    dgamma_t = path.dgamma_dtau(tau)            # [B, 1]
    var_t = gamma_t ** 2                        # [B, 1]
    d_var_t = 2 * gamma_t * dgamma_t            # [B, 1]
    std_t = gamma_t                             # [B, 1]

    # path mean and drift direction
    mu_tau = alpha_t * x0 + beta_t * x1  # [B, D]
    Delta = d_alpha_t * x0 + d_beta_t * x1  # [B, D]

    # noisy sample
    x_tau = mu_tau + std_t * epsilon            # [B, D]

    # closed-form target computation
    epsilon_sq = (epsilon ** 2).sum(dim=-1, keepdim=True)   # [B, 1]
    delta_dot_epsilon = (Delta * epsilon).sum(dim=-1, keepdim=True)  # [B, 1]
    dim = epsilon.shape[-1]

    # factor-aware temp from dre-prob-paths reference (see
    # notes/dre_prob_paths_audit.md). factor=2 -> temp == 1 uniformly,
    # removing the close-pair outlier-spike (lambda_t.max/median ~ 532 -> ~1)
    # without changing the bayes-optimal target (per-sample identity
    # target/lambda_t = d_tau log p_tau is invariant under temp choice).
    factor = 2.0
    one_minus_tau = 1.0 - tau
    temp = torch.sqrt(1.0 - 4.0 * tau + 4.0 * tau * tau
                      + 2.0 * factor * tau * one_minus_tau)         # [B, 1]

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

    # noise variance and its derivative (derived from path)
    gamma_t = path.gamma(t1, t2)                    # [B, 1]
    dgamma_t_dt1 = path.dgamma_dt1(t1, t2)         # [B, 1]
    dgamma_t_dt2 = path.dgamma_dt2(t1, t2)         # [B, 1]
    var_t = gamma_t ** 2                           # [B, 1]
    d_var_t_dt1 = 2 * gamma_t * dgamma_t_dt1      # [B, 1]
    d_var_t_dt2 = 2 * gamma_t * dgamma_t_dt2      # [B, 1]
    std_t = gamma_t                                # [B, 1]

    # noisy sample on path
    x    = mu + std_t * epsilon                     # [B, D]

    # partial derivatives of mu w.r.t. t_1, t_2
    dmu_dt1 = (1.0 - t2) * (x1 - x0)                # [B, D]
    dmu_dt2 = xstar - i_1                           # [B, D]

    # quantities reused in target
    eps_sq        = (epsilon ** 2).sum(dim=-1, keepdim=True)         # [B, 1]
    dim           = epsilon.shape[-1]
    dmu_dt1_dot_e = (dmu_dt1 * epsilon).sum(dim=-1, keepdim=True)    # [B, 1]
    dmu_dt2_dot_e = (dmu_dt2 * epsilon).sum(dim=-1, keepdim=True)    # [B, 1]

    # Option A2 (uniform weighting) — currently active
    target_1 = d_var_t_dt1 * (eps_sq - dim) / 2.0 + std_t * dmu_dt1_dot_e  # [B, 1]
    target_2 = d_var_t_dt2 * (eps_sq - dim) / 2.0 + std_t * dmu_dt2_dot_e  # [B, 1]
    target   = torch.cat([target_1, target_2], dim=-1)                      # [B, 2]
    lambda_t = var_t.expand(-1, 2)                                          # [B, 2]

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

    nets are called space-first (x, t) to match MLP.forward(x, t) and the
    training closures (model(x_t, tau)).

    vmap: in_dims=(None, None, None, 0, 0, None), out_dims=0.
    """
    b = net_b(x, tau)  # [B, D]
    eta = net_eta(x, tau)  # [B, D]

    # div_fn vmaps both x and tau; vecfield takes single-sample (xx, t) -> [D]
    div_b = div_fn(
        lambda xx, t: net_b(xx.unsqueeze(0), t.unsqueeze(0)).squeeze(0),
        x, state=tau,
    )

    gamma_t = path.gamma(tau)  # [B, 1]

    # normalize by gamma; squeeze to [B]
    numerator = (b * eta).sum(-1)  # [B]
    denominator = gamma_t.squeeze(-1)  # [B]

    # handle both [B] and [B, 1] from div_fn
    if div_b.dim() > 1:
        div_b = div_b.squeeze(-1)  # [B]

    return -div_b + numerator / denominator


def vfm_orthros_time_score_1d(
    orthros_net: Callable,
    path: DirectPath1D,
    x: Tensor,
    tau: Tensor,
    div_fn: Callable,
) -> Tensor:
    """Compute negative time-score for VFMOrthros 1D inference (stable parameterization).

    The two-head network predicts an endpoint posterior E[x0|x_t] and the denoiser
    E[z|x_t] directly. The x1 endpoint is *derived* from the interpolant constraint
    x_t = alpha*x0 + beta*x1 + gamma*z:

        x1_hat = (x - alpha*x0_hat - gamma*eta_hat) / beta

    The velocity is v_hat = d_alpha*x0_hat + d_beta*x1_hat + dgamma*eta_hat, and the
    time-score is -div(v_hat) + (v_hat . eta_hat) / gamma.

    Predicting the denoiser as a head (rather than reconstructing it from the two
    endpoints) keeps eta_hat O(1): the score -eta_hat/gamma carries only the single,
    unavoidable 1/gamma of any interpolant marginal score (VFM-level), instead of the
    1/gamma^2 a constraint-derived denoiser would incur. The derived endpoint
    contributes a 1/beta factor to v_hat at the tau->0 corner only, bounded by the
    path eps.

    Called inside torch.vmap for Hutchinson divergence estimation.

    Args:
        orthros_net: callable taking (x [B, D], tau [B, 1]) and returning tuple
                     (x0_hat [B, D], eta_hat [B, D]) -- endpoint posterior and
                     denoiser. space-first arg order matches OrthrosNet.forward(x, t).
        path: DirectPath1D instance (callables: weights, gamma, dgamma_dtau).
        x: noisy sample, shape [B, D].
        tau: time parameter, shape [B, 1].
        div_fn: Hutchinson divergence estimator; takes (closure, x, state=tau).

    Returns:
        time-score tensor, shape [B]: -div(v_hat) + (v_hat . eta_hat) / gamma.

    vmap: in_dims=(None, None, 0, 0, None), out_dims=0.
          orthros_net, path, div_fn are not vmapped; x and tau are vmapped over batch.
    """
    # retrieve weights and path scalars
    w = path.weights(tau)
    gamma_t = path.gamma(tau)  # [B, 1]
    dgamma_t = path.dgamma_dtau(tau)  # [B, 1]

    # heads: endpoint posterior E[x0|x_t] and denoiser E[z|x_t]
    x0_hat, eta_hat = orthros_net(x, tau)  # each [B, D]

    # derive the x1 endpoint from the interpolant constraint
    x1_hat = (x - w.alpha * x0_hat - gamma_t * eta_hat) / w.beta  # [B, D]

    # reconstruct velocity
    v_hat = w.d_alpha * x0_hat + w.d_beta * x1_hat + dgamma_t * eta_hat  # [B, D]

    # single-sample velocity closure for div_fn (re-derives x1 per sample)
    def _v_closure(xx, t):
        xx_b = xx.unsqueeze(0)  # [1, D]
        t_b = t.unsqueeze(0)  # [1, 1]
        x0_s, eta_s = orthros_net(xx_b, t_b)  # each [1, D]
        x0_s = x0_s.squeeze(0)  # [D]
        eta_s = eta_s.squeeze(0)  # [D]

        w_s = path.weights(t_b)
        gamma_s = path.gamma(t_b).squeeze(0)  # [1]
        dgamma_s = path.dgamma_dtau(t_b).squeeze(0)  # [1]
        alpha_s = w_s.alpha.squeeze(0)  # [1]
        beta_s = w_s.beta.squeeze(0)  # [1]
        d_alpha_s = w_s.d_alpha.squeeze(0)  # [1]
        d_beta_s = w_s.d_beta.squeeze(0)  # [1]

        x1_s = (xx - alpha_s * x0_s - gamma_s * eta_s) / beta_s  # [D]
        v_s = d_alpha_s * x0_s + d_beta_s * x1_s + dgamma_s * eta_s  # [D]
        return v_s  # [D]

    # divergence estimation
    div_v = div_fn(_v_closure, x, state=tau)

    # time-score computation
    numerator = (v_hat * eta_hat).sum(-1)  # [B]
    denominator = gamma_t.squeeze(-1)  # [B]

    if div_v.dim() > 1:
        div_v = div_v.squeeze(-1)  # [B]

    return -div_v + numerator / denominator


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
    # nets are called space-first (x, t1, t2) to match MLP2D.forward and training
    b1 = net_b1(x, t1, t2)  # [B, D]
    b2 = net_b2(x, t1, t2)  # [B, D]
    eta = net_eta(x, t1, t2)  # [B, D]

    # for 2d, stack (t1, t2) into a per-sample 2-vector state
    state12 = torch.cat([t1, t2], dim=-1)  # [B, 2]
    def _vf1(xx, s):
        return net_b1(xx.unsqueeze(0), s[0:1].unsqueeze(0), s[1:2].unsqueeze(0)).squeeze(0)
    def _vf2(xx, s):
        return net_b2(xx.unsqueeze(0), s[0:1].unsqueeze(0), s[1:2].unsqueeze(0)).squeeze(0)
    div_b1 = div_fn(_vf1, x, state=state12)
    div_b2 = div_fn(_vf2, x, state=state12)

    gamma_t = path.gamma(t1, t2)  # [B, 1]

    # velocity-denoiser interaction normalized by gamma; squeeze to [B].
    # both components divide by gamma (matching vfm_time_score_1d) -- the gamma
    # partials dgamma/dt1, dgamma/dt2 vanish on the domain and would yield nan.
    numerator_1 = (b1 * eta).sum(-1)  # [B]
    numerator_2 = (b2 * eta).sum(-1)  # [B]
    denominator = gamma_t.squeeze(-1)  # [B]

    # handle both [B] and [B, 1] from div_fn
    if div_b1.dim() > 1:
        div_b1 = div_b1.squeeze(-1)  # [B]
    if div_b2.dim() > 1:
        div_b2 = div_b2.squeeze(-1)  # [B]

    s1 = -div_b1 + numerator_1 / denominator
    s2 = -div_b2 + numerator_2 / denominator

    return s1, s2
