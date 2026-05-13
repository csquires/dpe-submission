"""pure factory functions and convenience constructors for path dataclasses.

produces callable weight bundles and complete path instances from hyperparameters.
all builders use keyword-only args (no *args, no **kwargs) to surface typos as
typeerror. closures capture hyperparameters; no object state is kept beyond pytorch's.
designs follow byte-identical porting from legacy modules (triangular_continuous,
piecewise_sb, triangular_continuous_2d). all math is immutable once built.
"""
import torch
from torch import Tensor
from src.waypoints.dataclass_paths import (
    DirectPath1D, TriangularPath1D, TriangularPath2D,
    DirectWeights1D, TriangularWeights1D, TriangularWeights2D,
)


def make_barycentric_weights_1d(*, vertex: float = 0.5):
    """factory for triangular barycentric weight callable.

    returns a closure that takes tau [B, 1] and computes c^1 piecewise-quadratic
    bell peak at vertex, with weights and derivatives summing to 1 and 0 respectively.

    fast path (vertex == 0.5): h(tau) = 4*tau*(1-tau) (legacy symmetric bell).
    general path: asymmetric piecewise-quadratic with analytical derivatives.

    closure signature: (tau: Tensor) -> TriangularWeights1D.
    """
    def weights_fn(tau: Tensor) -> TriangularWeights1D:
        # [B, 1] tau input assumed pre-validated in [eps, 1-eps]
        if vertex == 0.5:
            # legacy fast path: byte-identical symmetric bell
            h = 4.0 * tau * (1.0 - tau)                          # [B, 1]
            h_prime = 4.0 * (1.0 - 2.0 * tau)                    # [B, 1]
        else:
            # general piecewise-quadratic bell with peak at vertex
            v = vertex
            u_left = tau / v                                      # [B, 1]
            u_right = (1.0 - tau) / (1.0 - v)                    # [B, 1]
            h_left = u_left * (2.0 - u_left)                     # [B, 1]
            h_right = u_right * (2.0 - u_right)                  # [B, 1]
            h = torch.where(tau <= v, h_left, h_right)           # [B, 1]
            # left leg: dh/dtau = (2/v)(1 - u_left)
            # right leg: dh/dtau = -(2/(1-v))(1 - u_right)
            h_prime_left = (2.0 / v) * (1.0 - u_left)            # [B, 1]
            h_prime_right = -(2.0 / (1.0 - v)) * (1.0 - u_right) # [B, 1]
            h_prime = torch.where(tau <= v, h_prime_left, h_prime_right)  # [B, 1]

        alpha = (1.0 - tau) * (1.0 - h)                          # [B, 1]
        beta = tau * (1.0 - h)                                   # [B, 1]
        w_star = h                                               # [B, 1]
        d_alpha = -(1.0 - h) - (1.0 - tau) * h_prime             # [B, 1]
        d_beta = (1.0 - h) - tau * h_prime                       # [B, 1]
        d_w_star = h_prime                                       # [B, 1]

        return TriangularWeights1D(alpha, beta, w_star, d_alpha, d_beta, d_w_star)
    return weights_fn


def make_piecewise_sb_weights_1d(*, vertex: float, eps: float):
    """factory for piecewise-linear sb weights (two legs joined hard at vertex).

    leg 1 (tau < vertex): local t1 = tau / vertex, weights interpolate x0 -> xstar.
    leg 2 (tau >= vertex): local t2 = (tau - vertex) / (1 - vertex), weights interpolate xstar -> x1.
    local times clamped to [eps, 1-eps] per leg to avoid singularity.

    closure signature: (tau: Tensor) -> TriangularWeights1D.
    """
    def weights_fn(tau: Tensor) -> TriangularWeights1D:
        # [B, 1] tau input; per-leg local time clamped to [eps, 1-eps]
        t1 = tau / vertex                                        # [B, 1]
        t2 = (tau - vertex) / (1.0 - vertex)                     # [B, 1]
        t1_clamped = torch.clamp(t1, eps, 1.0 - eps)             # [B, 1]
        t2_clamped = torch.clamp(t2, eps, 1.0 - eps)             # [B, 1]

        # leg 1: alpha = 1 - t1, beta = 0, w_star = t1
        alpha_leg1 = 1.0 - t1_clamped                            # [B, 1]
        beta_leg1 = torch.zeros_like(tau)                        # [B, 1]
        w_star_leg1 = t1_clamped                                 # [B, 1]
        d_alpha_leg1 = -torch.ones_like(tau) / vertex            # [B, 1]
        d_beta_leg1 = torch.zeros_like(tau)                      # [B, 1]
        d_w_star_leg1 = torch.ones_like(tau) / vertex            # [B, 1]

        # leg 2: alpha = 0, beta = t2, w_star = 1 - t2
        alpha_leg2 = torch.zeros_like(tau)                       # [B, 1]
        beta_leg2 = t2_clamped                                   # [B, 1]
        w_star_leg2 = 1.0 - t2_clamped                           # [B, 1]
        d_alpha_leg2 = torch.zeros_like(tau)                     # [B, 1]
        d_beta_leg2 = torch.ones_like(tau) / (1.0 - vertex)      # [B, 1]
        d_w_star_leg2 = -torch.ones_like(tau) / (1.0 - vertex)   # [B, 1]

        # select leg via torch.where(tau < vertex, leg1, leg2)
        mask = tau < vertex                                      # [B, 1]
        alpha = torch.where(mask, alpha_leg1, alpha_leg2)        # [B, 1]
        beta = torch.where(mask, beta_leg1, beta_leg2)           # [B, 1]
        w_star = torch.where(mask, w_star_leg1, w_star_leg2)     # [B, 1]
        d_alpha = torch.where(mask, d_alpha_leg1, d_alpha_leg2)  # [B, 1]
        d_beta = torch.where(mask, d_beta_leg1, d_beta_leg2)     # [B, 1]
        d_w_star = torch.where(mask, d_w_star_leg1, d_w_star_leg2)  # [B, 1]

        return TriangularWeights1D(alpha, beta, w_star, d_alpha, d_beta, d_w_star)
    return weights_fn


def make_direct_weights_1d():
    """factory for linear interpolant weights (direct, no w_star).

    returns constant weights: alpha = 1 - tau, beta = tau.
    derivatives: d_alpha = -1, d_beta = 1 (constant).

    closure signature: (tau: Tensor) -> DirectWeights1D.
    """
    def weights_fn(tau: Tensor) -> DirectWeights1D:
        # [B, 1] tau input
        alpha = 1.0 - tau                                        # [B, 1]
        beta = tau                                               # [B, 1]
        d_alpha = -torch.ones_like(tau)                          # [B, 1]
        d_beta = torch.ones_like(tau)                            # [B, 1]
        return DirectWeights1D(alpha, beta, d_alpha, d_beta)
    return weights_fn


def make_stacked_2d_weights(*, t2_max: float, eps: float):
    """factory for 2d stacked-interpolant weights.

    interpolant: mu = (1-t2)*((1-t1)*x0 + t1*x1) + t2*xstar.
    weights factored out; t2_max and eps stored for interface but not used in computation.

    closure signature: (t1: Tensor, t2: Tensor) -> TriangularWeights2D.
    """
    # capture hyperparameters (unused in computation; stored for path interface)
    _ = (t2_max, eps)

    def weights_fn(t1: Tensor, t2: Tensor) -> TriangularWeights2D:
        # [B, 1] t1, t2 inputs
        alpha = (1.0 - t2) * (1.0 - t1)                          # [B, 1]
        beta = (1.0 - t2) * t1                                   # [B, 1]
        w_star = t2                                              # [B, 1]

        # partial derivatives w.r.t. t1
        d_alpha_dt1 = -(1.0 - t2)                                # [B, 1]
        d_beta_dt1 = (1.0 - t2)                                  # [B, 1]
        d_w_star_dt1 = torch.zeros_like(t2)                      # [B, 1]

        # partial derivatives w.r.t. t2
        d_alpha_dt2 = -(1.0 - t1)                                # [B, 1]
        d_beta_dt2 = -t1                                         # [B, 1]
        d_w_star_dt2 = torch.ones_like(t2)                       # [B, 1]

        return TriangularWeights2D(
            alpha, beta, w_star,
            d_alpha_dt1, d_beta_dt1, d_w_star_dt1,
            d_alpha_dt2, d_beta_dt2, d_w_star_dt2,
        )
    return weights_fn


def make_vfm_gamma(*, k: float):
    """factory for vfm sigmoid-product gamma schedule.

    returns (gamma_fn, dgamma_dtau_fn) pair.
    gamma(tau) = (1 - exp(-k*tau)) * (1 - exp(-k*(1-tau))).
    uses torch.expm1 for numerical stability.

    closure signatures: (tau: Tensor) -> Tensor (both gamma and derivative).
    """
    def gamma_fn(tau: Tensor) -> Tensor:
        # [B, 1] tau input
        return (-torch.expm1(-k * tau)) * (-torch.expm1(-k * (1.0 - tau)))  # [B, 1]

    def dgamma_dtau_fn(tau: Tensor) -> Tensor:
        # [B, 1] tau input
        exp_kt = torch.exp(-k * tau)                             # [B, 1]
        exp_k1t = torch.exp(-k * (1.0 - tau))                    # [B, 1]
        return k * exp_kt * (1.0 - exp_k1t) - k * exp_k1t * (1.0 - exp_kt)  # [B, 1]

    return gamma_fn, dgamma_dtau_fn


def make_ctsm_variance_gamma(*, sigma: float):
    """factory for ctsm variance schedule gamma(tau) = sigma * sqrt(tau * (1 - tau)).

    returns (gamma_fn, dgamma_dtau_fn) pair.

    closure signatures: (tau: Tensor) -> Tensor (both gamma and derivative).
    """
    def gamma_fn(tau: Tensor) -> Tensor:
        # [B, 1] tau input
        g_t = tau * (1.0 - tau)                                  # [B, 1]
        return sigma * torch.sqrt(g_t)                           # [B, 1]

    def dgamma_dtau_fn(tau: Tensor) -> Tensor:
        # [B, 1] tau input
        g_t = tau * (1.0 - tau)                                  # [B, 1]
        dg_dtau = 1.0 - 2.0 * tau                                # [B, 1]
        return sigma * dg_dtau / (2.0 * torch.sqrt(g_t))         # [B, 1]

    return gamma_fn, dgamma_dtau_fn


def make_piecewise_sb_gamma(*, sigma: float, vertex: float, gamma_min: float, inner_eps: float, eps: float):
    """factory for piecewise-sb gamma with hard floor gamma_min.

    gamma computed per leg with analytical derivatives; clamped from below.
    derivative forced to zero inside clamp window.

    returns (gamma_fn, dgamma_dtau_fn) pair.

    closure signatures: (tau: Tensor) -> Tensor (both gamma and derivative).
    """
    def gamma_fn(tau: Tensor) -> Tensor:
        # [B, 1] tau input
        t1 = tau / vertex                                        # [B, 1]
        t2 = (tau - vertex) / (1.0 - vertex)                     # [B, 1]
        gamma_leg1 = sigma * torch.sqrt(t1 * (1.0 - t1))         # [B, 1]
        gamma_leg2 = sigma * torch.sqrt(t2 * (1.0 - t2))         # [B, 1]
        gamma_raw = torch.where(tau < vertex, gamma_leg1, gamma_leg2)  # [B, 1]
        return torch.clamp_min(gamma_raw, gamma_min)             # [B, 1]

    def dgamma_dtau_fn(tau: Tensor) -> Tensor:
        # [B, 1] tau input
        t1 = tau / vertex                                        # [B, 1]
        t2 = (tau - vertex) / (1.0 - vertex)                     # [B, 1]
        gamma_leg1_raw = sigma * torch.sqrt(t1 * (1.0 - t1))     # [B, 1]
        gamma_leg2_raw = sigma * torch.sqrt(t2 * (1.0 - t2))     # [B, 1]
        mask = tau < vertex                                      # [B, 1]
        gamma_raw = torch.where(mask, gamma_leg1_raw, gamma_leg2_raw)  # [B, 1]
        clamped = gamma_raw < gamma_min                          # [B, 1]

        d_leg1 = (sigma / vertex) * (1.0 - 2.0 * t1) / (2.0 * torch.sqrt(t1 * (1.0 - t1)))  # [B, 1]
        d_leg2 = (sigma / (1.0 - vertex)) * (1.0 - 2.0 * t2) / (2.0 * torch.sqrt(t2 * (1.0 - t2)))  # [B, 1]
        dgamma_raw = torch.where(mask, d_leg1, d_leg2)           # [B, 1]
        return torch.where(clamped, torch.zeros_like(dgamma_raw), dgamma_raw)  # [B, 1]

    return gamma_fn, dgamma_dtau_fn


def make_stacked_2d_gamma(*, k: float, gamma_schedule: str = "linear-stiff", t2_max: float, eps: float):
    """factory for 2d stacked vfm gamma with parametric dispatch.

    currently only gamma_schedule="linear-stiff" implemented.
    returns (gamma_fn, dgamma_dt1_fn, dgamma_dt2_fn) triple.
    bak bug fix: dgamma_dt2 returns zeros (not -gamma) in linear-stiff schedule.

    closure signatures: (t1: Tensor, t2: Tensor) -> Tensor (all three).
    """
    if gamma_schedule != "linear-stiff":
        raise ValueError(f"gamma_schedule must be 'linear-stiff', got {gamma_schedule!r}")
    # capture hyperparameters (t2_max and eps stored for path interface but unused)
    _ = (t2_max, eps)

    def gamma_fn(t1: Tensor, t2: Tensor) -> Tensor:
        # [B, 1] t1, t2 inputs; gamma independent of t2
        return (-torch.expm1(-k * t1)) * (-torch.expm1(-k * (1.0 - t1)))  # [B, 1]

    def dgamma_dt1_fn(t1: Tensor, t2: Tensor) -> Tensor:
        # [B, 1] t1, t2 inputs
        e1 = torch.exp(-k * t1)                                  # [B, 1]
        e2 = torch.exp(-k * (1.0 - t1))                          # [B, 1]
        return k * e1 * (1.0 - e2) - k * e2 * (1.0 - e1)         # [B, 1]

    def dgamma_dt2_fn(t1: Tensor, t2: Tensor) -> Tensor:
        # [B, 1] t1, t2 inputs; gamma has no t2 dependence
        # bak bug fix: was returning -gamma. correct value is zero.
        return torch.zeros_like(t2)                              # [B, 1]

    return gamma_fn, dgamma_dt1_fn, dgamma_dt2_fn


def make_stacked_2d_ctsm_gamma(*, sigma: float, t2_max: float, eps: float):
    """factory for 2d ctsm variance-style gamma schedule.

    ports byte-identical math from Stacked2DCtsm.gamma/dgamma_dt1/dgamma_dt2 legacy.

    returns (gamma_fn, dgamma_dt1_fn, dgamma_dt2_fn) triple.

    closure signatures: (t1: Tensor, t2: Tensor) -> Tensor (all three).
    """
    # capture hyperparameters (unused; stored for interface)
    _ = (t2_max, eps)

    def gamma_fn(t1: Tensor, t2: Tensor) -> Tensor:
        # [B, 1] t1, t2 inputs; ctsm 2d variance schedule g(t1, t2) = t1(1-t1)(1-t2)
        g = t1 * (1.0 - t1) * (1.0 - t2)                         # [B, 1]
        return sigma * torch.sqrt(g)                             # [B, 1]

    def dgamma_dt1_fn(t1: Tensor, t2: Tensor) -> Tensor:
        # [B, 1] t1, t2 inputs; d/dt1 [t1(1-t1)(1-t2)] = (1-2t1)(1-t2)
        g = t1 * (1.0 - t1) * (1.0 - t2)                         # [B, 1]
        dg_dt1 = (1.0 - 2.0 * t1) * (1.0 - t2)                   # [B, 1]
        return sigma * dg_dt1 / (2.0 * torch.sqrt(g))            # [B, 1]

    def dgamma_dt2_fn(t1: Tensor, t2: Tensor) -> Tensor:
        # [B, 1] t1, t2 inputs; d/dt2 [t1(1-t1)(1-t2)] = -t1(1-t1)
        g = t1 * (1.0 - t1) * (1.0 - t2)                         # [B, 1]
        dg_dt2 = -t1 * (1.0 - t1)                                # [B, 1]
        return sigma * dg_dt2 / (2.0 * torch.sqrt(g))            # [B, 1]

    return gamma_fn, dgamma_dt1_fn, dgamma_dt2_fn


def barycentric_triangular_path_1d(*, k: float = 20.0, vertex: float = 0.5, eps: float = 1e-3) -> TriangularPath1D:
    """convenience constructor for triangular vfm v1 path (barycentric weights + sigmoid gamma).

    validates hyperparameters; constructs path instance bundling weights and gamma callables.
    """
    if not (0.0 < vertex < 1.0):
        raise ValueError(f"vertex must be in (0, 1), got {vertex}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")

    weights_fn = make_barycentric_weights_1d(vertex=vertex)
    gamma_fn, dgamma_dtau_fn = make_vfm_gamma(k=k)
    return TriangularPath1D(
        weights=weights_fn,
        gamma=gamma_fn,
        dgamma_dtau=dgamma_dtau_fn,
        eps=eps,
    )


def piecewise_sb_triangular_path_1d(*, sigma: float = 1.0, vertex: float = 0.5, gamma_min: float = 5e-2, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """convenience constructor for triangular vfm v2 path (piecewise-sb weights + floored gamma).

    validates hyperparameters; constructs path instance bundling weights and gamma callables.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if not (0.0 < vertex < 1.0):
        raise ValueError(f"vertex must be in (0, 1), got {vertex}")
    if gamma_min <= 0:
        raise ValueError(f"gamma_min must be > 0, got {gamma_min}")
    if inner_eps < 0:
        raise ValueError(f"inner_eps must be >= 0, got {inner_eps}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")
    if eps >= min(vertex, 1.0 - vertex):
        raise ValueError(f"eps must be < min(vertex, 1-vertex), got {eps}")
    if inner_eps + eps >= 1.0:
        raise ValueError(f"inner_eps + eps must be < 1, got {inner_eps + eps}")

    weights_fn = make_piecewise_sb_weights_1d(vertex=vertex, eps=eps)
    gamma_fn, dgamma_dtau_fn = make_piecewise_sb_gamma(
        sigma=sigma, vertex=vertex, gamma_min=gamma_min, inner_eps=inner_eps, eps=eps
    )
    return TriangularPath1D(
        weights=weights_fn,
        gamma=gamma_fn,
        dgamma_dtau=dgamma_dtau_fn,
        eps=eps,
    )


def stacked_2d_triangular_path(*, k: float = 20.0, gamma_schedule: str = "linear-stiff", t2_max: float = 0.3, eps: float = 1e-3) -> TriangularPath2D:
    """convenience constructor for triangular vfm v3 path (stacked 2d weights + 2d gamma).

    validates hyperparameters; constructs path instance bundling weights and gamma callables.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if gamma_schedule not in {"linear-stiff"}:
        raise ValueError(f"gamma_schedule must be 'linear-stiff', got {gamma_schedule!r}")
    if not (0.0 < t2_max < 1.0):
        raise ValueError(f"t2_max must be in (0, 1), got {t2_max}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")

    weights_fn = make_stacked_2d_weights(t2_max=t2_max, eps=eps)
    gamma_fn, dgamma_dt1_fn, dgamma_dt2_fn = make_stacked_2d_gamma(
        k=k, gamma_schedule=gamma_schedule, t2_max=t2_max, eps=eps
    )
    return TriangularPath2D(
        weights=weights_fn,
        gamma=gamma_fn,
        dgamma_dt1=dgamma_dt1_fn,
        dgamma_dt2=dgamma_dt2_fn,
        eps=eps,
        t2_max=t2_max,
    )


def direct_path_1d(*, k: float = 0.5, eps: float = 1e-3) -> DirectPath1D:
    """convenience constructor for stock vfm path (linear weights + sigmoid gamma).

    note: default k=0.5 differs from triangular variants (k=20.0) to reflect stock vfm hpo range.

    validates hyperparameters; constructs path instance bundling weights and gamma callables.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")

    weights_fn = make_direct_weights_1d()
    gamma_fn, dgamma_dtau_fn = make_vfm_gamma(k=k)
    return DirectPath1D(
        weights=weights_fn,
        gamma=gamma_fn,
        dgamma_dtau=dgamma_dtau_fn,
        eps=eps,
    )


def barycentric_ctsm_path_1d(*, sigma: float = 1.0, vertex: float = 0.5, eps: float = 1e-3) -> TriangularPath1D:
    """convenience constructor for v1-ctsm path (barycentric weights + ctsm variance gamma).

    same barycentric weights as vfm v1, but with ctsm-classical variance schedule.

    validates hyperparameters; constructs path instance bundling weights and gamma callables.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if not (0.0 < vertex < 1.0):
        raise ValueError(f"vertex must be in (0, 1), got {vertex}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")

    weights_fn = make_barycentric_weights_1d(vertex=vertex)
    gamma_fn, dgamma_dtau_fn = make_ctsm_variance_gamma(sigma=sigma)
    return TriangularPath1D(
        weights=weights_fn,
        gamma=gamma_fn,
        dgamma_dtau=dgamma_dtau_fn,
        eps=eps,
    )


def piecewise_sb_ctsm_path_1d(*, sigma: float = 1.0, vertex: float = 0.5, gamma_min: float = 5e-2, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """convenience constructor for v2-ctsm path (piecewise-sb weights + piecewise-sb gamma).

    piecewise-sb weights and piecewise-sb variance schedule with hard floor.

    validates hyperparameters; constructs path instance bundling weights and gamma callables.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if not (0.0 < vertex < 1.0):
        raise ValueError(f"vertex must be in (0, 1), got {vertex}")
    if gamma_min <= 0:
        raise ValueError(f"gamma_min must be > 0, got {gamma_min}")
    if inner_eps < 0:
        raise ValueError(f"inner_eps must be >= 0, got {inner_eps}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")
    if eps >= min(vertex, 1.0 - vertex):
        raise ValueError(f"eps must be < min(vertex, 1-vertex), got {eps}")
    if inner_eps + eps >= 1.0:
        raise ValueError(f"inner_eps + eps must be < 1, got {inner_eps + eps}")

    weights_fn = make_piecewise_sb_weights_1d(vertex=vertex, eps=eps)
    gamma_fn, dgamma_dtau_fn = make_piecewise_sb_gamma(
        sigma=sigma, vertex=vertex, gamma_min=gamma_min, inner_eps=inner_eps, eps=eps
    )
    return TriangularPath1D(
        weights=weights_fn,
        gamma=gamma_fn,
        dgamma_dtau=dgamma_dtau_fn,
        eps=eps,
    )


def stacked_2d_ctsm_path(*, sigma: float = 1.0, t2_max: float = 0.3, eps: float = 1e-3) -> TriangularPath2D:
    """convenience constructor for v3-ctsm path (stacked 2d weights + ctsm 2d gamma).

    stacked linear interpolant with 2d variance-style noise schedule.

    validates hyperparameters; constructs path instance bundling weights and gamma callables.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if not (0.0 < t2_max < 1.0):
        raise ValueError(f"t2_max must be in (0, 1), got {t2_max}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")

    weights_fn = make_stacked_2d_weights(t2_max=t2_max, eps=eps)
    gamma_fn, dgamma_dt1_fn, dgamma_dt2_fn = make_stacked_2d_ctsm_gamma(
        sigma=sigma, t2_max=t2_max, eps=eps
    )
    return TriangularPath2D(
        weights=weights_fn,
        gamma=gamma_fn,
        dgamma_dt1=dgamma_dt1_fn,
        dgamma_dt2=dgamma_dt2_fn,
        eps=eps,
        t2_max=t2_max,
    )
