"""pure-math atoms for path interpolants. no clamping, no eps.

core schedule and weight atoms used by path builders to compose regularized
diffusion paths. all functions are pure; derivatives are a.e. analytic with
no discontinuities in the differentiable interior. boundary behavior at
tau=0,1 or t=0,1 is unspecified and handled by builders via clamps.

weight atoms (return NamedTuple bundles from dataclass_paths):
  - bary_weights(tau, *, vertex=0.5) -> TriangularWeights1D
  - psb_legs(tau, *, vertex=0.5) -> LegSplit
  - dir_weights(tau) -> DirectWeights1D
  - stack2d_weights(t1, t2) -> TriangularWeights2D

schedule atoms (1d): var_sqrt + d_var_sqrt (sigma * sqrt(t(1-t))),
sigm_prod + d_sigm_prod ((1 - exp(-k*t))*(1 - exp(-k*(1-t)))).

schedule atoms (2d): stack2d_var family (variance sigma**2 * t1(1-t1)(1-t2))
and stack2d_stiff family (linear-stiff (1 - exp(-k*t1))(1 - exp(-k*(1-t1)))).

float32 stability: sigm_prod and stack2d_stiff families use torch.expm1 to
avoid catastrophic cancellation in 1 - exp(-k*t) near t=0.
"""

import torch
from torch import Tensor
from typing import NamedTuple

from src.waypoints.dataclass_paths import (
    DirectWeights1D, TriangularWeights1D, TriangularWeights2D,
)


class LegSplit(NamedTuple):
    """piecewise-sb leg split: mask selecting leg 1 + per-leg local-tau pair.

    when a global time tau is mapped to two legs (0 -> vertex, vertex -> 1),
    mask_leg1 picks leg 1; t_loc1 = tau/vertex and t_loc2 = (tau-vertex)/(1-vertex).
    """
    mask_leg1: Tensor
    t_loc1: Tensor
    t_loc2: Tensor


def bary_weights(tau: Tensor, *, vertex: float = 0.5) -> TriangularWeights1D:
    """asymmetric piecewise-quadratic bell weights for triangular interpolants.

    alpha = (1 - tau)(1 - h), beta = tau(1 - h), w_star = h, where h is an
    asymmetric bell with peak at tau = vertex. h is C^1 (h'(v-) = h'(v+) = 0).
      vertex == 0.5:  h = 4 tau (1 - tau)              (legacy symmetric)
      general v:      h = (tau/v)(2 - tau/v)           for tau <= v
                        = ((1-tau)/(1-v))(2 - (1-tau)/(1-v))  for tau > v

    invariants: alpha + beta + w_star = 1, d_alpha + d_beta + d_w_star = 0.
    """
    v = vertex
    if v == 0.5:
        # fast path: legacy symmetric bell
        h = 4.0 * tau * (1.0 - tau)
        h_prime = 4.0 * (1.0 - 2.0 * tau)
    else:
        # piecewise bell: u_left = tau/v, u_right = (1-tau)/(1-v)
        u_left = tau / v
        u_right = (1.0 - tau) / (1.0 - v)
        h_left = u_left * (2.0 - u_left)
        h_right = u_right * (2.0 - u_right)
        h = torch.where(tau <= v, h_left, h_right)
        # derivatives: left: (2/v)(1 - tau/v), right: -(2/(1-v))(1 - (1-tau)/(1-v))
        h_prime_left = (2.0 / v) * (1.0 - u_left)
        h_prime_right = -(2.0 / (1.0 - v)) * (1.0 - u_right)
        h_prime = torch.where(tau <= v, h_prime_left, h_prime_right)

    alpha = (1.0 - tau) * (1.0 - h)
    beta = tau * (1.0 - h)
    w_star = h

    d_alpha = -(1.0 - h) - (1.0 - tau) * h_prime
    d_beta = (1.0 - h) - tau * h_prime
    d_w_star = h_prime

    return TriangularWeights1D(
        alpha=alpha, beta=beta, w_star=w_star,
        d_alpha=d_alpha, d_beta=d_beta, d_w_star=d_w_star,
    )


def psb_legs(tau: Tensor, *, vertex: float = 0.5) -> LegSplit:
    """piecewise-sb local-time split: leg 1 (tau < vertex), leg 2 (tau >= vertex).

    leg 1 spans [0, vertex] (x0 -> xstar), leg 2 spans [vertex, 1] (xstar -> x1).
    returns raw (unclamped) local times; builders apply inner_eps clamping.
    """
    mask_leg1 = tau < vertex
    t_loc1 = tau / vertex
    t_loc2 = (tau - vertex) / (1.0 - vertex)
    return LegSplit(mask_leg1=mask_leg1, t_loc1=t_loc1, t_loc2=t_loc2)


def dir_weights(tau: Tensor) -> DirectWeights1D:
    """linear (non-singular) direct weights: alpha = 1 - tau, beta = tau.

    invariants: alpha + beta = 1, d_alpha + d_beta = 0.
    """
    alpha = 1.0 - tau
    beta = tau
    d_alpha = torch.ones_like(tau) * (-1.0)
    d_beta = torch.ones_like(tau)
    return DirectWeights1D(alpha=alpha, beta=beta, d_alpha=d_alpha, d_beta=d_beta)


def stack2d_weights(t1: Tensor, t2: Tensor) -> TriangularWeights2D:
    """stacked-interpolant 2d weights with two partials each.

    two-tier interpolation:
      I_1 = (1 - t1) x_0 + t1 x_1             (first leg)
      mu = (1 - t2) I_1 + t2 x_*              (second leg)
    so alpha = (1-t2)(1-t1), beta = (1-t2) t1, w_star = t2.
    invariants: alpha + beta + w_star = 1, sum(d*_dti) = 0 for each i.
    """
    alpha = (1.0 - t2) * (1.0 - t1)
    beta = (1.0 - t2) * t1
    w_star = t2 * torch.ones_like(alpha)

    ones_joint = torch.ones_like(alpha)
    d_alpha_dt1 = -(1.0 - t2) * ones_joint
    d_beta_dt1 = (1.0 - t2) * ones_joint
    d_w_star_dt1 = torch.zeros_like(alpha)

    d_alpha_dt2 = -(1.0 - t1) * ones_joint
    d_beta_dt2 = -t1 * ones_joint
    d_w_star_dt2 = ones_joint

    return TriangularWeights2D(
        alpha=alpha, beta=beta, w_star=w_star,
        d_alpha_dt1=d_alpha_dt1, d_beta_dt1=d_beta_dt1, d_w_star_dt1=d_w_star_dt1,
        d_alpha_dt2=d_alpha_dt2, d_beta_dt2=d_beta_dt2, d_w_star_dt2=d_w_star_dt2,
    )


def var_sqrt(tau: Tensor, *, sigma: float) -> Tensor:
    """variance-sqrt schedule: sigma * sqrt(tau (1 - tau)).

    noise amplitude for gaussian paths with variance vanishing at tau=0,1.
    """
    return sigma * torch.sqrt(tau * (1.0 - tau))


def d_var_sqrt(tau: Tensor, *, sigma: float) -> Tensor:
    """d/dtau [sigma sqrt(tau (1 - tau))] = sigma (1 - 2 tau) / (2 sqrt(tau (1 - tau)))."""
    g = tau * (1.0 - tau)
    dg_dtau = 1.0 - 2.0 * tau
    return sigma * 0.5 * dg_dtau / torch.sqrt(g)


def sigm_prod(tau: Tensor, *, k: float) -> Tensor:
    """sigmoid-product schedule: (1 - exp(-k tau))(1 - exp(-k (1 - tau))).

    smooth stochasticity vanishing at endpoints; stiffness k > 0.
    uses torch.expm1 for stability: 1 - exp(-x) = -expm1(-x).
    """
    sig1 = -torch.expm1(-k * tau)
    sig2 = -torch.expm1(-k * (1.0 - tau))
    return sig1 * sig2


def d_sigm_prod(tau: Tensor, *, k: float) -> Tensor:
    """d/dtau [(1 - exp(-k tau))(1 - exp(-k (1 - tau)))].

    with s1 = 1 - exp(-k tau), s2 = 1 - exp(-k (1 - tau)), s1' = k exp(-k tau),
    s2' = -k exp(-k (1 - tau)): derivative is s1' s2 + s1 s2'.
    """
    e1 = torch.exp(-k * tau)
    e2 = torch.exp(-k * (1.0 - tau))
    return (k * e1) * (1.0 - e2) + (1.0 - e1) * (-k * e2)


def stack2d_var(t1: Tensor, t2: Tensor, *, sigma: float) -> Tensor:
    """2d variance schedule: sigma**2 * t1(1-t1)(1-t2).

    note: returns variance (not std); builders take sqrt to get gamma.
    """
    return (sigma ** 2) * t1 * (1.0 - t1) * (1.0 - t2)


def d_stack2d_var_dt1(t1: Tensor, t2: Tensor, *, sigma: float) -> Tensor:
    """d/dt1 [sigma**2 t1(1-t1)(1-t2)] = sigma**2 (1 - 2 t1)(1 - t2)."""
    return (sigma ** 2) * (1.0 - 2.0 * t1) * (1.0 - t2)


def d_stack2d_var_dt2(t1: Tensor, t2: Tensor, *, sigma: float) -> Tensor:
    """d/dt2 [sigma**2 t1(1-t1)(1-t2)] = -sigma**2 t1(1-t1)."""
    return -(sigma ** 2) * t1 * (1.0 - t1)


def stack2d_stiff(t1: Tensor, t2: Tensor, *, k: float) -> Tensor:
    """2d linear-stiff schedule: (1 - exp(-k t1))(1 - exp(-k (1 - t1))).

    independent of t2; t2 arg kept for api uniformity.
    """
    sig1 = -torch.expm1(-k * t1)
    sig2 = -torch.expm1(-k * (1.0 - t1))
    return sig1 * sig2


def d_stack2d_stiff_dt1(t1: Tensor, t2: Tensor, *, k: float) -> Tensor:
    """d/dt1 [(1 - exp(-k t1))(1 - exp(-k (1 - t1)))] = k e1 (1 - e2) - k e2 (1 - e1)."""
    e1 = torch.exp(-k * t1)
    e2 = torch.exp(-k * (1.0 - t1))
    return k * e1 * (1.0 - e2) - k * e2 * (1.0 - e1)


def d_stack2d_stiff_dt2(t1: Tensor, t2: Tensor, *, k: float) -> Tensor:
    """d/dt2 [stack2d_stiff] = 0 (schedule has no t2 dependence)."""
    return torch.zeros_like(t2)
