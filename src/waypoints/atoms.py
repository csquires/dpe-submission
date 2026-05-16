"""pure-math atoms for path interpolants. no clamping, no eps.

core schedule and weight atoms used by path builders to compose regularized
diffusion paths. all functions are pure; derivatives are a.e. analytic with
no discontinuities in the differentiable interior. boundary behavior at
tau=0,1 or t=0,1 is unspecified and handled by builders via clamps.

weight atoms (return NamedTuple bundles from dataclass_paths):
  - bary_weights(tau, *, vertex=0.5) -> TriangularWeights1D
  - psb_legs(tau, *, vertex=0.5) -> LegSplit
  - dir_weights(tau) -> DirectWeights1D
  - rect_weights(t1, t2) -> TriangularWeights2D

schedule atoms (1d, canonical shapes -- no sigma):
  bridge + d_bridge: sqrt(t(1-t)).
  stiff + d_stiff: (1 - exp(-k*t))(1 - exp(-k*(1-t))) (takes k).

schedule atoms (2d, canonical shapes -- no sigma):
  bridge_2d family: sqrt(t1(1-t1)(1-t2)).
  stiff_2d family: (1 - exp(-k*t1))(1 - exp(-k*(1-t1))) (takes k).

amplitude scaling (sigma and any analytic constant like sqrt(2) for
variance-preserving normalization) lives in the noise-schedule factories
in path_builders.py, not in the atoms.

float32 stability: stiff and stiff_2d use torch.expm1 to avoid
catastrophic cancellation in 1 - exp(-k*t) near t=0.
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


def rect_weights(t1: Tensor, t2: Tensor) -> TriangularWeights2D:
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


def bridge(tau: Tensor) -> Tensor:
    """canonical 1d bridge shape: sqrt(tau (1 - tau)).

    pure shape with no amplitude; the noise-schedule factory applies any
    overall scaling (e.g. sqrt(2) for vp-normalized, sigma for user scale).
    """
    return torch.sqrt(tau * (1.0 - tau))


def d_bridge(tau: Tensor) -> Tensor:
    """d/dtau [sqrt(tau (1 - tau))] = (1 - 2 tau) / (2 sqrt(tau (1 - tau)))."""
    g = tau * (1.0 - tau)
    return 0.5 * (1.0 - 2.0 * tau) / torch.sqrt(g)


def stiff(tau: Tensor, *, k: float) -> Tensor:
    """1d stiff schedule: (1 - exp(-k tau))(1 - exp(-k (1 - tau))).

    smooth, endpoint-vanishing; stiffness k > 0 controls how steeply the
    schedule saturates toward 1 in the interior. uses torch.expm1 for
    float32 stability near the endpoints.
    """
    sig1 = -torch.expm1(-k * tau)
    sig2 = -torch.expm1(-k * (1.0 - tau))
    return sig1 * sig2


def d_stiff(tau: Tensor, *, k: float) -> Tensor:
    """d/dtau [(1 - exp(-k tau))(1 - exp(-k (1 - tau)))] = k e1 (1 - e2) - k e2 (1 - e1).

    where e1 = exp(-k tau), e2 = exp(-k (1 - tau)).
    """
    e1 = torch.exp(-k * tau)
    e2 = torch.exp(-k * (1.0 - tau))
    return k * e1 * (1.0 - e2) - k * e2 * (1.0 - e1)


def bridge_2d(t1: Tensor, t2: Tensor) -> Tensor:
    """canonical 2d bridge shape: sqrt(t1 (1 - t1)(1 - t2)).

    pure shape; the schedule factory applies overall scaling.
    """
    return torch.sqrt(t1 * (1.0 - t1) * (1.0 - t2))


def d_bridge_2d_dt1(t1: Tensor, t2: Tensor) -> Tensor:
    """d/dt1 [sqrt(t1(1-t1)(1-t2))] = (1 - 2 t1)(1 - t2) / (2 sqrt(...))."""
    v = t1 * (1.0 - t1) * (1.0 - t2)
    return (1.0 - 2.0 * t1) * (1.0 - t2) / (2.0 * torch.sqrt(v))


def d_bridge_2d_dt2(t1: Tensor, t2: Tensor) -> Tensor:
    """d/dt2 [sqrt(t1(1-t1)(1-t2))] = -t1(1-t1) / (2 sqrt(...))."""
    v = t1 * (1.0 - t1) * (1.0 - t2)
    return -t1 * (1.0 - t1) / (2.0 * torch.sqrt(v))


def stiff_2d(t1: Tensor, t2: Tensor, *, k: float) -> Tensor:
    """2d linear-stiff schedule: (1 - exp(-k t1))(1 - exp(-k (1 - t1))).

    independent of t2; t2 arg kept for api uniformity.
    """
    sig1 = -torch.expm1(-k * t1)
    sig2 = -torch.expm1(-k * (1.0 - t1))
    return sig1 * sig2


def d_stiff_2d_dt1(t1: Tensor, t2: Tensor, *, k: float) -> Tensor:
    """d/dt1 [(1 - exp(-k t1))(1 - exp(-k (1 - t1)))] = k e1 (1 - e2) - k e2 (1 - e1)."""
    e1 = torch.exp(-k * t1)
    e2 = torch.exp(-k * (1.0 - t1))
    return k * e1 * (1.0 - e2) - k * e2 * (1.0 - e1)


def d_stiff_2d_dt2(t1: Tensor, t2: Tensor, *, k: float) -> Tensor:
    """d/dt2 [stiff_2d] = 0 (schedule has no t2 dependence)."""
    return torch.zeros_like(t2)
