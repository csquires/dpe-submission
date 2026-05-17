"""path builders: compose atoms + clamps + schedules into frozen path dataclasses.

two layers:

  layer 1 (noise schedules) -- factories returning a `Sched1D` or `Sched2D`
    NamedTuple of gamma/dgamma callables baked with their params:
      1d:  bridge_noise(sigma), stiff_noise(k), vp_bary_noise(vertex)
      2d:  bridge_noise_2d(sigma), stiff_noise_2d(k), vp_noise_2d()
    note: vp for a linear segment (direct/psb per leg) is exactly
    bridge_noise(sigma) -- sqrt(2) is baked into bridge_noise.

  layer 2 (paths) -- general path constructors per *weights* family, each
    accepting a `sched=` of the matching dimension and uniform clamp kwargs
    `inner_eps` and `gamma_min`:
      direct_1d(*, sched, ...)         -> DirectPath1D
      bary_1d(*, sched, vertex, ...)   -> TriangularPath1D
      psb_1d(*, sched, vertex, ...)    -> TriangularPath1D
      rect_2d(*, sched, ...)           -> TriangularPath2D

named legacy pairs (bary_vfm, bary_ctsm, psb, etc.) are one-line wrappers
preserved for hpo configs and existing call sites.

clamp semantics:
  inner_eps >= 0  coord-clamp tau (or local_tau for psb, t1 for 2d) to
                  [inner_eps, 1 - inner_eps] before evaluating gamma and (for
                  psb only) weights.
  gamma_min >= 0  pointwise lower bound on gamma; below the floor dgamma=0.

clamps bake in at construction via python-level branch dispatch; runtime
closures contain no branches over hyperparams.
"""
from math import sqrt
from typing import Callable, NamedTuple, Tuple

import torch
from torch import Tensor

from src.waypoints.atoms import (
    bary_weights, dir_weights, rect_weights, psb_legs,
    bridge, d_bridge, stiff, d_stiff,
    bridge_2d, d_bridge_2d_dt1, d_bridge_2d_dt2,
    stiff_2d, d_stiff_2d_dt1,
)
from src.waypoints.dataclass_paths import (
    DirectPath1D, TriangularPath1D, TriangularPath2D, TriangularWeights1D,
)

Atom1D = Callable[[Tensor], Tensor]
Atom2DPart = Callable[[Tensor, Tensor], Tensor]


# ============================================================================
# schedule namedtuples + factories
# ============================================================================


class Sched1D(NamedTuple):
    """1d noise schedule: gamma(tau) and dgamma/dtau as bound callables."""
    gamma: Atom1D
    dgamma: Atom1D


class Sched2D(NamedTuple):
    """2d noise schedule: gamma(t1, t2) and its two partials as bound callables."""
    gamma: Atom2DPart
    dgamma_dt1: Atom2DPart
    dgamma_dt2: Atom2DPart


# every noise factory accepts sigma=1.0 (the user-facing amplitude knob).
# bridge factories also bake in the analytic sqrt(2) so that sigma=1.0
# corresponds to vp normalization for a single linear segment between two
# unit-variance endpoints. vp factories embed the per-family analytic gamma
# and accept sigma the same way (sigma=1.0 is the strict vp point).
_VP_SIGMA = sqrt(2.0)


def bridge_noise(sigma: float = 1.0) -> Sched1D:
    """1d bridge schedule: gamma(tau) = sigma * sqrt(2 tau (1 - tau)).

    sqrt(2) is baked in so sigma=1 corresponds to vp normalization (var(mu) +
    gamma^2 = 1 for direct/psb linear segments). sigma scales multiplicatively.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    scale = sigma * _VP_SIGMA
    return Sched1D(
        gamma=lambda t: scale * bridge(t),
        dgamma=lambda t: scale * d_bridge(t),
    )


def stiff_noise(k: float, sigma: float = 1.0) -> Sched1D:
    """1d stiff schedule: gamma(tau) = sigma * (1 - exp(-k tau))(1 - exp(-k (1 - tau)))."""
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    return Sched1D(
        gamma=lambda t: sigma * stiff(t, k=k),
        dgamma=lambda t: sigma * d_stiff(t, k=k),
    )


def bridge_noise_2d(sigma: float = 1.0) -> Sched2D:
    """2d bridge schedule: gamma(t1, t2) = sigma * sqrt(2 t1 (1 - t1)(1 - t2))."""
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    scale = sigma * _VP_SIGMA
    return Sched2D(
        gamma=lambda t1, t2: scale * bridge_2d(t1, t2),
        dgamma_dt1=lambda t1, t2: scale * d_bridge_2d_dt1(t1, t2),
        dgamma_dt2=lambda t1, t2: scale * d_bridge_2d_dt2(t1, t2),
    )


def stiff_noise_2d(k: float, sigma: float = 1.0) -> Sched2D:
    """2d stiff schedule: gamma(t1, t2) = sigma * (1 - exp(-k t1))(1 - exp(-k (1 - t1)))."""
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    return Sched2D(
        gamma=lambda t1, t2: sigma * stiff_2d(t1, t2, k=k),
        dgamma_dt1=lambda t1, t2: sigma * d_stiff_2d_dt1(t1, t2, k=k),
        dgamma_dt2=lambda t1, t2: torch.zeros_like(t2),
    )


# ----- variance-preserving (VP) schedules -----
# vp means Var(mu) + gamma^2 = 1 at every interior tau (or (t1, t2)), assuming
# x_0, x_1, x_* independent unit-variance, AT sigma=1.0. for sigma != 1, the
# returned schedule is sigma * gamma_vp (no longer strictly vp, but the
# natural uniform scaling).


def vp_bary_noise(vertex: float = 0.5, sigma: float = 1.0) -> Sched1D:
    """vp schedule for bary_1d at matching vertex.

    at sigma=1: gamma^2(tau) = 2 (1 - h) [h + (1 - h) tau (1 - tau)] where
    h(tau, vertex) is the asymmetric bell from bary_weights. sigma scales
    multiplicatively. caller must pass the same vertex to both this factory
    and bary_1d.
    """
    if not (0.0 < vertex < 1.0):
        raise ValueError(f"vertex must be in (0, 1), got {vertex}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    def _bell(tau):
        v = vertex
        if v == 0.5:
            return 4.0 * tau * (1.0 - tau)
        u_left = tau / v
        u_right = (1.0 - tau) / (1.0 - v)
        return torch.where(
            tau <= v,
            u_left * (2.0 - u_left),
            u_right * (2.0 - u_right),
        )

    def _bell_and_deriv(tau):
        v = vertex
        if v == 0.5:
            return 4.0 * tau * (1.0 - tau), 4.0 * (1.0 - 2.0 * tau)
        u_left = tau / v
        u_right = (1.0 - tau) / (1.0 - v)
        h = torch.where(
            tau <= v,
            u_left * (2.0 - u_left),
            u_right * (2.0 - u_right),
        )
        dh = torch.where(
            tau <= v,
            (2.0 / v) * (1.0 - u_left),
            -(2.0 / (1.0 - v)) * (1.0 - u_right),
        )
        return h, dh

    tiny = 1e-12

    def g(tau):
        h = _bell(tau)
        gsq = 2.0 * (1.0 - h) * (h + (1.0 - h) * tau * (1.0 - tau))
        return sigma * torch.sqrt(gsq.clamp_min(tiny))

    def dg(tau):
        h, dh = _bell_and_deriv(tau)
        u = 1.0 - h
        v_ = tau * (1.0 - tau)
        dv = 1.0 - 2.0 * tau
        w = h + u * v_
        dw = dh * (1.0 - v_) + u * dv
        dgsq = 2.0 * (-dh * w + u * dw)
        gamma = torch.sqrt((2.0 * u * w).clamp_min(tiny))
        return sigma * dgsq / (2.0 * gamma)

    return Sched1D(gamma=g, dgamma=dg)


def vp_noise_2d(sigma: float = 1.0) -> Sched2D:
    """vp schedule for rect_2d.

    at sigma=1: gamma^2(t1, t2) = 2 (1 - t2) [t1 (1 - t1)(1 - t2) + t2].
    sigma scales multiplicatively.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    tiny = 1e-12

    def g(t1, t2):
        gsq = 2.0 * (1.0 - t2) * (t1 * (1.0 - t1) * (1.0 - t2) + t2)
        return sigma * torch.sqrt(gsq.clamp_min(tiny))

    def dg_dt1(t1, t2):
        return sigma * (1.0 - t2) ** 2 * (1.0 - 2.0 * t1) / (g(t1, t2) / sigma)

    def dg_dt2(t1, t2):
        gsq_dt2 = -2.0 * (t1 * (1.0 - t1) * (1.0 - t2) + t2) \
                  + 2.0 * (1.0 - t2) * (1.0 - t1 * (1.0 - t1))
        return sigma * gsq_dt2 / (2.0 * (g(t1, t2) / sigma))

    return Sched2D(gamma=g, dgamma_dt1=dg_dt1, dgamma_dt2=dg_dt2)


# ============================================================================
# clamp composition helpers
# ============================================================================


def _wrap_1d(g: Atom1D, dg: Atom1D, *, inner_eps: float, gamma_min: float) -> Tuple[Atom1D, Atom1D]:
    """compose 1d gamma/dgamma with coord-clamp on tau and value floor.

    subgradient: dgamma = 0 where coord-clamp or value floor is active.
    branches at construction; runtime closure is branch-free.
    """
    has_in = inner_eps > 0
    has_fl = gamma_min > 0
    if not has_in and not has_fl:
        return g, dg
    if has_in and not has_fl:
        lo, hi = inner_eps, 1.0 - inner_eps
        def gamma_fn(tau):
            return g(torch.clamp(tau, lo, hi))
        def dgamma_fn(tau):
            in_win = (tau < lo) | (tau > hi)
            d = dg(torch.clamp(tau, lo, hi))
            return torch.where(in_win, torch.zeros_like(d), d)
        return gamma_fn, dgamma_fn
    if not has_in and has_fl:
        def gamma_fn(tau):
            return torch.clamp_min(g(tau), gamma_min)
        def dgamma_fn(tau):
            gv = g(tau); d = dg(tau)
            return torch.where(gv < gamma_min, torch.zeros_like(d), d)
        return gamma_fn, dgamma_fn
    lo, hi = inner_eps, 1.0 - inner_eps
    def gamma_fn(tau):
        return torch.clamp_min(g(torch.clamp(tau, lo, hi)), gamma_min)
    def dgamma_fn(tau):
        in_win = (tau < lo) | (tau > hi)
        t = torch.clamp(tau, lo, hi)
        gv = g(t); d = dg(t)
        return torch.where(in_win | (gv < gamma_min), torch.zeros_like(d), d)
    return gamma_fn, dgamma_fn


def _wrap_2d(
    g: Atom2DPart, dg1: Atom2DPart, dg2: Atom2DPart,
    *, inner_eps: float, gamma_min: float,
) -> Tuple[Atom2DPart, Atom2DPart, Atom2DPart]:
    """2d analog of _wrap_1d. coord-clamp on t1 only; value floor pointwise."""
    has_in = inner_eps > 0
    has_fl = gamma_min > 0
    if not has_in and not has_fl:
        return g, dg1, dg2
    lo, hi = inner_eps, 1.0 - inner_eps
    if has_in and not has_fl:
        def gamma_fn(t1, t2): return g(torch.clamp(t1, lo, hi), t2)
        def dg1_fn(t1, t2):
            in_win = (t1 < lo) | (t1 > hi)
            d = dg1(torch.clamp(t1, lo, hi), t2)
            return torch.where(in_win, torch.zeros_like(d), d)
        def dg2_fn(t1, t2):
            in_win = (t1 < lo) | (t1 > hi)
            d = dg2(torch.clamp(t1, lo, hi), t2)
            return torch.where(in_win, torch.zeros_like(d), d)
        return gamma_fn, dg1_fn, dg2_fn
    if not has_in and has_fl:
        def gamma_fn(t1, t2): return torch.clamp_min(g(t1, t2), gamma_min)
        def dg1_fn(t1, t2):
            gv = g(t1, t2); d = dg1(t1, t2)
            return torch.where(gv < gamma_min, torch.zeros_like(d), d)
        def dg2_fn(t1, t2):
            gv = g(t1, t2); d = dg2(t1, t2)
            return torch.where(gv < gamma_min, torch.zeros_like(d), d)
        return gamma_fn, dg1_fn, dg2_fn
    def gamma_fn(t1, t2):
        return torch.clamp_min(g(torch.clamp(t1, lo, hi), t2), gamma_min)
    def dg1_fn(t1, t2):
        in_win = (t1 < lo) | (t1 > hi)
        t = torch.clamp(t1, lo, hi)
        gv = g(t, t2); d = dg1(t, t2)
        return torch.where(in_win | (gv < gamma_min), torch.zeros_like(d), d)
    def dg2_fn(t1, t2):
        in_win = (t1 < lo) | (t1 > hi)
        t = torch.clamp(t1, lo, hi)
        gv = g(t, t2); d = dg2(t, t2)
        return torch.where(in_win | (gv < gamma_min), torch.zeros_like(d), d)
    return gamma_fn, dg1_fn, dg2_fn


def _check_1d(*, vertex: float, inner_eps: float, gamma_min: float, eps: float) -> None:
    """common 1d hyperparam guards."""
    if not (0.0 < vertex < 1.0):
        raise ValueError(f"vertex must be in (0, 1), got {vertex}")
    if inner_eps < 0 or inner_eps >= 0.5:
        raise ValueError(f"inner_eps must be in [0, 0.5), got {inner_eps}")
    if gamma_min < 0:
        raise ValueError(f"gamma_min must be >= 0, got {gamma_min}")
    if eps >= min(vertex, 1.0 - vertex):
        raise ValueError(f"eps must be < min(vertex, 1-vertex), got {eps}")
    if inner_eps + eps >= 1.0:
        raise ValueError(f"inner_eps + eps must be < 1, got {inner_eps + eps}")


# ============================================================================
# general path builders: any weights family x any compatible schedule
# ============================================================================


def direct_1d(*, sched: Sched1D, inner_eps: float = 0.0, gamma_min: float = 0.0, eps: float = 1e-3) -> DirectPath1D:
    """linear (no-anchor) weights + user-supplied 1d schedule."""
    if inner_eps < 0 or inner_eps >= 0.5:
        raise ValueError(f"inner_eps must be in [0, 0.5), got {inner_eps}")
    if gamma_min < 0:
        raise ValueError(f"gamma_min must be >= 0, got {gamma_min}")
    gamma_fn, dgamma_fn = _wrap_1d(sched.gamma, sched.dgamma, inner_eps=inner_eps, gamma_min=gamma_min)
    return DirectPath1D(weights=dir_weights, gamma=gamma_fn, dgamma_dtau=dgamma_fn, eps=eps)


def bary_1d(*, sched: Sched1D, vertex: float = 0.5, inner_eps: float = 0.0, gamma_min: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """asymmetric-bell barycentric weights + user-supplied 1d schedule."""
    _check_1d(vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)
    weights_fn = lambda tau: bary_weights(tau, vertex=vertex)
    gamma_fn, dgamma_fn = _wrap_1d(sched.gamma, sched.dgamma, inner_eps=inner_eps, gamma_min=gamma_min)
    return TriangularPath1D(weights=weights_fn, gamma=gamma_fn, dgamma_dtau=dgamma_fn, eps=eps)


def psb_1d(*, sched: Sched1D, vertex: float = 0.5, inner_eps: float = 0.0, gamma_min: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """piecewise-sb weights (two legs at tau=vertex) + user-supplied 1d schedule.

    schedule is applied per-leg on the local_tau. inner_eps clamps the local
    time in both the weight closure (zeroing derivatives inside the clamp
    window) and the gamma closure. dgamma is chain-rule scaled by 1/vertex
    (leg 1) or 1/(1-vertex) (leg 2).
    """
    _check_1d(vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)

    v = vertex
    lo, hi = inner_eps, 1.0 - inner_eps
    has_in = inner_eps > 0
    has_fl = gamma_min > 0
    g_atom, dg_atom = sched.gamma, sched.dgamma

    if has_in:
        def _local_clamped(tau):
            legs = psb_legs(tau, vertex=v)
            return torch.clamp(legs.t_loc1, lo, hi), torch.clamp(legs.t_loc2, lo, hi), legs.mask_leg1
    else:
        def _local_clamped(tau):
            legs = psb_legs(tau, vertex=v)
            return legs.t_loc1, legs.t_loc2, legs.mask_leg1

    def weights_fn(tau):
        t1, t2, m1 = _local_clamped(tau)
        alpha = torch.where(m1, 1.0 - t1, torch.zeros_like(t2))
        beta = torch.where(m1, torch.zeros_like(t1), t2)
        w_star = torch.where(m1, t1, 1.0 - t2)

        d_alpha_1 = (-1.0 / v) * torch.ones_like(t1)
        d_w_star_1 = (1.0 / v) * torch.ones_like(t1)
        d_beta_2 = (1.0 / (1.0 - v)) * torch.ones_like(t2)
        d_w_star_2 = (-1.0 / (1.0 - v)) * torch.ones_like(t2)
        d_alpha = torch.where(m1, d_alpha_1, torch.zeros_like(t2))
        d_beta = torch.where(m1, torch.zeros_like(t1), d_beta_2)
        d_w_star = torch.where(m1, d_w_star_1, d_w_star_2)

        if has_in:
            raw = psb_legs(tau, vertex=v)
            in_win = torch.where(
                m1, (raw.t_loc1 < lo) | (raw.t_loc1 > hi),
                (raw.t_loc2 < lo) | (raw.t_loc2 > hi),
            )
            zero = torch.zeros_like(d_alpha)
            d_alpha = torch.where(in_win, zero, d_alpha)
            d_beta = torch.where(in_win, zero, d_beta)
            d_w_star = torch.where(in_win, zero, d_w_star)

        return TriangularWeights1D(
            alpha=alpha, beta=beta, w_star=w_star,
            d_alpha=d_alpha, d_beta=d_beta, d_w_star=d_w_star,
        )

    def gamma_fn(tau):
        t1, t2, m1 = _local_clamped(tau)
        g = torch.where(m1, g_atom(t1), g_atom(t2))
        return torch.clamp_min(g, gamma_min) if has_fl else g

    def dgamma_fn(tau):
        legs = psb_legs(tau, vertex=v)
        t1_raw, t2_raw, m1 = legs.t_loc1, legs.t_loc2, legs.mask_leg1
        t1 = torch.clamp(t1_raw, lo, hi) if has_in else t1_raw
        t2 = torch.clamp(t2_raw, lo, hi) if has_in else t2_raw
        d = torch.where(m1, dg_atom(t1) / v, dg_atom(t2) / (1.0 - v))
        if has_in:
            in_win = torch.where(
                m1, (t1_raw < lo) | (t1_raw > hi),
                (t2_raw < lo) | (t2_raw > hi),
            )
            d = torch.where(in_win, torch.zeros_like(d), d)
        if has_fl:
            g = torch.where(m1, g_atom(t1), g_atom(t2))
            d = torch.where(g < gamma_min, torch.zeros_like(d), d)
        return d

    return TriangularPath1D(weights=weights_fn, gamma=gamma_fn, dgamma_dtau=dgamma_fn, eps=eps)


def rect_2d(*, sched: Sched2D, inner_eps: float = 0.0, gamma_min: float = 0.0, eps: float = 1e-3) -> TriangularPath2D:
    """stacked-2d rectangular weights + user-supplied 2d schedule.

    pure geometry on the open square (t1, t2) in (0, 1)^2; the t2-integration
    bound (t2_max < 1) lives on the sampler / curve / estimator.
    """
    if inner_eps < 0 or inner_eps >= 0.5:
        raise ValueError(f"inner_eps must be in [0, 0.5), got {inner_eps}")
    if gamma_min < 0:
        raise ValueError(f"gamma_min must be >= 0, got {gamma_min}")
    gamma_fn, dg1_fn, dg2_fn = _wrap_2d(
        sched.gamma, sched.dgamma_dt1, sched.dgamma_dt2,
        inner_eps=inner_eps, gamma_min=gamma_min,
    )
    return TriangularPath2D(
        weights=rect_weights, gamma=gamma_fn,
        dgamma_dt1=dg1_fn, dgamma_dt2=dg2_fn,
        eps=eps,
    )


# ============================================================================
# named legacy pairs -- thin wrappers for hpo configs + existing call sites
# ============================================================================


def direct_vfm(*, k: float = 0.5, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> DirectPath1D:
    """stock vfm: direct weights + stiff (sigmoid-product) gamma."""
    return direct_1d(sched=stiff_noise(k), inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)


def direct_ctsm(*, sigma: float = 1.0, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> DirectPath1D:
    """stock ctsm: direct weights + bridge (variance-sqrt) gamma."""
    return direct_1d(sched=bridge_noise(sigma), inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)


def bary_vfm(*, k: float = 20.0, vertex: float = 0.5, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """v1 vfm: barycentric weights + stiff gamma."""
    return bary_1d(sched=stiff_noise(k), vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)


def bary_ctsm(*, sigma: float = 1.0, vertex: float = 0.5, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """v1 ctsm: barycentric weights + bridge gamma."""
    return bary_1d(sched=bridge_noise(sigma), vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)


def psb(*, sigma: float = 1.0, vertex: float = 0.5, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """v2 piecewise-sb: psb weights + bridge gamma per leg.

    legacy psb-vfm and ctsm both use this same gamma; the vfm-ness lives in
    the estimator (velocity + denoiser networks), not in path geometry.
    """
    return psb_1d(sched=bridge_noise(sigma), vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)


def rect_vfm(*, k: float = 20.0, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath2D:
    """v3 vfm: rect weights + 2d stiff gamma."""
    return rect_2d(sched=stiff_noise_2d(k), inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)


def rect_ctsm(*, sigma: float = 1.0, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath2D:
    """v3 ctsm: rect weights + 2d bridge gamma."""
    return rect_2d(sched=bridge_noise_2d(sigma), inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)
