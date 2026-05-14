"""path builders: compose atoms + clamps + schedules into frozen path dataclasses.

two layers:

  layer 1 (schedules) -- named factories returning a `Sched1D` or `Sched2D`
    NamedTuple of gamma/dgamma callables baked with their scalar params:
      sched_var_sqrt(sigma)       -> Sched1D
      sched_sigm_prod(k)          -> Sched1D
      sched_stack2d_var(sigma)    -> Sched2D
      sched_stack2d_stiff(k)      -> Sched2D

  layer 2 (paths) -- general path constructors per *weights* family, each
    accepting a `sched=` of the matching dimension and uniform clamp kwargs
    `inner_eps` and `gamma_min`:
      bary_path_1d(*, sched, vertex, ...)    -> TriangularPath1D
      psb_path_1d(*, sched, vertex, ...)     -> TriangularPath1D
      direct_path_1d(*, sched, ...)          -> DirectPath1D
      stack2d_path(*, sched, t2_max, ...)    -> TriangularPath2D

named legacy pairs (vfm_bary_path, ctsm_bary_path, psb_path, etc.) are
one-line wrappers preserved for hpo configs and existing call sites.

clamp semantics:
  inner_eps >= 0  coord-clamp tau (or local_tau for psb, t1 for 2d) to
                  [inner_eps, 1 - inner_eps] before evaluating gamma and (for
                  psb only) weights.
  gamma_min >= 0  pointwise lower bound on gamma; below the floor dgamma=0.

clamps bake in at construction via python-level branch dispatch; runtime
closures contain no branches over hyperparams.
"""
from typing import Callable, NamedTuple, Tuple

import torch
from torch import Tensor

from src.waypoints.atoms import (
    bary_weights, dir_weights, stack2d_weights, psb_legs,
    var_sqrt, d_var_sqrt, sigm_prod, d_sigm_prod,
    stack2d_var, d_stack2d_var_dt1, d_stack2d_var_dt2,
    stack2d_stiff, d_stack2d_stiff_dt1,
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


def sched_var_sqrt(sigma: float) -> Sched1D:
    """gamma(tau) = sigma * sqrt(tau (1 - tau)) (ctsm-style)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    return Sched1D(
        gamma=lambda t: var_sqrt(t, sigma=sigma),
        dgamma=lambda t: d_var_sqrt(t, sigma=sigma),
    )


def sched_sigm_prod(k: float) -> Sched1D:
    """gamma(tau) = (1 - exp(-k tau))(1 - exp(-k (1 - tau))) (vfm-style)."""
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    return Sched1D(
        gamma=lambda t: sigm_prod(t, k=k),
        dgamma=lambda t: d_sigm_prod(t, k=k),
    )


def sched_stack2d_var(sigma: float) -> Sched2D:
    """gamma(t1, t2) = sigma * sqrt(t1 (1-t1)(1-t2)) (2d ctsm-style).

    wraps the variance atom with sqrt and chain-rules its partials; the path
    layer's gamma callable returns std, matching the 1d ctsm convention.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    tiny = 1e-12
    def g(t1, t2):
        return torch.sqrt(stack2d_var(t1, t2, sigma=sigma) + tiny)
    def dg1(t1, t2):
        return d_stack2d_var_dt1(t1, t2, sigma=sigma) / (2.0 * g(t1, t2))
    def dg2(t1, t2):
        return d_stack2d_var_dt2(t1, t2, sigma=sigma) / (2.0 * g(t1, t2))
    return Sched2D(gamma=g, dgamma_dt1=dg1, dgamma_dt2=dg2)


def sched_stack2d_stiff(k: float) -> Sched2D:
    """gamma(t1, t2) = (1 - exp(-k t1))(1 - exp(-k (1 - t1))) (2d vfm-style, no t2 dep)."""
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    return Sched2D(
        gamma=lambda t1, t2: stack2d_stiff(t1, t2, k=k),
        dgamma_dt1=lambda t1, t2: d_stack2d_stiff_dt1(t1, t2, k=k),
        dgamma_dt2=lambda t1, t2: torch.zeros_like(t2),
    )


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
    """2d analog of _wrap_1d. coord-clamp acts on t1 only (t2 is already
    bounded by t2_max in the integrator). value floor applies pointwise.
    """
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
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")
    if eps >= min(vertex, 1.0 - vertex):
        raise ValueError(f"eps must be < min(vertex, 1-vertex), got {eps}")
    if inner_eps + eps >= 1.0:
        raise ValueError(f"inner_eps + eps must be < 1, got {inner_eps + eps}")


# ============================================================================
# general path builders: any weights family x any compatible schedule
# ============================================================================


def direct_path_1d(*, sched: Sched1D, inner_eps: float = 0.0, gamma_min: float = 0.0, eps: float = 1e-3) -> DirectPath1D:
    """linear (no-anchor) weights + user-supplied 1d schedule."""
    if inner_eps < 0 or inner_eps >= 0.5:
        raise ValueError(f"inner_eps must be in [0, 0.5), got {inner_eps}")
    if gamma_min < 0:
        raise ValueError(f"gamma_min must be >= 0, got {gamma_min}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")
    gamma_fn, dgamma_fn = _wrap_1d(sched.gamma, sched.dgamma, inner_eps=inner_eps, gamma_min=gamma_min)
    return DirectPath1D(weights=dir_weights, gamma=gamma_fn, dgamma_dtau=dgamma_fn, eps=eps)


def bary_path_1d(*, sched: Sched1D, vertex: float = 0.5, inner_eps: float = 0.0, gamma_min: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """asymmetric-bell barycentric weights + user-supplied 1d schedule."""
    _check_1d(vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)
    weights_fn = lambda tau: bary_weights(tau, vertex=vertex)
    gamma_fn, dgamma_fn = _wrap_1d(sched.gamma, sched.dgamma, inner_eps=inner_eps, gamma_min=gamma_min)
    return TriangularPath1D(weights=weights_fn, gamma=gamma_fn, dgamma_dtau=dgamma_fn, eps=eps)


def psb_path_1d(*, sched: Sched1D, vertex: float = 0.5, inner_eps: float = 0.0, gamma_min: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """piecewise-sb weights (two legs at tau=vertex) + user-supplied 1d schedule
    applied per-leg on local_tau.

    inner_eps clamps the per-leg local time in both the weight closure (zeroing
    derivatives inside the clamp window) and the gamma closure. dgamma is
    chain-rule scaled by 1/vertex (leg 1) or 1/(1-vertex) (leg 2).
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

        # chain-rule: d/dtau = (1/v) d/dt_loc1 on leg 1, (1/(1-v)) d/dt_loc2 on leg 2.
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


def stack2d_path(*, sched: Sched2D, t2_max: float = 0.3, inner_eps: float = 0.0, gamma_min: float = 0.0, eps: float = 1e-3) -> TriangularPath2D:
    """stacked-2d weights + user-supplied 2d schedule."""
    if not (0.0 < t2_max < 1.0):
        raise ValueError(f"t2_max must be in (0, 1), got {t2_max}")
    if inner_eps < 0 or inner_eps >= 0.5:
        raise ValueError(f"inner_eps must be in [0, 0.5), got {inner_eps}")
    if gamma_min < 0:
        raise ValueError(f"gamma_min must be >= 0, got {gamma_min}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")
    gamma_fn, dg1_fn, dg2_fn = _wrap_2d(
        sched.gamma, sched.dgamma_dt1, sched.dgamma_dt2,
        inner_eps=inner_eps, gamma_min=gamma_min,
    )
    return TriangularPath2D(
        weights=stack2d_weights, gamma=gamma_fn,
        dgamma_dt1=dg1_fn, dgamma_dt2=dg2_fn,
        eps=eps, t2_max=t2_max,
    )


# ============================================================================
# named legacy pairs -- thin wrappers preserved for hpo configs + existing call sites
# ============================================================================


def vfm_direct_path(*, k: float = 0.5, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> DirectPath1D:
    """stock vfm: direct weights + sigmoid-product gamma."""
    return direct_path_1d(sched=sched_sigm_prod(k), inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)


def ctsm_direct_path(*, sigma: float = 1.0, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> DirectPath1D:
    """stock ctsm: direct weights + variance-sqrt gamma."""
    return direct_path_1d(sched=sched_var_sqrt(sigma), inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)


def vfm_bary_path(*, k: float = 20.0, vertex: float = 0.5, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """v1 vfm: barycentric weights + sigmoid-product gamma."""
    return bary_path_1d(sched=sched_sigm_prod(k), vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)


def ctsm_bary_path(*, sigma: float = 1.0, vertex: float = 0.5, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """v1 ctsm: barycentric weights + variance-sqrt gamma."""
    return bary_path_1d(sched=sched_var_sqrt(sigma), vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)


def psb_path(*, sigma: float = 1.0, vertex: float = 0.5, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """v2 piecewise-sb: psb weights + variance-sqrt gamma per leg.

    legacy psb-vfm and ctsm both use this same gamma; the vfm-ness lives in
    the estimator (velocity + denoiser networks), not in path geometry.
    """
    return psb_path_1d(sched=sched_var_sqrt(sigma), vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)


def vfm_stack2d_path(*, k: float = 20.0, t2_max: float = 0.3, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath2D:
    """v3 vfm: stacked-2d weights + linear-stiff gamma."""
    return stack2d_path(sched=sched_stack2d_stiff(k), t2_max=t2_max, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)


def ctsm_stack2d_path(*, sigma: float = 1.0, t2_max: float = 0.3, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath2D:
    """v3 ctsm: stacked-2d weights + variance-sqrt 2d gamma."""
    return stack2d_path(sched=sched_stack2d_var(sigma), t2_max=t2_max, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)
