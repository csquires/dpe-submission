"""path builders: compose atoms + clamps into frozen path dataclasses.

one composition layer over `atoms` and `dataclass_paths`. seven public builders
cover the {vfm, ctsm} x {direct, barycentric, piecewise-sb, stacked-2d} matrix
(piecewise-sb is shared between vfm and ctsm since legacy psb-vfm uses the same
variance-sqrt gamma as ctsm; the "vfm-ness" lives in the estimator, not in
gamma geometry).

each builder accepts uniform clamp kwargs:
  inner_eps >= 0  coord-clamp tau (or local_tau for psb, t1/t2 for 2d) to
                  [inner_eps, 1 - inner_eps] before evaluating gamma and (for
                  psb only) weights.
  gamma_min >= 0  pointwise lower bound on gamma; below the floor dgamma=0.

both clamps are baked in at construction via python-level branch dispatch;
the returned closures contain no runtime branches over hyperparams.
"""
from typing import Callable, Tuple

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


def _wrap_1d(g: Atom1D, dg: Atom1D, *, inner_eps: float, gamma_min: float) -> Tuple[Atom1D, Atom1D]:
    """compose 1d gamma/dgamma atoms with coord-clamp on tau and value floor.

    subgradient convention: dgamma = 0 where the coord-clamp or value floor is
    active. branches at construction; runtime closure is branch-free.
    """
    has_in = inner_eps > 0
    has_fl = gamma_min > 0
    if not has_in and not has_fl:
        return g, dg
    if has_in and not has_fl:
        lo, hi = inner_eps, 1.0 - inner_eps
        def gamma_fn(tau: Tensor) -> Tensor:
            return g(torch.clamp(tau, lo, hi))
        def dgamma_fn(tau: Tensor) -> Tensor:
            in_win = (tau < lo) | (tau > hi)
            d = dg(torch.clamp(tau, lo, hi))
            return torch.where(in_win, torch.zeros_like(d), d)
        return gamma_fn, dgamma_fn
    if not has_in and has_fl:
        def gamma_fn(tau: Tensor) -> Tensor:
            return torch.clamp_min(g(tau), gamma_min)
        def dgamma_fn(tau: Tensor) -> Tensor:
            gv = g(tau); d = dg(tau)
            return torch.where(gv < gamma_min, torch.zeros_like(d), d)
        return gamma_fn, dgamma_fn
    lo, hi = inner_eps, 1.0 - inner_eps
    def gamma_fn(tau: Tensor) -> Tensor:
        return torch.clamp_min(g(torch.clamp(tau, lo, hi)), gamma_min)
    def dgamma_fn(tau: Tensor) -> Tensor:
        in_win = (tau < lo) | (tau > hi)
        t = torch.clamp(tau, lo, hi)
        gv = g(t); d = dg(t)
        zero = torch.zeros_like(d)
        return torch.where(in_win | (gv < gamma_min), zero, d)
    return gamma_fn, dgamma_fn


def _wrap_2d(
    g: Atom2DPart, dg1: Atom2DPart, dg2: Atom2DPart,
    *, inner_eps: float, gamma_min: float,
) -> Tuple[Atom2DPart, Atom2DPart, Atom2DPart]:
    """2d analog of _wrap_1d. coord-clamp acts on t1 only (t2 is already bounded
    by t2_max in the integrator). value floor applies pointwise.
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


def vfm_direct_path(*, k: float = 0.5, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> DirectPath1D:
    """stock vfm path: linear (direct) weights + sigmoid-product gamma."""
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if inner_eps < 0 or inner_eps >= 0.5:
        raise ValueError(f"inner_eps must be in [0, 0.5), got {inner_eps}")
    if gamma_min < 0:
        raise ValueError(f"gamma_min must be >= 0, got {gamma_min}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")
    gamma_fn, dgamma_fn = _wrap_1d(
        lambda t: sigm_prod(t, k=k),
        lambda t: d_sigm_prod(t, k=k),
        inner_eps=inner_eps, gamma_min=gamma_min,
    )
    return DirectPath1D(weights=dir_weights, gamma=gamma_fn, dgamma_dtau=dgamma_fn, eps=eps)


def ctsm_direct_path(*, sigma: float = 1.0, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> DirectPath1D:
    """stock ctsm path: linear (direct) weights + sigma * sqrt(tau (1-tau)) gamma."""
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if inner_eps < 0 or inner_eps >= 0.5:
        raise ValueError(f"inner_eps must be in [0, 0.5), got {inner_eps}")
    if gamma_min < 0:
        raise ValueError(f"gamma_min must be >= 0, got {gamma_min}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")
    gamma_fn, dgamma_fn = _wrap_1d(
        lambda t: var_sqrt(t, sigma=sigma),
        lambda t: d_var_sqrt(t, sigma=sigma),
        inner_eps=inner_eps, gamma_min=gamma_min,
    )
    return DirectPath1D(weights=dir_weights, gamma=gamma_fn, dgamma_dtau=dgamma_fn, eps=eps)


def vfm_bary_path(*, k: float = 20.0, vertex: float = 0.5, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """v1 vfm path: barycentric (asymmetric bell) weights + sigmoid-product gamma."""
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    _check_1d(vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)
    weights_fn = lambda tau: bary_weights(tau, vertex=vertex)
    gamma_fn, dgamma_fn = _wrap_1d(
        lambda t: sigm_prod(t, k=k),
        lambda t: d_sigm_prod(t, k=k),
        inner_eps=inner_eps, gamma_min=gamma_min,
    )
    return TriangularPath1D(weights=weights_fn, gamma=gamma_fn, dgamma_dtau=dgamma_fn, eps=eps)


def ctsm_bary_path(*, sigma: float = 1.0, vertex: float = 0.5, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """v1 ctsm path: barycentric weights + sigma * sqrt(tau (1-tau)) gamma."""
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    _check_1d(vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)
    weights_fn = lambda tau: bary_weights(tau, vertex=vertex)
    gamma_fn, dgamma_fn = _wrap_1d(
        lambda t: var_sqrt(t, sigma=sigma),
        lambda t: d_var_sqrt(t, sigma=sigma),
        inner_eps=inner_eps, gamma_min=gamma_min,
    )
    return TriangularPath1D(weights=weights_fn, gamma=gamma_fn, dgamma_dtau=dgamma_fn, eps=eps)


def psb_path(*, sigma: float = 1.0, vertex: float = 0.5, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath1D:
    """v2 piecewise-sb path (shared vfm/ctsm).

    each leg is a schroedinger bridge with sb stochasticity gamma_leg =
    sigma * sqrt(t_local (1 - t_local)). inner_eps clamps local_tau in BOTH the
    weight closure and the gamma closure (mirrors legacy ctsm-psb's
    sample_and_target semantics).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    _check_1d(vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=eps)

    v = vertex
    lo, hi = inner_eps, 1.0 - inner_eps
    has_in = inner_eps > 0
    has_fl = gamma_min > 0

    if has_in:
        def _local_clamped(tau: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            legs = psb_legs(tau, vertex=v)
            t1 = torch.clamp(legs.t_loc1, lo, hi)
            t2 = torch.clamp(legs.t_loc2, lo, hi)
            return t1, t2, legs.mask_leg1
    else:
        def _local_clamped(tau: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            legs = psb_legs(tau, vertex=v)
            return legs.t_loc1, legs.t_loc2, legs.mask_leg1

    def weights_fn(tau: Tensor) -> TriangularWeights1D:
        t1, t2, m1 = _local_clamped(tau)
        # leg-local barycentric anchors: leg 1 mixes x0 <-> xstar (alpha,w_star),
        # leg 2 mixes xstar <-> x1 (w_star,beta). chain-rule scaling on derivatives
        # converts d/dt_local back to d/dtau (factor 1/v or 1/(1-v)).
        alpha_1 = 1.0 - t1
        w_star_1 = t1
        beta_1 = torch.zeros_like(t1)
        beta_2 = t2
        w_star_2 = 1.0 - t2
        alpha_2 = torch.zeros_like(t2)
        alpha = torch.where(m1, alpha_1, alpha_2)
        beta = torch.where(m1, beta_1, beta_2)
        w_star = torch.where(m1, w_star_1, w_star_2)

        d_alpha_1 = -1.0 / v * torch.ones_like(t1)
        d_w_star_1 = 1.0 / v * torch.ones_like(t1)
        d_beta_1 = torch.zeros_like(t1)
        d_beta_2 = 1.0 / (1.0 - v) * torch.ones_like(t2)
        d_w_star_2 = -1.0 / (1.0 - v) * torch.ones_like(t2)
        d_alpha_2 = torch.zeros_like(t2)
        d_alpha = torch.where(m1, d_alpha_1, d_alpha_2)
        d_beta = torch.where(m1, d_beta_1, d_beta_2)
        d_w_star = torch.where(m1, d_w_star_1, d_w_star_2)

        if has_in:
            raw_legs = psb_legs(tau, vertex=v)
            in_win = torch.where(
                m1,
                (raw_legs.t_loc1 < lo) | (raw_legs.t_loc1 > hi),
                (raw_legs.t_loc2 < lo) | (raw_legs.t_loc2 > hi),
            )
            zero = torch.zeros_like(d_alpha)
            d_alpha = torch.where(in_win, zero, d_alpha)
            d_beta = torch.where(in_win, zero, d_beta)
            d_w_star = torch.where(in_win, zero, d_w_star)

        return TriangularWeights1D(
            alpha=alpha, beta=beta, w_star=w_star,
            d_alpha=d_alpha, d_beta=d_beta, d_w_star=d_w_star,
        )

    def gamma_fn(tau: Tensor) -> Tensor:
        t1, t2, m1 = _local_clamped(tau)
        g = torch.where(m1, var_sqrt(t1, sigma=sigma), var_sqrt(t2, sigma=sigma))
        if has_fl:
            return torch.clamp_min(g, gamma_min)
        return g

    def dgamma_fn(tau: Tensor) -> Tensor:
        legs = psb_legs(tau, vertex=v)
        t1_raw, t2_raw, m1 = legs.t_loc1, legs.t_loc2, legs.mask_leg1
        t1 = torch.clamp(t1_raw, lo, hi) if has_in else t1_raw
        t2 = torch.clamp(t2_raw, lo, hi) if has_in else t2_raw
        d1 = d_var_sqrt(t1, sigma=sigma) / v
        d2 = d_var_sqrt(t2, sigma=sigma) / (1.0 - v)
        d = torch.where(m1, d1, d2)
        if has_in:
            in_win = torch.where(
                m1, (t1_raw < lo) | (t1_raw > hi),
                (t2_raw < lo) | (t2_raw > hi),
            )
            d = torch.where(in_win, torch.zeros_like(d), d)
        if has_fl:
            g = torch.where(m1, var_sqrt(t1, sigma=sigma), var_sqrt(t2, sigma=sigma))
            d = torch.where(g < gamma_min, torch.zeros_like(d), d)
        return d

    return TriangularPath1D(weights=weights_fn, gamma=gamma_fn, dgamma_dtau=dgamma_fn, eps=eps)


def vfm_stack2d_path(*, k: float = 20.0, t2_max: float = 0.3, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath2D:
    """v3 vfm path: stacked-2d weights + linear-stiff (sigmoid-product 2d) gamma."""
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if not (0.0 < t2_max < 1.0):
        raise ValueError(f"t2_max must be in (0, 1), got {t2_max}")
    if inner_eps < 0 or inner_eps >= 0.5:
        raise ValueError(f"inner_eps must be in [0, 0.5), got {inner_eps}")
    if gamma_min < 0:
        raise ValueError(f"gamma_min must be >= 0, got {gamma_min}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")
    gamma_fn, dg1_fn, dg2_fn = _wrap_2d(
        lambda t1, t2: stack2d_stiff(t1, t2, k=k),
        lambda t1, t2: d_stack2d_stiff_dt1(t1, t2, k=k),
        lambda t1, t2: torch.zeros_like(t2),
        inner_eps=inner_eps, gamma_min=gamma_min,
    )
    return TriangularPath2D(
        weights=stack2d_weights, gamma=gamma_fn,
        dgamma_dt1=dg1_fn, dgamma_dt2=dg2_fn,
        eps=eps, t2_max=t2_max,
    )


def ctsm_stack2d_path(*, sigma: float = 1.0, t2_max: float = 0.3, gamma_min: float = 0.0, inner_eps: float = 0.0, eps: float = 1e-3) -> TriangularPath2D:
    """v3 ctsm path: stacked-2d weights + sigma * sqrt(t1(1-t1)(1-t2)) gamma.

    note: stack2d_var atom returns variance (sigma**2 * g); we wrap with sqrt
    so the path's gamma callable returns std, matching the 1d ctsm convention.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if not (0.0 < t2_max < 1.0):
        raise ValueError(f"t2_max must be in (0, 1), got {t2_max}")
    if inner_eps < 0 or inner_eps >= 0.5:
        raise ValueError(f"inner_eps must be in [0, 0.5), got {inner_eps}")
    if gamma_min < 0:
        raise ValueError(f"gamma_min must be >= 0, got {gamma_min}")
    if eps < 1e-3:
        raise ValueError(f"eps must be >= 1e-3, got {eps}")

    tiny = 1e-12
    def _g_std(t1, t2):
        return torch.sqrt(stack2d_var(t1, t2, sigma=sigma) + tiny)
    def _dg_dt1(t1, t2):
        g = _g_std(t1, t2)
        return d_stack2d_var_dt1(t1, t2, sigma=sigma) / (2.0 * g)
    def _dg_dt2(t1, t2):
        g = _g_std(t1, t2)
        return d_stack2d_var_dt2(t1, t2, sigma=sigma) / (2.0 * g)

    gamma_fn, dg1_fn, dg2_fn = _wrap_2d(
        _g_std, _dg_dt1, _dg_dt2,
        inner_eps=inner_eps, gamma_min=gamma_min,
    )
    return TriangularPath2D(
        weights=stack2d_weights, gamma=gamma_fn,
        dgamma_dt1=dg1_fn, dgamma_dt2=dg2_fn,
        eps=eps, t2_max=t2_max,
    )
