"""time-weighting helpers for Hyvarinen-style losses.

mirrors the ``ProbPath.get_time_weighting_quantities`` contract from
dre-prob-paths: each helper returns the quadruple
(lam_t, lam_t0, lam_t1, lam_dt) consumed by tsm-family losses to
reweight interior and boundary terms.

the per-mode contract:
  - ``identity``: lam_t = lam_t0 = lam_t1 = 1, lam_dt = 0; recovers the
    un-reweighted Hyvarinen loss (``reweight=False`` in tsm_loss).
  - ``path_var``: variance-of-path schedule for the linear interpolant
    ``x_tau = sqrt(1 - tau^2) x_0 + tau x_1``; lam_t = 1 - tau^2,
    lam_dt = -2 tau; lam_t0/lam_t1 are evaluated at the boundary times
    (with a tiny eps^2 regularizer at t=1). matches the
    ``reweight=True`` formulas previously inlined in tsm_loss /
    tri_tsm_loss and the OneVP ``path_var`` weights in dre-prob-paths.

the future ``obj_var`` mode (parameterized by a mixing factor) is the
analog of CTSM's optimal weighting; it is path-specific and not
implemented here. paths that grow a ``get_time_weighting_quantities``
method should return their own callable matching the same signature.
"""
from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor


@dataclass(frozen=True)
class Lambdas:
    """quadruple of time-weighting scalars consumed by Hyvarinen losses.

    Attributes:
        lam_t: [B] interior weight applied to terms at sampled tau.
        lam_t0: [B] or scalar boundary weight at t = t0.
        lam_t1: [B] or scalar boundary weight at t = t1.
        lam_dt: [B] interior weight on the score-times-derivative term.
    """

    lam_t: Union[Tensor, float]
    lam_t0: Union[Tensor, float]
    lam_t1: Union[Tensor, float]
    lam_dt: Union[Tensor, float]


def identity_lambdas() -> Lambdas:
    """no-reweight quadruple (lam_t=lam_t0=lam_t1=1, lam_dt=0).

    recovers the bare Hyvarinen loss. matches the ``reweight=False``
    branch of tsm_loss and tri_tsm_loss prior to this refactor.
    """
    return Lambdas(lam_t=1.0, lam_t0=1.0, lam_t1=1.0, lam_dt=0.0)


def path_var_lambdas(
    interior_tau: Tensor,
    t0: Tensor,
    t1: Tensor,
    eps: float = 1e-5,
) -> Lambdas:
    """path-variance weighting for the linear interpolant.

    matches the ``reweight=True`` formulas previously inlined in
    tsm_loss / tri_tsm_loss; equivalent to ``OneVP.path_var`` in
    dre-prob-paths.

    Args:
        interior_tau: [B, 1] sampled time used by interior terms.
        t0: [B, 1] left-boundary time (typically eps).
        t1: [B, 1] right-boundary time (typically 1).
        eps: regularizer added to lam_t1 to avoid an exact zero at t=1.

    Returns:
        Lambdas(lam_t=1-tau^2, lam_t0=1-t0^2, lam_t1=1-t1^2+eps^2, lam_dt=-2*tau).
        all tensor fields are squeezed to [B].
    """
    lam_t = (1 - interior_tau**2).squeeze(-1)
    lam_t0 = (1 - t0**2).squeeze(-1)
    lam_t1 = (1 - t1**2 + eps**2).squeeze(-1)
    lam_dt = (-2 * interior_tau).squeeze(-1)
    return Lambdas(lam_t=lam_t, lam_t0=lam_t0, lam_t1=lam_t1, lam_dt=lam_dt)


def resolve_lambdas(
    reweight: bool,
    interior_tau: Tensor,
    t0: Tensor,
    t1: Tensor,
    eps: float = 1e-5,
) -> Lambdas:
    """dispatch on reweight: True -> path_var_lambdas, False -> identity_lambdas.

    convenience wrapper for tsm-family losses that retain the
    ``reweight: bool`` api but want to delegate lambda computation.
    """
    if reweight:
        return path_var_lambdas(interior_tau, t0, t1, eps=eps)
    return identity_lambdas()


def outer_path_var(tau: Tensor) -> Tensor:
    """per-sample outer weight that inverts the linear-path variance.

    returns ``(1 - tau^2).squeeze(-1)``; the same path_var lambda used by
    tsm/tri_tsm but exposed as a single per-sample multiplier for losses
    that apply a single outer weight rather than a (lam_t0/lam_t1/lam_dt)
    quadruple. matches the dre-prob-paths ``path_var`` mode applied at
    the loss level. composes multiplicatively with the importance weight
    (iw) from non-uniform tau sampling.

    Args:
        tau: [B, 1] sampled time.

    Returns:
        [B] per-sample variance-inverting weight ``1 - tau^2``.
    """
    return (1 - tau**2).squeeze(-1)


def resolve_outer_lambda(reweight: bool, tau: Tensor) -> Tensor:
    """[B] outer per-sample weight; ones when reweight=False, else outer_path_var.

    used by ctsm/fm/velo/denoiser losses that compose a single outer
    multiplicative lambda with their per-sample loss expression.
    """
    if reweight:
        return outer_path_var(tau)
    return torch.ones(tau.shape[0], device=tau.device, dtype=tau.dtype)


def outer_path_var_v3(
    t1: Tensor,
    t2: Tensor,
    gamma_fn,
    gamma_eps: float = 1e-3,
    normalize: bool = True,
) -> Tensor:
    """per-sample variance-inverse reweight derived from the *actual* V3 path gamma.

    replaces the 1D linear-interpolant ``outer_path_var(tau) = 1 - tau^2`` for
    V3-family methods, whose path is a 2D stacked interpolant (rect_2d). the
    correct path-variance-inverse weight is ``1/gamma^2(t1, t2)``, where
    gamma is the path's own noise schedule -- bridge_2d, stiff_2d, or any other.

    normalization (default on) divides by the batch mean so the loss scale
    matches the reweight=False baseline; this preserves the effective learning
    rate, gradient-clip threshold, and EMA decay calibration.

    Args:
        t1: [B, 1] first time axis.
        t2: [B, 1] second time axis.
        gamma_fn: callable (t1, t2) -> [B, 1] noise schedule; typically
            ``self.path.gamma``.
        gamma_eps: floor on gamma to bound 1/gamma^2 below 1/gamma_eps^2;
            keeps outer finite when the schedule vanishes.
        normalize: if True, divide outer by its batch mean so mean(outer) = 1.

    Returns:
        [B] per-sample outer weight ``1/gamma(t1, t2)^2``, optionally batch-
        normalized.
    """
    g = gamma_fn(t1, t2)  # [B, 1]
    g2 = (g * g).clamp_min(gamma_eps * gamma_eps)
    outer = (1.0 / g2).squeeze(-1)  # [B]
    if normalize:
        outer = outer / outer.mean().clamp_min(1e-12)
    return outer
