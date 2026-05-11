"""pure loss functions for density-ratio estimators (TSM, CTSM, FM, VFM).

each loss exposes two module-level attributes used by `_trainer.train_loop`:
  - `required_keys`: subset of {"x0", "x1", "xstar"} the trainer must populate
  - `requires_tau_grad`: whether the loss enables autograd on tau internally
"""
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.waypoints.path_1d import Path1D

from src.waypoints.sb_bridge import sb_target


def _score_dt(
    model_call: Callable,
    x: torch.Tensor,
    tau: torch.Tensor,
    *extra: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """return (score, d score / d tau) with create_graph=True.

    tau_g = tau.clone().detach().requires_grad_(True)
    score = model_call(x, tau_g, *extra)
    dscore = autograd.grad(score.sum(), tau_g, create_graph=True)[0]
    """
    tau_g = tau.clone().detach().requires_grad_(True)
    score = model_call(x, tau_g, *extra)
    dscore = autograd.grad(score.sum(), tau_g, create_graph=True)[0]
    return score, dscore


def tsm_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    reweight: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Hyvarinen time-score-matching loss on the path x_tau = sqrt(1-tau^2) x0 + tau x1.

    loss = mean(term1 - term2 + term3 + term4 + term5) with
      term1 = 2 model(x0, eps) \\lambda_{t0}
      term2 = 2 model(x1, 1)   \\lambda_{t1}
      term3 = 2 d_tau model(x_tau, tau) \\lambda_t
      term4 = model(x_tau, tau) \\lambda_{dt}
      term5 = model(x_tau, tau)^2 \\lambda_t
    weights default to 1; reweight=True picks lambda_t=1-tau^2, lambda_{dt}=-2 tau.

    Args:
        model: f(x [B,D], t [B,1]) -> [B,1].
        batch: requires "x0", "x1".
        tau: [B,1].
        iw: importance weight, applied to interior (sampled-tau) terms only.
        reweight: enable time-dependent weighting.
        eps: tau=0 boundary value.

    Returns: scalar loss.
    """
    x0 = batch["x0"]
    x1 = batch["x1"]

    t0 = torch.zeros((x0.shape[0], 1), device=x0.device, dtype=x0.dtype) + eps
    t1 = torch.ones((x1.shape[0], 1), device=x1.device, dtype=x1.dtype)

    if reweight:
        lam_t = (1 - tau ** 2).squeeze()
        lam_t0 = (1 - t0.squeeze() ** 2)
        lam_t1 = (1 - t1.squeeze() ** 2 + eps ** 2)
        lam_dt = (-2 * tau).squeeze()
    else:
        lam_t = lam_t0 = lam_t1 = 1.0
        lam_dt = 0.0

    term1 = (2 * model(x0, t0)).squeeze() * lam_t0
    term2 = (2 * model(x1, t1)).squeeze() * lam_t1

    x_tau = torch.sqrt(1 - tau ** 2) * x0 + tau * x1
    score, dscore = _score_dt(model, x_tau, tau)

    term3 = (2 * dscore).squeeze() * lam_t
    term4 = score.squeeze() * lam_dt
    term5 = (score ** 2).squeeze() * lam_t

    # separate boundary (t0, t1) and interior (tau) terms; iw applied to interior only
    # when iw == ones(B,1), this is identical to mean(term1 - term2 + term3 + term4 + term5)
    boundary = (term1 - term2).mean()
    interior = (iw.squeeze(-1) * (term3 + term4 + term5)).mean()
    return boundary + interior


tsm_loss.required_keys = frozenset({"x0", "x1"})
tsm_loss.requires_tau_grad = True


def sb_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    sigma: float = 1.0,
    path: Optional["Path1D"] = None,
) -> torch.Tensor:
    """closed-form Schroedinger-bridge regression loss (CTSM family).

    sampling/target:
      path is None: (x_tau, target, lambda_t) = sb_target(x0, x1, sigma, tau, epsilon)
      path given : (x_tau, target, lambda_t) = path.sample_and_target(x0, x1, xstar, tau, epsilon)

    when path exposes `sample_tau`, the trainer-provided (tau, iw) is overridden by
    path.sample_tau(batch_size, path.eps, device).

    loss = mean(iw * (target - lambda_t * model(x_tau, tau))^2).

    Args:
        model: f(x [B,D], t [B,1]) -> [B,1].
        batch: requires "x0", "x1"; "xstar" when path is triangular.
        tau, iw: [B,1] each; may be replaced if path.sample_tau exists.
        sigma: noise amplitude when path is None.
        path: None or a Path1D subclass.

    Returns: scalar loss.
    """
    from src.waypoints.path_1d import Path1D

    x0 = batch["x0"]
    x1 = batch["x1"]

    if path is not None and not isinstance(path, Path1D):
        raise TypeError(f"path must be None or Path1D; got {type(path).__name__}")

    sampler = getattr(path, "sample_tau", None) if path is not None else None
    if sampler is not None:
        sampled = sampler(tau.shape[0], path.eps, tau.device)
        if isinstance(sampled, tuple):
            tau, iw = sampled
        else:
            tau = sampled
            iw = torch.ones_like(tau)

    epsilon = torch.randn_like(x0)

    if path is None:
        x_tau, target, lam_t = sb_target(x0, x1, sigma, tau, epsilon)
    else:
        xstar = batch["xstar"]
        x_tau, target, lam_t = path.sample_and_target(x0, x1, xstar, tau, epsilon)

    err = target - lam_t * model(x_tau, tau)
    return torch.mean(iw * err ** 2)


sb_loss.required_keys = frozenset({"x0", "x1"})
sb_loss.requires_tau_grad = False


def velo_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    path,
    antithetic: bool = False,
) -> torch.Tensor:
    """VFM velocity (b-phase) loss on a stochastic interpolant.

    x_t   = (1-tau) x0 + tau x1 + gamma(tau) z, z ~ N(0, I)
    v*    = (x1 - x0) + gamma'(tau) z
    loss  = mean(0.5 ||b(x_t)||^2 - <v*, b(x_t)>)
    antithetic averages over (z, -z).

    Args:
        model: f(t [B,1], x [B,D]) -> [B,D].
        batch: requires "x0", "x1".
        tau, iw: [B,1]; iw is importance weight applied to per-sample loss.
        path: object with gamma(tau), dgamma_dtau(tau) returning [B,1].
        antithetic: enable (z, -z) variance reduction.

    Returns: scalar loss.
    """
    x0 = batch["x0"]
    x1 = batch["x1"]
    g = path.gamma(tau)
    dg = path.dgamma_dtau(tau)
    mu = (1 - tau) * x0 + tau * x1
    z = torch.randn_like(x0)
    v_star = (x1 - x0) + dg * z

    if not antithetic:
        b = model(tau, mu + g * z)
        # per-sample loss with iw applied; when iw == ones(B,1) is identical to original mean
        loss_per_sample = 0.5 * (b ** 2).sum(-1) - (v_star * b).sum(-1)
        return (loss_per_sample * iw.squeeze(-1)).mean()

    b_p = model(tau, mu + g * z)
    b_m = model(tau, mu - g * z)
    v_star_m = (x1 - x0) - dg * z
    # compute loss per pair and apply iw
    lp = 0.25 * (b_p ** 2).sum(-1) - 0.5 * (v_star * b_p).sum(-1)
    lm = 0.25 * (b_m ** 2).sum(-1) - 0.5 * (v_star_m * b_m).sum(-1)
    loss_per_pair = 0.5 * (lp + lm)
    return (loss_per_pair * iw.squeeze(-1)).mean()


velo_loss.required_keys = frozenset({"x0", "x1"})
velo_loss.requires_tau_grad = False


def denoiser_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    path,
) -> torch.Tensor:
    """VFM denoiser (eta-phase) loss.

    x_t  = (1-tau) x0 + tau x1 + gamma(tau) z, z ~ N(0, I)
    loss = mean(0.5 ||eta(x_t)||^2 - <z, eta(x_t)>)

    Args:
        model: f(t [B,1], x [B,D]) -> [B,D].
        batch: requires "x0", "x1".
        tau, iw: [B,1]; iw is importance weight applied to per-sample loss.
        path: object with gamma(tau) returning [B,1].

    Returns: scalar loss.
    """
    x0 = batch["x0"]
    x1 = batch["x1"]
    z = torch.randn_like(x0)
    x_t = (1 - tau) * x0 + tau * x1 + path.gamma(tau) * z
    eta = model(tau, x_t)
    # per-sample loss with iw applied; when iw == ones(B,1) is identical to original mean
    loss_per_sample = 0.5 * (eta ** 2).sum(-1) - (z * eta).sum(-1)
    return (loss_per_sample * iw.squeeze(-1)).mean()


denoiser_loss.required_keys = frozenset({"x0", "x1"})
denoiser_loss.requires_tau_grad = False


def fm_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    score_weight: float = 1.0,
    p_uncond: float = 0.0,
    sentinel_cond: float = -1.0,
) -> torch.Tensor:
    """conditional flow-matching loss with optional CFG dropout (2 classes).

    x_t      = (1-tau) z + tau x_data, z ~ N(0, I), x_data ~ {p0, p1}
    v*       = x_data - z
    s*       = -z / (1 - tau)
    loss     = mse(v_pred, v*) + score_weight * mse(s_pred, s*)
    with prob p_uncond, c is replaced by sentinel_cond.

    Args:
        model: f(t [B,1], x [B,D], c [B,1]) -> (v [B,D], s [B,D]).
        batch: requires "x0", "x1". trainer passes equal-size halves.
        tau, iw: [B,1]; iw is importance weight, tiled to 2B and applied per-sample.
        score_weight, p_uncond, sentinel_cond: as documented.

    Returns: scalar loss.
    """
    x0 = batch["x0"]
    x1 = batch["x1"]
    x_data = torch.cat([x0, x1], dim=0)
    c = torch.cat(
        [
            torch.zeros(x0.shape[0], 1, device=x0.device, dtype=x0.dtype),
            torch.ones(x1.shape[0], 1, device=x1.device, dtype=x1.dtype),
        ],
        dim=0,
    )
    iw_tiled = iw.repeat(2, 1).squeeze(-1)  # [2B] importance weight, tiled to match tau.repeat
    tau = tau.repeat(2, 1)

    if p_uncond > 0.0:
        mask = torch.bernoulli(torch.full_like(c, p_uncond))
        c = torch.where(mask > 0.5, torch.full_like(c, sentinel_cond), c)

    z = torch.randn_like(x_data)
    x_t = (1 - tau) * z + tau * x_data
    v_star = x_data - z
    s_star = -z / (1 - tau)

    v, s = model(tau, x_t, c)
    # manual mse to apply per-sample iw; when iw == ones(B,1) is identical to F.mse_loss
    v_err = ((v - v_star) ** 2).mean(dim=-1)  # [2B]
    s_err = ((s - s_star) ** 2).mean(dim=-1)  # [2B]
    return (v_err * iw_tiled).mean() + score_weight * (s_err * iw_tiled).mean()


fm_loss.required_keys = frozenset({"x0", "x1"})
fm_loss.requires_tau_grad = False


def tri_fm_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    score_weight: float = 1.0,
    triangular_p_uncond: float = 0.0,
) -> torch.Tensor:
    """3-class flow-matching loss with masked score term.

    same form as `fm_loss` over x_data = cat([x0, x1, xstar]) with 3-way one-hot,
    but the score-MSE is masked to exclude the xstar rows (those are never queried
    at inference). with prob triangular_p_uncond, the one-hot row is zeroed.

    Args:
        model: must expose `forward_from_onehot(t, x, y_onehot)` -> (v, s).
        batch: requires "x0", "x1", "xstar".
        tau, iw: [B,1]; iw is importance weight, tiled to 3B and applied per-sample.
        score_weight, triangular_p_uncond: as documented.

    Returns: scalar loss.
    """
    x0 = batch["x0"]
    x1 = batch["x1"]
    xstar = batch["xstar"]
    x_data = torch.cat([x0, x1, xstar], dim=0)
    b0, b1, bs = x0.shape[0], x1.shape[0], xstar.shape[0]
    y_idx = torch.cat(
        [
            torch.zeros(b0, dtype=torch.long, device=x0.device),
            torch.ones(b1, dtype=torch.long, device=x1.device),
            torch.full((bs,), 2, dtype=torch.long, device=xstar.device),
        ],
        dim=0,
    )
    y_oh = F.one_hot(y_idx, num_classes=3).to(x_data.dtype)
    iw_tiled = iw.repeat(3, 1).squeeze(-1)  # [3B] importance weight, tiled to match tau.repeat
    tau = tau.repeat(3, 1)

    if triangular_p_uncond > 0.0:
        mask = torch.bernoulli(torch.full((y_oh.shape[0], 1), triangular_p_uncond, device=y_oh.device))
        y_oh = torch.where(mask > 0.5, torch.zeros_like(y_oh), y_oh)

    z = torch.randn_like(x_data)
    x_t = (1 - tau) * z + tau * x_data
    v_star = x_data - z
    s_star = -z / (1 - tau)

    v, s = model.forward_from_onehot(tau, x_t, y_oh)

    # velocity term: manual mse with iw; when iw == ones(B,1) is identical to F.mse_loss
    v_err = ((v - v_star) ** 2).mean(dim=-1)  # [3B]
    loss_v = (v_err * iw_tiled).mean()
    # score term: apply iw before masking; normalize by n_active
    s_mask = (y_idx != 2).to(x_data.dtype).unsqueeze(-1)
    diff = ((s - s_star) ** 2) * s_mask * iw_tiled.unsqueeze(-1)
    n_active = s_mask.sum() * x_data.shape[1]
    loss_s = diff.sum() / n_active.clamp(min=1.0)

    return loss_v + score_weight * loss_s


tri_fm_loss.required_keys = frozenset({"x0", "x1", "xstar"})
tri_fm_loss.requires_tau_grad = False


def tri_tsm_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    reweight: bool = False,
    eps: float = 1e-5,
    vertex: float = 0.5,
    peak_max: float = 1.0,
) -> torch.Tensor:
    """Hyvarinen time-score loss on the triangular bell path (TriangularTSM).

    t = clamp(tau, eps, 1); t' is the piecewise-quadratic bell at (vertex, peak_max).
    x_t   = sqrt(1 - t^2) x0 + t x1
    x_tau = sqrt(1 - t'^2) x_t + t' xstar
    loss form mirrors `tsm_loss`; model takes (x, t, t').

    Args:
        model: f(x [B,D], t [B,1], t' [B,1]) -> [B,1].
        batch: requires "x0", "x1", "xstar".
        tau, iw: [B,1]; iw unused.
        reweight, eps, vertex, peak_max: as documented.

    Returns: scalar loss.
    """
    x0 = batch["x0"]
    x1 = batch["x1"]
    xstar = batch["xstar"]

    t = torch.clamp(tau, min=eps, max=1.0)
    left = peak_max * (2.0 * (t / vertex) - (t / vertex) ** 2)
    right = peak_max * (1.0 - ((t - vertex) / (1.0 - vertex)) ** 2)
    t_prime = torch.clamp(torch.where(t <= vertex, left, right), min=0.0, max=1.0)

    t0 = torch.zeros_like(tau) + eps
    t1 = torch.ones_like(tau)
    tp0 = torch.zeros_like(tau) + eps
    tp1 = torch.ones_like(tau)

    if reweight:
        lam_t = (1 - t ** 2).squeeze(-1)
        lam_t0 = (1 - t0.squeeze(-1) ** 2)
        lam_t1 = (1 - t1.squeeze(-1) ** 2 + eps ** 2)
        lam_dt = (-2 * t).squeeze(-1)
    else:
        lam_t = lam_t0 = lam_t1 = 1.0
        lam_dt = 0.0

    term1 = 2 * model(x0, t0, tp0).squeeze(-1) * lam_t0
    term2 = 2 * model(x1, t1, tp1).squeeze(-1) * lam_t1

    sqrt_t = torch.sqrt(torch.clamp(1.0 - t ** 2, min=eps))
    sqrt_tp = torch.sqrt(torch.clamp(1.0 - t_prime ** 2, min=eps))
    x_t = sqrt_t * x0 + t * x1
    x_tau = sqrt_tp * x_t + t_prime * xstar

    score, dscore = _score_dt(model, x_tau, tau, t_prime)
    term3 = (2 * dscore).squeeze(-1) * lam_t
    term4 = score.squeeze(-1) * lam_dt
    term5 = (score ** 2).squeeze(-1) * lam_t

    return (term1 - term2 + term3 + term4 + term5).mean()


tri_tsm_loss.required_keys = frozenset({"x0", "x1", "xstar"})
tri_tsm_loss.requires_tau_grad = True
