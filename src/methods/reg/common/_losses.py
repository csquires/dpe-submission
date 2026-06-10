"""pure loss functions for density-ratio estimators (TSM, CTSM, FM, VFM).

each loss exposes two module-level attributes used by `_trainer.train_loop`:
  - `required_keys`: subset of {"x0", "x1", "xstar"} the trainer must populate
  - `requires_tau_grad`: whether the loss enables autograd on tau internally

losses with init-time choices (path vs sigma, antithetic on/off, cfg dropout
on/off) are exposed as factory functions ``make_<name>``: the factory binds
the choice once and returns a specialized closure with required_keys /
requires_tau_grad pre-set. estimators call the factory inside ``fit`` so the
hot-path loop sees a single specialized callable rather than re-checking the
same flag per step.

the two losses without init-time choices (``tsm_loss``, ``tri_tsm_loss``,
``denoiser_loss``) remain module-level functions for convenience.
"""
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from typing import Callable

from src.waypoints.dataclass_paths import DirectPath1D, TriangularPath1D, TriangularPath2D
from src.methods.reg.common._paradigm_funcs import (
    vfm_velocity_target_1d, vfm_velocity_target_direct_1d,
    ctsm_regression_target_1d, ctsm_regression_target_direct_1d,
)
from ..common._weighting import resolve_lambdas, resolve_outer_lambda


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
    weights default to 1 (reweight=False -> identity_lambdas);
    reweight=True picks path_var_lambdas (lambda_t=1-tau^2, lambda_{dt}=-2 tau).
    delegated to `_weighting.resolve_lambdas`; mirrors the
    `prob_path.get_time_weighting_quantities` contract from dre-prob-paths.

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

    lam = resolve_lambdas(reweight, tau, t0, t1, eps=eps)

    term1 = (2 * model(x0, t0)).squeeze() * lam.lam_t0
    term2 = (2 * model(x1, t1)).squeeze() * lam.lam_t1

    x_tau = torch.sqrt(1 - tau ** 2) * x0 + tau * x1
    score, dscore = _score_dt(model, x_tau, tau)

    term3 = (2 * dscore).squeeze() * lam.lam_t
    term4 = score.squeeze() * lam.lam_dt
    term5 = (score ** 2).squeeze() * lam.lam_t

    # boundary (t0, t1) and interior (tau) terms; iw applied to interior only.
    boundary = (term1 - term2).mean()
    interior = (iw.squeeze(-1) * (term3 + term4 + term5)).mean()
    return boundary + interior


tsm_loss.required_keys = frozenset({"x0", "x1"})
tsm_loss.requires_tau_grad = True


def make_sb_loss(*, path, reweight: bool = False) -> Callable:
    """factory: closed-form schroedinger-bridge regression loss with path bound.

    path must be one of the new dataclass paths (DirectPath1D, TriangularPath1D,
    or TriangularPath2D). returns a callable (model, batch, tau, iw) -> scalar
    with `required_keys` and `requires_tau_grad` attributes.
    """
    if isinstance(path, TriangularPath1D):
        def loss(model, batch, tau, iw):
            x0 = batch["x0"]; x1 = batch["x1"]; xstar = batch["xstar"]
            epsilon = torch.randn_like(x0)
            outer = resolve_outer_lambda(reweight, tau)
            x_t, target, lambda_t = ctsm_regression_target_1d(path, x0, x1, xstar, tau, epsilon)
            pred = model(x_t, tau)  # space-first: TimeScoreNetwork.forward(x, t)
            return (((target - lambda_t * pred) ** 2).squeeze(-1) * outer * iw.squeeze(-1)).mean()
        loss.required_keys = frozenset({"x0", "x1", "xstar"})
    elif isinstance(path, DirectPath1D):
        def loss(model, batch, tau, iw):
            x0 = batch["x0"]; x1 = batch["x1"]
            epsilon = torch.randn_like(x0)
            outer = resolve_outer_lambda(reweight, tau)
            x_t, target, lambda_t = ctsm_regression_target_direct_1d(path, x0, x1, tau, epsilon)
            pred = model(x_t, tau)  # space-first: TimeScoreNetwork.forward(x, t)
            return (((target - lambda_t * pred) ** 2).squeeze(-1) * outer * iw.squeeze(-1)).mean()
        loss.required_keys = frozenset({"x0", "x1"})
    elif isinstance(path, TriangularPath2D):
        # 2d loss wiring lives inside v3 estimator.
        def loss(model, batch, tau, iw):
            raise NotImplementedError("TriangularPath2D loss is wired inside the V3 estimator")
        loss.required_keys = frozenset({"x0", "x1", "xstar"})
    else:
        raise TypeError(
            f"path must be one of DirectPath1D, TriangularPath1D, TriangularPath2D; "
            f"got {type(path).__name__}"
        )
    loss.requires_tau_grad = False
    return loss


def make_velo_loss(*, path, antithetic: bool = False, reweight: bool = False, outer_weight: Callable | None = None) -> Callable:
    """factory: VFM velocity (b-phase) loss on a stochastic interpolant.

    x_t   = (1-tau) x0 + tau x1 + gamma(tau) z, z ~ N(0, I)
    v*    = (x1 - x0) + gamma'(tau) z
    loss  = mean(0.5 ||b(x_t)||^2 - <v*, b(x_t)>)
    antithetic averages over (z, -z) for variance reduction.

    bound at factory time:
        path: object with gamma(tau), dgamma_dtau(tau) returning [B,1], or new dataclass path.
        antithetic: select the antithetic body once.
        reweight: outer path_var lambda.
    """
    # bind weight_fn at factory time: either the outer_weight closure or default resolve-based
    if outer_weight is not None:
        weight_fn = outer_weight
    else:
        weight_fn = lambda t: resolve_outer_lambda(reweight, t)

    if isinstance(path, TriangularPath1D):
        if antithetic:
            def loss(model, batch, tau, iw):
                x0 = batch["x0"]
                x1 = batch["x1"]
                xstar = batch["xstar"]
                z = torch.randn_like(x0)
                outer = weight_fn(tau)
                x_t_plus, v_plus = vfm_velocity_target_1d(path, x0, x1, xstar, tau, z)
                x_t_minus, v_minus = vfm_velocity_target_1d(path, x0, x1, xstar, tau, -z)
                b_plus = model(x_t_plus, tau)
                b_minus = model(x_t_minus, tau)
                lp = 0.5 * (b_plus ** 2).sum(-1) - (v_plus * b_plus).sum(-1)
                lm = 0.5 * (b_minus ** 2).sum(-1) - (v_minus * b_minus).sum(-1)
                return (0.5 * (lp + lm) * outer * iw.squeeze(-1)).mean()
        else:
            def loss(model, batch, tau, iw):
                x0 = batch["x0"]
                x1 = batch["x1"]
                xstar = batch["xstar"]
                z = torch.randn_like(x0)
                outer = weight_fn(tau)
                x_t, v_star = vfm_velocity_target_1d(path, x0, x1, xstar, tau, z)
                b = model(x_t, tau)
                return ((0.5 * (b ** 2).sum(-1) - (v_star * b).sum(-1)) * outer * iw.squeeze(-1)).mean()

        loss.required_keys = frozenset({"x0", "x1", "xstar"})
    elif isinstance(path, DirectPath1D):
        if antithetic:
            def loss(model, batch, tau, iw):
                x0 = batch["x0"]
                x1 = batch["x1"]
                z = torch.randn_like(x0)
                outer = weight_fn(tau)
                x_t_plus, v_plus = vfm_velocity_target_direct_1d(path, x0, x1, tau, z)
                x_t_minus, v_minus = vfm_velocity_target_direct_1d(path, x0, x1, tau, -z)
                b_plus = model(x_t_plus, tau)
                b_minus = model(x_t_minus, tau)
                lp = 0.5 * (b_plus ** 2).sum(-1) - (v_plus * b_plus).sum(-1)
                lm = 0.5 * (b_minus ** 2).sum(-1) - (v_minus * b_minus).sum(-1)
                return (0.5 * (lp + lm) * outer * iw.squeeze(-1)).mean()
        else:
            def loss(model, batch, tau, iw):
                x0 = batch["x0"]
                x1 = batch["x1"]
                z = torch.randn_like(x0)
                outer = weight_fn(tau)
                x_t, v_star = vfm_velocity_target_direct_1d(path, x0, x1, tau, z)
                b = model(x_t, tau)
                return ((0.5 * (b ** 2).sum(-1) - (v_star * b).sum(-1)) * outer * iw.squeeze(-1)).mean()

        loss.required_keys = frozenset({"x0", "x1"})
    else:
        raise TypeError(
            f"path must be TriangularPath1D or DirectPath1D; got {type(path).__name__}"
        )
    loss.requires_tau_grad = False
    return loss


def make_denoiser_loss(*, path, reweight: bool = False, outer_weight: Callable | None = None) -> Callable:
    """factory: VFM denoiser (eta-phase) loss with path, reweight bound.

    x_t  = (1-tau) x0 + tau x1 + gamma(tau) z, z ~ N(0, I)
    loss = mean(0.5 ||eta(x_t)||^2 - <z, eta(x_t)>)
    """
    # bind weight_fn at factory time: either the outer_weight closure or default resolve-based
    if outer_weight is not None:
        weight_fn = outer_weight
    else:
        weight_fn = lambda t: resolve_outer_lambda(reweight, t)

    if isinstance(path, TriangularPath1D):
        def loss(model, batch, tau, iw):
            x0 = batch["x0"]
            x1 = batch["x1"]
            xstar = batch["xstar"]
            z = torch.randn_like(x0)
            outer = weight_fn(tau)
            w = path.weights(tau)
            mu = w.alpha * x0 + w.beta * x1 + w.w_star * xstar
            gamma_t = path.gamma(tau)
            x_t = mu + gamma_t * z
            eta = model(x_t, tau)
            return ((0.5 * (eta ** 2).sum(-1) - (z * eta).sum(-1)) * outer * iw.squeeze(-1)).mean()

        loss.required_keys = frozenset({"x0", "x1", "xstar"})
    elif isinstance(path, DirectPath1D):
        def loss(model, batch, tau, iw):
            x0 = batch["x0"]
            x1 = batch["x1"]
            z = torch.randn_like(x0)
            outer = weight_fn(tau)
            w = path.weights(tau)
            mu = w.alpha * x0 + w.beta * x1
            gamma_t = path.gamma(tau)
            x_t = mu + gamma_t * z
            eta = model(x_t, tau)
            return ((0.5 * (eta ** 2).sum(-1) - (z * eta).sum(-1)) * outer * iw.squeeze(-1)).mean()

        loss.required_keys = frozenset({"x0", "x1"})
    else:
        raise TypeError(
            f"path must be TriangularPath1D or DirectPath1D; got {type(path).__name__}"
        )
    loss.requires_tau_grad = False
    return loss


def make_fm_loss(
    *,
    score_weight: float = 1.0,
    p_uncond: float = 0.0,
    sentinel_cond: float = -1.0,
    reweight: bool = False,
    outer_weight: Callable | None = None,
) -> Callable:
    """factory: conditional flow-matching loss with optional CFG dropout (2 classes).

    x_t      = (1-tau) z + tau x_data, z ~ N(0, I), x_data ~ {p0, p1}
    v*       = x_data - z
    s*       = -z / (1 - tau)
    loss     = mse(v_pred, v*) + score_weight * mse(s_pred, s*)
    with prob p_uncond, c is replaced by sentinel_cond.

    bound at factory time:
        score_weight, sentinel_cond, reweight: scalar constants.
        p_uncond: if > 0, the bernoulli-dropout body is bound; else the identity.
          this removes the per-step ``if p_uncond > 0`` check.
    """
    if p_uncond > 0.0:
        def apply_uncond(c):
            mask = torch.bernoulli(torch.full_like(c, p_uncond))
            return torch.where(mask > 0.5, torch.full_like(c, sentinel_cond), c)
    else:
        def apply_uncond(c):
            return c

    # bind weight_fn at factory time: either the outer_weight closure or default resolve-based
    if outer_weight is not None:
        weight_fn = outer_weight
    else:
        weight_fn = lambda t: resolve_outer_lambda(reweight, t)

    def loss(model, batch, tau, iw):
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
        iw_tiled = iw.repeat(2, 1).squeeze(-1)
        tau2 = tau.repeat(2, 1)
        c = apply_uncond(c)
        z = torch.randn_like(x_data)
        x_t = (1 - tau2) * z + tau2 * x_data
        v_star = x_data - z
        s_star = -z / (1 - tau2)
        v, s = model(tau2, x_t, c)
        outer = weight_fn(tau2)
        v_err = ((v - v_star) ** 2).mean(dim=-1)
        s_err = ((s - s_star) ** 2).mean(dim=-1)
        return (v_err * outer * iw_tiled).mean() + score_weight * (s_err * outer * iw_tiled).mean()

    loss.required_keys = frozenset({"x0", "x1"})
    loss.requires_tau_grad = False
    return loss


def make_tri_fm_loss(
    *,
    score_weight: float = 1.0,
    triangular_p_uncond: float = 0.0,
    reweight: bool = False,
    outer_weight: Callable | None = None,
) -> Callable:
    """factory: 3-class flow-matching loss with masked score term.

    same form as ``make_fm_loss`` over ``x_data = cat([x0, x1, xstar])`` with
    3-way one-hot, but the score-MSE is masked to exclude the xstar rows
    (those are never queried at inference). with prob triangular_p_uncond,
    the one-hot row is zeroed.

    bound at factory time:
        score_weight, reweight: scalar constants.
        triangular_p_uncond: if > 0, the bernoulli-dropout body is bound; else identity.
    """
    if triangular_p_uncond > 0.0:
        def apply_uncond(y_oh):
            mask = torch.bernoulli(
                torch.full((y_oh.shape[0], 1), triangular_p_uncond, device=y_oh.device)
            )
            return torch.where(mask > 0.5, torch.zeros_like(y_oh), y_oh)
    else:
        def apply_uncond(y_oh):
            return y_oh

    # bind weight_fn at factory time: either the outer_weight closure or default resolve-based
    if outer_weight is not None:
        weight_fn = outer_weight
    else:
        weight_fn = lambda t: resolve_outer_lambda(reweight, t)

    def loss(model, batch, tau, iw):
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
        y_oh = apply_uncond(y_oh)
        iw_tiled = iw.repeat(3, 1).squeeze(-1)
        tau3 = tau.repeat(3, 1)
        z = torch.randn_like(x_data)
        x_t = (1 - tau3) * z + tau3 * x_data
        v_star = x_data - z
        s_star = -z / (1 - tau3)
        v, s = model.forward_from_onehot(tau3, x_t, y_oh)
        outer = weight_fn(tau3)
        v_err = ((v - v_star) ** 2).mean(dim=-1)
        loss_v = (v_err * outer * iw_tiled).mean()
        s_mask = (y_idx != 2).to(x_data.dtype).unsqueeze(-1)
        diff = ((s - s_star) ** 2) * s_mask * (outer * iw_tiled).unsqueeze(-1)
        n_active = s_mask.sum() * x_data.shape[1]
        loss_s = diff.sum() / n_active.clamp(min=1.0)
        return loss_v + score_weight * loss_s

    loss.required_keys = frozenset({"x0", "x1", "xstar"})
    loss.requires_tau_grad = False
    return loss


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
        tau: [B,1] sampled time.
        iw: [B,1] importance weight, applied to interior (sampled-tau) terms only.
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
    # boundary (t, t') must lie on the bell path used at inference. supervising
    # an off-manifold point (e.g. tp1=1 when bell(1)=0) biases the score the
    # ldr integral consumes. bell(1)=0 exactly; bell(eps) on the left branch.
    bell_eps_l = peak_max * (2.0 * (eps / vertex) - (eps / vertex) ** 2)
    bell_eps_r = peak_max * (1.0 - ((eps - vertex) / (1.0 - vertex)) ** 2)
    bell_eps = min(1.0, max(0.0, bell_eps_l if eps <= vertex else bell_eps_r))
    tp0 = torch.zeros_like(tau) + bell_eps
    tp1 = torch.zeros_like(tau)

    lam = resolve_lambdas(reweight, t, t0, t1, eps=eps)

    term1 = 2 * model(x0, t0, tp0).squeeze(-1) * lam.lam_t0
    term2 = 2 * model(x1, t1, tp1).squeeze(-1) * lam.lam_t1

    # fixed tiny floor (decoupled from the searched time-eps): matches TSM's
    # un-clamped sqrt path so the only train-time difference is the bell path.
    sqrt_t = torch.sqrt(torch.clamp(1.0 - t ** 2, min=1e-8))
    sqrt_tp = torch.sqrt(torch.clamp(1.0 - t_prime ** 2, min=1e-8))
    x_t = sqrt_t * x0 + t * x1
    x_tau = sqrt_tp * x_t + t_prime * xstar

    # compute full chain-rule derivative d/dtau s(x_tau, t(tau), t'(tau)):
    # d/dtau = (partial/partial_t) * (dt/dtau) + (partial/partial_t') * (dt'/dtau)
    tau_g = tau.clone().detach().requires_grad_(True)
    t_g = torch.clamp(tau_g, min=eps, max=1.0)
    left_g = peak_max * (2.0 * (t_g / vertex) - (t_g / vertex) ** 2)
    right_g = peak_max * (1.0 - ((t_g - vertex) / (1.0 - vertex)) ** 2)
    t_prime_g = torch.clamp(torch.where(t_g <= vertex, left_g, right_g), min=0.0, max=1.0)
    # x_tau detached from tau_g: IBP treats x_tau as evaluation point, not a function of tau
    score = model(x_tau.detach(), t_g, t_prime_g)
    dscore = autograd.grad(score.sum(), tau_g, create_graph=True)[0]

    term3 = (2 * dscore).squeeze(-1) * lam.lam_t
    term4 = score.squeeze(-1) * lam.lam_dt
    term5 = (score ** 2).squeeze(-1) * lam.lam_t

    boundary = (term1 - term2).mean()
    interior = (iw.squeeze(-1) * (term3 + term4 + term5)).mean()
    return boundary + interior


tri_tsm_loss.required_keys = frozenset({"x0", "x1", "xstar"})
tri_tsm_loss.requires_tau_grad = True
