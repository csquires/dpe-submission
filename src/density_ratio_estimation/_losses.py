"""Pure loss functions for density ratio estimation.

Modularizes loss computation from DRE estimators (TSM, CTSM, etc.) into reusable
standalone functions with clear tensor flow, gradient semantics, and path abstractions.
"""
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.waypoints.path_1d import Path1D

from src.waypoints.sb_bridge import sb_bridge_target


def _score_time_derivative(
    model_call: Callable,
    x: torch.Tensor,
    tau: torch.Tensor,
    *extra_time_args: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (score, d_score/d_tau) with create_graph=True for gradient flow.

    Procedure:
      - tau_g = tau.clone().detach().requires_grad_(True)
      - score = model_call(x, tau_g, *extra_time_args)  # [B, 1]
      - score_dt = autograd.grad(score.sum(), tau_g, create_graph=True)[0]  # [B, 1]
      - return score, score_dt

    Args:
        model_call: Callable that accepts (x, tau_g, *extra_time_args) -> [B, 1].
        x: [B, D] input tensor.
        tau: [B, 1] time parameter (will be cloned and re-enabled for gradients).
        extra_time_args: additional time arguments (e.g., t_prime for 2D-time paths).

    Returns:
        score: [B, 1] model output.
        score_dt: [B, 1] gradient of score.sum() w.r.t. tau_g.
    """
    tau_g = tau.clone().detach().requires_grad_(True)
    score = model_call(x, tau_g, *extra_time_args)  # [B, 1]
    score_dt = autograd.grad(score.sum(), tau_g, create_graph=True)[0]  # [B, 1]
    return score, score_dt


def hyvarinen_time_score_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    reweight: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Time-score matching loss with optional time-reweighting (Hyvarinen 2005).

    Computes the time-score loss for density ratio estimation via score matching
    on continuous-time interpolant paths. Handles automatic differentiation on tau
    to extract score time-derivatives.

    Math (extracted verbatim from TSM.time_score_loss, tsm.py:53-86):

    Given time tau in (0, 1) and interpolated path point x_tau sampled at tau,
    the loss penalizes:
      L = term1 - term2 + term3 + term4 + term5

    where (with optional lambda_t reweighting):
      lambda_t = 1.0                           (if not reweight)
      lambda_t = (1 - tau^2)                   (if reweight)
      lambda_t0 = (1 - eps^2)                  (boundary term at tau=0)
      lambda_t1 = (1 - 1^2 + eps^2) = eps^2   (boundary term at tau=1)
      lambda_dt = 0.0                          (if not reweight)
      lambda_dt = -2*tau                       (if reweight)

      term1 = 2 * model(x0, t0) * lambda_t0            [x0 at t0=eps]
      term2 = 2 * model(x1, t1) * lambda_t1            [x1 at t1=1]
      term3 = 2 * d/d_tau[model(x_tau, tau)] * lambda_t
      term4 = model(x_tau, tau) * lambda_dt
      term5 = model(x_tau, tau)^2 * lambda_t

      loss = mean(term1 - term2 + term3 + term4 + term5)

    The key computation:
      - x_tau is interpolated from batch["x0"] and batch["x1"] using the TSM path:
        x_tau = sqrt(1 - tau^2) * x0 + tau * x1
      - tau is cloned, detached, and marked requires_grad_(True) to enable
        torch.autograd.grad(model(x_tau, tau).sum(), tau, create_graph=True)
        to compute the score time-derivative.

    Importance weight `iw` is currently UNUSED for Hyvarinen loss (documented for
    future variants; all terms share uniform weight of 1.0). Passed to match the
    trainer's generic loss signature.

    Args:
        model: Callable with signature model(x: [B, D], t: [B, 1]) -> [B, 1].
        batch: dict of tensors from trainer, required keys {"x0", "x1"}.
               - x0: [B, D] samples from p0 (bootstrap endpoint).
               - x1: [B, D] samples from p1 (bootstrap endpoint).
        tau: [B, 1] time parameter sampled from trainer's time_sampler.
             Trainer is responsible for clamping tau to (eps, 1-eps) boundary.
        iw: [B, 1] importance weight from trainer's time_sampler.
            Currently unused for Hyvarinen; included for signature consistency.
        reweight: if True, apply time-dependent lambda_t and lambda_dt weights.
                  Default False (uniform weighting).
        eps: scalar float boundary epsilon. Default 1e-5.
             Used for boundary condition times t0 = eps, t1 = 1.0 (no need to clamp).

    Returns:
        torch.Tensor: scalar loss value. Graph is active (suitable for .backward()).

    Raises:
        KeyError: if batch does not contain required keys {"x0", "x1"}.

    Notes:
        - tau MUST be requires_grad_(True) after cloning and detaching to enable
          autograd.grad(score.sum(), tau, create_graph=True).
        - Path construction (x_tau) is hardcoded to TSM's 2-source formula:
          x_tau = sqrt(1 - tau^2) * x0 + tau * x1. This is NOT generic; TSM is
          currently the only Hyvarinen-loss consumer. If a future variant uses
          a different path family (e.g., geodesic, exponential), this loss will
          need refactoring (e.g., accepting a path_sampler callable or reading
          x_tau from batch). This is a known design boundary.
        - iw is not used; TSM originally did not support importance-weighted time
          sampling. Retaining iw in signature for trainer compatibility.
        - The loss is scalar (mean over batch).

    Source: tsm.py:53-86 (TSM.time_score_loss), tsm.py:105 (x_t path computation).
    """
    # extract and validate batch
    x0 = batch["x0"]  # [B, D]
    x1 = batch["x1"]  # [B, D]

    # prepare boundary times [B, 1]
    t0 = torch.zeros((x0.shape[0], 1), device=x0.device, dtype=x0.dtype) + eps
    t1 = torch.ones((x1.shape[0], 1), device=x1.device, dtype=x1.dtype)

    # compute lambda weights based on reweight flag
    if reweight:
        lambda_t = (1 - tau**2).squeeze()  # [B]
        lambda_t0 = (1 - t0.squeeze() ** 2)  # [B]
        lambda_t1 = (1 - t1.squeeze() ** 2 + eps**2)  # [B]
        lambda_dt = (-2 * tau).squeeze()  # [B]
    else:
        lambda_t = lambda_t0 = lambda_t1 = 1.0  # scalar float
        lambda_dt = 0.0  # scalar float

    # compute boundary term 1: 2 * model(x0, t0) * lambda_t0 [B]
    score_t0 = model(x0, t0)  # [B, 1]
    term1 = (2 * score_t0).squeeze() * lambda_t0  # [B]

    # compute boundary term 2: 2 * model(x1, t1) * lambda_t1 [B]
    score_t1 = model(x1, t1)  # [B, 1]
    term2 = (2 * score_t1).squeeze() * lambda_t1  # [B]

    # construct x_tau and compute score + time-derivative
    # x_tau = sqrt(1 - tau^2) * x0 + tau * x1 [B, D]
    score_tau, score_tau_dt = _score_time_derivative(model, torch.sqrt(1 - tau**2) * x0 + tau * x1, tau)  # both [B, 1]

    # compute remaining terms
    term3 = (2 * score_tau_dt).squeeze() * lambda_t  # [B]
    # term4 handles both scalar and tensor lambda_dt
    term4 = score_tau.squeeze() * lambda_dt  # [B]
    term5 = (score_tau**2).squeeze() * lambda_t  # [B]

    # aggregate and return scalar loss
    loss = term1 - term2 + term3 + term4 + term5  # [B]
    return loss.mean()  # scalar


# attach module-level constants
hyvarinen_time_score_loss.required_keys = frozenset({"x0", "x1"})
hyvarinen_time_score_loss.requires_tau_grad = True


def closed_form_sb_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    sigma: float = 1.0,
    path: Optional["Path1D"] = None,
) -> torch.Tensor:
    """Closed-form Schrodinger-Bridge MSE loss for 2-source and 3-source paths.

    Computes MSE-based regression loss for score-based density ratio estimation
    on Schrödinger-Bridge paths. Supports:
      - 2-source paths (standard CTSM): no path object, pure sb_bridge_target helper.
      - 3-source triangular paths: delegates to path.sample_and_target(x0, x1, xstar, tau, epsilon).

    Math (extracted from ctsm.py:93-189, generalized to 3-source):

    For 2-source (path is None):
      x_tau, target, lambda_t = sb_bridge_target(x0, x1, sigma, tau, epsilon)
      where sb_bridge_target is a pure helper computing the closed-form SB path
      and regression target/weight (see src/waypoints/sb_bridge.py).

    For 3-source (path is Path1D):
      x_tau, target, lambda_t = path.sample_and_target(x0, x1, xstar, tau, epsilon)
      where path encapsulates the triangular interpolant math.

    Regression loss (both 2-source and 3-source):
      mse = mean(iw * (target - lambda_t * model(x_tau, tau))^2)

    V1 Path Tau Override (Dynamic Sampling):
    If the path object exposes a `sample_tau` method (e.g., for band-limited vertex
    sampling in PiecewiseSBCtsm1D), the loss will OVERRIDE the trainer-provided (tau, iw):
      tau_new, iw_new = path.sample_tau(batch_size, eps, device)
      # use tau_new, iw_new for the entire loss computation instead
    This enables specialized path variants to customize time sampling without modifying
    the trainer. Document this behavior when using custom paths.

    Args:
        model: Callable with signature model(x: [B, D], t: [B, 1]) -> [B, 1].
        batch: dict of tensors from trainer, required keys {"x0", "x1"}.
               Optional key "xstar" if path is a 3-source Path1D.
               - x0: [B, D] samples from p0.
               - x1: [B, D] samples from p1.
               - xstar: [B, D] samples from p* (midpoint mixture). Only needed if path is 3-source.
        tau: [B, 1] time parameter from trainer's time_sampler.
             May be OVERRIDDEN if path.sample_tau() is defined (V1 path override).
        iw: [B, 1] importance weight from trainer's time_sampler.
            May be OVERRIDDEN if path.sample_tau() is defined.
        sigma: scalar float, noise amplitude in SB path. Default 1.0 (CTSM default).
               Only used if path is None (2-source). If path is a CtsmPath1D,
               path.sample_and_target already bakes sigma into computation.
        path: None (2-source CTSM path) or Path1D subclass (triangular path).
              - If None: loss uses sb_bridge_target(x0, x1, sigma, tau, epsilon).
              - If Path1D: loss calls path.sample_and_target(x0, x1, xstar, tau, epsilon).
              Default None (2-source).

    Returns:
        torch.Tensor: scalar loss value (MSE). Graph is active for backprop.

    Raises:
        KeyError: if batch does not contain "x0" and "x1", or if path is 3-source
                  and batch does not contain "xstar".
        TypeError: if path is not None and not a Path1D subclass.

    Notes:
        - Epsilon sampling: `epsilon = torch.randn_like(x0)` (standard Gaussian [B, D]).
        - Device: all operations inferred from input tensor devices; no explicit .to() needed.
        - Both target and lambda_t are DETACHED from the autograd graph; the prediction
          mse is differentiable w.r.t. model(x_tau, tau) and x_tau (but not target).
        - iw is used directly; trainer/path is responsible for ensuring iw > 0 where needed.
        - V1 Path Tau Override: if path has a `sample_tau` callable, it is called ONCE
          per loss evaluation, BEFORE accessing batch["xstar"]. This allows custom paths
          to sample tau and conditionally provide xstar. Batch must still include all
          required keys for the eventual sample_and_target call.
        - 3-source validation: if path is provided, we do NOT statically pre-check
          batch["xstar"]. Instead, at runtime, if sample_and_target requires xstar
          and it is missing, path.sample_and_target will raise. This defers validation
          to the path implementation.
        - Path subclass check: if path is not None and not isinstance(path, Path1D),
          a TypeError is raised (to catch typos like passing a callable).

    Source: ctsm.py:93-189 (_epsilon_target, fit method), src/waypoints/sb_bridge.py
            (sb_bridge_target), src/waypoints/path_1d.py (CtsmPath1D.sample_and_target).
    """
    from src.waypoints.path_1d import Path1D

    # extract and validate batch
    x0 = batch["x0"]  # [B, D]
    x1 = batch["x1"]  # [B, D]

    # validate path type if provided
    if path is not None and not isinstance(path, Path1D):
        raise TypeError(
            f"path must be None or Path1D subclass, got {type(path).__name__}"
        )

    # handle V1 path tau override. PiecewiseSBCtsm1D.sample_tau returns just
    # tau (a tensor); other potential samplers may return (tau, iw). accept both.
    tau_sampler = getattr(path, "sample_tau", None) if path is not None else None
    if tau_sampler is not None:
        batch_size = tau.shape[0]
        eps_path = path.eps
        device = tau.device
        sampled = tau_sampler(batch_size, eps_path, device)
        if isinstance(sampled, tuple):
            tau, iw = sampled
        else:
            tau = sampled  # tensor [B, 1]
            iw = torch.ones_like(tau)

    # sample epsilon [B, D]
    epsilon = torch.randn_like(x0)

    # dispatch based on path
    if path is None:
        # 2-source: use sb_bridge_target
        x_tau, target, lambda_t = sb_bridge_target(x0, x1, sigma, tau, epsilon)
    else:
        # 3-source: use path.sample_and_target
        xstar = batch["xstar"]  # [B, D], may raise KeyError if missing
        x_tau, target, lambda_t = path.sample_and_target(x0, x1, xstar, tau, epsilon)

    # compute model prediction [B, 1]
    pred = model(x_tau, tau)

    # compute MSE with importance weighting [B, 1]
    err = target - lambda_t * pred  # [B, 1]
    loss = torch.mean(iw * (err**2))  # scalar

    return loss


# attach module-level constants
closed_form_sb_loss.required_keys = frozenset({"x0", "x1"})
closed_form_sb_loss.requires_tau_grad = False


def velo_matching_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    path,
    antithetic: bool = False,
) -> torch.Tensor:
    """VFM velocity-matching loss (b-phase).

    Computes b-phase MSE loss for velocity prediction on stochastic interpolant.

    Math:
      Given x_t = I_t + gamma_t * z where I_t = (1-t)*x0 + t*x1, z ~ N(0,I):
      target_v = (x1 - x0) + dgamma_dt * z
      loss = 0.5*||b(x_t)||^2 - <target_v, b(x_t)>

    With antithetic=True, averages over (z, -z) pairs for variance reduction.

    Args:
        model: Callable with signature model(t: [B, 1], x: [B, D]) -> [B, D].
        batch: dict with required keys {"x0", "x1"}.
        tau: [B, 1] time parameter (trainer must clamp to (eps, 1-eps)).
        iw: [B, 1] importance weight (unused; for signature consistency).
        path: VfmPath1D object with gamma(tau), dgamma_dtau(tau) methods.
        antithetic: if True, sample (z, -z) and average.

    Returns:
        scalar loss tensor with active gradient.
    """
    x0 = batch["x0"]  # [B, D]
    x1 = batch["x1"]  # [B, D]

    # extract path properties
    gamma_t = path.gamma(tau)  # [B, 1]
    dgamma_dt = path.dgamma_dtau(tau)  # [B, 1]

    # linear path component
    I_t = (1 - tau) * x0 + tau * x1  # [B, D]

    # sample noise
    z = torch.randn_like(x0)  # [B, D]

    # interpolated point
    x_t_plus = I_t + gamma_t * z  # [B, D]

    # target velocity
    v_target = (x1 - x0) + dgamma_dt * z  # [B, D]

    if not antithetic:
        # single forward pass
        b_pred = model(tau, x_t_plus)  # [B, D]
        b_norm_sq = (b_pred ** 2).sum(dim=-1)  # [B]
        target_dot_b = (v_target * b_pred).sum(dim=-1)  # [B]
        loss = (0.5 * b_norm_sq - target_dot_b).mean()  # scalar
    else:
        # antithetic variant: average over (z, -z)
        x_t_minus = I_t - gamma_t * z  # [B, D]
        b_plus = model(tau, x_t_plus)  # [B, D]
        b_minus = model(tau, x_t_minus)  # [B, D]
        b_norm_sq_plus = (b_plus ** 2).sum(dim=-1)  # [B]
        b_norm_sq_minus = (b_minus ** 2).sum(dim=-1)  # [B]
        target_dot_b_plus = (v_target * b_plus).sum(dim=-1)  # [B]
        target_dot_b_minus = ((x1 - x0 - dgamma_dt * z) * b_minus).sum(dim=-1)  # [B]
        loss = (
            0.25 * b_norm_sq_plus
            - 0.5 * target_dot_b_plus
            + 0.25 * b_norm_sq_minus
            - 0.5 * target_dot_b_minus
        ).mean()  # scalar

    return loss


velo_matching_loss.required_keys = frozenset({"x0", "x1"})
velo_matching_loss.requires_tau_grad = False


def denoiser_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    path,
) -> torch.Tensor:
    """VFM denoising loss (eta-phase).

    Computes eta-phase MSE loss for noise prediction.

    Math:
      Given x_t = (1-t)*x0 + t*x1 + gamma_t*z where z ~ N(0,I):
      target = z
      loss = 0.5*||eta(x_t)||^2 - <z, eta(x_t)>

    Args:
        model: Callable with signature model(t: [B, 1], x: [B, D]) -> [B, D].
        batch: dict with required keys {"x0", "x1"}.
        tau: [B, 1] time parameter.
        iw: [B, 1] importance weight (unused; for signature consistency).
        path: VfmPath1D object with gamma(tau) method.

    Returns:
        scalar loss tensor with active gradient.
    """
    x0 = batch["x0"]  # [B, D]
    x1 = batch["x1"]  # [B, D]

    # extract path property
    gamma_t = path.gamma(tau)  # [B, 1]

    # sample noise
    z = torch.randn_like(x0)  # [B, D]

    # interpolated sample
    x_t = (1 - tau) * x0 + tau * x1 + gamma_t * z  # [B, D]

    # forward pass
    eta_pred = model(tau, x_t)  # [B, D]

    # compute loss
    eta_norm_sq = (eta_pred ** 2).sum(dim=-1)  # [B]
    z_dot_eta = (z * eta_pred).sum(dim=-1)  # [B]
    loss = (0.5 * eta_norm_sq - z_dot_eta).mean()  # scalar

    return loss


denoiser_loss.required_keys = frozenset({"x0", "x1"})
denoiser_loss.requires_tau_grad = False


def flow_matching_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    score_weight: float = 1.0,
    p_uncond: float = 0.0,
    sentinel_cond: float = -1.0,
) -> torch.Tensor:
    """Conditional flow matching loss with optional CFG dropout (2-class).

    Computes joint velocity + score MSE loss for binary FM.

    Math:
      Path: x_t = (1-t)*z + t*x_data where z ~ N(0,I), x_data ~ {p_0, p_1}
      Targets: v_target = x_data - z, s_target = -z/(1-t)
      Loss: MSE(v_pred, v_target) + score_weight * MSE(s_pred, s_target)

    CFG: With prob p_uncond per sample, replace c with sentinel_cond.

    Args:
        model: Callable(t: [B, 1], x: [B, D], c: [B, 1]) -> (v_pred, s_pred) [B, D] each.
        batch: dict with required keys {"x0", "x1"}.
               Trainer constructs x_data = cat([x0, x1]) and c = cat([zeros, ones]).
        tau: [B, 1] time parameter (trainer clamps to [eps, 1-eps]).
        iw: [B, 1] importance weight (unused; for signature consistency).
        score_weight: float coefficient for score loss. Default 1.0.
        p_uncond: float, per-sample CFG dropout probability. Default 0.0.
        sentinel_cond: float, sentinel value for unconditional mode. Default -1.0.

    Returns:
        scalar loss tensor with active gradient.
    """
    x0 = batch["x0"]  # [B0, D]
    x1 = batch["x1"]  # [B1, D]

    # construct batch data and condition
    x_data = torch.cat([x0, x1], dim=0)  # [B0+B1, D]
    B0 = x0.shape[0]
    c = torch.cat([
        torch.zeros(B0, 1, device=x0.device, dtype=x0.dtype),
        torch.ones(x1.shape[0], 1, device=x1.device, dtype=x1.dtype),
    ], dim=0)  # [B0+B1, 1]

    # tile tau and iw to match concatenated batch (trainer passes [B,1]; here B = B0 = B1)
    tau = tau.repeat(2, 1)  # [B0+B1, 1]
    iw = iw.repeat(2, 1)    # [B0+B1, 1] (currently unused below; kept for symmetry)

    # apply CFG dropout
    if p_uncond > 0.0:
        mask = torch.bernoulli(torch.full_like(c, p_uncond))  # [B, 1]
        c = torch.where(mask > 0.5, torch.full_like(c, sentinel_cond), c)

    # sample noise
    z = torch.randn_like(x_data)  # [B, D]

    # interpolate
    x_t = (1 - tau) * z + tau * x_data  # [B, D]

    # compute targets
    v_target = x_data - z  # [B, D]
    s_target = -z / (1 - tau)  # [B, D]

    # forward pass
    v_pred, s_pred = model(tau, x_t, c)  # [B, D], [B, D]

    # compute losses
    loss_v = F.mse_loss(v_pred, v_target)  # scalar
    loss_s = F.mse_loss(s_pred, s_target)  # scalar

    loss = loss_v + score_weight * loss_s  # scalar

    return loss


flow_matching_loss.required_keys = frozenset({"x0", "x1"})
flow_matching_loss.requires_tau_grad = False


def triangular_flow_matching_loss(
    model: Callable[..., torch.Tensor],
    batch: dict[str, torch.Tensor],
    tau: torch.Tensor,
    iw: torch.Tensor,
    *,
    score_weight: float = 1.0,
    triangular_p_uncond: float = 0.0,
) -> torch.Tensor:
    """Triangular (3-class) flow matching loss with masked score loss.

    Computes joint velocity + masked-score MSE loss for 3-class FM.
    Score loss only applies to p_0 and p_1; p_* contributes zero via masking.

    Math:
      Path: x_t = (1-t)*z + t*x_data where x_data ~ {p_0, p_1, p_*}
      Targets: v_target = x_data - z, s_target = -z/(1-t)
      Velocity loss: MSE(v_pred, v_target)
      Score loss (masked): MSE(s_pred, s_target) only for p_0, p_1

    Args:
        model: Callable.forward_from_onehot(t: [B, 1], x: [B, D], y_onehot: [B, 3])
               -> (v_pred, s_pred) [B, D] each.
        batch: dict with required keys {"x0", "x1", "xstar"}.
               Trainer constructs x_data = cat([x0, x1, xstar]).
        tau: [B, 1] time parameter (trainer clamps to [eps, 1-eps]).
        iw: [B, 1] importance weight (unused; for signature consistency).
        score_weight: float coefficient for masked score loss. Default 1.0.
        triangular_p_uncond: float, per-sample CFG dropout (zero one-hot row). Default 0.0.

    Returns:
        scalar loss tensor with active gradient.
    """
    x0 = batch["x0"]  # [B0, D]
    x1 = batch["x1"]  # [B1, D]
    xstar = batch["xstar"]  # [B*, D]

    # construct batch data and class indices
    x_data = torch.cat([x0, x1, xstar], dim=0)  # [B0+B1+B*, D]
    B0, B1 = x0.shape[0], x1.shape[0]
    y_idx = torch.cat([
        torch.zeros(B0, dtype=torch.long, device=x0.device),
        torch.ones(B1, dtype=torch.long, device=x1.device),
        torch.full((xstar.shape[0],), 2, dtype=torch.long, device=xstar.device),
    ], dim=0)  # [B]

    # convert to one-hot
    y_onehot = F.one_hot(y_idx, num_classes=3).to(x_data.dtype)  # [B, 3]

    # tile tau and iw to match concatenated batch (trainer passes [B,1]; here B=B0=B1=B*)
    tau = tau.repeat(3, 1)  # [B0+B1+B*, 1]
    iw = iw.repeat(3, 1)    # [B0+B1+B*, 1]

    # apply CFG dropout on one-hot
    if triangular_p_uncond > 0.0:
        mask = torch.bernoulli(torch.full((y_onehot.shape[0], 1), triangular_p_uncond, device=y_onehot.device))  # [B, 1]
        y_onehot = torch.where(mask > 0.5, torch.zeros_like(y_onehot), y_onehot)

    # sample noise
    z = torch.randn_like(x_data)  # [B, D]

    # interpolate
    x_t = (1 - tau) * z + tau * x_data  # [B, D]

    # compute targets
    v_target = x_data - z  # [B, D]
    s_target = -z / (1 - tau)  # [B, D]

    # forward pass
    v_pred, s_pred = model.forward_from_onehot(tau, x_t, y_onehot)  # [B, D] each

    # compute velocity loss (unmasked)
    loss_v = F.mse_loss(v_pred, v_target)  # scalar

    # compute masked score loss: only p_0 and p_1
    s_mask = (y_idx != 2).to(x_data.dtype).unsqueeze(-1)  # [B, 1]
    diff_s = (s_pred - s_target) ** 2  # [B, D]
    diff_s = diff_s * s_mask  # zero out p_* rows
    n_active = s_mask.sum() * x_data.shape[1]  # count of nonzero entries
    loss_s = diff_s.sum() / n_active.clamp(min=1.0)  # scalar

    loss = loss_v + score_weight * loss_s  # scalar

    return loss


triangular_flow_matching_loss.required_keys = frozenset({"x0", "x1", "xstar"})
triangular_flow_matching_loss.requires_tau_grad = False


def triangular_hyvarinen_time_score_loss(
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
    """Hyvarinen time-score loss on 3-source bell path (TriangularTSM).

    Extends hyvarinen_time_score_loss to 3-source paths with bell-curve time warping.

    Path math (triangular_tsm.py:73-100):
      t, t_prime = path_t_tprime(tau, vertex, peak_max)
      x_t = sqrt(1 - t^2) * x0 + t * x1
      x_tau = sqrt(1 - t_prime^2) * x_t + t_prime * xstar

    Hyvarinen loss structure mirrors 2-source variant with 2D-time model interface.

    Args:
        model: Callable(x: [B, D], t: [B, 1], t_prime: [B, 1]) -> [B, 1].
        batch: dict with required keys {"x0", "x1", "xstar"}.
        tau: [B, 1] time parameter (trainer clamps to (eps, 1-eps)).
        iw: [B, 1] importance weight (unused; for signature consistency).
        reweight: if True, apply time-dependent lambda_t and lambda_dt weights.
        eps: scalar float boundary epsilon. Default 1e-5.
        vertex: float, position of bell curve peak in [0, 1]. Default 0.5.
        peak_max: float, maximum value of bell curve. Default 1.0.

    Returns:
        scalar loss tensor with active gradient.
    """
    x0 = batch["x0"]  # [B, D]
    x1 = batch["x1"]  # [B, D]
    xstar = batch["xstar"]  # [B, D]

    # compute t, t_prime via bell path (hardcoded from triangular_tsm.py:73-100)
    v = vertex
    m = peak_max
    tau_clamped = torch.clamp(tau, min=eps, max=1.0)
    left = m * (2.0 * (tau_clamped / v) - (tau_clamped / v) ** 2)
    right = m * (1.0 - ((tau_clamped - v) / (1.0 - v)) ** 2)
    t_prime = torch.where(tau_clamped <= v, left, right)
    t_prime = torch.clamp(t_prime, min=0.0, max=1.0)
    t = tau_clamped  # [B, 1]

    # prepare boundary times
    t0 = torch.zeros_like(tau) + eps
    t1 = torch.ones_like(tau)
    tp0 = torch.zeros_like(tau) + eps
    tp1 = torch.ones_like(tau)

    # compute lambda weights
    if reweight:
        lambda_t = (1 - tau_clamped ** 2).squeeze(-1)  # [B]
        lambda_t0 = (1 - t0.squeeze(-1) ** 2)  # [B]
        lambda_t1 = (1 - t1.squeeze(-1) ** 2 + eps ** 2)  # [B]
        lambda_dt = (-2 * tau_clamped).squeeze(-1)  # [B]
    else:
        lambda_t = lambda_t0 = lambda_t1 = 1.0
        lambda_dt = 0.0

    # boundary term 1: at x0 (t0, tp0)
    x_t0 = torch.sqrt(torch.clamp(1.0 - t0 ** 2, min=eps)) * x0 + t0 * x1
    score_t0_tp0 = model(x0, t0, tp0).squeeze(-1)  # [B]
    term1 = 2 * score_t0_tp0 * lambda_t0  # [B]

    # boundary term 2: at x1 (t1, tp1)
    x_t1 = torch.sqrt(torch.clamp(1.0 - t1 ** 2, min=eps)) * x0 + t1 * x1
    score_t1_tp1 = model(x1, t1, tp1).squeeze(-1)  # [B]
    term2 = 2 * score_t1_tp1 * lambda_t1  # [B]

    # interior: construct x_tau and score time-derivative
    sqrt_1_minus_t2 = torch.sqrt(torch.clamp(1.0 - t ** 2, min=eps))  # [B, 1]
    sqrt_1_minus_tp2 = torch.sqrt(torch.clamp(1.0 - t_prime ** 2, min=eps))  # [B, 1]
    x_t = sqrt_1_minus_t2 * x0 + t * x1  # [B, D]
    x_tau = sqrt_1_minus_tp2 * x_t + t_prime * xstar  # [B, D]

    # compute score and time-derivative using helper
    score_tau, score_tau_dt = _score_time_derivative(
        lambda x, tau_g, tp_g: model(x, tau_g, tp_g),
        x_tau,
        tau,
        t_prime,
    )  # [B, 1] each

    term3 = (2 * score_tau_dt).squeeze(-1) * lambda_t  # [B]
    term4 = score_tau.squeeze(-1) * lambda_dt  # [B]
    term5 = (score_tau ** 2).squeeze(-1) * lambda_t  # [B]

    # aggregate and return
    loss = (term1 - term2 + term3 + term4 + term5).mean()  # scalar

    return loss


triangular_hyvarinen_time_score_loss.required_keys = frozenset({"x0", "x1", "xstar"})
triangular_hyvarinen_time_score_loss.requires_tau_grad = True
