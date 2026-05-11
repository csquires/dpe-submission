"""unified training loop for score-matching and flow-matching DRE estimators."""
from typing import Callable

import torch
from torch import nn, optim

from ._ema import EMA, maybe_clip_grad


def train_score_flow(
    model: Callable[..., torch.Tensor],
    samples_p0: torch.Tensor,
    samples_p1: torch.Tensor,
    samples_pstar: torch.Tensor | None,
    loss_fn: Callable,
    optim: torch.optim.Optimizer,
    n_steps: int,
    batch_size: int,
    time_sampler: Callable[[int, float, torch.device], tuple[torch.Tensor, torch.Tensor]],
    *,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ema: EMA | None = None,
    grad_clip_norm: float | None = None,
    eps: float = 1e-3,
    loss_kwargs: dict | None = None,
    model_module: nn.Module | None = None,
) -> None:
    """unified training loop for score-matching and flow-matching DRE estimators.

    handles batching, device management, time sampling, loss computation,
    backpropagation, and optional auxiliary updates (grad clipping, EMA,
    learning-rate scheduling). the trainer is agnostic to the loss function,
    delegating all loss-specific logic to the `loss_fn` callable via a
    declarative interface (required_keys, requires_tau_grad).

    procedure:
      1. validate sample dimensions and loss function interface.
      2. cast samples to .float(), move to model.device.
      3. set model.train().
      4. for each of n_steps:
         - sample batch indices (with replacement).
         - build batch_dict with keys required by loss_fn.
         - sample tau and importance weights via time_sampler.
         - compute loss = loss_fn(model, batch_dict, tau, iw, **loss_kwargs).
         - backprop: zero_grad, backward, clip_grad (if enabled), step.
         - update EMA (if enabled) and schedule (if enabled).
      5. set model.eval().

    loss function contract:
      loss_fn must be callable:
        loss_fn(model, batch_dict, tau, iw, **loss_kwargs) -> torch.Tensor (scalar)

      and must have module-level constants:
        - loss_fn.required_keys: frozenset[str]
          subset of {"x0", "x1", "xstar"}. trainer populates only these keys.

        - loss_fn.requires_tau_grad: bool
          if True, loss internally sets tau.requires_grad_(True).
          if False, tau is non-differentiable. trainer does not modify tau.requires_grad.

    Args:
        model: trainable callable (score or velocity network). accepts (x, tau, ...)
               and returns [B, 1] or compatible shape. can be nn.Module or closure.
        samples_p0: source distribution samples [N0, D].
        samples_p1: target distribution samples [N1, D].
        samples_pstar: optional intermediate distribution samples [Nstar, D] for 3-source losses.
                       if None and loss requires "xstar", raises ValueError.
        loss_fn: loss function callable with required_keys and requires_tau_grad.
        optim: optimizer with zero_grad(), step() methods.
        n_steps: number of gradient steps (mini-batches).
        batch_size: mini-batch size. with-replacement sampling allows batch_size > dataset size.
        time_sampler: callable(batch_size, eps, device) -> (tau [B,1], iw [B,1]).
        scheduler: optional learning-rate scheduler; .step() called after each optim.step().
        ema: optional exponential-moving-average helper; .update() called after each optim.step().
        grad_clip_norm: gradient max-norm for clipping. None or <= 0 disables clipping.
        eps: time-domain margin for tau sampling [eps, 1-eps].
        loss_kwargs: extra keyword arguments forwarded to loss_fn.
        model_module: underlying nn.Module for optimizer/EMA access when model is a closure.
                      if None and model is an nn.Module, defaults to model. required if
                      model is a closure and ema/grad_clip_norm are used.

    Raises:
        ValueError: if sample dimensions do not match, if loss requires "xstar" but
                    samples_pstar is None, if loss_fn.required_keys is invalid.
        AttributeError: if loss_fn does not have required_keys or requires_tau_grad.

    Example (smoke test):
        >>> import torch
        >>> from torch import nn, optim
        >>> from src.density_ratio_estimation._trainer import train_score_flow
        >>>
        >>> model = nn.Linear(3, 1)
        >>> def my_loss(model, batch, tau, iw):
        ...     return ((model(batch["x0"]) - 0)**2).mean()
        >>> my_loss.required_keys = frozenset({"x0", "x1"})
        >>> my_loss.requires_tau_grad = False
        >>>
        >>> train_score_flow(
        ...     model=model,
        ...     samples_p0=torch.randn(100, 3),
        ...     samples_p1=torch.randn(100, 3),
        ...     samples_pstar=None,
        ...     loss_fn=my_loss,
        ...     optim=optim.Adam(model.parameters(), lr=1e-3),
        ...     n_steps=10,
        ...     batch_size=8,
        ...     time_sampler=lambda B, eps, dev: (
        ...         torch.rand(B, 1, device=dev), torch.ones(B, 1, device=dev)
        ...     ),
        ... )
        # runs without error; model is in .eval() mode.
    """

    # ========== stage 1: validation ==========

    # check dimension match
    if samples_p0.shape[1] != samples_p1.shape[1]:
        raise ValueError(
            f"p0 dimension {samples_p0.shape[1]} != p1 dimension {samples_p1.shape[1]}"
        )

    # check loss_fn attributes
    if not hasattr(loss_fn, "required_keys") or not hasattr(loss_fn, "requires_tau_grad"):
        raise AttributeError(
            "loss_fn must have 'required_keys' and 'requires_tau_grad' attributes"
        )

    # check required_keys is frozenset and subset
    if not isinstance(loss_fn.required_keys, frozenset):
        raise ValueError(
            f"loss_fn.required_keys must be frozenset; got {type(loss_fn.required_keys)}"
        )
    if not loss_fn.required_keys.issubset({"x0", "x1", "xstar"}):
        raise ValueError(
            f"loss_fn.required_keys must be subset of {{'x0', 'x1', 'xstar'}}; "
            f"got {loss_fn.required_keys}"
        )

    # check xstar availability
    if "xstar" in loss_fn.required_keys:
        if samples_pstar is None:
            raise ValueError("loss_fn requires 'xstar' but samples_pstar is None")
        if samples_pstar.shape[1] != samples_p0.shape[1]:
            raise ValueError(
                f"samples_pstar dimension {samples_pstar.shape[1]} != "
                f"samples_p0 dimension {samples_p0.shape[1]}"
            )

    # ========== stage 2: device and type casting ==========

    # extract device from model if it's an nn.Module; otherwise from model_module
    if isinstance(model, nn.Module):
        device = next(model.parameters()).device
        if model_module is None:
            model_module = model
    else:
        # model is a closure; must have model_module for device/grad access
        if model_module is None:
            raise ValueError(
                "model_module must be provided when model is a Callable (not nn.Module)"
            )
        device = next(model_module.parameters()).device

    # cast and move samples to device
    samples_p0 = samples_p0.float().to(device)  # [N0, D]
    samples_p1 = samples_p1.float().to(device)  # [N1, D]
    if samples_pstar is not None:
        samples_pstar = samples_pstar.float().to(device)  # [Nstar, D]

    # sample counts
    n0 = samples_p0.shape[0]
    n1 = samples_p1.shape[0]
    n_star = samples_pstar.shape[0] if samples_pstar is not None else 0

    # ========== stage 3: training mode ==========

    if isinstance(model, nn.Module):
        model.train()
    elif model_module is not None:
        model_module.train()

    # ========== stage 4: main training loop ==========

    loss_kw = loss_kwargs if loss_kwargs is not None else {}

    for step in range(n_steps):
        # subsection 4a: index sampling (with replacement)
        idx0 = torch.randint(0, n0, (batch_size,), device=device)
        idx1 = torch.randint(0, n1, (batch_size,), device=device)

        # opportunistic xstar: pass through whenever samples_pstar was supplied,
        # even if loss.required_keys doesn't list it (e.g. closed_form_sb_loss
        # statically declares {"x0","x1"} but reads xstar at runtime when path
        # is 3-source).
        include_xstar = samples_pstar is not None
        if include_xstar:
            idx_star = torch.randint(0, n_star, (batch_size,), device=device)

        # subsection 4b: construct batch_dict
        batch_dict = {
            "x0": samples_p0[idx0],  # [B, D]
            "x1": samples_p1[idx1],  # [B, D]
        }
        if include_xstar:
            batch_dict["xstar"] = samples_pstar[idx_star]  # [B, D]

        # subsection 4c: sample tau and importance weights
        tau, iw = time_sampler(batch_size, eps, device)  # tau [B, 1], iw [B, 1]

        # do not modify tau.requires_grad here.
        # the loss function manages tau.requires_grad internally based on
        # loss_fn.requires_tau_grad.

        # subsection 4d: compute loss
        loss = loss_fn(model, batch_dict, tau, iw, **loss_kw)  # scalar []

        # subsection 4e: backpropagation
        optim.zero_grad()
        loss.backward()

        # subsection 4f: gradient clipping (if enabled)
        maybe_clip_grad(model_module.parameters(), grad_clip_norm)

        # subsection 4g: optimizer step
        optim.step()

        # subsection 4h: learning-rate scheduler (if enabled)
        if scheduler is not None:
            scheduler.step()

        # subsection 4i: EMA update (if enabled)
        if ema is not None:
            ema.update(model_module)

    # ========== stage 5: evaluation mode ==========

    if isinstance(model, nn.Module):
        model.eval()
    elif model_module is not None:
        model_module.eval()


def train_two_phase(
    model_b: nn.Module,
    model_eta: nn.Module,
    samples_p0: torch.Tensor,
    samples_p1: torch.Tensor,
    samples_pstar: torch.Tensor | None,
    loss_b: Callable,
    loss_eta: Callable,
    optim_b: optim.Optimizer,
    optim_eta: optim.Optimizer,
    n_steps_b: int,
    n_steps_eta: int,
    batch_size: int,
    time_sampler: Callable[[int, float, torch.device], tuple[torch.Tensor, torch.Tensor]],
    *,
    scheduler_b: optim.lr_scheduler._LRScheduler | None = None,
    scheduler_eta: optim.lr_scheduler._LRScheduler | None = None,
    ema_b: EMA | None = None,
    ema_eta: EMA | None = None,
    grad_clip_norm_b: float | None = None,
    grad_clip_norm_eta: float | None = None,
    eps: float = 1e-3,
    loss_kwargs_b: dict | None = None,
    loss_kwargs_eta: dict | None = None,
) -> None:
    """sequential velocity-then-denoiser training for VFM-family DRE estimators.

    procedure:
      phase 1 (b): freeze model_eta, train model_b via train_score_flow.
      phase 2 (eta): freeze model_b, train model_eta via train_score_flow.
      final: set both models to .eval().

    each phase delegates to train_score_flow for batching, time sampling, loss
    computation, and optional auxiliary updates (EMA, scheduler, grad clipping).

    invariants:
      - per-phase parameter ownership: optim_b MUST own only model_b.parameters();
        optim_eta MUST own only model_eta.parameters(). trainer asserts this.
      - no cross-net forward calls: loss_b MUST NOT call model_eta (and vice versa).
        this is a contract on the loss writer; trainer documents but cannot enforce.
      - explicit train/eval toggling: before phase 1, model_b.train() and
        model_eta.eval(). before phase 2, model_b.eval() and model_eta.train().
        after both: both .eval().

    Args:
        model_b: velocity network to train in phase 1 (frozen in phase 2).
        model_eta: denoiser network to train in phase 2 (frozen in phase 1).
        samples_p0: source distribution samples [N0, D].
        samples_p1: target distribution samples [N1, D].
        samples_pstar: optional intermediate distribution samples [Nstar, D].
                       if None and a loss requires "xstar", train_score_flow raises ValueError.
        loss_b: loss function for phase 1. must have required_keys and requires_tau_grad attributes.
        loss_eta: loss function for phase 2. must have required_keys and requires_tau_grad attributes.
        optim_b: optimizer for model_b.
        optim_eta: optimizer for model_eta.
        n_steps_b: number of gradient steps in phase 1.
        n_steps_eta: number of gradient steps in phase 2.
        batch_size: mini-batch size (shared across both phases).
        time_sampler: callable(batch_size, eps, device) -> (tau [B,1], iw [B,1]).
        scheduler_b: optional learning-rate scheduler for phase 1.
        scheduler_eta: optional learning-rate scheduler for phase 2.
        ema_b: optional exponential-moving-average helper for phase 1.
        ema_eta: optional exponential-moving-average helper for phase 2.
        grad_clip_norm_b: gradient clipping max-norm for phase 1.
        grad_clip_norm_eta: gradient clipping max-norm for phase 2.
        eps: time-domain margin for tau sampling [eps, 1-eps]. default: 1e-3.
        loss_kwargs_b: extra keyword arguments forwarded to loss_b.
        loss_kwargs_eta: extra keyword arguments forwarded to loss_eta.

    Raises:
        ValueError: if model_b and model_eta are on different devices.
        (other exceptions propagated from train_score_flow per phase.)
    """

    # ========== device validation ==========

    device_b = next(model_b.parameters()).device
    device_eta = next(model_eta.parameters()).device

    if device_b != device_eta:
        raise ValueError(
            f"model_b and model_eta must be on the same device; "
            f"got model_b on {device_b} and model_eta on {device_eta}"
        )

    # ========== parameter ownership assertion ==========

    # collect model_b and model_eta parameter ids
    model_b_param_ids = set(id(p) for p in model_b.parameters())
    model_eta_param_ids = set(id(p) for p in model_eta.parameters())

    # collect optim_b and optim_eta parameter ids
    optim_b_param_ids = set()
    for param_group in optim_b.param_groups:
        for p in param_group['params']:
            optim_b_param_ids.add(id(p))

    optim_eta_param_ids = set()
    for param_group in optim_eta.param_groups:
        for p in param_group['params']:
            optim_eta_param_ids.add(id(p))

    # assert invariant: optim_b owns only model_b.parameters()
    assert optim_b_param_ids == model_b_param_ids, (
        "optim_b must own exactly model_b.parameters(); "
        f"optim_b has {len(optim_b_param_ids)} params, model_b has {len(model_b_param_ids)} params"
    )

    # assert invariant: optim_eta owns only model_eta.parameters()
    assert optim_eta_param_ids == model_eta_param_ids, (
        "optim_eta must own exactly model_eta.parameters(); "
        f"optim_eta has {len(optim_eta_param_ids)} params, model_eta has {len(model_eta_param_ids)} params"
    )

    # ========== phase 1: train model_b, freeze model_eta ==========

    model_eta.eval()
    model_b.train()

    train_score_flow(
        model=model_b,
        samples_p0=samples_p0,
        samples_p1=samples_p1,
        samples_pstar=samples_pstar,
        loss_fn=loss_b,
        optim=optim_b,
        n_steps=n_steps_b,
        batch_size=batch_size,
        time_sampler=time_sampler,
        scheduler=scheduler_b,
        ema=ema_b,
        grad_clip_norm=grad_clip_norm_b,
        eps=eps,
        loss_kwargs=loss_kwargs_b,
    )

    # ========== phase 2: train model_eta, freeze model_b ==========

    model_b.eval()
    model_eta.train()

    train_score_flow(
        model=model_eta,
        samples_p0=samples_p0,
        samples_p1=samples_p1,
        samples_pstar=samples_pstar,
        loss_fn=loss_eta,
        optim=optim_eta,
        n_steps=n_steps_eta,
        batch_size=batch_size,
        time_sampler=time_sampler,
        scheduler=scheduler_eta,
        ema=ema_eta,
        grad_clip_norm=grad_clip_norm_eta,
        eps=eps,
        loss_kwargs=loss_kwargs_eta,
    )

    # ========== final state: both models to eval ==========

    model_b.eval()
    model_eta.eval()
