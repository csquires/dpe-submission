"""generic training loops for score-matching and flow-matching DRE estimators.

`train_loop` drives a single network through (sample, time, loss, step) iterations.
`train_two_phase` orchestrates two sequential `train_loop` calls for VFM-family
estimators that fit a velocity network then a denoiser.
"""
from typing import Callable

import torch
from torch import nn, optim

from ._ema import EMA, maybe_clip_grad


def train_loop(
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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: EMA | None = None,
    grad_clip_norm: float | None = None,
    eps: float = 1e-3,
    loss_kwargs: dict | None = None,
    model_module: nn.Module | None = None,
) -> None:
    """train one network for `n_steps` mini-batch steps.

    loss contract: loss_fn(model, batch_dict, tau, iw, **loss_kwargs) -> scalar,
    with module-level attributes:
      - required_keys: frozenset subset of {"x0", "x1", "xstar"}
      - requires_tau_grad: bool

    procedure:
      1. validate samples + loss_fn contract.
      2. cast samples to float on model device.
      3. model.train(); for n_steps:
           - bootstrap [B, D] subsets of p0/p1 (and pstar if available).
           - draw (tau, iw) from time_sampler.
           - loss = loss_fn(model, batch, tau, iw, **loss_kwargs); .backward();
             optional grad clip; optim.step(); optional scheduler / EMA step.
      4. model.eval().

    Args:
        model: trainable callable; if not an nn.Module, pass `model_module`.
        samples_p0, samples_p1: [N, D] each; pstar [N*, D] or None.
        loss_fn: see contract above.
        optim, n_steps, batch_size: as named.
        time_sampler: (B, eps, device) -> (tau [B,1], iw [B,1]).
        scheduler, ema, grad_clip_norm, loss_kwargs: optional auxiliaries.
        eps: tau-domain margin forwarded to time_sampler.
        model_module: nn.Module backing `model` (required when model is a closure).
    """
    if samples_p0.shape[1] != samples_p1.shape[1]:
        raise ValueError(
            f"p0 dimension {samples_p0.shape[1]} != p1 dimension {samples_p1.shape[1]}"
        )

    if not hasattr(loss_fn, "required_keys") or not hasattr(loss_fn, "requires_tau_grad"):
        raise AttributeError("loss_fn must have 'required_keys' and 'requires_tau_grad'")

    if not isinstance(loss_fn.required_keys, frozenset):
        raise ValueError(
            f"loss_fn.required_keys must be frozenset; got {type(loss_fn.required_keys)}"
        )
    if not loss_fn.required_keys.issubset({"x0", "x1", "xstar"}):
        raise ValueError(
            f"loss_fn.required_keys must be subset of {{'x0','x1','xstar'}}; "
            f"got {loss_fn.required_keys}"
        )

    if "xstar" in loss_fn.required_keys:
        if samples_pstar is None:
            raise ValueError("loss_fn requires 'xstar' but samples_pstar is None")
        if samples_pstar.shape[1] != samples_p0.shape[1]:
            raise ValueError(
                f"samples_pstar dimension {samples_pstar.shape[1]} != "
                f"samples_p0 dimension {samples_p0.shape[1]}"
            )

    if isinstance(model, nn.Module):
        device = next(model.parameters()).device
        if model_module is None:
            model_module = model
    else:
        if model_module is None:
            raise ValueError("model_module is required when model is not an nn.Module")
        device = next(model_module.parameters()).device

    samples_p0 = samples_p0.float().to(device)
    samples_p1 = samples_p1.float().to(device)
    if samples_pstar is not None:
        samples_pstar = samples_pstar.float().to(device)

    n0 = samples_p0.shape[0]
    n1 = samples_p1.shape[0]
    n_star = samples_pstar.shape[0] if samples_pstar is not None else 0

    if isinstance(model, nn.Module):
        model.train()
    elif model_module is not None:
        model_module.train()

    loss_kw = loss_kwargs if loss_kwargs is not None else {}

    for _ in range(n_steps):
        idx0 = torch.randint(0, n0, (batch_size,), device=device)
        idx1 = torch.randint(0, n1, (batch_size,), device=device)

        # pass xstar through when available, even if required_keys excludes it:
        # sb_loss declares {"x0","x1"} statically but reads xstar at runtime for
        # triangular paths.
        include_xstar = samples_pstar is not None
        if include_xstar:
            idx_star = torch.randint(0, n_star, (batch_size,), device=device)

        batch = {"x0": samples_p0[idx0], "x1": samples_p1[idx1]}
        if include_xstar:
            batch["xstar"] = samples_pstar[idx_star]

        tau, iw = time_sampler(batch_size, eps, device)

        loss = loss_fn(model, batch, tau, iw, **loss_kw)

        optim.zero_grad()
        loss.backward()
        maybe_clip_grad(model_module.parameters(), grad_clip_norm)
        optim.step()

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model_module)

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
    scheduler_b: optim.lr_scheduler.LRScheduler | None = None,
    scheduler_eta: optim.lr_scheduler.LRScheduler | None = None,
    ema_b: EMA | None = None,
    ema_eta: EMA | None = None,
    grad_clip_norm_b: float | None = None,
    grad_clip_norm_eta: float | None = None,
    eps: float = 1e-3,
    loss_kwargs_b: dict | None = None,
    loss_kwargs_eta: dict | None = None,
) -> None:
    """train velocity then denoiser sequentially via two `train_loop` calls.

    invariants enforced:
      - optim_b owns exactly model_b.parameters(); same for eta.
      - both models on the same device.

    invariants assumed (responsibility of the loss writer, not the trainer):
      - loss_b does not call model_eta and vice versa.

    phase 1: model_eta.eval(); train_loop(model_b, loss_b, optim_b, ...).
    phase 2: model_b.eval();   train_loop(model_eta, loss_eta, optim_eta, ...).
    final  : both .eval().
    """
    device_b = next(model_b.parameters()).device
    device_eta = next(model_eta.parameters()).device
    if device_b != device_eta:
        raise ValueError(
            f"model_b on {device_b} but model_eta on {device_eta}; must match"
        )

    b_ids = {id(p) for p in model_b.parameters()}
    eta_ids = {id(p) for p in model_eta.parameters()}
    ob_ids = {id(p) for g in optim_b.param_groups for p in g["params"]}
    oe_ids = {id(p) for g in optim_eta.param_groups for p in g["params"]}
    assert ob_ids == b_ids, "optim_b must own exactly model_b.parameters()"
    assert oe_ids == eta_ids, "optim_eta must own exactly model_eta.parameters()"

    model_eta.eval()
    model_b.train()
    train_loop(
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

    model_b.eval()
    model_eta.train()
    train_loop(
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

    model_b.eval()
    model_eta.eval()
