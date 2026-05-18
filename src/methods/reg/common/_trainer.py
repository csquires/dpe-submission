"""generic training loops for score-matching and flow-matching DRE estimators.

`train_loop` drives a single network through (sample, time, loss, step) iterations.
`train_two_phase` orchestrates two sequential `train_loop` calls for VFM-family
estimators that fit a velocity network then a denoiser.
`train_interleaved` advances both networks together in one loop so the held-out
eval (which needs both) is available throughout, making the whole trial
prunable by hyperband rather than only its second half.

also hosts `maybe_clip_grad`, a small gradient-clipping helper used by the inner
loop and by estimators with bespoke training bodies (TriangularCTSM2D, TriangularVFM2D).
"""
from typing import Any, Callable, Iterable

import torch
from torch import nn, optim

from ._ema import EMA
from ...common._report import _make_report, _make_report_pair


def _noop() -> None:
    """no-op callable used as a fallback when an optional hook is disabled."""
    return None


def maybe_clip_grad(
    params: Iterable[torch.nn.Parameter],
    max_norm: float | None,
) -> None:
    """clip gradient norm in-place if `max_norm` is set; no-op otherwise."""
    if max_norm is None or max_norm <= 0:
        return
    torch.nn.utils.clip_grad_norm_(list(params), max_norm=max_norm)


def _make_clip(params, max_norm: float | None) -> Callable[[], None]:
    """bind gradient-clipping into a 0-arg closure; returns _noop when disabled.

    materializes the parameter list once (rather than iterating an iterable per
    step) so the hot path performs only the clip call.
    """
    if max_norm is None or max_norm <= 0:
        return _noop
    param_list = list(params)

    def clip() -> None:
        torch.nn.utils.clip_grad_norm_(param_list, max_norm=max_norm)

    return clip


def _make_build_batch(
    samples_p0: torch.Tensor,
    samples_p1: torch.Tensor,
    samples_pstar: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
    needs_xstar: bool,
) -> Callable[[], dict[str, torch.Tensor]]:
    """bind batch-bootstrap into a 0-arg closure with xstar-inclusion decided once."""
    n0 = samples_p0.shape[0]
    n1 = samples_p1.shape[0]

    if needs_xstar:
        n_star = samples_pstar.shape[0]

        def build() -> dict[str, torch.Tensor]:
            idx0 = torch.randint(0, n0, (batch_size,), device=device)
            idx1 = torch.randint(0, n1, (batch_size,), device=device)
            idx_star = torch.randint(0, n_star, (batch_size,), device=device)
            return {
                "x0": samples_p0[idx0],
                "x1": samples_p1[idx1],
                "xstar": samples_pstar[idx_star],
            }
    else:
        def build() -> dict[str, torch.Tensor]:
            idx0 = torch.randint(0, n0, (batch_size,), device=device)
            idx1 = torch.randint(0, n1, (batch_size,), device=device)
            return {"x0": samples_p0[idx0], "x1": samples_p1[idx1]}

    return build


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
    step_cb: Callable[[int, float], None] | None = None,
    eval_fn: Callable[[Any], torch.Tensor] | None = None,
    step_cb_interval: int = 50,
) -> None:
    """train one network for `n_steps` mini-batch steps.

    loss contract: loss_fn(model, batch, tau, iw, **loss_kwargs) -> scalar,
    with module-level attributes:
      - required_keys: frozenset subset of {"x0", "x1", "xstar"}
      - requires_tau_grad: bool

    procedure:
      1. validate samples + loss_fn contract.
      2. cast samples to float on model device.
      3. pre-bind hot-path callables (build_batch, do_clip, do_sched, do_ema, do_report)
         so the inner loop performs no `is not None` checks.
      4. model.train(); for n_steps:
           - build_batch() draws (x0, x1[, xstar]).
           - draw (tau, iw) from time_sampler.
           - loss = loss_fn(model, batch, tau, iw, **loss_kwargs); .backward();
             do_clip(); optim.step(); do_sched(); do_ema(); do_report().
      5. on every step_cb_interval steps (starting from step_cb_interval, not step 0),
         invoke do_report() to sample eval_fn, compute score, and forward to step_cb.
         TrialPruned and any other exception propagate uncaught.
      6. model.eval().

    Args:
        model: trainable callable; if not an nn.Module, pass `model_module`.
        samples_p0, samples_p1: [N, D] each; pstar [N*, D] or None.
        loss_fn: see contract above.
        optim, n_steps, batch_size: as named.
        time_sampler: (B, eps, device) -> (tau [B,1], iw [B,1]).
        scheduler, ema, grad_clip_norm, loss_kwargs: optional auxiliaries.
        eps: tau-domain margin forwarded to time_sampler.
        model_module: nn.Module backing `model` (required when model is a closure).
        step_cb, eval_fn, step_cb_interval: enable periodic held-out evaluation
        for Optuna trial pruning. step_cb=None (default) disables instrumentation;
        method's training loop performs zero-overhead no-op call per step.
        when step_cb is not None, eval_fn must also be provided.
    """
    if samples_p0.shape[1] != samples_p1.shape[1]:
        raise ValueError(
            f"p0 dimension {samples_p0.shape[1]} != p1 dimension {samples_p1.shape[1]}"
        )

    if not hasattr(loss_fn, "required_keys") or not hasattr(loss_fn, "requires_tau_grad"):
        raise AttributeError("loss_fn must have 'required_keys' and 'requires_tau_grad'")

    required_keys = loss_fn.required_keys
    if not isinstance(required_keys, frozenset):
        raise ValueError(
            f"loss_fn.required_keys must be frozenset; got {type(required_keys)}"
        )
    if not required_keys.issubset({"x0", "x1", "xstar"}):
        raise ValueError(
            f"loss_fn.required_keys must be subset of {{'x0','x1','xstar'}}; "
            f"got {required_keys}"
        )

    needs_xstar = "xstar" in required_keys
    if needs_xstar:
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

    if isinstance(model, nn.Module):
        model.train()
    elif model_module is not None:
        model_module.train()

    loss_kw = loss_kwargs if loss_kwargs is not None else {}

    build_batch = _make_build_batch(
        samples_p0, samples_p1, samples_pstar, batch_size, device, needs_xstar,
    )
    do_clip = _make_clip(model_module.parameters(), grad_clip_norm)
    do_sched = scheduler.step if scheduler is not None else _noop
    do_ema = (lambda: ema.update(model_module)) if ema is not None else _noop
    do_report = _make_report(step_cb, step_cb_interval, eval_fn, model, model_module)

    try:
        for _ in range(n_steps):
            batch = build_batch()
            tau, iw = time_sampler(batch_size, device)

            loss = loss_fn(model, batch, tau, iw, **loss_kw)

            optim.zero_grad()
            loss.backward()
            do_clip()
            optim.step()
            do_sched()
            do_ema()
            do_report()
    finally:
        # restore eval mode even if do_report raises optuna.TrialPruned.
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
    model_module_b: nn.Module | None = None,
    model_module_eta: nn.Module | None = None,
    step_cb: Callable[[int, float], None] | None = None,
    eval_fn: Callable[[Any], torch.Tensor] | None = None,
    step_cb_interval: int = 50,
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

    model_module_b / model_module_eta: backing nn.Module when model_b / model_eta
    is a non-Module callable (e.g. a preconditioning wrapper). used for device
    resolution, the optim-ownership assertion, train()/eval() toggling, and is
    forwarded to train_loop as `model_module`. defaults to model_b / model_eta.

    step_cb, eval_fn, step_cb_interval: optuna pruning hooks, forwarded to the
    phase-2 train_loop only. step_cb=None (default) disables instrumentation.
    phase 1 (model_b) is never instrumented: the held-out eval needs both
    networks, which is meaningless until model_eta is trained. reported step
    indices are phase-2-local (the phase-2 train_loop's own 0-based counter), so
    hyperband sees the same rung ladder for this method as for any single-phase
    method.
    """
    # backing nn.Module for each phase; defaults to the model itself when the
    # model is already an nn.Module (no preconditioning wrapper).
    mod_b = model_module_b if model_module_b is not None else model_b
    mod_eta = model_module_eta if model_module_eta is not None else model_eta

    device_b = next(mod_b.parameters()).device
    device_eta = next(mod_eta.parameters()).device
    if device_b != device_eta:
        raise ValueError(
            f"model_b on {device_b} but model_eta on {device_eta}; must match"
        )

    b_ids = {id(p) for p in mod_b.parameters()}
    eta_ids = {id(p) for p in mod_eta.parameters()}
    ob_ids = {id(p) for g in optim_b.param_groups for p in g["params"]}
    oe_ids = {id(p) for g in optim_eta.param_groups for p in g["params"]}
    assert ob_ids == b_ids, "optim_b must own exactly model_b.parameters()"
    assert oe_ids == eta_ids, "optim_eta must own exactly model_eta.parameters()"

    mod_eta.eval()
    mod_b.train()
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
        model_module=mod_b,
    )

    mod_b.eval()
    mod_eta.train()
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
        model_module=mod_eta,
        step_cb=step_cb,
        eval_fn=eval_fn,
        step_cb_interval=step_cb_interval,
    )

    mod_b.eval()
    mod_eta.eval()


def train_interleaved(
    model_b: Callable[..., torch.Tensor],
    model_eta: Callable[..., torch.Tensor],
    samples_p0: torch.Tensor,
    samples_p1: torch.Tensor,
    samples_pstar: torch.Tensor | None,
    loss_b: Callable,
    loss_eta: Callable,
    optim_b: optim.Optimizer,
    optim_eta: optim.Optimizer,
    n_steps: int,
    batch_size: int,
    time_sampler: Callable[..., tuple[torch.Tensor, torch.Tensor]],
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
    model_module_b: nn.Module | None = None,
    model_module_eta: nn.Module | None = None,
    step_cb: Callable[[int, float], None] | None = None,
    eval_fn: Callable[[Any], torch.Tensor] | None = None,
    step_cb_interval: int = 50,
) -> None:
    """train velocity (b) and denoiser (eta) jointly via one interleaved loop.

    loss_b and loss_eta are independent -- neither references the other network
    -- so advancing both per step is algebraically identical to two sequential
    train_loop calls: each net only ever sees its own loss and optimizer.
    interleaving only changes WHEN the held-out eval (which needs both nets)
    becomes available. with both nets advancing together it is computable from
    the start, so do_report forwards a score to step_cb every step_cb_interval
    steps and hyperband can prune the whole trial -- not just its second half.

    each iteration: one b-update, then one eta-update, then do_report().

    invariants:
      - optim_b owns exactly model_b.parameters(); same for eta.
      - both models on the same device.
      - both models are left in eval() on exit, INCLUDING when step_cb raises
        optuna.TrialPruned mid-loop (try/finally) -- callers rely on the
        "outside fit, models in eval" invariant.

    args mirror train_two_phase except n_steps replaces n_steps_b / n_steps_eta:
    b and eta advance in lockstep, so the two counts are necessarily equal.
    """
    mod_b = model_module_b if model_module_b is not None else model_b
    mod_eta = model_module_eta if model_module_eta is not None else model_eta

    device_b = next(mod_b.parameters()).device
    device_eta = next(mod_eta.parameters()).device
    if device_b != device_eta:
        raise ValueError(
            f"model_b on {device_b} but model_eta on {device_eta}; must match"
        )

    b_ids = {id(p) for p in mod_b.parameters()}
    eta_ids = {id(p) for p in mod_eta.parameters()}
    ob_ids = {id(p) for g in optim_b.param_groups for p in g["params"]}
    oe_ids = {id(p) for g in optim_eta.param_groups for p in g["params"]}
    assert ob_ids == b_ids, "optim_b must own exactly model_b.parameters()"
    assert oe_ids == eta_ids, "optim_eta must own exactly model_eta.parameters()"

    for name, fn in (("loss_b", loss_b), ("loss_eta", loss_eta)):
        if not hasattr(fn, "required_keys") or not hasattr(fn, "requires_tau_grad"):
            raise AttributeError(
                f"{name} must have 'required_keys' and 'requires_tau_grad'"
            )
        if not fn.required_keys.issubset({"x0", "x1", "xstar"}):
            raise ValueError(
                f"{name}.required_keys must be subset of {{'x0','x1','xstar'}}"
            )

    needs_xstar = ("xstar" in loss_b.required_keys) or ("xstar" in loss_eta.required_keys)
    if needs_xstar and samples_pstar is None:
        raise ValueError("a loss requires 'xstar' but samples_pstar is None")

    device = device_b
    samples_p0 = samples_p0.float().to(device)
    samples_p1 = samples_p1.float().to(device)
    if samples_pstar is not None:
        samples_pstar = samples_pstar.float().to(device)

    loss_kw_b = loss_kwargs_b if loss_kwargs_b is not None else {}
    loss_kw_eta = loss_kwargs_eta if loss_kwargs_eta is not None else {}

    build_batch = _make_build_batch(
        samples_p0, samples_p1, samples_pstar, batch_size, device, needs_xstar,
    )
    do_clip_b = _make_clip(mod_b.parameters(), grad_clip_norm_b)
    do_clip_eta = _make_clip(mod_eta.parameters(), grad_clip_norm_eta)
    do_sched_b = scheduler_b.step if scheduler_b is not None else _noop
    do_sched_eta = scheduler_eta.step if scheduler_eta is not None else _noop
    do_ema_b = (lambda: ema_b.update(mod_b)) if ema_b is not None else _noop
    do_ema_eta = (lambda: ema_eta.update(mod_eta)) if ema_eta is not None else _noop
    do_report = _make_report_pair(
        step_cb, step_cb_interval, eval_fn, [mod_b, mod_eta],
    )

    mod_b.train()
    mod_eta.train()
    try:
        for _ in range(n_steps):
            # b-update
            batch_b = build_batch()
            tau_b, iw_b = time_sampler(batch_size, device)
            loss = loss_b(model_b, batch_b, tau_b, iw_b, **loss_kw_b)
            optim_b.zero_grad()
            loss.backward()
            do_clip_b()
            optim_b.step()
            do_sched_b()
            do_ema_b()
            # eta-update
            batch_eta = build_batch()
            tau_eta, iw_eta = time_sampler(batch_size, device)
            loss = loss_eta(model_eta, batch_eta, tau_eta, iw_eta, **loss_kw_eta)
            optim_eta.zero_grad()
            loss.backward()
            do_clip_eta()
            optim_eta.step()
            do_sched_eta()
            do_ema_eta()
            # report; may raise optuna.TrialPruned, caught by the finally below.
            do_report()
    finally:
        mod_b.eval()
        mod_eta.eval()
