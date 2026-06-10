"""sequential (non-interleaved) trainer for two-phase optimization.

train_two_phase_3 runs phase 1 (b1+b2 joint, eta frozen) for n_steps_b iterations,
then phase 2 (b1+b2 frozen, eta training) for n_steps_eta iterations. each phase may
terminate early via per-phase early-stop detectors. returns None; exports final step
counts and stop reason via optional _meta_out dict.
"""
from typing import Any, Callable, Iterable

import torch
from torch import nn, optim

from ._ema import EMA
from ...common.early_stop import make_early_stopper


def train_two_phase_3(
    net_b1: nn.Module,
    net_b2: nn.Module,
    net_eta: nn.Module,
    loss_b: Callable[[], torch.Tensor],
    loss_eta: Callable[[], torch.Tensor],
    optim_b: optim.Optimizer,
    optim_eta: optim.Optimizer,
    n_steps_b: int,
    n_steps_eta: int,
    *,
    scheduler_b: optim.lr_scheduler.LRScheduler | None = None,
    scheduler_eta: optim.lr_scheduler.LRScheduler | None = None,
    ema_b1: EMA | None = None,
    ema_b2: EMA | None = None,
    ema_eta: EMA | None = None,
    b_params: Iterable[torch.nn.Parameter] | None = None,
    eta_params: Iterable[torch.nn.Parameter] | None = None,
    grad_clip_norm: float | None = None,
    step_cb: Callable[[int, float], None] | None = None,
    eval_fn: Callable[[Any], torch.Tensor] | None = None,
    step_cb_interval: int = 50,
    early_stop_cfg: dict | None = None,
    _meta_out: dict | None = None,
) -> None:
    """sequential trainer: phase 1 (b1+b2), then phase 2 (eta).

    procedure:
      phase 1: net_eta.eval(); net_b1.train(); net_b2.train().
        for step in range(n_steps_b):
          - loss_b() -> backward() -> clip() -> optim_b.step() -> scheduler -> ema.
          - early-stop check; break if should_stop.
      phase 2: net_b1.eval(); net_b2.eval(); net_eta.train().
        for step in range(n_steps_eta):
          - loss_eta() -> backward() -> clip() -> optim_eta.step() -> scheduler -> ema.
          - early-stop check; break if should_stop.
      finally: all nets to eval().

    args:
      net_b1, net_b2: velocity networks (trained together in phase 1).
      net_eta: denoiser network (trained in phase 2).
      loss_b, loss_eta: 0-arg callables returning scalar loss tensors.
      optim_b: owns net_b1.parameters() + net_b2.parameters().
      optim_eta: owns net_eta.parameters().
      n_steps_b, n_steps_eta: max iterations per phase.
      scheduler_b, scheduler_eta: optional LR schedulers, stepped after optim.step().
      ema_b1, ema_b2, ema_eta: optional EMA wrappers, updated after their group step.
      b_params, eta_params: pre-built param lists for grad clipping (None -> skip).
      grad_clip_norm: max-norm for gradient clipping. None or <=0 -> skip.
      step_cb, eval_fn, step_cb_interval: optuna pruning hooks (not wired; raises if step_cb not None).
      early_stop_cfg: dict enabling early-stop detectors. None -> no early stopping.
      _meta_out: optional dict mutated with "final_step" and "stop_reason" keys.

    returns: None (side effects only).

    raises:
      NotImplementedError: if step_cb is not None (not supported in sequential mode).
    """
    if step_cb is not None:
        raise NotImplementedError(
            "train_two_phase_3 does not support step_cb; use train_interleaved_3 for hyperband-pruned trials"
        )

    # phase 1: b1+b2 joint, eta frozen
    net_eta.eval()
    net_b1.train()
    net_b2.train()

    observe_b, should_stop_b = make_early_stopper(early_stop_cfg)
    final_step_b = n_steps_b
    reason_b = None

    try:
        for step in range(n_steps_b):
            optim_b.zero_grad()
            lb = loss_b()
            lb.backward()

            if grad_clip_norm is not None and b_params is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(list(b_params), max_norm=grad_clip_norm)

            optim_b.step()

            if scheduler_b is not None:
                scheduler_b.step()

            if ema_b1 is not None:
                ema_b1.update(net_b1)

            if ema_b2 is not None:
                ema_b2.update(net_b2)

            observe_b(step + 1, lb.item())
            stop, reason = should_stop_b()
            if stop:
                final_step_b = step + 1
                reason_b = reason
                break

        # phase 2: b1+b2 frozen, eta training
        net_b1.eval()
        net_b2.eval()
        net_eta.train()

        observe_e, should_stop_e = make_early_stopper(early_stop_cfg)
        final_step_e = n_steps_eta
        reason_e = None

        for step in range(n_steps_eta):
            optim_eta.zero_grad()
            le = loss_eta()
            le.backward()

            if grad_clip_norm is not None and eta_params is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(list(eta_params), max_norm=grad_clip_norm)

            optim_eta.step()

            if scheduler_eta is not None:
                scheduler_eta.step()

            if ema_eta is not None:
                ema_eta.update(net_eta)

            observe_e(step + 1, le.item())
            stop, reason = should_stop_e()
            if stop:
                final_step_e = step + 1
                reason_e = reason
                break

    finally:
        net_b1.eval()
        net_b2.eval()
        net_eta.eval()

    if _meta_out is not None:
        _meta_out["final_step"] = final_step_b + final_step_e
        _meta_out["stop_reason"] = reason_b if reason_b is not None else reason_e
