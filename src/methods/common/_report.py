"""evaluation reporting instrumentation factory for training loops."""
import torch
from torch import nn
from typing import Callable, Any


def _noop() -> None:
    """no-op callable used as fallback when step_cb is disabled."""
    return None


def _make_report(
    step_cb: Callable[[int, float], None] | None,
    step_cb_interval: int | set[int] | frozenset[int],
    eval_fn: Callable[[Any], torch.Tensor] | None,
    model: Any,
    model_module: nn.Module | None,
) -> Callable[[], None]:
    """bind evaluation reporting into a 0-arg closure; returns _noop when disabled.

    closure maintains nonlocal step counter and guards report by either
    (step > 0 and step % step_cb_interval == 0) when step_cb_interval is an int,
    or (step in step_cb_interval) when it is a set of explicit step indices
    (used by the optuna stage to fire only on Hyperband rung boundaries).
    snapshots model.training state before toggling to eval, wraps eval_fn in
    torch.no_grad(), extracts scalar via .item(), and restores training state in
    finally block.
    """
    if step_cb is None or eval_fn is None:
        return _noop

    if isinstance(step_cb_interval, int):
        gate = lambda s: s > 0 and s % step_cb_interval == 0
    else:
        rung_set = frozenset(step_cb_interval)
        gate = lambda s: s in rung_set

    step = 0

    def do_report() -> None:
        nonlocal step
        step += 1

        if gate(step):
            # snapshot training state before toggling
            if isinstance(model, nn.Module):
                was_training = model.training
            else:
                was_training = model_module.training

            try:
                # set eval mode via isinstance branch (mirror _trainer.py)
                if isinstance(model, nn.Module):
                    model.eval()
                else:
                    model_module.eval()

                # eval in no_grad context
                with torch.no_grad():
                    score_tensor = eval_fn(model)
                score = float(score_tensor.item())

                # invoke callback with (step, float_score)
                step_cb(step, score)
            finally:
                # restore prior training state
                if isinstance(model, nn.Module):
                    model.train(was_training)
                else:
                    model_module.train(was_training)

    return do_report


def _make_report_pair(
    step_cb: Callable[[int, float], None] | None,
    step_cb_interval: int | set[int] | frozenset[int],
    eval_fn: Callable[[Any], torch.Tensor] | None,
    modules: list[nn.Module],
) -> Callable[[], None]:
    """bind multi-network evaluation reporting into a 0-arg closure.

    mirrors _make_report but toggles EVERY module in `modules` to eval before
    the eval_fn call and restores each module's prior training state in a
    finally block. used by train_interleaved, whose held-out eval needs all
    networks (b and eta) in eval mode, not just one. eval_fn is called with
    None: interleaved eval closures ignore the model arg and capture the
    estimator's predict path directly. step_cb_interval accepts int (fire at
    K, 2K, ...) or set[int] (fire only at given step indices, e.g. Hyperband
    rungs).
    """
    if step_cb is None or eval_fn is None:
        return _noop

    if isinstance(step_cb_interval, int):
        gate = lambda s: s > 0 and s % step_cb_interval == 0
    else:
        rung_set = frozenset(step_cb_interval)
        gate = lambda s: s in rung_set

    step = 0

    def do_report() -> None:
        nonlocal step
        step += 1

        if gate(step):
            was = [m.training for m in modules]
            try:
                for m in modules:
                    m.eval()
                with torch.no_grad():
                    score = float(eval_fn(None).item())
                step_cb(step, score)
            finally:
                for m, w in zip(modules, was):
                    m.train(w)

    return do_report
