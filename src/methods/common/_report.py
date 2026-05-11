"""evaluation reporting instrumentation factory for training loops."""
import torch
from torch import nn
from typing import Callable, Any


def _noop() -> None:
    """no-op callable used as fallback when step_cb is disabled."""
    return None


def _make_report(
    step_cb: Callable[[int, float], None] | None,
    step_cb_interval: int,
    eval_fn: Callable[[Any], torch.Tensor] | None,
    model: Any,
    model_module: nn.Module | None,
) -> Callable[[], None]:
    """bind evaluation reporting into a 0-arg closure; returns _noop when disabled.

    closure maintains nonlocal step counter and guards report with
    (step > 0 and step % step_cb_interval == 0). snapshots model.training state
    before toggling to eval, wraps eval_fn in torch.no_grad(), extracts scalar
    via .item(), and restores training state in finally block.
    """
    # when step_cb is disabled, no instrumentation
    if step_cb is None or eval_fn is None:
        return _noop

    # closure-local state: step counter
    step = 0

    def do_report() -> None:
        nonlocal step
        step += 1

        # report only at intervals K, 2K, 3K, ...; guard skips step 0
        if step > 0 and step % step_cb_interval == 0:
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
