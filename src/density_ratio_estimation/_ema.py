"""minimal exponential-moving-average helper for score-based DRE estimators.

EMA tracks a shadow copy of the trained network's parameters with decay rate
`d`: shadow := d * shadow + (1-d) * current. At inference, the shadow is
swapped into the live module via `apply_to`, with `restore` returning the
training weights for any further training/diagnostics.

Standard recipe from EDM / score-based diffusion. Decay 0.999 is the toy
default; 0.9999 is canonical for production-scale runs.
"""
import torch
from torch import nn


class EMA:
    """exponential-moving-average helper for nn.Module parameters.

    Procedure:
        - on construction, snapshot module.parameters() as `shadow` (cloned, detached).
        - `update(module)` blends current params into shadow with `decay`.
        - `apply_to(module)` saves training weights to `_backup` and copies
          shadow into module.
        - `restore(module)` writes `_backup` back into module and clears it.

    Args:
        module: nn.Module whose parameters to track.
        decay: float in (0, 1). Higher = slower averaging, more memory of past.
    """

    def __init__(self, module: nn.Module, decay: float) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        self.decay = decay
        self.shadow: list[torch.Tensor] = [
            p.detach().clone() for p in module.parameters()
        ]
        self._backup: list[torch.Tensor] | None = None

    def update(self, module: nn.Module) -> None:
        """blend current `module` parameters into `shadow` in-place.

        shadow_i := decay * shadow_i + (1 - decay) * param_i.
        """
        with torch.no_grad():
            for s, p in zip(self.shadow, module.parameters()):
                s.mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    def apply_to(self, module: nn.Module) -> None:
        """copy `shadow` into `module`; save current params for `restore`."""
        if self._backup is not None:
            raise RuntimeError(
                "apply_to called twice without restore; would lose backup"
            )
        with torch.no_grad():
            self._backup = [p.detach().clone() for p in module.parameters()]
            for s, p in zip(self.shadow, module.parameters()):
                p.data.copy_(s.data)

    def restore(self, module: nn.Module) -> None:
        """restore the training-time params saved by `apply_to`. clears backup."""
        if self._backup is None:
            raise RuntimeError("restore called without prior apply_to")
        with torch.no_grad():
            for b, p in zip(self._backup, module.parameters()):
                p.data.copy_(b.data)
        self._backup = None


