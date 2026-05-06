"""minimal exponential-moving-average helper for score-based DRE estimators.

EMA tracks a shadow copy of the trained network's parameters with decay rate
`d`: shadow := d * shadow + (1-d) * current. At inference, the shadow is
swapped into the live module via `apply_to`, with `restore` returning the
training weights for any further training/diagnostics.

Standard recipe from EDM / score-based diffusion. Decay 0.999 is the toy
default; 0.9999 is canonical for production-scale runs.
"""
from typing import Iterable

import torch
from torch import nn
from torch.distributions import Beta


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


def maybe_clip_grad(
    params: Iterable[torch.nn.Parameter],
    max_norm: float | None,
) -> None:
    """clip gradient norm in-place if `max_norm` is set."""
    if max_norm is None or max_norm <= 0:
        return
    torch.nn.utils.clip_grad_norm_(list(params), max_norm=max_norm)


def sample_time_and_iw(
    time_dist: str,
    batch_size: int,
    eps: float,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """sample tau and compute importance weights for unbiased IS estimator.

    sample tau and return (tau [B,1], iw [B,1]) where iw is per-sample
    importance weight = p_uniform(tau) / q(tau).

    Args:
        time_dist: in {"uniform", "beta_2_2", "beta_5_5"}.
        batch_size: B, number of samples.
        eps: margin for tau bounds [eps, 1-eps].
        device: torch device.

    Returns:
        tuple (tau [B,1], iw [B,1]). for time_dist="uniform", iw=1.
        otherwise, tau ~ Beta(a,b) clamped to [eps, 1-eps] and iw =
        p_uniform(tau) / q(tau) where q is the truncated Beta pdf
        (approximated as untruncated for small eps).
    """
    if time_dist == "uniform":
        tau = torch.rand(batch_size, 1, device=device) * (1.0 - 2.0 * eps) + eps
        iw = torch.ones(batch_size, 1, device=device)
        return tau, iw

    # parse Beta(a, b) from time_dist string
    if time_dist == "beta_2_2":
        a, b = 2, 2
    elif time_dist == "beta_5_5":
        a, b = 5, 5
    else:
        raise ValueError(
            f"time_dist must be in {{'uniform', 'beta_2_2', 'beta_5_5'}}; "
            f"got {time_dist!r}"
        )

    # sample from Beta(a, b) on [0, 1]
    dist = Beta(torch.tensor(float(a)), torch.tensor(float(b)))
    tau_unclamped = dist.sample((batch_size,)).to(device)  # [B]

    # clamp to [eps, 1-eps] and reshape
    tau = torch.clamp(tau_unclamped, eps, 1.0 - eps).unsqueeze(-1)  # [B, 1]

    # compute importance weights: p_uniform(tau) / q(tau)
    # p_uniform = 1 / (1 - 2*eps)
    # q(tau) = pdf_Beta(tau; a, b)
    p_uniform = 1.0 / (1.0 - 2.0 * eps)
    log_q = dist.log_prob(tau.squeeze(-1))  # [B]
    q = torch.exp(log_q)  # [B]
    iw = (p_uniform / q).unsqueeze(-1)  # [B, 1]

    return tau, iw
