"""configuration dataclasses and factory functions for optimizer, scheduler, ema, and time-sampling hyperparameters."""
from dataclasses import dataclass
from typing import Callable

import torch
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch import Tensor

from ._ema import EMA, sample_time_and_iw


@dataclass(frozen=True, kw_only=True)
class OptimCfg:
    """optimizer hyperparameters. instantiate via make_optim factory."""

    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None

    def __post_init__(self) -> None:
        """validate optimizer hyperparameters."""
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0; got {self.lr}")
        if not (0.0 < self.betas[0] < 1.0):
            raise ValueError(f"betas[0] must be in (0, 1); got {self.betas[0]}")
        if not (0.0 < self.betas[1] < 1.0):
            raise ValueError(f"betas[1] must be in (0, 1); got {self.betas[1]}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0; got {self.weight_decay}")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError(f"grad_clip_norm must be > 0 or None; got {self.grad_clip_norm}")


@dataclass(frozen=True, kw_only=True)
class SchedCfg:
    """learning-rate scheduler hyperparameters. instantiate via make_sched factory."""

    cosine_min_factor: float = 1.0

    def __post_init__(self) -> None:
        """validate scheduler hyperparameters."""
        if not (0.0 <= self.cosine_min_factor <= 1.0):
            raise ValueError(f"cosine_min_factor must be in [0, 1]; got {self.cosine_min_factor}")


@dataclass(frozen=True, kw_only=True)
class EmaCfg:
    """exponential-moving-average hyperparameters. instantiate via make_ema factory."""

    decay: float | None = None

    def __post_init__(self) -> None:
        """validate ema hyperparameters."""
        if self.decay is not None and not (0.0 < self.decay < 1.0):
            raise ValueError(f"decay must be in (0, 1); got {self.decay}")


@dataclass(frozen=True, kw_only=True)
class TimeCfg:
    """time-sampling hyperparameters. instantiate via make_time_sampler factory."""

    dist: str = "uniform"
    eps: float = 1e-3

    def __post_init__(self) -> None:
        """validate time-sampling hyperparameters."""
        if self.dist not in {"uniform", "beta_2_2", "beta_5_5"}:
            raise ValueError(f"dist must be in {{'uniform', 'beta_2_2', 'beta_5_5'}}; got {self.dist!r}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0; got {self.eps}")


def make_optim(params, cfg: OptimCfg) -> torch.optim.Adam:
    """instantiate an adam optimizer with cfg hyperparameters.

    returns an Adam instance with lr, betas, weight_decay from cfg.
    eps is hardcoded to 1e-8 per codebase convention.

    note: cfg.grad_clip_norm is not applied here; caller reads and applies
    separately via maybe_clip_grad from _ema.
    """
    return torch.optim.Adam(
        params,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )


def make_sched(opt, n_steps: int, lr: float, cfg: SchedCfg) -> LRScheduler | None:
    """instantiate a learning-rate scheduler, or None if annealing is disabled.

    if cfg.cosine_min_factor == 1.0, returns None (annealing off).
    otherwise, returns a CosineAnnealingLR with T_max=n_steps and
    eta_min=lr * cfg.cosine_min_factor.
    """
    if cfg.cosine_min_factor == 1.0:
        return None
    else:
        return CosineAnnealingLR(
            opt,
            T_max=n_steps,
            eta_min=lr * cfg.cosine_min_factor,
        )


def make_ema(model, cfg: EmaCfg) -> EMA | None:
    """instantiate ema wrapper, or None if ema is disabled.

    if cfg.decay is None, returns None (ema off).
    otherwise, returns an EMA instance with cfg.decay.
    """
    if cfg.decay is None:
        return None
    else:
        return EMA(model, cfg.decay)


def make_time_sampler(cfg: TimeCfg) -> Callable[[int, float, torch.device], tuple[Tensor, Tensor]]:
    """return a closure that samples time and importance weights on demand.

    the returned callable has signature (batch_size, eps, device) and forwards
    all three to sample_time_and_iw(cfg.dist, ...). eps is passed at call time
    (from trainer), not from cfg.eps; cfg.eps is the per-estimator default that
    trainer reads and forwards to the sampler.

    returns:
        callable with signature (batch_size: int, eps: float, device: torch.device)
        -> tuple[Tensor, Tensor] yielding (tau [B,1], iw [B,1]).
    """
    def sampler(batch_size: int, eps: float, device: torch.device) -> tuple[Tensor, Tensor]:
        return sample_time_and_iw(cfg.dist, batch_size, eps, device)

    return sampler
