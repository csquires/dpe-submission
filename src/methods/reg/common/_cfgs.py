"""configuration dataclasses and factory functions for optimizer, scheduler, ema, and time-sampling hyperparameters."""
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch import Tensor

from ._ema import EMA
from ._time_samplers import (
    TimeSampler, UniformSampler, BetaSampler, PathSampler, NoIWSampler,
    _FuncSampler, sampler_from_dist, _BETA_DIST_PARAMS,
)


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
    """time-sampling cfg: holds a TimeSampler instance plus an iw toggle.

    the sampler abstracts the (q, iw) pair; concrete samplers in
    `_time_samplers.py` cover uniform, Beta, and path-driven cases, and the
    surface is open to user-supplied samplers (any TimeSampler subclass).

    `apply_iw` (default True) controls whether the sampler's iw is honored.
    when False, the sampler is wrapped with NoIWSampler so iw is forced to 1,
    making the loss reflect the q distribution directly (biased against the
    uniform integral; unbiased against the q-weighted integral).

    legacy string-based construction is available via TimeCfg.from_dist for
    convenience and HPO migration.
    """

    sampler: TimeSampler = field(default_factory=UniformSampler)
    apply_iw: bool = True

    def __post_init__(self) -> None:
        """validate sampler type."""
        if not isinstance(self.sampler, TimeSampler):
            raise TypeError(
                f"sampler must be a TimeSampler instance; got {type(self.sampler).__name__}"
            )

    @classmethod
    def from_dist(cls, dist: str = "uniform", eps: float = 1e-3, apply_iw: bool = True) -> "TimeCfg":
        """legacy convenience constructor: TimeCfg.from_dist('beta_2_2', eps=1e-3)."""
        return cls(sampler=sampler_from_dist(dist, eps), apply_iw=apply_iw)

    @property
    def effective_sampler(self) -> TimeSampler:
        """the sampler actually used by make_time_sampler.

        wraps the configured sampler with NoIWSampler when apply_iw is False.
        UniformSampler always returns iw=1 anyway, so wrapping is a no-op there;
        kept uniform for consistency / hashability.
        """
        if self.apply_iw:
            return self.sampler
        return NoIWSampler(base=self.sampler)

    @property
    def eps(self) -> float:
        """forward the sampler's eps for trainer/loss eps argument needs."""
        return float(getattr(self.sampler, "eps", 1e-3))

    @property
    def dist(self) -> str:
        """diagnostic label of the underlying sampler; not used by the trainer.

        returns the legacy enum string when the sampler is one of the canonical
        ones, otherwise a class-name-derived label.
        """
        s = self.sampler
        if isinstance(s, UniformSampler):
            return "uniform"
        if isinstance(s, BetaSampler):
            # recover the canonical enum (e.g. beta_half_half) when params match;
            # int() truncation would mislabel non-integer betas (0.5 -> "0").
            for name, (a, b) in _BETA_DIST_PARAMS.items():
                if (s.a, s.b) == (a, b):
                    return name
            return f"beta_{s.a:g}_{s.b:g}"
        if isinstance(s, _FuncSampler):
            return s.label or "density"
        if isinstance(s, PathSampler):
            return f"path:{type(s.path).__name__}"
        return type(s).__name__


def make_optim(params, cfg: OptimCfg) -> torch.optim.Adam:
    """instantiate an adam optimizer with cfg hyperparameters.

    returns an Adam instance with lr, betas, weight_decay from cfg.
    eps is hardcoded to 1e-8 per codebase convention.

    note: cfg.grad_clip_norm is not applied here; caller reads and applies
    separately via maybe_clip_grad from _trainer.
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


def make_time_sampler(cfg: TimeCfg) -> Callable[[int, torch.device], tuple[Tensor, Tensor]]:
    """return a callable that delegates to cfg.effective_sampler.sample(B, device).

    signature is (batch_size, device), matching the trainer's call shape in
    train_loop / train_two_phase. the sampler owns its own eps internally.

    when cfg.apply_iw is False, effective_sampler wraps the configured sampler
    with NoIWSampler to force iw=1.

    returns:
        callable with signature (batch_size: int, device: torch.device)
        -> tuple[Tensor, Tensor] yielding (tau [B,1], iw [B,1]).
    """
    sampler_obj = cfg.effective_sampler

    def sampler(batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        return sampler_obj.sample(batch_size, device)

    return sampler
