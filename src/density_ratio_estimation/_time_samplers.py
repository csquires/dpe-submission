"""time samplers: abstract (tau, iw) producers for the trainer.

contract: ``TimeSampler.sample(B, device) -> (tau [B,1], iw [B,1])`` with the
unbiasedness invariant ``E_{tau~q}[iw(tau) f(tau)] = int_S f(tau) p_target(tau) dtau``
for some support S and target measure p_target. concrete samplers fix both.

three concrete samplers:
  - UniformSampler: q = U([eps, 1-eps]); iw = 1. matches the legacy "uniform" dist.
  - BetaSampler: q = Beta(a, b) clamped to [eps, 1-eps]; iw = p_uniform / q.
    a=b=2 and a=b=5 recover the legacy "beta_2_2" and "beta_5_5".
  - PathSampler: delegates to ``path.sample_tau(B, eps, device)``. accepts the
    one-tensor (tau) or two-tensor (tau, iw) return shape already documented on
    sb_loss. iw defaults to ones if the path returns just tau.

composition / restriction (e.g. RestrictedSampler that wraps a base sampler with
a path-supplied exclusion zone, with iw normalized for the truncated support) is
a natural extension; not in this module to keep the initial surface small.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.distributions import Beta


class TimeSampler(ABC):
    """abstract (tau, iw) producer; subclasses must declare an ``eps`` field."""

    @abstractmethod
    def sample(self, batch_size: int, device) -> tuple[Tensor, Tensor]:
        """draw (tau [B,1], iw [B,1]). iw makes per-tau quantities unbiased
        estimators of the target measure (typically uniform on the sampler's support)."""

    # callers may treat the sampler as a callable for convenience.
    def __call__(self, batch_size: int, device) -> tuple[Tensor, Tensor]:
        return self.sample(batch_size, device)


@dataclass(frozen=True)
class UniformSampler(TimeSampler):
    """q = U([eps, 1-eps]); iw = 1. matches the legacy ``dist='uniform'`` case."""

    eps: float = 1e-3

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0; got {self.eps}")

    def sample(self, batch_size: int, device) -> tuple[Tensor, Tensor]:
        """uniform tau in [eps, 1-eps]; iw is ones since q == p_target."""
        tau = torch.rand(batch_size, 1, device=device) * (1.0 - 2.0 * self.eps) + self.eps
        iw = torch.ones(batch_size, 1, device=device)
        return tau, iw


@dataclass(frozen=True)
class BetaSampler(TimeSampler):
    """q = Beta(a, b) clamped to [eps, 1-eps]; iw = p_uniform / q (untruncated Beta pdf).

    Args:
        a, b: Beta concentration parameters. a == b yields a distribution
              symmetric about 0.5; larger values concentrate around the mean.
        eps: boundary margin; tau clamped to [eps, 1-eps].

    note: the iw is computed against the untruncated Beta pdf, matching the
    legacy ``sample_time_and_iw`` behavior. for sharp Beta the boundary mass
    is negligible so the truncation correction Z is ~1; this caveat carries
    forward unchanged.
    """

    a: float = 2.0
    b: float = 2.0
    eps: float = 1e-3

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0:
            raise ValueError(f"a, b must be > 0; got a={self.a}, b={self.b}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0; got {self.eps}")

    def sample(self, batch_size: int, device) -> tuple[Tensor, Tensor]:
        """sample Beta(a, b) tau and weight by p_uniform / q.

        concentration tensors live on `device` so the subsequent log_prob does
        not cross the cpu/cuda boundary (was the bug fixed in sample_time_and_iw).
        """
        dist = Beta(
            torch.tensor(float(self.a), device=device),
            torch.tensor(float(self.b), device=device),
        )
        tau_unclamped = dist.sample((batch_size,))  # [B] on device
        tau = torch.clamp(tau_unclamped, self.eps, 1.0 - self.eps).unsqueeze(-1)  # [B, 1]

        p_uniform = 1.0 / (1.0 - 2.0 * self.eps)
        log_q = dist.log_prob(tau.squeeze(-1))  # [B]
        q = torch.exp(log_q)  # [B]
        iw = (p_uniform / q).unsqueeze(-1)  # [B, 1]
        return tau, iw


@dataclass(frozen=True)
class PathSampler(TimeSampler):
    """delegates to ``path.sample_tau``; accepts (tau) or (tau, iw) return shape.

    when the path returns just tau, iw defaults to ones (the path is presumed
    to sample from its native target measure, so no correction is needed).
    when the path returns (tau, iw), pass them through.

    composition with TimeCfg-style distributions on the path's support is a
    future RestrictedSampler.
    """

    path: Any = field(hash=False, compare=False)  # path object; hash by identity below

    def __post_init__(self) -> None:
        if not callable(getattr(self.path, "sample_tau", None)):
            raise TypeError(
                f"path {type(self.path).__name__} does not provide a callable sample_tau"
            )

    @property
    def eps(self) -> float:
        """forward the path's own eps; falls back to 1e-3 if path lacks one."""
        return float(getattr(self.path, "eps", 1e-3))

    def sample(self, batch_size: int, device) -> tuple[Tensor, Tensor]:
        sampled = self.path.sample_tau(batch_size, self.eps, device)
        if isinstance(sampled, tuple):
            tau, iw = sampled
            return tau, iw
        tau = sampled
        iw = torch.ones(batch_size, 1, device=device)
        return tau, iw

    def __hash__(self) -> int:
        # path objects may not be hashable by structure; hash by identity for
        # run-id derivation use cases.
        return hash((type(self).__name__, id(self.path)))


@dataclass(frozen=True)
class NoIWSampler(TimeSampler):
    """wraps any TimeSampler; forces iw=1 regardless of the base's iw.

    use when non-uniform sampling is a deliberate loss-emphasis choice rather
    than a variance-reduction technique: the resulting estimator is biased
    against the uniform integral (the canonical CTSM / FM integrand) but
    unbiased against the q-weighted integral the sampler defines. composes
    with reweight (the lambda multiplier) the same way iw would.

    constructed implicitly via ``TimeCfg(apply_iw=False)``; may also be used
    directly: ``NoIWSampler(base=BetaSampler(a=2, b=2))``.
    """

    base: TimeSampler

    @property
    def eps(self) -> float:
        return self.base.eps

    def sample(self, batch_size: int, device) -> tuple[Tensor, Tensor]:
        """delegate to base for tau; replace iw with ones."""
        tau, _ = self.base.sample(batch_size, device)
        iw = torch.ones(batch_size, 1, device=device)
        return tau, iw


def sampler_from_dist(dist: str, eps: float = 1e-3) -> TimeSampler:
    """legacy string -> TimeSampler factory for {uniform, beta_2_2, beta_5_5}.

    convenience for code paths that previously used ``TimeCfg(dist=...)``;
    construct the matching sampler instance.
    """
    if dist == "uniform":
        return UniformSampler(eps=eps)
    if dist == "beta_2_2":
        return BetaSampler(a=2.0, b=2.0, eps=eps)
    if dist == "beta_5_5":
        return BetaSampler(a=5.0, b=5.0, eps=eps)
    raise ValueError(
        f"dist must be in {{'uniform', 'beta_2_2', 'beta_5_5'}} or use an explicit "
        f"TimeSampler; got {dist!r}"
    )


def sample_time_and_iw(
    time_dist: str,
    batch_size: int,
    eps: float,
    device,
) -> tuple[Tensor, Tensor]:
    """legacy string-driven (tau, iw) sampler; thin shim over sampler_from_dist.

    kept for callers (analysis scripts, external code) that prefer the flat
    function form. behaviorally identical to ``sampler_from_dist(time_dist,
    eps).sample(batch_size, device)``.
    """
    return sampler_from_dist(time_dist, eps).sample(batch_size, device)
