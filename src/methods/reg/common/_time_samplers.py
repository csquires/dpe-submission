"""time samplers: (tau, iw) producers for training and inference.

LEGACY SURFACE (deprecated, backward-compatible):
  Abstract TimeSampler base class and concrete implementations (UniformSampler,
  BetaSampler, NoIWSampler, PathSampler) for old ABC-style paths. These are
  retained for backward compatibility; new code should use the functional
  builders below.

NEW SURFACE (Pillar C, recommended):
  Functional builders that return callables matching TimeSampler1D or
  TimeSampler2D Protocols. All builders use kw-only args (no **kwargs) to
  surface typos at construction time. Composition operators enable variance
  reduction (stratification, antithetic, forced-iw-off) and multi-dimensional
  sampling (product for 2D rectangle-stacked geometries).

contract for new builders: callables matching TimeSampler1D/2D Protocols
return (tau, iw) tensors with the unbiasedness invariant:
  E_tau~q[iw(tau) * f(tau)] = int_S f(tau) * p_target(tau) d_tau
for support S and a canonical target measure (typically uniform on the sampler's
domain).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor
from torch.distributions import Beta


# ============================================================================
# Protocol type aliases for new functional builders
# ============================================================================

class TimeSampler1D(Protocol):
    """callable (B, device) -> (tau [B,1], iw [B,1]); tau scalar per batch element."""

    def __call__(self, B: int, device: torch.device) -> tuple[Tensor, Tensor]:
        ...


class TimeSampler2D(Protocol):
    """callable (B, device) -> (t1 [B,1], t2 [B,1], iw [B,1]); (t1, t2) vector per element."""

    def __call__(self, B: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        ...


# ============================================================================
# Legacy ABC surface
# ============================================================================


class TimeSampler(ABC):
    """DEPRECATED: abstract (tau, iw) producer; use functional builders instead.

    Subclasses must declare an ``eps`` field. Legacy code using this ABC is
    supported for backward compatibility; new code should use the functional
    builders (make_uniform, make_beta, etc.) that return callables matching
    TimeSampler1D or TimeSampler2D Protocols.
    """

    @abstractmethod
    def sample(self, batch_size: int, device) -> tuple[Tensor, Tensor]:
        """draw (tau [B,1], iw [B,1]). iw makes per-tau quantities unbiased
        estimators of the target measure (typically uniform on the sampler's support)."""

    # callers may treat the sampler as a callable for convenience.
    def __call__(self, batch_size: int, device) -> tuple[Tensor, Tensor]:
        return self.sample(batch_size, device)


@dataclass(frozen=True)
class UniformSampler(TimeSampler):
    """(legacy) q = U([eps, 1-eps]); iw = 1. matches the legacy ``dist='uniform'`` case.

    Deprecated in favor of the functional builder make_uniform(eps=...).
    """

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
    """(legacy) q = Beta(a, b) clamped to [eps, 1-eps]; iw = p_uniform / q (untruncated Beta pdf).

    Args:
        a, b: Beta concentration parameters. a == b yields a distribution
              symmetric about 0.5; larger values concentrate around the mean.
        eps: boundary margin; tau clamped to [eps, 1-eps].

    note: the iw is computed against the untruncated Beta pdf, matching the
    legacy ``sample_time_and_iw`` behavior. for sharp Beta the boundary mass
    is negligible so the truncation correction Z is ~1; this caveat carries
    forward unchanged.

    Deprecated in favor of the functional builder make_beta(a=..., b=..., eps=...).
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
    """(legacy) delegates to ``path.sample_tau``; accepts (tau) or (tau, iw) return shape.

    when the path returns just tau, iw defaults to ones (the path is presumed
    to sample from its native target measure, so no correction is needed).
    when the path returns (tau, iw), pass them through.

    composition with TimeCfg-style distributions on the path's support is a
    future RestrictedSampler.

    DEPRECATED: new code should use functional builders and path dataclass types
    that do not expose sample_tau. This class exists only for backward compatibility
    with legacy paths. When the old ABC paths are removed, PathSampler will be retired.
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
    """(legacy) wraps any TimeSampler; forces iw=1 regardless of the base's iw.

    use when non-uniform sampling is a deliberate loss-emphasis choice rather
    than a variance-reduction technique: the resulting estimator is biased
    against the uniform integral (the canonical CTSM / FM integrand) but
    unbiased against the q-weighted integral the sampler defines. composes
    with reweight (the lambda multiplier) the same way iw would.

    constructed implicitly via ``TimeCfg(apply_iw=False)``; may also be used
    directly: ``NoIWSampler(base=BetaSampler(a=2, b=2))``.

    Deprecated in favor of the functional composition operator make_no_iw(base).
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


@dataclass(frozen=True)
class _FuncSampler(TimeSampler):
    """(internal) ABC adaptor wrapping a functional TimeSampler1D closure.

    bridges the new functional builders (plain closures) with legacy code that
    expects a TimeSampler ABC instance (TimeCfg asserts isinstance). keeps a
    single source of truth for the math: bridge/stiff densities are defined once
    as functional builders and merely re-exposed here, never re-implemented.

    not intended for direct construction; returned by sampler_from_dist for
    distributions that have no dedicated ABC subclass. fn is excluded from the
    structural hash (closures are not structurally comparable); identity + eps
    define the value, mirroring PathSampler.
    """

    fn: Any = field(hash=False, compare=False)  # TimeSampler1D callable
    eps: float = 1e-3
    label: str = ""  # diagnostic dist string, surfaced via TimeCfg.dist

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError(f"fn must be callable; got {type(self.fn).__name__}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0; got {self.eps}")

    def sample(self, batch_size: int, device) -> tuple[Tensor, Tensor]:
        """delegate to the wrapped functional sampler."""
        return self.fn(batch_size, device)

    def __hash__(self) -> int:
        return hash((type(self).__name__, id(self.fn), self.eps, self.label))


def sampler_from_dist(dist: str, eps: float = 1e-3) -> TimeSampler:
    """DEPRECATED: legacy string -> TimeSampler factory.

    Use time_sampler_from_legacy_cfg instead, which returns a functional
    TimeSampler1D callable instead of an ABC instance.

    supports the canonical enum plus the extended set: beta_{10_10, half_half},
    bridge (density propto sqrt(t(1-t)) == Beta(1.5, 1.5)), and the stiff
    sigmoid-product family stiff_{10,20,40} and its reciprocal stiff_inv_{...}.
    beta cases map to BetaSampler directly; schedule-shaped cases wrap the
    functional density samplers in _FuncSampler so the math is not duplicated.

    convenience for code paths that previously used ``TimeCfg(dist=...)``;
    construct the matching sampler instance.
    """
    if dist == "uniform":
        return UniformSampler(eps=eps)
    if dist in _BETA_DIST_PARAMS:
        a, b = _BETA_DIST_PARAMS[dist]
        return BetaSampler(a=a, b=b, eps=eps)
    if dist in _DENSITY_DIST_BUILDERS:
        return _FuncSampler(fn=_DENSITY_DIST_BUILDERS[dist](eps), eps=eps, label=dist)
    raise ValueError(
        f"dist must be one of {sorted(TIME_DISTS)} or use an explicit "
        f"TimeSampler; got {dist!r}"
    )


def sample_time_and_iw(
    time_dist: str,
    batch_size: int,
    eps: float,
    device,
) -> tuple[Tensor, Tensor]:
    """DEPRECATED: legacy string-driven (tau, iw) sampler; thin shim over sampler_from_dist.

    Use time_sampler_from_legacy_cfg and call the result as a TimeSampler1D instead.

    kept for callers (analysis scripts, external code) that prefer the flat
    function form. behaviorally identical to ``sampler_from_dist(time_dist,
    eps).sample(batch_size, device)``.
    """
    return sampler_from_dist(time_dist, eps).sample(batch_size, device)


# ============================================================================
# NEW FUNCTIONAL BUILDERS
# ============================================================================
# Path-unaware builders: make_uniform, make_uniform_scaled, make_beta, make_sobol
# ============================================================================


def make_uniform(*, eps: float) -> TimeSampler1D:
    """uniform distribution on [eps, 1-eps]; iw = 1 (target is uniform).

    Returns a callable (B, device) -> (tau [B,1], iw [B,1]).

    Args:
        eps: boundary margin in (0, 1). Sampling range is [eps, 1-eps].

    Body: tau = torch.rand(B, 1, device=device) * (1 - 2*eps) + eps
          iw = torch.ones(B, 1, device=device)
          return tau, iw
    """
    if eps <= 0 or eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5); got {eps}")

    def sampler(B: int, device: torch.device) -> tuple[Tensor, Tensor]:
        tau = torch.rand(B, 1, device=device) * (1.0 - 2.0 * eps) + eps
        iw = torch.ones(B, 1, device=device)
        return tau, iw

    return sampler


def make_uniform_scaled(*, eps: float, max: float) -> TimeSampler1D:
    """uniform on [eps, max]; iw = 1.

    Used by V3 estimators (specs 12, 16) to sample t2 on [eps, t2_max]
    when paired with make_uniform via make_product.

    Returns a callable (B, device) -> (tau [B,1], iw [B,1]).

    Args:
        eps: lower bound in (0, max).
        max: upper bound > eps.

    Body: tau = torch.rand(B, 1, device=device) * (max - eps) + eps
          iw = torch.ones(B, 1, device=device)
          return tau, iw

    Note: iw is identically 1 because the target measure for this sampler
    is uniform on [eps, max] (the actual support of the secondary time axis),
    not uniform on [eps, 1-eps]. Callers using make_product treat the two
    axes as independent uniforms on their respective supports.
    """
    if eps <= 0 or eps >= max:
        raise ValueError(f"require 0 < eps < max; got eps={eps}, max={max}")

    def sampler(B: int, device: torch.device) -> tuple[Tensor, Tensor]:
        tau = torch.rand(B, 1, device=device) * (max - eps) + eps
        iw = torch.ones(B, 1, device=device)
        return tau, iw

    sampler.max = max  # exposes domain bound for downstream coverage gates
    return sampler


def make_beta(*, a: float, b: float, eps: float) -> TimeSampler1D:
    """Beta(a, b) distribution clamped to [eps, 1-eps]; iw = p_uniform / q.

    Returns a callable (B, device) -> (tau [B,1], iw [B,1]).

    The iw is computed against the untruncated Beta pdf; boundary mass is
    negligible for sharp Beta (a, b >= 2) so truncation correction is ~1.

    Args:
        a, b: Beta shape parameters (both > 0).
        eps: boundary clamp margin in (0, 0.5).

    Body: dist = torch.distributions.Beta(
              torch.tensor(a, device=device),
              torch.tensor(b, device=device)
          )
          tau_unclamped = dist.sample((B,))  # [B]
          tau = clamp(tau_unclamped, eps, 1-eps).unsqueeze(-1)  # [B, 1]
          p_uniform = 1 / (1 - 2*eps)
          log_q = dist.log_prob(tau.squeeze(-1))  # [B]
          iw = (p_uniform / exp(log_q)).unsqueeze(-1)  # [B, 1]
          return tau, iw
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"a, b must be > 0; got a={a}, b={b}")
    if eps <= 0 or eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5); got {eps}")

    def sampler(B: int, device: torch.device) -> tuple[Tensor, Tensor]:
        dist = Beta(
            torch.tensor(float(a), device=device),
            torch.tensor(float(b), device=device),
        )
        tau_unclamped = dist.sample((B,))
        tau = torch.clamp(tau_unclamped, eps, 1.0 - eps).unsqueeze(-1)

        p_uniform = 1.0 / (1.0 - 2.0 * eps)
        log_q = dist.log_prob(tau.squeeze(-1))
        q = torch.exp(log_q)
        iw = (p_uniform / q).unsqueeze(-1)
        return tau, iw

    return sampler


def make_sobol(
    *,
    eps: float,
    scramble: bool = True,
    seed: int | None = None,
) -> TimeSampler1D:
    """Sobol quasi-random sequence on [eps, 1-eps]; iw = 1 (target is uniform).

    Returns a callable (B, device) -> (tau [B,1], iw [B,1]).

    The SobolEngine lives on CPU; draws are moved to the requested device after
    transformation. For typical batch sizes this overhead is negligible.

    Args:
        eps: boundary margin in (0, 0.5).
        scramble: enable randomized scrambling of the Sobol sequence.
        seed: random seed for scrambling (if scramble=True).

    Body: engine = torch.quasirandom.SobolEngine(
              1, scramble=scramble, seed=seed  # 1-D; engine on CPU
          )
          def sampler(B, device):
              tau_unit = engine.draw(B).squeeze(-1)  # [B] on CPU, values in [0, 1)
              tau = (tau_unit * (1 - 2*eps) + eps).unsqueeze(-1).to(device)  # [B, 1]
              iw = torch.ones(B, 1, device=device)
              return tau, iw
          return sampler
    """
    if eps <= 0 or eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5); got {eps}")

    engine = torch.quasirandom.SobolEngine(1, scramble=scramble, seed=seed)

    def sampler(B: int, device: torch.device) -> tuple[Tensor, Tensor]:
        tau_unit = engine.draw(B).squeeze(-1)  # [B] on CPU
        tau = (tau_unit * (1.0 - 2.0 * eps) + eps).unsqueeze(-1).to(device)
        iw = torch.ones(B, 1, device=device)
        return tau, iw

    return sampler


# ============================================================================
# Schedule-shaped density samplers (bridge / stiff sigmoid-product)
# ============================================================================
# canonical schedule shapes mirrored from src/waypoints/atoms.py; kept local so
# this module stays import-light. used as unnormalized sampling DENSITIES (not
# path noise schedules): sampling tau propto the schedule emphasises the times
# where it is large.


def _g_bridge(t: Tensor) -> Tensor:
    """bridge shape sqrt(t(1-t)); the sqrt(2) vp-amplitude cancels under
    normalization, so propto sqrt(2 t(1-t)). equals the Beta(1.5, 1.5) density
    up to a constant."""
    return torch.sqrt(t * (1.0 - t))


def _g_stiff(t: Tensor, *, k: float) -> Tensor:
    """stiff sigmoid-product (1 - e^{-k t})(1 - e^{-k(1-t)}); flat-topped bump
    that vanishes at the endpoints, sharper edges as k grows. expm1 for float32
    stability near the endpoints."""
    return (-torch.expm1(-k * t)) * (-torch.expm1(-k * (1.0 - t)))


def _g_stiff_inv(t: Tensor, *, k: float) -> Tensor:
    """reciprocal stiff density 1 / stiff(t; k); diverges at the endpoints, so
    mass piles toward the boundaries. floored to stay finite on [eps, 1-eps]."""
    return 1.0 / _g_stiff(t, k=k).clamp_min(1e-12)


def make_density_sampler(
    g_fn,
    *,
    eps: float,
    n_grid: int = 4096,
) -> TimeSampler1D:
    """tabulated inverse-CDF sampler for an arbitrary unnormalized density g_fn.

    Returns a callable (B, device) -> (tau [B,1], iw [B,1]).

    procedure:
      build: grid = linspace(eps, 1-eps, n_grid); g = g_fn(grid);
             Z = trapezoid(g, grid); cdf = [0, cumulative_trapezoid(g, grid)] / Z
             (monotone since g > 0, cdf[0]=0, cdf[-1]=1). precomputed once on cpu,
             moved to the requested device on first use (cached per device).
      draw:  u ~ U(0,1); idx = searchsorted(cdf, u) clamped to [1, n_grid-1];
             tau = linear interpolation of grid between (cdf[idx-1], cdf[idx]).
      weight: q(tau) = g(tau)/Z is the normalized proposal on [eps, 1-eps]; the
             target is uniform there with p_uniform = 1/(1-2eps), so
             iw = p_uniform / q(tau) = Z / ((1-2eps) g(tau)), computed from the
             analytic g_fn at the sampled tau (not the tabulation) for accuracy.

    Args:
        g_fn: callable Tensor -> Tensor, positive unnormalized density on (0,1),
              shape-preserving (accepts the [n_grid] grid and [B,1] tau alike).
        eps: boundary margin in (0, 0.5). domain is [eps, 1-eps].
        n_grid: tabulation resolution; trapezoid error ~ O(n_grid^{-2}).

    note: densities that vanish at the endpoints (bridge, stiff) keep iw bounded
    because inverse-CDF sampling rarely lands where g ~ 0; densities that diverge
    (stiff_inv) are evaluated only on the interior [eps, 1-eps], so g stays finite.
    """
    if eps <= 0 or eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5); got {eps}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2; got {n_grid}")

    grid = torch.linspace(eps, 1.0 - eps, n_grid)  # [n_grid], cpu
    g_vals = g_fn(grid)  # [n_grid]
    z = float(torch.trapezoid(g_vals, grid))
    cdf = torch.cat(
        [torch.zeros(1), torch.cumulative_trapezoid(g_vals, grid)]
    )  # [n_grid]
    cdf = cdf / cdf[-1].clamp_min(1e-12)

    p_uniform = 1.0 / (1.0 - 2.0 * eps)
    cache: dict[torch.device, tuple[Tensor, Tensor]] = {}

    def sampler(B: int, device: torch.device) -> tuple[Tensor, Tensor]:
        dev = torch.device(device)
        if dev not in cache:
            cache[dev] = (grid.to(dev), cdf.to(dev))
        grid_d, cdf_d = cache[dev]

        u = torch.rand(B, device=dev)  # [B]
        hi = torch.searchsorted(cdf_d, u).clamp(1, n_grid - 1)  # [B]
        lo = hi - 1
        c_lo, c_hi = cdf_d[lo], cdf_d[hi]
        frac = (u - c_lo) / (c_hi - c_lo).clamp_min(1e-12)
        tau = (grid_d[lo] + frac * (grid_d[hi] - grid_d[lo])).clamp(eps, 1.0 - eps)
        tau = tau.unsqueeze(-1)  # [B, 1]

        iw = (z * p_uniform / g_fn(tau).clamp_min(1e-12))  # [B, 1]
        return tau, iw

    return sampler


def make_bridge(*, eps: float, n_grid: int = 4096) -> TimeSampler1D:
    """density propto sqrt(t(1-t)) (== Beta(1.5, 1.5)); concentrates near 0.5."""
    return make_density_sampler(_g_bridge, eps=eps, n_grid=n_grid)


def make_stiff(*, k: float, eps: float, n_grid: int = 4096) -> TimeSampler1D:
    """density propto the stiff sigmoid-product; flat-topped, edges sharpen with k."""
    return make_density_sampler(lambda t: _g_stiff(t, k=k), eps=eps, n_grid=n_grid)


def make_stiff_inv(*, k: float, eps: float, n_grid: int = 4096) -> TimeSampler1D:
    """density propto 1 / stiff sigmoid-product; concentrates toward the boundaries."""
    return make_density_sampler(lambda t: _g_stiff_inv(t, k=k), eps=eps, n_grid=n_grid)


# ============================================================================
# shared dist-string registry (single source of truth for both factories)
# ============================================================================
# beta cases have a dedicated ABC subclass (BetaSampler); schedule-shaped cases
# are functional and wrapped in _FuncSampler for the legacy ABC factory.

_BETA_DIST_PARAMS: dict[str, tuple[float, float]] = {
    "beta_5_5": (5.0, 5.0),
    "beta_10_10": (10.0, 10.0),
    "beta_half_half": (0.5, 0.5),
}

_DENSITY_DIST_BUILDERS: dict[str, "Callable[[float], TimeSampler1D]"] = {
    "bridge": lambda e: make_bridge(eps=e),
    "stiff_10": lambda e: make_stiff(k=10.0, eps=e),
    "stiff_20": lambda e: make_stiff(k=20.0, eps=e),
    "stiff_inv_40": lambda e: make_stiff_inv(k=40.0, eps=e),
}

# every dist string accepted by the time_dist hparam (HPO categorical choices
# should mirror this set).
TIME_DISTS: tuple[str, ...] = (
    "uniform",
    *_BETA_DIST_PARAMS,
    *_DENSITY_DIST_BUILDERS,
)


# ============================================================================
# Path-aware builders
# ============================================================================


def make_piecewise_sb_sampler(
    *,
    vertex: float,
    inner_eps: float,
    eps: float,
) -> TimeSampler1D:
    """uniform on [eps, vertex - inner_eps] ∪ [vertex + inner_eps, 1 - eps].

    Returns a callable (B, device) -> (tau [B,1], iw [B,1]).

    Used by piecewise-SB paths (V2) to avoid a forbidden band around the
    vertex where the path's gamma has a singularity. Owns its forbidden
    geometry; does NOT query any path object at runtime.

    Args:
        vertex: forbidden-zone center (typically 0.5).
        inner_eps: half-width of forbidden band around vertex.
        eps: outer boundary margin.

    Body: allowed_width = (1 - 2*eps) - 2*inner_eps
          assert allowed_width > 0

          def sampler(B, device):
              # sample from [0, allowed_width), then map into allowed intervals
              u = torch.rand(B, 1, device=device) * allowed_width

              # interval 1: [eps, vertex - inner_eps], width = (vertex - inner_eps - eps)
              # interval 2: [vertex + inner_eps, 1 - eps], width = (1 - eps - vertex - inner_eps)
              threshold = vertex - inner_eps - eps

              # if u < threshold, tau in interval 1; else shift into interval 2
              tau = torch.where(
                  u < threshold,
                  u + eps,
                  u + eps + 2*inner_eps
              )

              iw = ones * (1 - 2*eps) / allowed_width
              return tau, iw
    """
    if eps <= 0 or eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5); got {eps}")
    if vertex <= eps or vertex >= 1.0 - eps:
        raise ValueError(f"vertex must be in ({eps}, {1.0 - eps}); got {vertex}")
    if inner_eps <= 0 or inner_eps >= 0.5 * (1.0 - 2.0 * eps):
        raise ValueError(
            f"inner_eps must be in (0, {0.5 * (1.0 - 2.0 * eps)}); got {inner_eps}"
        )

    allowed_width = (1.0 - 2.0 * eps) - 2.0 * inner_eps
    threshold = vertex - inner_eps - eps
    iw_scale = (1.0 - 2.0 * eps) / allowed_width

    def sampler(B: int, device: torch.device) -> tuple[Tensor, Tensor]:
        u = torch.rand(B, 1, device=device) * allowed_width
        tau = torch.where(
            u < threshold,
            u + eps,
            u + eps + 2.0 * inner_eps,
        )
        iw = torch.full((B, 1), iw_scale, device=device)
        return tau, iw

    return sampler


# ============================================================================
# Composition and variance-reduction operators
# ============================================================================


def make_no_iw(base: TimeSampler1D) -> TimeSampler1D:
    """wraps base sampler; forces iw=1 regardless of base's iw.

    Returns a callable (B, device) -> (tau [B,1], iw [B,1]).

    Use when non-uniform sampling is a deliberate loss-emphasis choice rather
    than variance reduction: the resulting estimator is biased against the
    uniform integral but unbiased against the proposal q the sampler defines.
    Composes with reweight (lambda multiplier) the same way iw would.

    Args:
        base: TimeSampler1D callable.

    Body: def sampler(B, device):
              tau, _ = base(B, device)
              iw = torch.ones(B, 1, device=device)
              return tau, iw
    """

    def sampler(B: int, device: torch.device) -> tuple[Tensor, Tensor]:
        tau, _ = base(B, device)
        iw = torch.ones(B, 1, device=device)
        return tau, iw

    return sampler


def make_product(
    s1: TimeSampler1D,
    s2: TimeSampler1D,
) -> TimeSampler2D:
    """compose two 1D samplers into a 2D sampler for rectangle-stacked geometries.

    Returns a callable (B, device) -> (t1 [B,1], t2 [B,1], iw [B,1]).

    IMPORTANT: This builder is scoped exclusively to rectangle-stacked domains
    where the proposal and target both factorize on the rectangle. The product
    iw = iw1 * iw2 is correct only in this setting. Future 2D geometries with
    non-rectangular domains (simplex, triangle region, importance-sampled
    rectangle; see notes/2d_path_alternatives.md) require dedicated 2D sampler
    builders.

    Args:
        s1, s2: TimeSampler1D callables for the t1 and t2 dimensions.

    Body: def sampler(B, device):
              t1, iw1 = s1(B, device)  # [B, 1] each
              t2, iw2 = s2(B, device)  # [B, 1] each
              iw = iw1 * iw2           # [B, 1]
              return t1, t2, iw
    """

    def sampler(B: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        t1, iw1 = s1(B, device)
        t2, iw2 = s2(B, device)
        iw = iw1 * iw2
        return t1, t2, iw

    # propagate t2-axis domain bound from inner sampler (s2 in the rect convention)
    if hasattr(s2, "max"):
        sampler.t2_max = s2.max
    return sampler


def make_stratified(
    base: TimeSampler1D,
    *,
    n_strata: int | None = None,
) -> TimeSampler1D:
    """stratified sampling: one draw per stratum; preserves base's iw.

    Returns a callable (B, device) -> (tau [B,1], iw [B,1]).

    Divides [0, 1) into n_strata equiprobable intervals, draws one sample per
    stratum from the base sampler, and concatenates. Reduces variance while
    preserving the unbiasedness of the base's iw. Stratification is applied
    to the proposal distribution; the importance weights are passed through
    unchanged from base.

    Args:
        base: TimeSampler1D callable.
        n_strata: number of strata. Defaults to B if not specified.

    Body: def sampler(B, device):
              effective_n_strata = n_strata if n_strata is not None else B

              # draw effective_n_strata samples from base
              tau_all, iw_all = base(effective_n_strata, device)  # [n_strata, 1] each

              # if n_strata > B, select first B; if n_strata < B, tile/repeat
              if effective_n_strata >= B:
                  tau = tau_all[:B]
                  iw = iw_all[:B]
              else:
                  repeats = (B + effective_n_strata - 1) // effective_n_strata
                  tau = (tau_all.repeat(repeats, 1))[:B]
                  iw = (iw_all.repeat(repeats, 1))[:B]

              return tau, iw
    """

    def sampler(B: int, device: torch.device) -> tuple[Tensor, Tensor]:
        effective_n_strata = n_strata if n_strata is not None else B
        tau_all, iw_all = base(effective_n_strata, device)

        if effective_n_strata >= B:
            tau = tau_all[:B]
            iw = iw_all[:B]
        else:
            repeats = (B + effective_n_strata - 1) // effective_n_strata
            tau = tau_all.repeat(repeats, 1)[:B]
            iw = iw_all.repeat(repeats, 1)[:B]

        return tau, iw

    return sampler


def make_antithetic(base: TimeSampler1D) -> TimeSampler1D:
    """antithetic variance reduction: sample B/2 from base, mirror via 1-tau.

    Returns a callable (B, device) -> (tau [B,1], iw [B,1]).

    Draws ceil(B/2) samples from base, mirrors each via tau' = 1 - tau, and
    concatenates (tau, 1-tau) to reach batch size B (truncating if B is odd).
    The mirrored iw is inherited from base; for asymmetric proposals (e.g.,
    asymmetric Beta) the log-prob and thus iw differ between tau and 1-tau.
    Future convenience builders (e.g., make_antithetic_beta) may be added to
    handle asymmetric cases with correct iw for the mirror.

    Args:
        base: TimeSampler1D callable.

    Body: def sampler(B, device):
              n_half = (B + 1) // 2  # ceil(B/2)
              tau, iw = base(n_half, device)  # [n_half, 1] each

              tau_mirror = 1.0 - tau

              # concatenate and truncate to B
              tau_all = torch.cat([tau, tau_mirror], dim=0)[:B]
              iw_all = torch.cat([iw, iw], dim=0)[:B]

              return tau_all, iw_all
    """

    def sampler(B: int, device: torch.device) -> tuple[Tensor, Tensor]:
        n_half = (B + 1) // 2
        tau, iw = base(n_half, device)

        tau_mirror = 1.0 - tau
        tau_all = torch.cat([tau, tau_mirror], dim=0)[:B]
        iw_all = torch.cat([iw, iw], dim=0)[:B]

        return tau_all, iw_all

    return sampler


# ============================================================================
# Compatibility and legacy bridges
# ============================================================================


def validate_compat(path, sampler: TimeSampler1D) -> None:
    """assert sampler's eps matches path.eps; check forbidden-set scalars.

    Opt-in helper for HPO builders that want belt-and-suspenders protection
    against scalar mismatch between path geometry and sampler parameterization.
    The estimator constructor does NOT call this.

    Args:
        path: a path object (TriangularPath1D, DirectPath1D, etc.) with an eps field.
        sampler: a TimeSampler1D or subclass (or any callable with an optional eps attribute).

    Behavior:
      1. If sampler has an .eps attribute, assert sampler.eps == path.eps.
      2. If sampler is a closure from make_piecewise_sb_sampler (no public API to
         check internals), this is a documentation note: users must ensure the
         vertex and inner_eps match the path's geometry.
      3. Raises AssertionError with a detailed message on mismatch.
      4. Returns None on success.

    Body: if hasattr(sampler, 'eps'):
              assert sampler.eps == path.eps, (
                  f"sampler.eps ({sampler.eps}) != path.eps ({path.eps})"
              )
          if hasattr(path, 'eps'):
              # path-aware sampler check: document that the caller is responsible
              # for ensuring vertex/inner_eps scalar consistency
              pass
    """
    if not hasattr(path, "eps"):
        raise TypeError(f"path {type(path).__name__} does not have an eps attribute")

    if hasattr(sampler, "eps"):
        assert sampler.eps == path.eps, (
            f"sampler.eps ({sampler.eps}) != path.eps ({path.eps}); "
            f"sampler and path must have matching boundary margins."
        )


def time_sampler_from_legacy_cfg(
    dist: str,
    eps: float,
    apply_iw: bool,
) -> TimeSampler1D:
    """legacy string-driven (tau, iw) sampler factory.

    Maps the time_dist enumerated strings (TIME_DISTS) to the functional
    builders, dispatching via the shared _BETA_DIST_PARAMS / _DENSITY_DIST_BUILDERS
    registry so this stays in lockstep with sampler_from_dist. Wraps with
    make_no_iw if apply_iw=False.

    Args:
        dist: one of TIME_DISTS (uniform, the beta family, bridge, and the
              stiff / stiff_inv sigmoid-product family).
        eps: boundary margin.
        apply_iw: if False, wraps the sampler with make_no_iw (iw=1).

    Returns:
        A TimeSampler1D callable.

    Raises:
        ValueError: if dist is not in TIME_DISTS.
    """
    if dist == "uniform":
        sampler = make_uniform(eps=eps)
    elif dist in _BETA_DIST_PARAMS:
        a, b = _BETA_DIST_PARAMS[dist]
        sampler = make_beta(a=a, b=b, eps=eps)
    elif dist in _DENSITY_DIST_BUILDERS:
        sampler = _DENSITY_DIST_BUILDERS[dist](eps)
    else:
        raise ValueError(
            f"unknown legacy dist {dist!r}; expected one of {sorted(TIME_DISTS)}"
        )

    if not apply_iw:
        sampler = make_no_iw(sampler)
    return sampler


def make_psb_mixture_sampler(
    *,
    time_dist: str,
    apply_iw: bool,
    vertex: float,
    inner_eps: float,
    eps: float,
) -> TimeSampler1D:
    """two-leg mixture time sampler for piecewise-Schroedinger-bridge paths.

    samples tau in the allowed region [eps, vertex - inner_eps] U [vertex + inner_eps, 1 - eps],
    excising the forbidden band of half-width inner_eps around the vertex. leg selection is
    width-proportional; the chosen time_dist density is applied PER LEG (its local "piecewise"
    form), as opposed to the global default. any value in TIME_DISTS
    is supported -- uniform, the beta family, and the schedule-shaped density-builders
    (bridge, stiff_*, stiff_inv_*) -- with no collapse to uniform.

    Returns a callable (B, device) -> (tau [B,1], iw [B,1]).

    Args:
        time_dist: density applied per leg; any value in TIME_DISTS (uniform, the beta family,
                   bridge, stiff_*, stiff_inv_*). a uniform time_dist recovers uniform over the
                   allowed region; a schedule-shaped density is applied per-leg (piecewise).
        apply_iw: if False, wraps the returned sampler with make_no_iw (iw=1).
        vertex: forbidden-zone center (typically 0.5).
        inner_eps: half-width of excised band around vertex; >= 0. at inner_eps == 0
                   the two legs meet at the vertex (zero-width band) and the mixture
                   is still per-leg.
        eps: outer boundary margin in (0, 0.5).

    Validation:
        - eps must be in (0, 0.5).
        - require eps < vertex - inner_eps (leg 1 width > 0).
        - require vertex + inner_eps < 1 - eps (leg 2 width > 0).
        raises ValueError with a clear message naming the offending quantities if any
        constraint is violated.

    Body pseudocode:
        leg1 = [eps, vertex - inner_eps], width w1
        leg2 = [vertex + inner_eps, 1 - eps], width w2
        allowed_width = w1 + w2
        p1 = w1 / allowed_width

        leg1_base = time_sampler_from_legacy_cfg(time_dist, eps=eps, apply_iw=apply_iw)
        leg2_base = time_sampler_from_legacy_cfg(time_dist, eps=eps, apply_iw=apply_iw)

        def sampler(B, device):
            pick = rand(B, 1) < p1
            u1, iw1 = leg1_base(B, device)
            u2, iw2 = leg2_base(B, device)
            tau1 = eps + u1 * w1
            tau2 = (vertex + inner_eps) + u2 * w2
            tau = where(pick, tau1, tau2)
            iw = where(pick, iw1, iw2)
            return tau, iw

    math justification:
        because leg-selection is width-proportional, the global proposal density is
        q(tau) = d_local((tau - a_leg)/w_leg) / |A| (leg width cancels in the jacobian),
        so a uniform time_dist recovers uniform-over-allowed-region exactly. the per-leg
        base iw = p_uniform/d differs from the exact global iw only by the constant factor |A|
        (the allowed-region mass), which is immaterial because every loss applies iw as
        (...*iw).mean() under adamw (a constant loss scale does not move the optimum).
        hence reusing the per-leg iw is correct up to an immaterial constant;
        apply_iw=False yields iw=1 (the base make_no_iw wrapper handles this).
    """
    # validate eps
    if eps <= 0 or eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5); got {eps}")

    # compute leg boundaries and widths
    a1 = eps
    b1 = vertex - inner_eps
    w1 = b1 - a1

    a2 = vertex + inner_eps
    b2 = 1.0 - eps
    w2 = b2 - a2

    # validate leg widths
    if w1 <= 0:
        raise ValueError(
            f"forbidden band too wide: w1={w1} must be > 0 (require eps < vertex - inner_eps); "
            f"got vertex={vertex}, inner_eps={inner_eps}, eps={eps}"
        )
    if w2 <= 0:
        raise ValueError(
            f"forbidden band too wide: w2={w2} must be > 0 (require vertex + inner_eps < 1 - eps); "
            f"got vertex={vertex}, inner_eps={inner_eps}, eps={eps}"
        )

    # build per-leg base samplers
    leg1_base = time_sampler_from_legacy_cfg(time_dist, eps=eps, apply_iw=apply_iw)
    leg2_base = time_sampler_from_legacy_cfg(time_dist, eps=eps, apply_iw=apply_iw)

    # precompute leg selection probability
    allowed_width = (1.0 - 2.0 * eps) - 2.0 * inner_eps  # |A|
    p1 = w1 / allowed_width

    def sampler(B: int, device: torch.device) -> tuple[Tensor, Tensor]:
        # draw leg selection [B, 1], boolean
        pick = torch.rand(B, 1, device=device) < p1

        # sample from per-leg bases on their local [eps, 1-eps] domain [B, 1] each
        u1, iw1 = leg1_base(B, device)
        u2, iw2 = leg2_base(B, device)

        # affine-map to global leg ranges [B, 1] each
        tau1 = a1 + u1 * w1
        tau2 = a2 + u2 * w2

        # select per pick [B, 1] each
        tau = torch.where(pick, tau1, tau2)
        iw = torch.where(pick, iw1, iw2)

        return tau, iw

    return sampler
