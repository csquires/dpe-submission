"""path archetypes for vfm and ctsm: frozen dataclasses bundling callable
weight/gamma fields and a scalar eps boundary.

three path families:
- DirectPath1D: two-source path (no anchor); stock vfm/ctsm.
- TriangularPath1D: three-source path with x_star anchor; v1/v2.
- TriangularPath2D: 2d-time triangular path; v3.

weights are NamedTuple bundles to preserve joint shape across outputs and stay
vmap-friendly. builders in path_builders.py populate the Callable fields; the
frozen=True dataclasses prevent post-construction mutation.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple
import torch


class DirectWeights1D(NamedTuple):
    """weights and derivatives for direct (non-singular) 1d paths.

    invariant: alpha + beta == 1 and d_alpha + d_beta == 0 (to numerical tolerance).
    """
    alpha: torch.Tensor
    beta: torch.Tensor
    d_alpha: torch.Tensor
    d_beta: torch.Tensor


class TriangularWeights1D(NamedTuple):
    """weights and derivatives for triangular 1d paths (with singular point x_star).

    invariant: alpha + beta + w_star == 1 and d_alpha + d_beta + d_w_star == 0.
    """
    alpha: torch.Tensor
    beta: torch.Tensor
    w_star: torch.Tensor
    d_alpha: torch.Tensor
    d_beta: torch.Tensor
    d_w_star: torch.Tensor


class TriangularWeights2D(NamedTuple):
    """weights and partial derivatives for triangular 2d paths.

    each weight has two partials (one per time dimension). invariant: alpha + beta +
    w_star == 1 and for each time variable t_i, (d*_dti).sum() == 0.
    """
    alpha: torch.Tensor
    beta: torch.Tensor
    w_star: torch.Tensor
    d_alpha_dt1: torch.Tensor
    d_beta_dt1: torch.Tensor
    d_w_star_dt1: torch.Tensor
    d_alpha_dt2: torch.Tensor
    d_beta_dt2: torch.Tensor
    d_w_star_dt2: torch.Tensor


@dataclass(frozen=True)
class DirectPath1D:
    """frozen path for direct (two-source) vfm and ctsm.

    contains pure geometry only; builders populate the callables. frozen=True
    prevents accidental rebinding after construction.
    """
    weights: Callable[[torch.Tensor], DirectWeights1D]
    gamma: Callable[[torch.Tensor], torch.Tensor]
    dgamma_dtau: Callable[[torch.Tensor], torch.Tensor]
    eps: float


@dataclass(frozen=True)
class TriangularPath1D:
    """frozen path for triangular vfm and ctsm variants (v1, v2).

    includes singular point x_star. gamma(tau) > 0 on [eps, 1-eps].
    """
    weights: Callable[[torch.Tensor], TriangularWeights1D]
    gamma: Callable[[torch.Tensor], torch.Tensor]
    dgamma_dtau: Callable[[torch.Tensor], torch.Tensor]
    eps: float


@dataclass(frozen=True)
class TriangularPath2D:
    """frozen path for triangular 2d vfm and ctsm (v3).

    pure geometry: the path is well-defined on the open square (t1, t2) in
    (0, 1)^2. integration-domain bounds (e.g. t2_max < 1 to avoid the pure-
    anchor corner) live on the sampler/curve/estimator, not here.
    """
    weights: Callable[[torch.Tensor, torch.Tensor], TriangularWeights2D]
    gamma: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    dgamma_dt1: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    dgamma_dt2: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    eps: float
