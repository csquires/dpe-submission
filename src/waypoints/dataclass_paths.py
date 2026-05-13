"""
path archetypes for vfm and ctsm: frozen geometry containers with callable weight/gamma fields.

three path families are defined:
- DirectPath1D: two-source path without singular point (vfm/ctsm direct mode).
- TriangularPath1D: triangular path with x_star singular point (vfm/ctsm v1/v2).
- TriangularPath2D: 2d triangular path over (t1, t2) domain (vfm/ctsm v3).

weights are encapsulated in NamedTuple bundles (DirectWeights1D, TriangularWeights1D,
TriangularWeights2D) to preserve joint data dependency across outputs and ensure
clean vmap compatibility. builders in path_builders.py populate the Callable fields;
paths are immutable after construction via frozen=True.

each path callable takes time argument(s) and returns either a weight bundle or a
scalar tensor. gamma schedules enforce positivity on interior [eps, 1-eps] (1d) or
[eps, 1-eps] x [eps, t2_max] (2d) to support logarithmic time reparameterization
and flow matching.
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

    two-argument callables for a rectangular or height-bounded time domain.
    gamma is positive on [eps, 1-eps] x [eps, t2_max].
    """
    weights: Callable[[torch.Tensor, torch.Tensor], TriangularWeights2D]
    gamma: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    dgamma_dt1: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    dgamma_dt2: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    eps: float
    t2_max: float
