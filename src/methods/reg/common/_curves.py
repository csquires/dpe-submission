"""Parameterized inference curves for V2 (1D) and V3 (2D) estimators.

Curve protocol and implementations for mapping a batch of scalar times tau in [0, 1]
to curve coordinates in the (t_1, t_2) time domain. Tensor-native refactor of
src/waypoints/curve_2d.py.
"""

from typing import Protocol

import torch
from torch import Tensor


class Curve(Protocol):
    """Protocol for parameterized curves tau -> (t_1, ..., t_d).

    Any object with a `dim` attribute and `points`, `derivatives` methods
    satisfies this protocol.
    """

    dim: int

    def points(self, tau: Tensor) -> Tensor:
        """Map tau batch to curve coordinates.

        Args:
            tau: shape [n], batch of n scalar times in [0, 1].

        Returns:
            shape [n, dim], where row i is the dim-dimensional curve point at tau[i].
        """
        ...

    def derivatives(self, tau: Tensor) -> Tensor:
        """Map tau batch to curve tangent vectors.

        Args:
            tau: shape [n], batch of n scalar times in [0, 1].

        Returns:
            shape [n, dim], where row i is the dim-dimensional tangent vector
            d(point) / d(tau) at tau[i].
        """
        ...


class IdentityCurve1D:
    """Trivial 1D curve: tau -> [tau].

    Used by 1D estimators (VFM V1, V2, CTSM V1, V2). The curve parameter
    and time are identical; stateless singleton.
    """

    dim = 1

    def points(self, tau: Tensor) -> Tensor:
        """Return tau reshaped to [n, 1]."""
        # [n] -> [n, 1]
        return tau.unsqueeze(-1)

    def derivatives(self, tau: Tensor) -> Tensor:
        """Return constant derivative 1, shape [n, 1]."""
        # [n] -> [n, 1]
        return torch.ones_like(tau).unsqueeze(-1)


class LowArcCurve2D:
    """Tensor-native low-arc curve: tau -> (tau, h * tau * (1 - tau)).

    Traces a smooth arc below the diagonal in the 2D time square [0, 1]^2.
    Used by V3 estimators (VFM V3, CTSM V3) for line-integral inference.
    Refactored from src/waypoints/curve_2d.py (Python-float based) to
    operate natively on Tensor batches.

    Constructor argument:
        path_height: float, default 1.0. Arc height multiplier h.
            At tau = 0.5, t_2 = h/4 (peak). h = 0 collapses to bottom edge.
    """

    dim = 2

    def __init__(self, path_height: float = 1.0) -> None:
        """Initialize with arc height.

        Args:
            path_height: float, arc height multiplier. Must be >= 0 (not validated).
        """
        self.path_height = path_height

    def points(self, tau: Tensor) -> Tensor:
        """Map tau batch to (t_1, t_2) coordinates on the low arc.

        Args:
            tau: shape [n], batch of times in [0, 1].

        Returns:
            shape [n, 2]. Row i is [tau[i], h * tau[i] * (1 - tau[i])].
        """
        h = self.path_height
        # [n] and [n] -> [n, 2]
        return torch.stack([tau, h * tau * (1.0 - tau)], dim=-1)

    def derivatives(self, tau: Tensor) -> Tensor:
        """Map tau batch to tangent vectors on the low arc.

        Args:
            tau: shape [n], batch of times in [0, 1].

        Returns:
            shape [n, 2]. Row i is [1, h * (1 - 2*tau[i])].
            Tangent vector is zero at tau = 0.5 (peak).
        """
        h = self.path_height
        # [n] and [n] -> [n, 2]
        return torch.stack([torch.ones_like(tau), h * (1.0 - 2.0 * tau)], dim=-1)

    def peak_t2(self) -> float:
        """Return the maximum t_2 value attained on the curve.

        For h tau (1 - tau), peak is h/4 at tau = 0.5. Callers can use this
        to check coverage against a sampler's t2_max if they wish.

        Returns:
            float, the maximum t_2 value on the curve.
        """
        return 0.25 * self.path_height
