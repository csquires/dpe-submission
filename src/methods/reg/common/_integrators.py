"""tensor-native quadrature integrators for score-based DRE.

provides ``integrator_mean``, ``integrator_trapezoid``, ``integrator_simpson``:
pure functions returning the signed integral. negation and device handling
are the caller's responsibility (e.g., ``predict_ldr_via_curve`` in Pillar D).
"""
from typing import Callable

import torch
from torch import Tensor

Integrator = Callable[[Tensor, Tensor], Tensor]


def integrator_mean(scores: Tensor, taus: Tensor) -> Tensor:
    """rectangle rule via uniform mean over effective interval [eps, 1-eps].

    infers eps from taus[0]; assumes taus[0] >= eps and taus[-1] = 1 - eps
    by convention (set by the caller, e.g., ``predict_ldr_via_curve``).

    args:
        scores: [n_points, n_samples], integrand values.
        taus: [n_points], tau nodes (ignored except for eps extraction).

    returns:
        [n_samples], the integral (1 - 2*eps) * mean(scores, dim=0).
    """
    eps = taus[0]  # infer eps from first node
    scale = 1.0 - 2.0 * eps  # effective interval length
    return scale * scores.mean(dim=0)  # [n_samples]


def integrator_trapezoid(scores: Tensor, taus: Tensor) -> Tensor:
    """trapezoidal rule, composite form.

    args:
        scores: [n_points, n_samples], integrand values.
        taus: [n_points], tau nodes (arbitrary spacing supported).

    returns:
        [n_samples], the trapezoidal approximation to the integral.
    """
    return torch.trapezoid(scores, taus, dim=0)  # [n_samples]


def integrator_simpson(scores: Tensor, taus: Tensor) -> Tensor:
    """Simpson's 1/3 rule, assuming uniform tau spacing.

    precondition: scores.shape[0] must be odd.

    args:
        scores: [n_points, n_samples], integrand values.
        taus: [n_points], tau nodes (uniform spacing assumed).

    returns:
        [n_samples], the Simpson 1/3 approximation to the integral.

    raises:
        ValueError: if n_points is even.
    """
    n_points = scores.shape[0]
    if n_points % 2 != 1:
        raise ValueError(
            f"Simpson's rule requires an odd number of points; got {n_points}. "
            f"Provide an integrator grid with 2k+1 points (e.g., n_points=9, 17, 33)."
        )

    # uniform spacing: h = (taus[-1] - taus[0]) / (n_points - 1)
    h = (taus[-1] - taus[0]) / (n_points - 1)

    # build weight vector: [1, 4, 2, 4, 2, ..., 4, 1]
    weights = torch.ones(n_points, device=scores.device, dtype=scores.dtype)
    weights[1::2] = 4.0  # odd indices
    weights[2::2] = 2.0  # even indices (starting at 2)
    weights[-1] = 1.0  # ensure endpoint is 1

    # weighted sum: (h / 3) * sum(weights[i] * scores[i])
    integral = (h / 3.0) * torch.einsum("i,ij->j", weights, scores)  # [n_samples]
    return integral
