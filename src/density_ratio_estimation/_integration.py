"""init-time bindings for tau-quadrature in score-based DRE estimators.

provides ``build_integrator``: selects mean / trapezoid / Simpson 1/3 once and
returns a callable ``(time_scores, t_vals) -> tensor`` that yields the negated
integral on CPU. used by VFM and TriangularVFM (which expose all three options)
and is a drop-in replacement for the hard-coded ``torch.trapezoid`` calls in
TSM, CTSM, TriangularCTSM (V1/V2), TriangularCTSM2D, and TriangularVFM2D
(which only use trapezoid today).

the divergence-estimator factory lives in ``src.models.flow.div_estimators``
(``build_div_fn``) since it is shared with the FMDRE ``ratio_ode*`` family;
import it from there.
"""
from typing import Callable

import torch


def build_integrator(integration_type: str) -> Callable:
    """select the tau-integration scheme once; return a single callable.

    Args:
        integration_type: "1" for mean, "2" for trapezoid, "3" for Simpson 1/3.

    Returns:
        Callable ``(time_scores, t_vals) -> tensor on CPU`` that yields
        ``-int score dtau`` (the sign matches predict_ldr semantics).
    """
    if integration_type == "3":
        def integrator(time_scores, t_vals):
            n_points = time_scores.shape[0]
            t_np = t_vals.cpu().numpy()
            h = (t_np[-1] - t_np[0]) / (n_points - 1)
            integrand = time_scores.cpu().numpy()
            integral = integrand[0] + integrand[-1]
            for i in range(1, n_points - 1):
                integral += (2 if i % 2 == 0 else 4) * integrand[i]
            integral *= h / 3
            return -torch.from_numpy(integral)
        return integrator
    if integration_type == "2":
        def integrator(time_scores, t_vals):
            return -torch.trapz(time_scores, t_vals, dim=0).cpu()
        return integrator

    def integrator(time_scores, t_vals):
        return -time_scores.mean(dim=0).cpu()
    return integrator
