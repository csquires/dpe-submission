from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


def predict_ldr_via_curve(
    time_score_fn: Callable,
    path,
    curve,
    integrator: Callable,
    n_points: int,
    samples: Tensor,
    excise_band: Optional[Tuple[float, float]] = None,
) -> Tensor:
    """Integrate time-score along curve in [path.eps, 1-path.eps]; return -integral on CPU.

    Implements chunked vmap over tau grid points, chain-rule combination for 2D curves,
    and tensor-native integration. EMA apply/restore is the caller's responsibility.

    When `excise_band = (lo, hi)` is given (with `path.eps <= lo < hi <= 1 - path.eps`),
    the integration domain is split into `[path.eps, lo] u [hi, 1 - path.eps]`. Each
    leg gets a linspace with the requested point density (proportional to leg width
    so that the per-leg point count sums to n_points), is integrated with the supplied
    integrator, and the two leg integrals are summed. Used by V1 (psb) inference to
    mirror the training sampler's excised forbidden band around the vertex.

    Args:
        time_score_fn: Callable(path, ts, samples) -> Tensor. Takes curve points ts
            (shape [chunk_len, curve.dim]) and samples (shape [n_samples, data_dim]),
            returns raw time-scores. For 1D curves: [chunk_len, n_samples]. For 2D curves:
            [chunk_len, n_samples, 2] (one per time-axis component, raw before chain rule).
        path: Path object with .eps attribute (float); defines integration bounds.
        curve: Curve object with .points(tau) and .derivatives(tau) methods. Both take
            tau Tensor of shape [n] and return [n, dim]. dim is 1 (identity 1D) or 2 (stacked 2D).
        integrator: Callable(time_scores, taus) -> Tensor on CPU. Implements one of
            mean / trapezoid / Simpson. Receives combined scores [n_points, n_samples]
            and tau grid [n_points], returns signed integral [n_samples].
        n_points: Number of quadrature points on [eps, 1-eps]. For Simpson, caller
            must ensure n_points % 2 == 1 if integrator is Simpson; this function
            defers the guard to integrator's own assertion.
        samples: Test points [n_samples, data_dim] on device D.
        excise_band: Optional (lo, hi) pair; when set, integrate over two legs
            avoiding (lo, hi). Both bounds must satisfy path.eps <= lo < hi <= 1 - path.eps.

    Returns:
        Tensor [n_samples] on CPU. Log-density-ratio estimate.

    Raises:
        ValueError: if n_points < 2, or excise_band is invalid.

    Device behavior: All computation on samples.device; integrator returns CPU tensor.
    """
    # step 1: validate n_points
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2; got {n_points}")

    # step 2: branch on excise_band
    if excise_band is None:
        # single-leg path: integrate over [path.eps, 1 - path.eps]
        ldr = _integrate_leg(
            time_score_fn=time_score_fn,
            path=path,
            curve=curve,
            integrator=integrator,
            n_points=n_points,
            samples=samples,
            tau_lo=path.eps,
            tau_hi=1.0 - path.eps,
        )
        return ldr

    # two-leg path: validate band and split n_points proportional to leg width
    lo, hi = float(excise_band[0]), float(excise_band[1])
    if not (path.eps <= lo < hi <= 1.0 - path.eps):
        raise ValueError(
            f"excise_band must satisfy path.eps <= lo < hi <= 1 - path.eps; "
            f"got eps={path.eps}, lo={lo}, hi={hi}"
        )
    w1 = lo - path.eps           # leg-1 width [eps, lo]
    w2 = (1.0 - path.eps) - hi   # leg-2 width [hi, 1 - eps]
    total = w1 + w2
    # split n_points proportionally; each leg keeps at least 2 points
    n1 = max(2, int(round(n_points * w1 / total)))
    n2 = max(2, n_points - n1)
    ldr1 = _integrate_leg(
        time_score_fn=time_score_fn,
        path=path,
        curve=curve,
        integrator=integrator,
        n_points=n1,
        samples=samples,
        tau_lo=path.eps,
        tau_hi=lo,
    )
    ldr2 = _integrate_leg(
        time_score_fn=time_score_fn,
        path=path,
        curve=curve,
        integrator=integrator,
        n_points=n2,
        samples=samples,
        tau_lo=hi,
        tau_hi=1.0 - path.eps,
    )
    return ldr1 + ldr2


def _integrate_leg(
    time_score_fn: Callable,
    path,
    curve,
    integrator: Callable,
    n_points: int,
    samples: Tensor,
    tau_lo: float,
    tau_hi: float,
) -> Tensor:
    """integrate the time-score along curve over [tau_lo, tau_hi]; return -integral on CPU.

    inner kernel of predict_ldr_via_curve; identical math to the single-leg case but
    parameterised by explicit tau bounds so the two-leg excision case can reuse it.
    """
    with torch.no_grad():
        tau = torch.linspace(
            tau_lo,
            tau_hi,
            steps=n_points,
            device=samples.device,
            dtype=samples.dtype,
        )

        ts = curve.points(tau)
        dts = curve.derivatives(tau)

        n_samples = samples.shape[0]
        chunk_size = max(1, 100000 // n_samples)

        scores_chunks = []
        for i in range(0, n_points, chunk_size):
            ts_chunk = ts[i : i + chunk_size]
            chunk_out = time_score_fn(path, ts_chunk, samples)
            scores_chunks.append(chunk_out)

        scores = torch.cat(scores_chunks, dim=0)

        if scores.dim() == 3:
            combined = (scores * dts.unsqueeze(1)).sum(dim=-1)
        else:
            combined = scores

        integral = integrator(combined, tau)
        return -integral.cpu()
