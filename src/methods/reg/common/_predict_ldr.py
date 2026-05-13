from typing import Callable

import torch
from torch import Tensor


def predict_ldr_via_curve(
    time_score_fn: Callable,
    path,
    curve,
    integrator: Callable,
    n_points: int,
    samples: Tensor,
) -> Tensor:
    """Integrate time-score along curve in [path.eps, 1-path.eps]; return -integral on CPU.

    Implements chunked vmap over tau grid points, chain-rule combination for 2D curves,
    and tensor-native integration. EMA apply/restore is the caller's responsibility.

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

    Returns:
        Tensor [n_samples] on CPU. Log-density-ratio estimate.

    Raises:
        ValueError: if n_points < 2.

    Device behavior: All computation on samples.device; integrator returns CPU tensor.
    """
    # step 1: validate n_points
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2; got {n_points}")

    # step 2: build tau grid
    tau = torch.linspace(
        path.eps,
        1 - path.eps,
        steps=n_points,
        device=samples.device,
        dtype=samples.dtype,
    )

    # step 3: evaluate curve at tau
    ts = curve.points(tau)      # [n_points, curve.dim]
    dts = curve.derivatives(tau)  # [n_points, curve.dim]

    # step 4: compute chunked time-scores via vmap
    n_samples = samples.shape[0]
    chunk_size = max(1, 100000 // n_samples)

    def call_time_score_fn_on_chunk(ts_chunk):
        # ts_chunk: [chunk_len, curve.dim]
        # returns: [chunk_len, n_samples] or [chunk_len, n_samples, 2]
        return time_score_fn(path, ts_chunk, samples)

    vmapped_fn = torch.vmap(
        call_time_score_fn_on_chunk,
        in_dims=0,
        out_dims=0,
        randomness="different",
    )

    scores_chunks = []
    for i in range(0, n_points, chunk_size):
        ts_chunk = ts[i : i + chunk_size]  # [chunk_len, curve.dim]
        chunk_out = vmapped_fn(ts_chunk)  # [chunk_len, n_samples, {1,2}] or [chunk_len, n_samples]
        scores_chunks.append(chunk_out)

    scores = torch.cat(scores_chunks, dim=0)  # [n_points, n_samples, {1,2}] or [n_points, n_samples]

    # step 5: apply chain rule if 2d (dim==2)
    if scores.dim() == 3:
        # 2d case: scores [n_points, n_samples, 2], dts [n_points, 2]
        # d log rho / d tau = sum_i (d log rho / d s_i) * (d s_i / d tau)
        combined = (scores * dts.unsqueeze(1)).sum(dim=-1)  # [n_points, n_samples]
    else:
        # 1d case: [n_points, n_samples]
        combined = scores

    # step 6: integrate and negate
    integral = integrator(combined, tau)  # signed integral, [n_samples]
    return -integral.cpu()  # log(p0/p1) = -integral
