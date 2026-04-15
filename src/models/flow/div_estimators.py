"""
divergence estimation utilities for computing trace(Jacobian) of vector fields.

provides three methods for divergence estimation:
- exact_div: exact Jacobian trace via torch.func.jacrev
- hutch_div: stochastic Hutchinson trace estimator with Gaussian or Rademacher noise
- div_chunked: batch-chunking wrapper to prevent OOM on large batches

all functions accept a vecfield callable [D] -> [D] for single samples, handle
batching via vmap internally, and return divergence estimates [B].
"""

import torch
import math
from torch import Tensor
from typing import Callable


def exact_div(
    vecfield: Callable[[Tensor], Tensor],
    x: Tensor,
) -> Tensor:
    """
    exact divergence via Jacobian trace computation.

    computes divergence (trace of Jacobian) of vecfield at each point via
    torch.func.jacrev and torch.trace, vmapped over batch dimension.

    Inputs:
      vecfield: callable [D] -> [D], pure function representing velocity field
      x: batch of points [B, D] where divergence is computed

    Output:
      tensor [B,] of exact divergence values, one per sample
    """

    def div_single(z_single: Tensor) -> Tensor:
        """jacobian trace for single sample [D] -> scalar."""
        jac = torch.func.jacrev(vecfield)(z_single)  # [D, D]
        return torch.trace(jac)

    # vmap over batch dimension
    return torch.vmap(div_single)(x)  # [B]


def hutch_div(
    vecfield: Callable[[Tensor], Tensor],
    x: Tensor,
    noise: str = "gaussian",
) -> Tensor:
    """
    stochastic divergence via Hutchinson trace estimator.

    approximates trace(Jacobian) via E[eps^T * J * eps] where eps is sampled
    noise and J is the Jacobian. computes via VJP: (vjp(eps) · eps).
    vmapped over batch with independent noise per sample.

    Inputs:
      vecfield: callable [D] -> [D], pure function representing velocity field
      x: batch of points [B, D] where divergence is computed
      noise: noise distribution, either "gaussian" (N(0,I)) or "rademacher" ({-1,+1}^D)

    Output:
      tensor [B,] of stochastic divergence estimates
    """

    def hutch_single(z_single: Tensor) -> Tensor:
        """Hutchinson estimator for single sample."""
        # sample noise with specified distribution
        if noise == "gaussian":
            eps = torch.randn_like(z_single)  # [D] ~ N(0, I)
        elif noise == "rademacher":
            shape = z_single.shape
            eps = (
                torch.randint(0, 2, shape, dtype=z_single.dtype, device=z_single.device)
                * 2.0
                - 1.0
            )  # [D] in {-1, +1}
        else:
            raise ValueError(f"noise must be 'gaussian' or 'rademacher', got {noise}")

        # compute VJP and apply to noise
        _, vjp_func = torch.func.vjp(vecfield, z_single)  # vjp_func: [D] -> [D]
        grad_z = vjp_func(eps)[0]  # [D] (vjp_func returns tuple)

        # dot product: (vjp(eps) · eps) is unbiased estimator of trace
        return (grad_z * eps).sum()  # scalar

    # vmap over batch with independent noise per sample
    return torch.vmap(hutch_single)(x)  # [B]


def div_chunked(
    div_fn: Callable[[Callable[[Tensor], Tensor], Tensor], Tensor],
    vecfield: Callable[[Tensor], Tensor],
    x: Tensor,
    chunk_size: int = 500,
) -> Tensor:
    """
    chunked divergence computation to prevent OOM on large batches.

    splits batch into chunks, applies div_fn to each chunk, and concatenates
    results. if batch size <= chunk_size, applies div_fn directly without
    chunking.

    Inputs:
      div_fn: divergence function (vecfield, x_batch) -> [B,]
              e.g., exact_div, hutch_div, or partial(hutch_div, noise="gaussian")
      vecfield: callable [D] -> [D], pure function representing velocity field
      x: full batch [B, D]
      chunk_size: maximum samples per vmap call, default 500

    Output:
      tensor [B,] concatenated from all chunks
    """
    B = x.shape[0]

    # no chunking needed
    if B <= chunk_size:
        return div_fn(vecfield, x)  # [B]

    # split into chunks and process
    div_list = []
    num_chunks = math.ceil(B / chunk_size)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, B)

        x_chunk = x[start_idx:end_idx]  # [C, D]
        div_chunk = div_fn(vecfield, x_chunk)  # [C]
        div_list.append(div_chunk)

    return torch.cat(div_list, dim=0)  # [B]
