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
import warnings
from functools import partial
from torch import Tensor
from typing import Callable, Optional


def build_div_fn(
    method: str,
    *,
    noise: Optional[str] = None,
    n_samples: int = 1,
) -> Callable:
    """select a divergence estimator once and return a single callable.

    accepts two API styles:
      - separated: method in {"exact", "hutchinson"}; noise required for
        "hutchinson"; n_samples > 1 averages independent Hutchinson estimates.
        used by VFM-family (which exposes `div_method`, `div_noise`,
        `n_hutch_samples` as three independent knobs).
      - combined-key: method in {"exact", "hutch_gaussian", "hutch_rademacher"}.
        used by FMDRE-family `ratio_ode*` which pack method+noise into a single
        string in their public API.

    binds all dispatch at construction; the returned callable signature is
    `(vecfield, x) -> [B]` and is safe to use inside vmap or in the body of an
    integration loop without any per-step branches.
    """
    if method == "exact":
        return exact_div
    if method == "hutch_gaussian":
        return partial(hutch_div, noise="gaussian")
    if method == "hutch_rademacher":
        return partial(hutch_div, noise="rademacher")
    if method == "hutchinson":
        if noise not in ("gaussian", "rademacher"):
            raise ValueError(
                f"noise must be 'gaussian' or 'rademacher' when method='hutchinson'; "
                f"got {noise!r}"
            )
        if n_samples == 1:
            return partial(hutch_div, noise=noise)

        def avg_div(vecfield, x):
            out = hutch_div(vecfield, x, noise=noise)
            for _ in range(n_samples - 1):
                out = out + hutch_div(vecfield, x, noise=noise)
            return out / n_samples
        return avg_div
    raise ValueError(
        f"method must be one of "
        f"{{'exact', 'hutchinson', 'hutch_gaussian', 'hutch_rademacher'}}; "
        f"got {method!r}"
    )


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

    approximates trace(Jacobian) via E[eps^T J eps] where eps is sampled noise
    and J is the Jacobian; computed via VJP as <vjp(eps), eps>. vmapped over
    batch with independent noise per sample.

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

        # <vjp(eps), eps> is an unbiased estimator of the trace
        return (grad_z * eps).sum()

    # vmap over batch with independent noise per sample
    return torch.vmap(hutch_single, randomness='different')(x)  # [B]


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


def compute_divergence(
    output: Tensor,
    x: Tensor,
    epsilon: Tensor = None,
) -> Tensor:
    """
    Computes exact divergence via per-dimension autograd loop.

    **Deprecated:** Use `exact_div(f, x)` or `hutch_div(f, x)` for new code.
    This function is maintained for backward compatibility and computes
    divergence by summing per-dimension gradients: sum_i d(output_i)/dx_i.

    Uses create_graph=False since this is only called during inference,
    not during training where backprop through the divergence is needed.

    Inputs:
      output: vector field outputs [B, D]
      x: input points [B, D] (must have requires_grad=True)
      epsilon: ignored (kept for API compatibility)

    Output:
      divergence estimates [B] (one per sample)
    """
    warnings.warn(
        "compute_divergence is deprecated; prefer exact_div(f, x) or hutch_div(f, x) for new code.",
        DeprecationWarning,
        stacklevel=2,
    )

    batch_size, dim = x.shape
    divergence = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

    for i in range(dim):
        grad_i = torch.autograd.grad(
            outputs=output[:, i].sum(),
            inputs=x,
            create_graph=False,
            retain_graph=False,
        )[0]
        divergence = divergence + grad_i[:, i]

    return divergence
