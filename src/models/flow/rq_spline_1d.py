"""rational-quadratic monotone splines (Durkan et al. arXiv:1906.04032).

all funcs operate on unit square [0,1]x[0,1]

affinity to [-2,2] responsibility belongs to `flow_policy_model`

key invariants:
- widths, heights, derivatives all strictly positive (normalized, clamped)
- bin boundaries exact 0, 1 via forced padding
- monotonicity guaranteed by param constraints; no separate monotone check
- derivative clamp [MIN_DERIV, MAX_DERIV] bounds |log density| for cross-policy evals
- float64 canonical path: all constants and intermediates preserve dtype
"""
import math

import torch
from torch import Tensor


# constants (dtype-agnostic, created at use-time)
MIN_BIN = 1e-3
MIN_DERIV = 1e-3
MAX_DERIV = 1e3


def softplus_inverse(y: float) -> float:
    """inverse of softplus on positive scalars: x s.t. log(1 + e^x) = y.

    used for identity-init offsets (ud init so that MIN_DERIV + softplus(ud)
    equals 1). scalar math; callers embed the result in init tensors.
    """
    return math.log(math.expm1(y))


def normalize_params(uw: Tensor, uh: Tensor, ud: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """normalize raw unconstrained params to spline coeffs.

    input: uw, uh [B, K]; ud [B, K+1]
    output: widths, heights, derivs [B, K], [B, K], [B, K+1], dtype-matched

    procedure:
      validate K and min_bin*K < 1 (ensures positive gap for softmax).
      widths = min_bin + (1 - min_bin*K) * softmax(uw, dim=-1).
      heights = min_bin + (1 - min_bin*K) * softmax(uh, dim=-1).
      derivs = softplus(ud) + MIN_DERIV, clamped to [MIN_DERIV, MAX_DERIV].
      return (widths, heights, derivs).
    """
    dtype = uw.dtype
    device = uw.device
    K = uw.shape[-1]

    assert uh.shape[-1] == K, f"uh last dim {uh.shape[-1]} != K {K}"
    assert ud.shape[-1] == K + 1, f"ud last dim {ud.shape[-1]} != K+1 {K + 1}"
    assert MIN_BIN * K < 1.0, f"MIN_BIN*K = {MIN_BIN*K} >= 1 (softmax gap invalid)"

    min_bin_t = torch.tensor(MIN_BIN, dtype=dtype, device=device)
    gap_scale = 1.0 - MIN_BIN * K
    gap_scale_t = torch.tensor(gap_scale, dtype=dtype, device=device)

    # widths, heights: [B, K]
    widths = min_bin_t + gap_scale_t * torch.softmax(uw, dim=-1)
    heights = min_bin_t + gap_scale_t * torch.softmax(uh, dim=-1)

    # derivs: [B, K+1], softplus + clamp
    min_deriv_t = torch.tensor(MIN_DERIV, dtype=dtype, device=device)
    max_deriv_t = torch.tensor(MAX_DERIV, dtype=dtype, device=device)

    derivs = torch.nn.functional.softplus(ud) + min_deriv_t
    derivs = torch.clamp(derivs, min=min_deriv_t, max=max_deriv_t)

    return widths, heights, derivs


def cumsum_with_boundaries(vals: Tensor) -> Tensor:
    """cumsum with exact 0 at start, exact 1 at end.

    input: vals [B, K]
    output: cumsum [B, K+1] with exact 0 at index 0, exact 1 at index K

    procedure:
      compute cumsum along dim=-1.
      prepend exact 0 at dim 1.
      force last column to exact 1.
      return [B, K+1] cumsum.
    """
    dtype = vals.dtype
    device = vals.device
    B = vals.shape[0]

    # cumsum: [B, K]
    cs = torch.cumsum(vals, dim=-1)

    # prepend exact 0: [B, 1]
    zeros = torch.zeros(B, 1, dtype=dtype, device=device)
    cs_padded = torch.cat([zeros, cs], dim=-1)  # [B, K+1]

    # force last to exact 1
    one_t = torch.tensor(1.0, dtype=dtype, device=device)
    cs_padded[:, -1] = one_t

    return cs_padded


def rq_forward(
    u: Tensor, widths: Tensor, heights: Tensor, derivs: Tensor
) -> tuple[Tensor, Tensor]:
    """forward pass: u in [0,1] -> x in [0,1], log-det.

    input: u [B]; widths, heights [B, K]; derivs [B, K+1]
    output: x [B], logdet [B], dtype-matched

    procedure:
      compute cumwidths, cumheights, cumderivs via cumsum_with_boundaries.
      locate bin k via searchsorted on cumwidths; clamp to [0, K-1].
      gather per-bin params.
      compute within-bin coordinate theta.
      apply rational-quadratic spline formula.
      clamp x to [0, 1].
      return (x, logdet).
    """
    K = widths.shape[-1]

    cumwidths = cumsum_with_boundaries(widths)  # [B, K+1]
    cumheights = cumsum_with_boundaries(heights)  # [B, K+1]

    # locate bin: interval convention [cw_k, cw_{k+1}); right=True then -1 so
    # u=0 -> bin 0 and u=1 -> bin K-1 after clamping.
    k = torch.searchsorted(cumwidths, u.unsqueeze(-1), right=True) - 1  # [B, 1]
    k = torch.clamp(k.squeeze(-1), min=0, max=K - 1)  # [B]

    # gather per-bin params
    cumwidth_k = torch.gather(cumwidths, 1, k.unsqueeze(-1)).squeeze(-1)  # [B]
    width_k = torch.gather(widths, 1, k.unsqueeze(-1)).squeeze(-1)  # [B]
    cumheight_k = torch.gather(cumheights, 1, k.unsqueeze(-1)).squeeze(-1)  # [B]
    height_k = torch.gather(heights, 1, k.unsqueeze(-1)).squeeze(-1)  # [B]
    d_k = torch.gather(derivs, 1, k.unsqueeze(-1)).squeeze(-1)  # [B]
    d_kp1 = torch.gather(derivs, 1, (k + 1).unsqueeze(-1)).squeeze(-1)  # [B]

    # within-bin coordinate; params are bounded below by construction
    # (normalize_params), so no epsilon guards: they would poison exactness.
    theta = (u - cumwidth_k) / width_k  # [B]
    delta_k = height_k / width_k  # [B]
    tt = theta * (1 - theta)  # [B]

    denom = delta_k + (d_kp1 + d_k - 2 * delta_k) * tt  # [B], provably > 0
    x = cumheight_k + height_k * (delta_k * theta ** 2 + d_k * tt) / denom  # [B]

    # dy/du = delta_k^2 (d_kp1 theta^2 + 2 delta_k tt + d_k (1-theta)^2)/denom^2
    # (Durkan eq. with delta = h/w already yields d/du; no extra width term.)
    numerator_deriv = (
        delta_k ** 2
        * (d_kp1 * theta ** 2 + 2 * delta_k * tt + d_k * (1 - theta) ** 2)
    )  # [B], provably > 0
    logdet = torch.log(numerator_deriv) - 2 * torch.log(denom)  # [B]

    x = torch.clamp(x, min=0.0, max=1.0)  # float-noise guard only

    return x, logdet


def rq_inverse(
    x: Tensor, widths: Tensor, heights: Tensor, derivs: Tensor
) -> tuple[Tensor, Tensor]:
    """inverse pass: x in [0,1] -> u in [0,1], log-det.

    input: x [B]; widths, heights [B, K]; derivs [B, K+1]
    output: u [B], logdet [B], dtype-matched

    procedure:
      locate bin via searchsorted on cumheights; clamp to [0, K-1].
      gather per-bin params.
      solve quadratic for theta using stable root formula.
      recover u and logdet (negative of forward).
      clamp u to [0, 1].
      return (u, logdet).
    """
    K = widths.shape[-1]

    cumwidths = cumsum_with_boundaries(widths)  # [B, K+1]
    cumheights = cumsum_with_boundaries(heights)  # [B, K+1]

    # locate bin via cumheights (same interval convention as rq_forward)
    k = torch.searchsorted(cumheights, x.unsqueeze(-1), right=True) - 1  # [B, 1]
    k = torch.clamp(k.squeeze(-1), min=0, max=K - 1)  # [B]

    # gather per-bin params
    cumwidth_k = torch.gather(cumwidths, 1, k.unsqueeze(-1)).squeeze(-1)  # [B]
    width_k = torch.gather(widths, 1, k.unsqueeze(-1)).squeeze(-1)  # [B]
    cumheight_k = torch.gather(cumheights, 1, k.unsqueeze(-1)).squeeze(-1)  # [B]
    height_k = torch.gather(heights, 1, k.unsqueeze(-1)).squeeze(-1)  # [B]
    d_k = torch.gather(derivs, 1, k.unsqueeze(-1)).squeeze(-1)  # [B]
    d_kp1 = torch.gather(derivs, 1, (k + 1).unsqueeze(-1)).squeeze(-1)  # [B]

    # solve quadratic for theta
    # y = cumheight_k + height_k*(delta_k*theta^2 + d_k*theta*(1-theta)) / (denom)
    # where denom = delta_k + (d_kp1 + d_k - 2*delta_k)*theta*(1-theta)
    #
    # rearranging: denom*y = denom*cumheight_k + height_k*(delta_k*theta^2 + d_k*theta*(1-theta))
    # substitute x for y and solve for theta
    #
    # expanded form:
    # a = (x - cumheight_k)*(d_k + d_kp1 - 2*delta_k) + height_k*(delta_k - d_k)
    # b = height_k*d_k - (x - cumheight_k)*(d_k + d_kp1 - 2*delta_k)
    # c = -delta_k*(x - cumheight_k)

    delta_k = height_k / width_k  # [B]

    x_shifted = x - cumheight_k  # [B]
    common_coeff = d_k + d_kp1 - 2 * delta_k  # [B]

    a = x_shifted * common_coeff + height_k * (delta_k - d_k)  # [B]
    b = height_k * d_k - x_shifted * common_coeff  # [B]
    c = -delta_k * x_shifted  # [B]

    # discriminant clamped at 0 (float noise only; analytically >= 0)
    disc = torch.clamp(b ** 2 - 4 * a * c, min=0.0)  # [B]

    # stable root 2c/(-b - sqrt(disc)): avoids cancellation for small a.
    # at x_shifted = 0 the denominator is -2*height_k*d_k < 0, never zero.
    theta = 2 * c / (-b - torch.sqrt(disc))  # [B]
    theta = torch.clamp(theta, min=0.0, max=1.0)  # float-noise guard only

    # recover u
    u = cumwidth_k + width_k * theta  # [B]

    # logdet = negative of forward logdet at theta
    # forward logdet at the recovered theta (Durkan dy/du; no width term),
    # negated for the inverse direction.
    tt = theta * (1 - theta)  # [B]
    denom_fwd = delta_k + common_coeff * tt  # [B], provably > 0
    numerator_deriv_fwd = (
        delta_k ** 2
        * (d_kp1 * theta ** 2 + 2 * delta_k * tt + d_k * (1 - theta) ** 2)
    )  # [B], provably > 0

    logdet = -(torch.log(numerator_deriv_fwd) - 2 * torch.log(denom_fwd))  # [B]

    u = torch.clamp(u, min=0.0, max=1.0)  # float-noise guard only

    return u, logdet


def rq_transform(
    u: Tensor, widths: Tensor, heights: Tensor, derivs: Tensor, inverse: bool = False
) -> tuple[Tensor, Tensor]:
    """dispatcher: forward iff inverse=False, else rq_inverse.

    input: u or x [B]; widths, heights [B, K]; derivs [B, K+1]; inverse [bool]
    output: x, logdet or u, logdet [B], [B], dtype-matched

    returns rq_forward(u, ...) if inverse=False, else rq_inverse(x, ...).
    """
    if inverse:
        return rq_inverse(u, widths, heights, derivs)
    else:
        return rq_forward(u, widths, heights, derivs)
