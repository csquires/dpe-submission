"""karras/edm preconditioning for denoising-style regression losses.

the module provides moment estimation and coefficient computation to wrap
regression networks as optimal affine predictors of targets (noise, velocity,
x0, score) from noisy inputs.

all functions are purely functional (closures, no mutable state). coefficients
remain tau-differentiable: no detach(), no torch.no_grad().
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple

import torch
from torch import Tensor


# module constants (numerical stability)
_EPS_VAR = 1e-8          # floor for Var[x_t] before c_in inversion
_EPS_COUT2 = 1e-8        # floor for c_out^2 before sqrt
_EPS_MOMENTS = 1e-8      # fail-loud threshold for near-zero endpoint variance


@dataclass(frozen=True)
class Moments:
    """per-dimension endpoint moment summaries (variance + covariance).

    the best-affine EDM predictor (Section 2) depends only on variance of the
    target, covariance of target with the noisy input, and variance of the noisy
    input — all per-dimension to yield a diagonal preconditioner.

    Attributes:
        var: dict[str, Tensor] name -> [D] per-dimension variance (float32, on device).
        cov: dict[tuple[str,str], Tensor] sorted (name_i, name_j) -> [D] per-dim covariance.
    """
    var: dict[str, Tensor]
    cov: dict[tuple[str, Tensor], Tensor]


def endpoint_moments(samples: dict[str, Tensor]) -> Moments:
    """compute per-dimension variance and covariance from endpoint samples.

    for each sample set, move to device (inferred from values) and cast to float32.
    compute unbiased sample variance via Tensor.var(dim=0, correction=1). compute
    per-dimension cross-covariance via ((a - a.mean(0)) * (b - b.mean(0))).sum(0) / (N-1).

    fail loudly (ValueError) if:
    - any N < 2 (cannot compute unbiased variance).
    - any feature has variance < _EPS_MOMENTS (numerical safety).

    Args:
        samples: dict[name: str] -> tensor [N, D]. N >= 2, all on same device.

    Returns:
        Moments with:
          var[name] = [D] unbiased variance
          cov[(name_i, name_j)] = [D] unbiased covariance (sorted tuple keys).
    """
    # infer device from first sample
    device = next(iter(samples.values())).device

    # move to device, cast float32, extract N
    data = {}
    for name, x in samples.items():
        x = x.to(device=device, dtype=torch.float32)
        if x.shape[0] < 2:
            raise ValueError(f"endpoint_moments: {name} has N={x.shape[0]} < 2")
        data[name] = x

    # compute variances
    var_dict = {}
    for name, x in data.items():
        v = x.var(dim=0, correction=1)  # [D] unbiased variance
        if (v < _EPS_MOMENTS).any():
            raise ValueError(f"endpoint_moments: {name} has ~0 variance; check data distribution")
        var_dict[name] = v

    # compute covariances (sorted tuple keys)
    cov_dict = {}
    names = sorted(data.keys())
    for i, name_i in enumerate(names):
        for name_j in names[i:]:
            xi, xj = data[name_i], data[name_j]
            N = xi.shape[0]
            if name_i == name_j:
                # diagonal: variance (same computation)
                cov = xi.var(dim=0, correction=1)  # [D]
            else:
                # off-diagonal: cross-covariance
                xi_c = xi - xi.mean(0)  # [N, D] centered
                xj_c = xj - xj.mean(0)  # [N, D] centered
                cov = (xi_c * xj_c).sum(0) / (N - 1)  # [D] unbiased covariance
            cov_dict[(name_i, name_j)] = cov

    return Moments(var=var_dict, cov=cov_dict)


def _bilinear(a: dict[str, Tensor], b: dict[str, Tensor], moments: Moments) -> Tensor:
    """compute bilinear form sum_i a_i b_i var_i + sum_{i<j} (a_i b_j + a_j b_i) cov_ij.

    a, b: dict[name:str] -> [D] per-dimension coefficient vectors (same names).
    moments: pre-computed Moments.

    returns [D] per-dimension result (same dtype/device as moments.var).
    """
    # diagonal terms
    result = None
    for name in a.keys():
        term = a[name] * b[name] * moments.var[name]  # [D]
        result = term if result is None else result + term

    # off-diagonal: Cov[sum_i a_i X_i, sum_j b_j X_j] cross term is
    # (a_i b_j + a_j b_i) cov_ij -- NOT 2 a_i b_j cov_ij, which holds only for a == b.
    names = sorted(a.keys())
    for i, name_i in enumerate(names):
        for name_j in names[i+1:]:
            cov_ij = moments.cov[(name_i, name_j)]  # [D]
            term = (a[name_i] * b[name_j] + a[name_j] * b[name_i]) * cov_ij
            result = term if result is None else result + term

    return result


Coeffs = NamedTuple('Coeffs', [
    ('c_in', Tensor),      # [B, D] or [D] per-dim input gain
    ('c_out', Tensor),     # [B, D] or [D] per-dim output gain
    ('c_skip', Tensor),    # [B, D] or [D] per-dim skip connection
    ('c_noise', Tensor),   # [B, 1] noise embedding
])


def make_coeffs(interp, moments: Moments, target_kind: str) -> Callable:
    """build closure tau -> Coeffs via the unifying EDM formula.

    unifying formula (all targets):
    c_in = Var[x_t]^{-1/2}
    c_skip = Cov[x_t, y] / Var[x_t]
    c_out^2 = Var[y] - Cov[x_t, y]^2 / Var[x_t]
    c_noise = 0.25 * ln(clamp(gamma, min=eps))

    Args:
        interp: path object (DirectPath1D or TriangularPath1D) OR "fm" (FMDRE sentinel).
        moments: pre-computed Moments from endpoint_moments().
        target_kind: "noise", "velocity", "x0", or "score".

    Returns:
        closure tau [B, 1] -> Coeffs with shapes [B, D] or [D].

    arithmetic is plain tensor ops (no torch.no_grad, no .detach): coeffs remain
    tau-differentiable.
    """

    if interp == "fm":
        # FMDRE: conditional flow-matching interpolant
        # x_t = (1-tau)*z + tau*x_data, z ~ N(0, I)

        def coeff_fn_fm(tau: Tensor) -> Coeffs:
            """tau [B, 1] -> Coeffs."""
            # extract var_xdata [D]
            var_xdata = moments.var["x_data"]

            # Var[x_t] = tau^2 * var_xdata + (1-tau)^2, clamp to _EPS_VAR
            tau_sq = tau ** 2  # [B, 1]
            one_minus_tau = 1.0 - tau  # [B, 1]
            var_xt = tau_sq * var_xdata + one_minus_tau ** 2  # [B, D]
            var_xt = torch.clamp(var_xt, min=_EPS_VAR)

            # dispatch on target_kind
            if target_kind == "velocity":
                # Cov[x_t, y] = tau*var_xdata - (1-tau), Var[y] = var_xdata + 1
                cov_xt_y = tau * var_xdata - one_minus_tau  # [B, D]
                var_y = var_xdata + 1.0  # [D]
            elif target_kind == "score":
                # Cov[x_t, y] = -1, Var[y] = (1-tau)^{-2}
                cov_xt_y = torch.tensor(-1.0, device=tau.device, dtype=tau.dtype)  # [1]
                var_y = 1.0 / (one_minus_tau ** 2)  # [B, D]
            else:
                raise ValueError(f"unsupported target_kind for FMDRE: {target_kind}")

            # unifying formula
            c_in = 1.0 / torch.sqrt(var_xt)  # [B, D]
            c_skip = cov_xt_y / var_xt  # [B, D]
            c_out_sq = var_y - cov_xt_y ** 2 / var_xt  # [B, D]
            c_out_sq = torch.clamp(c_out_sq, min=_EPS_COUT2)
            c_out = torch.sqrt(c_out_sq)  # [B, D]

            # c_noise = 0.25 * ln(clamp(gamma, min=eps))
            # gamma_fm = 1 - tau
            gamma_fm = one_minus_tau  # [B, 1]
            c_noise = 0.25 * torch.log(torch.clamp(gamma_fm, min=_EPS_VAR))  # [B, 1]

            return Coeffs(c_in=c_in, c_out=c_out, c_skip=c_skip, c_noise=c_noise)

        return coeff_fn_fm

    else:
        # VFM family: stochastic interpolant with endpoint moments

        def coeff_fn_vfm(tau: Tensor) -> Coeffs:
            """tau [B, 1] -> Coeffs."""
            # extract path.weights(tau) -> NamedTuple (DirectWeights1D or TriangularWeights1D)
            w_nt = interp.weights(tau)  # NamedTuple with alpha, beta, [w_star], d_alpha, d_beta, [d_w_star]

            # extract path.gamma(tau), path.dgamma_dtau(tau)
            gamma = interp.gamma(tau)  # [B, 1]
            dgamma = interp.dgamma_dtau(tau)  # [B, 1]

            # convert NamedTuple fields to dict keyed by endpoint name
            # keys MUST match moments.var exactly: "x0", "x1", and (3-endpoint only) "xstar"
            # map alpha->x0, beta->x1, w_star->xstar; d_alpha->x0, d_beta->x1, d_w_star->xstar
            w_dict = {"x0": w_nt.alpha, "x1": w_nt.beta}  # [B, 1] or [1] depending on batch
            dw_dict = {"x0": w_nt.d_alpha, "x1": w_nt.d_beta}  # [B, 1] or [1]

            # detect 3-endpoint case from NamedTuple fields
            if hasattr(w_nt, "w_star"):
                w_dict["xstar"] = w_nt.w_star  # [B, 1]
                dw_dict["xstar"] = w_nt.d_w_star  # [B, 1]

            # build e0 = unit vector at x0
            e0_dict = {}
            for name in moments.var.keys():
                if name == "x0":
                    e0_dict[name] = torch.ones_like(w_dict["x0"])  # [B, 1]
                else:
                    e0_dict[name] = torch.zeros_like(w_dict.get(name, w_dict["x0"]))  # [B, 1]

            # dispatch on target_kind
            if target_kind == "noise":
                # Cov[x_t, y] = gamma, Var[y] = 1
                cov_xt_y = gamma  # [B, 1]
                var_y = 1.0  # scalar
                var_xt = _bilinear(w_dict, w_dict, moments) + gamma ** 2  # [D]

            elif target_kind == "velocity":
                # Cov[x_t, y] = bilinear(w, dw) + gamma*dgamma
                # Var[y] = bilinear(dw, dw) + dgamma^2
                # Var[x_t] = bilinear(w, w) + gamma^2
                cov_xt_y = _bilinear(w_dict, dw_dict, moments) + gamma * dgamma  # [D]
                var_y = _bilinear(dw_dict, dw_dict, moments) + dgamma ** 2  # [D]
                var_xt = _bilinear(w_dict, w_dict, moments) + gamma ** 2  # [D]

            elif target_kind == "x0":
                # Cov[x_t, y] = bilinear(w, e0) where e0 is [1, 0, ...]
                # Var[y] = Var[x0]
                # Var[x_t] = bilinear(w, w) + gamma^2
                cov_xt_y = _bilinear(w_dict, e0_dict, moments)  # [D]
                var_y = moments.var["x0"]  # [D]
                var_xt = _bilinear(w_dict, w_dict, moments) + gamma ** 2  # [D]

            else:
                raise ValueError(f"unsupported target_kind for VFM: {target_kind}")

            # unifying formula with two clamps
            var_xt = torch.clamp(var_xt, min=_EPS_VAR)  # [D] or [B, D]
            c_in = 1.0 / torch.sqrt(var_xt)  # [D] or [B, D]
            c_skip = cov_xt_y / var_xt  # [D] or [B, D]
            c_out_sq = var_y - cov_xt_y ** 2 / var_xt  # [D] or [B, D]
            c_out_sq = torch.clamp(c_out_sq, min=_EPS_COUT2)
            c_out = torch.sqrt(c_out_sq)  # [D] or [B, D]

            # c_noise = 0.25 * ln(clamp(gamma, min=eps))
            c_noise = 0.25 * torch.log(torch.clamp(gamma, min=_EPS_VAR))  # [B, 1]

            return Coeffs(c_in=c_in, c_out=c_out, c_skip=c_skip, c_noise=c_noise)

        return coeff_fn_vfm


def make_lambda(coeff_fn: Callable) -> Callable:
    """build closure tau -> lambda = 1 / mean_D(c_out^2).

    computes the per-sample EDM loss weight (Section 2: lambda = c_out^{-2}).
    the mean over dimension D yields a uniform effective training weight across
    tau (Section 2.3: "uniform effective loss weight property").

    Args:
        coeff_fn: closure tau -> Coeffs.

    Returns:
        closure tau [B, 1] -> [B] per-sample lambda values.
    """
    def lambda_fn(tau: Tensor) -> Tensor:
        coeffs = coeff_fn(tau)
        # c_out [B, D]; mean over D -> [B]
        cout2_mean = (coeffs.c_out ** 2).mean(dim=-1)
        return 1.0 / cout2_mean

    return lambda_fn


def wrap(net: Callable, coeff_fn: Callable) -> Callable:
    """precondition single-head network via D(x_t, tau) = c_skip*x_t + c_out*F(c_in*x_t, c_noise).

    the wrapper is parameter-free and vmap-safe (all-or-nothing). forward_pass
    reconstructs coefficients from tau and runs the preconditioned network.

    space-first: the wrapped callable has signature (x, tau) -- a drop-in
    replacement for the raw VFM nets, which are called net(x, tau) by the loss
    closures and by vfm_time_score_1d. the inner net call is also space-first:
    net(c_in*x, c_noise).

    Args:
        net: callable (x_precond, c_noise) -> F_theta(x_precond, c_noise).
        coeff_fn: closure tau -> Coeffs.

    Returns:
        callable (x, tau) -> D(x, tau) with x [B, D], tau [B, 1].
    """
    def forward(x: Tensor, tau: Tensor) -> Tensor:
        coeffs = coeff_fn(tau)
        x_in = coeffs.c_in * x  # [B, D]
        f_out = net(x_in, coeffs.c_noise)  # [B, D]
        return coeffs.c_skip * x + coeffs.c_out * f_out  # [B, D]

    return forward


def wrap_2head(net: Callable, coeff_fn0: Callable, coeff_fn1: Callable) -> Callable:
    """precondition 2-head network (e.g., VFMOrthros); share c_in/c_noise, per-head c_skip/c_out.

    net returns tuple (head0_raw, head1_raw) [B, D]. coefficients are sourced:
    - c_in, c_noise from coeff_fn0 (shared)
    - c_skip0, c_out0 from coeff_fn0; c_skip1, c_out1 from coeff_fn1

    sharing c_in/c_noise reduces redundant forward passes and reflects the
    fact that both heads operate on the same input embedding.

    space-first: the wrapped callable has signature (x, tau) -- a drop-in
    replacement for the raw OrthrosNet, called net(x, tau) by the loss closure
    and by vfm_orthros_time_score_1d. the inner net call is space-first:
    net(c_in*x, c_noise).

    Args:
        net: callable (x_precond, c_noise) -> (f0_raw, f1_raw).
        coeff_fn0: closure tau -> Coeffs (head 0).
        coeff_fn1: closure tau -> Coeffs (head 1).

    Returns:
        callable (x, tau) -> (D0(x, tau), D1(x, tau)).
    """
    def forward(x: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
        coeffs0 = coeff_fn0(tau)
        coeffs1 = coeff_fn1(tau)

        x_in = coeffs0.c_in * x  # [B, D]
        f0_raw, f1_raw = net(x_in, coeffs0.c_noise)  # each [B, D]

        d0 = coeffs0.c_skip * x + coeffs0.c_out * f0_raw  # [B, D]
        d1 = coeffs1.c_skip * x + coeffs1.c_out * f1_raw  # [B, D]

        return (d0, d1)

    return forward


def wrap_fm(net: Callable, coeff_fn_v: Callable, coeff_fn_s: Callable, *,
            onehot: bool = False) -> object:
    """precondition 2-head time-first network (FMDRE: velocity & score heads).

    net signature depends on onehot:
    - False: net(c_noise, c_in*x, tau_or_encoding) -> (v_raw, s_raw)
    - True: net(c_noise, c_in*x, y_onehot) -> (v_raw, s_raw)

    time-first: the FMDRE model is called model(tau, x_t, c); the wrapper is a
    drop-in replacement and keeps that order. wrapper exposes:
    - __call__(tau, x, c) -> (v, s)
    - if onehot: forward_from_onehot(tau, x, y_oh) -> (v, s)

    Args:
        net: callable (c_noise, x_in, c) -> (v_raw, s_raw).
        coeff_fn_v: closure tau -> Coeffs (velocity head).
        coeff_fn_s: closure tau -> Coeffs (score head).
        onehot: if True, net takes onehot encoding instead of tau.

    Returns:
        callable __call__(tau, x, c) -> (v_out, s_out).
        and if onehot: method forward_from_onehot(tau, x, y_oh) -> (v_out, s_out).
    """
    class FMWrapper:
        def __call__(self, tau: Tensor, x: Tensor, c) -> tuple[Tensor, Tensor]:
            """time-first call: tau [B, 1], x [B, D], c context (tau or onehot).

            returns (v_out, s_out) both [B, D].
            """
            coeffs_v = coeff_fn_v(tau)
            coeffs_s = coeff_fn_s(tau)

            x_in = coeffs_v.c_in * x  # [B, D]
            v_raw, s_raw = net(coeffs_v.c_noise, x_in, c)  # each [B, D]

            v_out = coeffs_v.c_skip * x + coeffs_v.c_out * v_raw  # [B, D]
            s_out = coeffs_s.c_skip * x + coeffs_s.c_out * s_raw  # [B, D]

            return (v_out, s_out)

        def forward_from_onehot(self, tau: Tensor, x: Tensor, y_oh) -> tuple[Tensor, Tensor]:
            """preconditioned call with an already-one-hot class encoding.

            routes the inner net call through net.forward_from_onehot (which
            accepts a float one-hot), not net.__call__ (which expects a long
            class index and would re-one-hot). only defined if onehot=True.
            """
            if not onehot:
                raise RuntimeError("forward_from_onehot only available when onehot=True")
            coeffs_v = coeff_fn_v(tau)
            coeffs_s = coeff_fn_s(tau)

            x_in = coeffs_v.c_in * x  # [B, D]
            v_raw, s_raw = net.forward_from_onehot(coeffs_v.c_noise, x_in, y_oh)

            v_out = coeffs_v.c_skip * x + coeffs_v.c_out * v_raw  # [B, D]
            s_out = coeffs_s.c_skip * x + coeffs_s.c_out * s_raw  # [B, D]

            return (v_out, s_out)

        def eval(self) -> "FMWrapper":
            """delegate eval-mode to the wrapped net; the wrapper is parameter-free.

            ratio_ode / ratio_ode_s2 / ratio_ode_triangular call model.eval()
            on whatever they are handed; the precond wrapper forwards it so the
            FMDRE-family precond inference path works as a drop-in.
            """
            net.eval()
            return self

        def train(self, mode: bool = True) -> "FMWrapper":
            """delegate train-mode to the wrapped net (mirrors eval())."""
            net.train(mode)
            return self

    return FMWrapper()
