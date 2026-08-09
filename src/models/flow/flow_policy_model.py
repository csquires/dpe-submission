import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rq_spline_1d import (
    normalize_params,
    rq_forward,
    rq_inverse,
    softplus_inverse,
    MIN_DERIV,
)


def featurize_state(s: Tensor) -> Tensor:
    """featurize pendulum state for flow policy hypernet.

    input: s [..., 2] where [..., 0] = theta, [..., 1] = theta_dot.
    output: f [..., 3] = [cos(theta), sin(theta), theta_dot / 8].
    all torch tensors; numpy callers must convert to/from torch.

    single source of truth for state featurization.
    """
    theta = s[..., 0]
    theta_dot = s[..., 1]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    scaled_dot = theta_dot / 8.0
    return torch.stack([cos_theta, sin_theta, scaled_dot], dim=-1)


class FlowPolicyModel(nn.Module):
    """conditional RQ-spline hypernet for bounded pendulum action policy.

    maps (u, s) to action a in [-2, 2] via rational-quadratic monotone spline
    with state-dependent bin widths, heights, derivatives. identity-initialized
    so log_prob(a, s) \approx -log(4) everywhere (uniform base on action support).

    architecture:
      featurize_state(s) [B, 3]
        -> backbone: Linear(3, hidden_dim) -> n_layers blocks of LayerNorm/GELU
        -> Linear(hidden_dim, 3*n_bins+1)
      unpack [B, 3*n_bins+1] into widths, heights, derivatives for RQ spline.
      push_forward: rq_forward(u [0,1]) -> x [0,1] -> a = -2 + 4*x [-2, 2].
      log_prob: reverse, a [-2,2] -> x -> rq_inverse -> log|da/du| with base -log(4).
    """

    def __init__(self, hidden_dim: int = 128, n_layers: int = 3, n_bins: int = 32) -> None:
        """initialize conditional spline hypernet.

        args:
            hidden_dim: width of hidden layers (default 128).
            n_layers: number of hidden Linear/LayerNorm/GELU blocks (default 3).
            n_bins: number of bins for RQ spline (default 32).

        procedure:
          1. store self.hidden_dim, self.n_layers, self.n_bins.
          2. record self.eval_dtype = "float64" as checkpoint attribute.
          3. build self.backbone: Linear(3, hidden_dim) -> [n_layers blocks of
             LayerNorm/GELU] -> final Linear(hidden_dim, 3*n_bins+1).
          4. zero-init final layer weights/bias.
          5. add fixed offsets to final layer bias:
             - bias[0:n_bins] += log(1/n_bins) (uniform bin widths).
             - bias[n_bins:2*n_bins] += log(1/n_bins) (uniform heights).
             - bias[2*n_bins:3*n_bins+1] += softplus_inverse(1-MIN_DERIV) (unit derivs).
          6. verify identity-init via forward pass: log_prob \approx -log(4) everywhere.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_bins = n_bins
        self.eval_dtype = "float64"

        # build backbone: [3] -> hidden_dim -> [n_layers hidden blocks] -> [3*n_bins+1]
        layers: list[nn.Module] = []
        prev = 3
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            prev = hidden_dim
        layers.append(nn.Linear(hidden_dim, 3 * n_bins + 1))
        self.backbone = nn.Sequential(*layers)

        # zero-init final layer
        self.backbone[-1].weight.data.zero_()
        self.backbone[-1].bias.data.zero_()

        # add fixed offsets for identity-init
        ud_init = softplus_inverse(1.0 - MIN_DERIV)
        log_uniform_bin = math.log(1.0 / n_bins)
        with torch.no_grad():
            # widths offset (first n_bins entries)
            self.backbone[-1].bias[0:n_bins] += log_uniform_bin
            # heights offset (next n_bins entries)
            self.backbone[-1].bias[n_bins : 2 * n_bins] += log_uniform_bin
            # derivatives offset (last n_bins+1 entries)
            self.backbone[-1].bias[2 * n_bins : 3 * n_bins + 1] += ud_init

        # verify identity-init
        self._verify_identity_init()

    def _verify_identity_init(self) -> None:
        """assert log_prob \approx -log(4) for random state, a \approx 0.

        sample [1, 3] random feats, forward, unpack spline params,
        call rq_inverse at x=0.5, verify logdet \approx 0.
        """
        with torch.no_grad():
            feats = torch.randn(1, 3)
            raw = self.backbone(feats)
            # unpack params
            uw = raw[:, : self.n_bins]
            uh = raw[:, self.n_bins : 2 * self.n_bins]
            ud = raw[:, 2 * self.n_bins :]
            widths, heights, derivs = normalize_params(uw, uh, ud)
            # test at x=0.5 (middle of support, a=0)
            x = torch.tensor([0.5], dtype=feats.dtype)  # [1]
            u, logdet = rq_inverse(x, widths, heights, derivs)  # [1], [1]
            # identity init => logdet \approx 0
            assert torch.abs(logdet[0]) < 1e-6, (
                f"identity-init logdet {logdet[0].item()} not ~0"
            )

    def push_forward(self, u: Tensor, s: Tensor) -> Tensor:
        """deterministic map u [0,1] + state s -> action a [-2, 2].

        args:
            u: uniform samples [B], dtype float32 or float64.
            s: state [B, 2] = [theta, theta_dot].

        returns:
            a: actions [B, ], dtype matches u/s input.

        procedure:
          1. featurize_state(s) -> [B, 3].
          2. backbone(feats) -> [B, 3*n_bins+1].
          3. unpack into widths, heights, derivatives.
          4. rq_forward(u, widths, heights, derivs) -> x [0, 1].
          5. a = -2 + 4*x.
        """
        bb_dtype = next(self.backbone.parameters()).dtype
        feats = featurize_state(s.to(bb_dtype))  # [B, 3]
        raw = self.backbone(feats).to(torch.float64)  # [B, 3*n_bins+1]
        # unpack spline params
        uw = raw[:, : self.n_bins]  # [B, n_bins]
        uh = raw[:, self.n_bins : 2 * self.n_bins]  # [B, n_bins]
        ud = raw[:, 2 * self.n_bins :]  # [B, n_bins+1]
        widths, heights, derivs = normalize_params(uw, uh, ud)
        # forward transform: u -> x
        x, _ = rq_forward(u, widths, heights, derivs)  # [B]
        # affine to [-2, 2]
        a = -2.0 + 4.0 * x  # [B]
        return a

    def log_prob(self, a: Tensor, s: Tensor) -> Tensor:
        """log probability of action a given state s (float64 canonical).

        args:
            a: actions [B], must be in [-2, 2].
            s: state [B, 2] = [theta, theta_dot].

        returns:
            lp: log p(a | s) [B], float64 dtype.

        error: if any a < -2 - 1e-12 or a > 2 + 1e-12, raise ValueError.

        procedure (float64 path):
          1. check a in [-2, 2] with 1e-12 tolerance.
          2. cast a, s to float64.
          3. featurize_state, backbone forward, unpack spline params.
          4. x = (a + 2) / 4 (inverse affine).
          5. clamp x to [0, 1] iff |x - clamp(x)| <= 1e-12.
          6. rq_inverse(x) -> u, logdet_inv.
          7. log_prob_a = -log(4) + logdet_inv.

        docstring derivation (chain rule):
          base density on u in [0, 1] is uniform (log p_u = 0).
          affine a = -2 + 4*x => da/dx = 4 => logdet_forward = log(4).
          reverse: x = (a+2)/4 => dx/da = 1/4 => log|dx/da| = -log(4).
          spline inverse gives log|du/dx| = logdet_inv.
          chain: log p(a) = log p_u(u(x(a))) + log|du/dx(a)| + log|dx/da|
                         = 0 + logdet_inv - log(4) = logdet_inv - log(4).
        """
        # error check: a in [-2, 2]
        tol = 1e-12
        if torch.any(a < -2.0 - tol) or torch.any(a > 2.0 + tol):
            raise ValueError(f"action outside [-2,2]: min={a.min().item()}, max={a.max().item()}")

        # cast to float64
        a_f64 = a.to(torch.float64)  # [B]
        s_f64 = s.to(torch.float64)  # [B, 2]

        # featurize in the backbone's native dtype (float32 hypernet), then
        # cast the emitted params to float64: the exactness lives in the
        # spline math, not the hypernet forward.
        bb_dtype = next(self.backbone.parameters()).dtype
        feats = featurize_state(s.to(bb_dtype))  # [B, 3]
        raw = self.backbone(feats).to(torch.float64)  # [B, 3*n_bins+1]

        # unpack spline params
        uw = raw[:, : self.n_bins]  # [B, n_bins]
        uh = raw[:, self.n_bins : 2 * self.n_bins]  # [B, n_bins]
        ud = raw[:, 2 * self.n_bins :]  # [B, n_bins+1]
        widths, heights, derivs = normalize_params(uw, uh, ud)

        # inverse affine: a [-2, 2] -> x [0, 1]
        x = (a_f64 + 2.0) / 4.0  # [B]

        # clamp x to [0, 1] iff float noise (|x - clip(x)| <= 1e-12)
        x_clamped = torch.clamp(x, 0.0, 1.0)
        x_noise = torch.abs(x - x_clamped)
        x = torch.where(x_noise <= tol, x_clamped, x)

        # spline inverse
        u, logdet_inv = rq_inverse(x, widths, heights, derivs)  # [B], [B]

        # log_prob_a = -log(4) + logdet_inv
        log_prob_a = -math.log(4.0) + logdet_inv  # [B], float64
        return log_prob_a

    def cdf(self, a: Tensor, s: Tensor) -> Tensor:
        """empirical CDF F(a | s) for PIT calibration (float64 canonical).

        args:
            a: actions [B], must be in [-2, 2].
            s: state [B, 2] = [theta, theta_dot].

        returns:
            u: CDF values [B] in [0, 1], float64 dtype.

        error: if any a < -2 - 1e-12 or a > 2 + 1e-12, raise ValueError.

        procedure: same as log_prob up to rq_inverse, return u (not logdet).

        docstring: recovered u from rq_inverse is the empirical CDF value;
        used by PIT calibration gate to assess distributional fit.
        """
        # error check: a in [-2, 2]
        tol = 1e-12
        if torch.any(a < -2.0 - tol) or torch.any(a > 2.0 + tol):
            raise ValueError(f"action outside [-2,2]: min={a.min().item()}, max={a.max().item()}")

        # cast to float64
        a_f64 = a.to(torch.float64)  # [B]
        s_f64 = s.to(torch.float64)  # [B, 2]

        # featurize state (preserves dtype -> float64)
        bb_dtype = next(self.backbone.parameters()).dtype
        feats = featurize_state(s.to(bb_dtype))  # [B, 3]

        # pass feats through backbone (may auto-cast based on model dtype)
        # then cast output to float64 for spline computation
        raw = self.backbone(feats).to(torch.float64)  # [B, 3*n_bins+1]

        # unpack spline params
        uw = raw[:, : self.n_bins]  # [B, n_bins]
        uh = raw[:, self.n_bins : 2 * self.n_bins]  # [B, n_bins]
        ud = raw[:, 2 * self.n_bins :]  # [B, n_bins+1]
        widths, heights, derivs = normalize_params(uw, uh, ud)

        # inverse affine: a [-2, 2] -> x [0, 1]
        x = (a_f64 + 2.0) / 4.0  # [B]

        # clamp x to [0, 1] iff float noise (|x - clip(x)| <= 1e-12)
        x_clamped = torch.clamp(x, 0.0, 1.0)
        x_noise = torch.abs(x - x_clamped)
        x = torch.where(x_noise <= tol, x_clamped, x)

        # spline inverse -> u is the CDF
        u, _ = rq_inverse(x, widths, heights, derivs)  # [B]
        return u
