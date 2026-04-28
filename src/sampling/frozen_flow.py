import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FrozenFlow(nn.Module):
    """Fixed multi-layer normalizing flow with closed-form log-determinant Jacobian.

    Composition of per-dimension affine transformations (softplus) and Householder
    reflections, initialized randomly and then frozen (no learnable parameters).

    Architecture: forward(z) = R_n @ T_n @ ... @ R_1 @ T_1 (z)
    where T_i(y) = a_i * softplus(y) + b_i (element-wise)
          R_i(x) = x - 2 * (x @ v_i) / (v_i @ v_i) * v_i (Householder reflection).

    Both forward and inverse return batched log-determinant Jacobians computed
    in closed form.
    """

    def __init__(self, dim: int, n_layers: int = 4, seed: int = 1729):
        """Initialize a frozen multi-layer normalizing flow.

        Args:
            dim: input/output dimension.
            n_layers: number of (T_i, R_i) layer pairs. default 4. if 0, identity.
            seed: integer seed for torch.Generator for reproducibility.

        Registers three buffers (if n_layers > 0):
            scales: [n_layers, dim] per-dimension scale factors a_i.
            biases: [n_layers, dim] per-dimension bias terms b_i.
            householders: [n_layers, dim] Householder reflection vectors v_i.
        """
        super().__init__()

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        self.dim = dim
        self.n_layers = n_layers

        if n_layers == 0:
            return

        # initialize per-layer parameters as lists, then stack.
        scales = []
        biases = []
        householders = []

        for _ in range(n_layers):
            # scale: a_i ~ exp(Normal(0, 0.1)) to stay near 1 and positive.
            log_a = torch.randn(dim, generator=gen, dtype=torch.float32) * 0.1
            a = torch.exp(log_a)
            a = torch.clamp(a, min=1e-3)
            scales.append(a)

            # bias: b_i ~ Normal(0, 0.5), unconstrained.
            b = torch.randn(dim, generator=gen, dtype=torch.float32) * 0.5
            biases.append(b)

            # Householder vector: v_i ~ Normal(0, 1), non-zero.
            v = torch.randn(dim, generator=gen, dtype=torch.float32)
            while (v == 0).all():
                v = torch.randn(dim, generator=gen, dtype=torch.float32)
            householders.append(v)

        # stack into three buffers for clean indexed access.
        # shape: [n_layers, dim] each.
        self.register_buffer("scales", torch.stack(scales, dim=0))
        self.register_buffer("biases", torch.stack(biases, dim=0))
        self.register_buffer("householders", torch.stack(householders, dim=0))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the flow.

        Args:
            z: input tensor [N, dim], float32 or float64.

        Returns:
            x: output tensor [N, dim], same dtype as z.
            log_det_J: log|det J| of forward map [N], same dtype as z.

        Semantics:
            x = forward_flow(z)
            log_det_J[i] = log|det dz/dx| evaluated at z[i], i.e.,
            the log absolute determinant of the Jacobian of the entire composition.
        """
        N, d = z.shape
        assert d == self.dim, f"expected dim {self.dim}, got {d}"

        if self.n_layers == 0:
            log_det_J = torch.zeros(N, dtype=z.dtype, device=z.device)
            return z, log_det_J

        curr = z.clone()  # [N, dim]
        total_log_det = 0.0  # accumulate scalar

        for i in range(self.n_layers):
            # layer T_i: y = a_i * softplus(curr) + b_i
            a_i = self.scales[i].to(dtype=z.dtype)  # [dim]
            b_i = self.biases[i].to(dtype=z.dtype)  # [dim]

            sp = F.softplus(curr)  # [N, dim]
            y = a_i * sp + b_i  # [N, dim]

            # log|det J_T_i| = sum_k log(a_i[k]) + sum_k log(sigmoid(curr[k]))
            # since d/dz [a*softplus(z)] = a*sigmoid(z)
            log_det_T = torch.sum(torch.log(a_i)) + torch.sum(torch.log(torch.sigmoid(curr)), dim=1)  # [N]
            total_log_det = total_log_det + log_det_T

            # layer R_i: Householder reflection w = y - 2 * (y @ v) / (v @ v) * v
            v_i = self.householders[i].to(dtype=z.dtype)  # [dim]

            v_dot_v = torch.sum(v_i ** 2)  # scalar
            scale = 2.0 / v_dot_v  # scalar
            y_dot_v = torch.matmul(y, v_i)  # [N]
            w = y - scale * y_dot_v[:, None] * v_i  # [N, dim]
            # Householder is orthogonal; log|det R_i| = 0 (contributes nothing)

            curr = w

        log_det_J = total_log_det  # [N]
        return curr, log_det_J

    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass through the flow.

        Args:
            x: input tensor [N, dim], float32 or float64.

        Returns:
            z: output tensor [N, dim], same dtype as x.
            log_det_inv_J: log|det dz/dx| [N], same dtype as x.

        Semantics:
            z = inverse_flow(x)
            log_det_inv_J = log|det J^{-1}| = -log_det_J(forward(z)),
            where z is reconstructed by this inverse operation.

        Raises:
            ValueError if softplus_inv produces NaN, indicating x is outside
            the image of forward() or a numerical error occurred.
        """
        N, d = x.shape
        assert d == self.dim, f"expected dim {self.dim}, got {d}"

        if self.n_layers == 0:
            log_det_inv_J = torch.zeros(N, dtype=x.dtype, device=x.device)
            return x, log_det_inv_J

        curr = x.clone()  # [N, dim]
        total_log_det = 0.0  # accumulate scalar

        for i in range(self.n_layers - 1, -1, -1):
            # inverse of R_i: Householder is self-inverse
            v_i = self.householders[i].to(dtype=x.dtype)  # [dim]

            v_dot_v = torch.sum(v_i ** 2)  # scalar
            scale = 2.0 / v_dot_v  # scalar
            curr_dot_v = torch.matmul(curr, v_i)  # [N]
            y = curr - scale * curr_dot_v[:, None] * v_i  # [N, dim]

            # inverse of T_i: y = a_i * softplus(z) + b_i  =>  z = softplus_inv((y - b_i) / a_i)
            a_i = self.scales[i].to(dtype=x.dtype)  # [dim]
            b_i = self.biases[i].to(dtype=x.dtype)  # [dim]

            u = (y - b_i) / a_i  # [N, dim]

            # softplus_inv(u) with numerical stability for u > 0
            z = torch.where(
                u < 20,
                torch.log(torch.expm1(u)),
                u + torch.log1p(-torch.exp(-u))
            )  # [N, dim]

            # check for NaN (indicates u <= 0 or other issue)
            if torch.isnan(z).any():
                raise ValueError(
                    f"inverse: softplus_inv produced NaN; "
                    f"input x may be outside the image of forward(). "
                    f"u range: [{u.min():.6e}, {u.max():.6e}]"
                )

            # log|det J_T_i^{-1}| = -log|det J_T_i|
            # = -(sum_k log(a_i[k]) + sum_k log(sigmoid(z[k])))
            log_det_T_inv = -(torch.sum(torch.log(a_i)) + torch.sum(torch.log(torch.sigmoid(z)), dim=1))  # [N]
            total_log_det = total_log_det + log_det_T_inv

            curr = z

        log_det_inv_J = total_log_det  # [N]
        return curr, log_det_inv_J
