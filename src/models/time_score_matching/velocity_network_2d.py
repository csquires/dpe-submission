"""
2D-time MLP network for V3-VFM velocity and denoiser heads.

Raw concatenation of (t_1, t_2, x) with no Fourier embedding, matching
V3-CTSM ScoreNetwork2D precedent. Outputs a d-vector [B, output_dim].
Used by TriangularVFM2D to instantiate three independent heads
(b_1, b_2, eta) via separate instantiation.
"""
import torch
import torch.nn as nn
from torch import Tensor


class MLP2D(nn.Module):
    """
    Multi-layer perceptron for 2D-time vector field estimation.

    Mirrors the 1D MLP at src/density_ratio_estimation/spatial_velo_denoiser2.py
    line 18 (3 hidden Linear+GELU blocks, hardcoded). Differs only in the input
    projection: `+ 2` for two time scalars instead of `+ 1`.

    Procedure:
        concatenate [t1, t2, x] along last dimension -> [B, D + 2]
        pass through nn.Sequential of (Linear+GELU) x 3 + Linear
        return [B, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int | None = None,
        n_hidden_layers: int = 3,
        activation: str = "gelu",
    ) -> None:
        """
        Args:
            input_dim: spatial dimension D.
            hidden_dim: hidden layer width (default 256).
            output_dim: output dimension (defaults to input_dim).
            n_hidden_layers: number of hidden linear+activation blocks (default 3).
            activation: activation function {"elu", "gelu", "silu"}; default "gelu" for byte-identical behavior.
        """
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}")

        # map activation string to nn module
        act_map = {
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }

        layers = [nn.Linear(input_dim + 2, hidden_dim), act_map[activation]]  # +2 for t1, t2
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_map[activation])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t1: Tensor, t2: Tensor, x: Tensor) -> Tensor:
        """
        Compute output at (t_1, t_2, x).

        Args:
            t1: [B, 1] first time coordinate.
            t2: [B, 1] second time coordinate.
            x: [B, D] spatial coordinates.

        Returns:
            [B, output_dim] vector field output.
        """
        tx = torch.cat([t1, t2, x], dim=-1)  # [B, D+2]
        return self.net(tx)  # [B, output_dim]
