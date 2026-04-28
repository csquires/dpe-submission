import torch
import torch.nn as nn


class ScoreNetwork2D(nn.Module):
    """Score network for V3 (2D-time stacked-interpolant) triangular CTSM.

    Produces a 2-vector score (s_1, s_2) where component i estimates
    \\partial_{t_i} \\log \\rho(x | t_1, t_2). Used by TriangularCTSM2D in
    src/density_ratio_estimation/triangular_ctsm_2d.py.

    Distinct from time_score_net_2d.py in this directory: that file has SCALAR
    output and is used by TriangularTSM. This file has VECTOR output [B, 2] and
    is used by TriangularCTSM2D. The "2d" in both filenames refers to two time
    inputs; the output dimensionality differs.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """Initialize score network.

        Args:
            input_dim: spatial dimension D.
            hidden_dim: hidden layer width (default 256).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Compute 2-vector score at (x, t_1, t_2).

        Args:
            x: [B, D] spatial coordinates.
            t1: [B, 1] first time coordinate.
            t2: [B, 1] second time coordinate.

        Returns:
            score: [B, 2] with score[:, 0] approximating \\partial_{t_1} \\log \\rho
                   and score[:, 1] approximating \\partial_{t_2} \\log \\rho.
        """
        xt = torch.cat([x, t1, t2], dim=-1)
        return self.net(xt)
