import torch
import torch.nn as nn
from torch import Tensor


class CondVelScoreMLP(nn.Module):
    """
    Shared backbone with dual output heads for velocity and score.

    Architecture:
      input [t, x, c] (dim D+2)
        -> backbone: n_hidden_layers hidden layers with GELU
        -> v_head: Linear(hidden_dim, D) [velocity]
        -> s_head: Linear(hidden_dim, D) [score]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_hidden_layers: int = 3) -> None:
        """
        Initialize CondVelScoreMLP.

        Args:
            input_dim: dimensionality D of data samples.
            hidden_dim: width of hidden layers (default 256).
            n_hidden_layers: number of hidden layers (default 3, must be >= 1).

        Behavior:
          1. Validate n_hidden_layers >= 1.
          2. Store input_dim, hidden_dim, and n_hidden_layers as instance attributes.
          3. Build shared backbone as nn.Sequential with n_hidden_layers layers.
             first layer: Linear(D+2, hidden_dim) + GELU.
             remaining layers: (n_hidden_layers-1) times [Linear(hidden_dim, hidden_dim) + GELU].
          4. Build velocity head: Linear(hidden_dim, D).
          5. Build score head: Linear(hidden_dim, D).
        """
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        layers = [nn.Linear(input_dim + 2, hidden_dim), nn.GELU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.backbone = nn.Sequential(*layers)

        self.v_head = nn.Linear(hidden_dim, input_dim)
        self.s_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, t: Tensor, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute velocity and score from time, data, and condition.

        Given time t in [0,1], data x in R^D, and condition c in {0,1},
        compute velocity v(t,x,c) and score s(t,x,c).

        Procedure:
          1. Concatenate [t, x, c] along dim -1 -> shape [B, D+2].
          2. Pass through backbone -> shape [B, hidden_dim].
          3. Apply velocity head -> shape [B, D].
          4. Apply score head -> shape [B, D].
          5. Return tuple (velocity, score).

        Args:
          t: time [B, 1]
          x: data [B, D]
          c: condition [B, 1]

        Returns:
          (velocity [B, D], score [B, D])
        """
        h = torch.cat([t, x, c], dim=-1)  # [B, D+2]
        h = self.backbone(h)  # [B, hidden_dim]
        velocity = self.v_head(h)  # [B, D]
        score = self.s_head(h)  # [B, D]
        return velocity, score
