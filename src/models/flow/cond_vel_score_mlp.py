import torch
import torch.nn as nn
from torch import Tensor


class CondVelScoreMLP(nn.Module):
    """
    Shared backbone with dual output heads for velocity and score.

    Architecture:
      input [t, x, c] (dim D+2)
        -> backbone: 3 hidden layers with ReLU
        -> v_head: Linear(hidden_dim, D) [velocity]
        -> s_head: Linear(hidden_dim, D) [score]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        """
        Initialize CondVelScoreMLP.

        Args:
            input_dim: dimensionality D of data samples.
            hidden_dim: width of hidden layers (default 256).

        Behavior:
          1. Store input_dim and hidden_dim as instance attributes.
          2. Build shared backbone as nn.Sequential with 3 hidden layers.
             first layer: Linear(D+2, hidden_dim) + ReLU.
             second layer: Linear(hidden_dim, hidden_dim) + ReLU.
             third layer: Linear(hidden_dim, hidden_dim) + ReLU.
          3. Build velocity head: Linear(hidden_dim, D).
          4. Build score head: Linear(hidden_dim, D).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.backbone = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

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
