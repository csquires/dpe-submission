import torch
import torch.nn as nn
from torch import Tensor


class CondVelScoreMLP(nn.Module):
    """
    Shared backbone with dual output heads for velocity and score.

    Architecture (n_shared_layers of n_hidden_layers in the shared trunk):
      input [t, x, c] (dim D+2)
        -> backbone: n_shared_layers x [Linear, GELU]
        -> v_head / s_head: each
              (n_hidden_layers - n_shared_layers) x [Linear, GELU]
              + Linear(hidden_dim, D)

    when n_shared_layers == n_hidden_layers the head reduces to a single
    Linear projection (byte-identical to the pre-split CondVelScoreMLP, same
    `self.v_head` / `self.s_head` module names and parameter init order).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_shared_layers: int = 3,
    ) -> None:
        """
        Initialize CondVelScoreMLP.

        Args:
            input_dim: dimensionality D of data samples.
            hidden_dim: width of hidden layers (default 256).
            n_hidden_layers: total hidden Linear+GELU rounds across backbone +
                each head (default 3). final Linear output projection is not
                counted.
            n_shared_layers: hidden rounds in the shared backbone; the remaining
                n_hidden_layers - n_shared_layers rounds live in each head
                before the output projection. must satisfy
                1 <= n_shared_layers <= n_hidden_layers (default 3 = fully
                shared, matches the pre-split architecture).
        """
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")
        if not (1 <= n_shared_layers <= n_hidden_layers):
            raise ValueError(
                f"n_shared_layers must satisfy 1 <= n_shared_layers <= n_hidden_layers "
                f"({n_hidden_layers}); got {n_shared_layers}"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_shared_layers = n_shared_layers

        # shared backbone: n_shared_layers Linear+GELU rounds; first layer
        # absorbs the [t, x, c] concatenation.
        layers = [nn.Linear(input_dim + 2, hidden_dim), nn.GELU()]
        for _ in range(n_shared_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.backbone = nn.Sequential(*layers)

        # heads: bare Linear when nothing else is left to do, else Sequential
        # of (head-specific Linear+GELU) ending in the output projection. the
        # bare-Linear branch keeps byte-identity for the fully-shared default.
        n_head_hidden = n_hidden_layers - n_shared_layers
        if n_head_hidden == 0:
            self.v_head = nn.Linear(hidden_dim, input_dim)
            self.s_head = nn.Linear(hidden_dim, input_dim)
        else:
            self.v_head = self._build_head(hidden_dim, input_dim, n_head_hidden)
            self.s_head = self._build_head(hidden_dim, input_dim, n_head_hidden)

    @staticmethod
    def _build_head(hidden_dim: int, output_dim: int, n_head_hidden: int) -> nn.Sequential:
        """build a per-head Sequential of n_head_hidden Linear+GELU rounds plus
        a final Linear(hidden_dim, output_dim) projection."""
        layers: list[nn.Module] = []
        for _ in range(n_head_hidden):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

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
