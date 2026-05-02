import torch
import torch.nn as nn


class TimeScoreNetwork2D(nn.Module):
    """
    Time score network for triangular TSM.

    Inputs: (x, t, t')
    Outputs: scalar time score s_theta(x, t, t')

    Builds backbone of n_hidden_layers (Linear + ELU) pairs, then output projection.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_hidden_layers: int = 3):
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")

        layers = []
        # input layer
        layers.append(nn.Linear(input_dim + 2, hidden_dim))
        layers.append(nn.ELU())
        # backbone pairs
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ELU())
        # output projection
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, t_prime: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t, t_prime], dim=-1)
        return self.net(xt)
