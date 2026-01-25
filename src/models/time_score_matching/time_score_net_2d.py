import torch
import torch.nn as nn


class TimeScoreNetwork2D(nn.Module):
    """
    Time score network for triangular TSM.

    Inputs: (x, t, t')
    Outputs: scalar time score s_theta(x, t, t')
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, t_prime: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t, t_prime], dim=-1)
        return self.net(xt)
