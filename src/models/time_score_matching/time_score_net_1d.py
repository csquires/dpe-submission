import torch
import torch.nn as nn

class TimeScoreNetwork1D(nn.Module):
    """
    MLP-based time score network from the DRE-infinity toy implementation.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)
