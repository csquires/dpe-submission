import torch
import torch.nn as nn

class TimeScoreNetwork1D(nn.Module):
    """
    MLP-based time score network from the DRE-infinity toy implementation.

    Build: [Linear(input_dim+1, hidden_dim), ELU] +
           (n_hidden_layers-1) * [Linear(hidden_dim, hidden_dim), ELU] +
           [Linear(hidden_dim, 1)].

    n_hidden_layers=3 (default) matches original 3 hidden ELU + 4 Linear modules.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_hidden_layers: int = 3):
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")

        layers = []
        # input layer: input_dim + 1 (concatenated time) -> hidden_dim
        layers.append(nn.Linear(input_dim + 1, hidden_dim))
        layers.append(nn.ELU())

        # hidden layers: hidden_dim -> hidden_dim (repeated n_hidden_layers - 1 times)
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ELU())

        # output layer: hidden_dim -> 1
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)
