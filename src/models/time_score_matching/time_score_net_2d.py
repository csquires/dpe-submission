import torch
import torch.nn as nn


class TimeScoreNetwork2D(nn.Module):
    """
    Time score network for triangular TSM.

    Inputs: (x, t, t')
    Outputs: scalar time score s_theta(x, t, t')

    Build: [Linear(input_dim+2, hidden_dim), activation] +
           (n_hidden_layers-1) * [Linear(hidden_dim, hidden_dim), activation] +
           [Linear(hidden_dim, 1)].

    activation: one of {"elu", "gelu", "silu"}; default "silu".
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_hidden_layers: int = 3, activation: str = "silu"):
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}")

        act_map = {
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        Act = act_map[activation]

        layers = [nn.Linear(input_dim + 2, hidden_dim), Act()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Act())
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, t_prime: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t, t_prime], dim=-1)
        return self.net(xt)
