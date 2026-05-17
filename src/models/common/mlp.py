"""MLP for time-conditioned vector field estimation."""
from typing import Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """time-conditioned MLP: cat([x, t]) -> hidden stack -> output.

    procedure:
        [x, t] -> linear(input_dim+1 -> hidden_dim)
              -> [(optional layernorm) activation linear(hidden_dim -> hidden_dim)] x (n_hidden_layers-1)
              -> linear(hidden_dim -> output_dim)

    Args:
        input_dim: spatial dimension.
        hidden_dim: hidden width.
        output_dim: defaults to input_dim.
        n_hidden_layers: >= 1.
        activation: one of {"elu", "gelu", "silu"}.
        layernorm: one of {"off", "pre", "post"}; "pre"/"post" insert LayerNorm
            before/after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None,
        n_hidden_layers: int = 3,
        activation: str = "silu",
        layernorm: str = "off",
    ) -> None:
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(f"activation must be in {{'elu','gelu','silu'}}; got {activation!r}")
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off','pre','post'}}; got {layernorm!r}")

        self.n_hidden_layers = n_hidden_layers
        act = {"elu": nn.ELU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[activation]

        layers: list[nn.Module] = [nn.Linear(input_dim + 1, hidden_dim)]
        if layernorm == "pre":
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act)
        if layernorm == "post":
            layers.append(nn.LayerNorm(hidden_dim))

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layernorm == "pre":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act)
            if layernorm == "post":
                layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: [B, input_dim], t: [B, 1] -> [B, output_dim]. space-first."""
        return self.net(torch.cat([x, t], dim=-1))
