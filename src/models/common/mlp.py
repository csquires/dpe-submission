"""
MLP for time-conditioned vector field estimation.
"""
from typing import Optional, Literal

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron for time-conditioned vector field estimation.

    Time and spatial inputs are concatenated, passed through hidden layers,
    and projected to output. Supports optional layer normalization.

    Args:
        input_dim: spatial dimension
        hidden_dim: hidden layer width (default 256)
        output_dim: output dimension; defaults to input_dim if None
        n_hidden_layers: number of hidden layers in backbone (>= 1; default 3)
        activation: activation function in {"elu", "gelu", "silu"} (default "silu")
        layernorm: layer normalization placement in {"off", "pre", "post"} (default "off")
            "off": no normalization (byte-identical to pre-S7 behavior)
            "pre": LayerNorm applied BEFORE each hidden activation
            "post": LayerNorm applied AFTER each hidden activation

    Procedure:
        [t, x] concatenation -> [batch, input_dim+1]
        -> linear(input_dim+1 -> hidden_dim)
        -> [activation, (optional layernorm), linear(hidden_dim -> hidden_dim)] x (n_hidden_layers-1)
        -> activation, (optional layernorm)
        -> linear(hidden_dim -> output_dim)
        -> [batch, output_dim]
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None,
        n_hidden_layers: int = 3,
        activation: str = "silu",
        layernorm: str = "off"
    ) -> None:
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}")
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")

        self.n_hidden_layers = n_hidden_layers

        # map activation string to nn module
        act_map = {
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }

        # build backbone: input -> hidden, then (n_hidden_layers - 1) hidden -> hidden.
        # layernorm "off" preserves byte-identical pre-S7 behavior. "pre" inserts
        # a LayerNorm BEFORE each hidden activation; "post" inserts AFTER.
        layers = []

        # input projection: concatenated [t, x] has size input_dim + 1
        layers.append(nn.Linear(input_dim + 1, hidden_dim))
        if layernorm == "pre":
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_map[activation])
        if layernorm == "post":
            layers.append(nn.LayerNorm(hidden_dim))

        # hidden layers: n_hidden_layers - 1 times
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layernorm == "pre":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_map[activation])
            if layernorm == "post":
                layers.append(nn.LayerNorm(hidden_dim))

        # output projection
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [batch, 1]
            x: Spatial values [batch, input_dim]
        Returns:
            Vector field [batch, output_dim]
        """
        # concatenate time and spatial along last dimension
        tx = torch.cat([t, x], dim=-1)  # [batch, input_dim + 1]
        return self.net(tx)
