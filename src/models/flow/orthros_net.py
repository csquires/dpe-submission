"""Time-conditioned two-head regression network for VFMOrthros."""
from typing import Optional

import torch
import torch.nn as nn


class OrthrosNet(nn.Module):
    """time-conditioned two-head regression net: cat([x, t]) -> shared backbone -> (head0, head1).

    procedure:
        [x, t] -> linear(input_dim+1 -> hidden_dim)
              -> [(optional layernorm) activation linear(hidden_dim -> hidden_dim)] x (n_shared_layers-1)
              -> [backbone output]
              -> head0: [(optional layernorm) activation linear(hidden_dim -> hidden_dim)] x (n_hidden_layers - n_shared_layers)
                       -> linear(hidden_dim -> output_dim)
              -> head1: [(optional layernorm) activation linear(hidden_dim -> hidden_dim)] x (n_hidden_layers - n_shared_layers)
                       -> linear(hidden_dim -> output_dim)

    Args:
        input_dim: spatial dimension.
        hidden_dim: hidden width.
        output_dim: output dimension for each head; defaults to input_dim.
        n_hidden_layers: total number of hidden layers in each head (>= 1).
        n_shared_layers: number of shared layers in the backbone (1 <= n_shared_layers <= n_hidden_layers).
        activation: one of {"elu", "gelu", "silu"}.
        layernorm: one of {"off", "pre", "post"}; "pre"/"post" insert LayerNorm
            before/after each hidden activation.

    Returns:
        tuple of (head0_out, head1_out), each [B, output_dim]; semantics are
        caller-defined (VFMOrthros: head 0 = velocity b, head 1 = denoiser eta).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None,
        n_hidden_layers: int = 3,
        n_shared_layers: int = 2,
        activation: str = "silu",
        layernorm: str = "off",
    ) -> None:
        super().__init__()

        # validation
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")
        if n_shared_layers < 1:
            raise ValueError("n_shared_layers must be >= 1")
        if n_shared_layers > n_hidden_layers:
            raise ValueError("n_shared_layers must be <= n_hidden_layers")
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(f"activation must be in {{'elu','gelu','silu'}}; got {activation!r}")
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off','pre','post'}}; got {layernorm!r}")

        # store configuration
        self.n_hidden_layers = n_hidden_layers
        self.n_shared_layers = n_shared_layers
        if output_dim is None:
            output_dim = input_dim
        self.output_dim = output_dim

        # activation factory
        act = {"elu": nn.ELU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[activation]

        # build shared backbone
        backbone_layers: list[nn.Module] = [nn.Linear(input_dim + 1, hidden_dim)]
        if layernorm == "pre":
            backbone_layers.append(nn.LayerNorm(hidden_dim))
        backbone_layers.append(act)
        if layernorm == "post":
            backbone_layers.append(nn.LayerNorm(hidden_dim))

        for _ in range(n_shared_layers - 1):
            backbone_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layernorm == "pre":
                backbone_layers.append(nn.LayerNorm(hidden_dim))
            backbone_layers.append(act)
            if layernorm == "post":
                backbone_layers.append(nn.LayerNorm(hidden_dim))

        self.backbone = nn.Sequential(*backbone_layers)

        # build head 0
        head0_layers: list[nn.Module] = []
        for _ in range(n_hidden_layers - n_shared_layers):
            head0_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layernorm == "pre":
                head0_layers.append(nn.LayerNorm(hidden_dim))
            head0_layers.append({"elu": nn.ELU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[activation])
            if layernorm == "post":
                head0_layers.append(nn.LayerNorm(hidden_dim))
        head0_layers.append(nn.Linear(hidden_dim, output_dim))
        self.head0 = nn.Sequential(*head0_layers)

        # build head 1
        head1_layers: list[nn.Module] = []
        for _ in range(n_hidden_layers - n_shared_layers):
            head1_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layernorm == "pre":
                head1_layers.append(nn.LayerNorm(hidden_dim))
            head1_layers.append({"elu": nn.ELU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[activation])
            if layernorm == "post":
                head1_layers.append(nn.LayerNorm(hidden_dim))
        head1_layers.append(nn.Linear(hidden_dim, output_dim))
        self.head1 = nn.Sequential(*head1_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, input_dim], t: [B, 1] -> ([B, output_dim], [B, output_dim]). space-first."""
        shared_input = torch.cat([x, t], dim=-1)  # [B, input_dim+1]
        features = self.backbone(shared_input)  # [B, hidden_dim]
        # generic two-head outputs; the caller assigns semantics (VFMOrthros
        # reads head 0 as the x0 endpoint posterior, head 1 as the denoiser).
        head0_out = self.head0(features)  # [B, output_dim]
        head1_out = self.head1(features)  # [B, output_dim]
        return (head0_out, head1_out)
