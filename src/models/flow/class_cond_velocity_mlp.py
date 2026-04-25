import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ClassCondVelocityMLP(nn.Module):
    """velocity field mlp for k-way class-conditional flow matching.

    maps (z, t, y) tuples to velocity predictions where z is latent state,
    t is time, and y is class label. uses one-hot label encoding so the
    first linear layer's class-slice acts as the implicit embedding.

    architecture (mirrors VelocityMLP):
      one-hot labels [B] -> [B, num_classes]
      input concatenation [z, t, y_onehot] [B, latent_dim + 1 + num_classes]
        -> backbone: n_layers blocks of Linear -> LayerNorm -> GELU
        -> v_head: Linear(hidden_dim, latent_dim)
      output: velocity [B, latent_dim]
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        n_layers: int = 4,
    ) -> None:
        """initialize ClassCondVelocityMLP.

        args:
            latent_dim: dimensionality D of latent space.
            num_classes: number of classes K in range 0..K-1.
            hidden_dim: width of hidden layers (default 512).
            n_layers: number of hidden Linear -> LayerNorm -> GELU blocks (default 4).

        behavior:
          1. build self.backbone as n_layers blocks of Linear -> LayerNorm -> GELU,
             first block ingesting latent_dim + 1 + num_classes,
             subsequent blocks hidden_dim -> hidden_dim.
          2. build self.v_head = nn.Linear(hidden_dim, latent_dim).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        layers: list[nn.Module] = []
        prev = latent_dim + 1 + num_classes
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            prev = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.v_head = nn.Linear(hidden_dim, latent_dim)

    def forward_from_onehot(self, z: Tensor, t: Tensor, y_onehot: Tensor) -> Tensor:
        """predict velocity from pre-computed one-hot labels.

        args:
            z: latent state [B, latent_dim].
            t: time [B].
            y_onehot: float one-hot labels [B, num_classes].

        returns:
            v: velocity predictions [B, latent_dim].

        purpose: vmap-safe entry point. F.one_hot uses scatter_ which does not
        compose with vmap, so callers (e.g. log_prob_class_cond) pre-compute
        the one-hot tensor outside the vmap scope and use this method.
        """
        h = torch.cat([z, t.unsqueeze(-1), y_onehot], dim=-1)
        h = self.backbone(h)
        return self.v_head(h)

    def forward(self, z: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """predict velocity from latent state, time, and class label.

        args:
            z: latent state [B, latent_dim].
            t: time [B].
            y: class indices [B], dtype long, range 0..num_classes-1.

        returns:
            v: velocity predictions [B, latent_dim].

        procedure: one-hot encode y, dispatch to forward_from_onehot.
        """
        y_onehot = F.one_hot(y, num_classes=self.num_classes).to(z.dtype)
        return self.forward_from_onehot(z, t, y_onehot)
