import torch
import torch.nn as nn
from torch import Tensor


class ClassCondVelocityMLP(nn.Module):
    """
    Velocity field MLP for K-way class-conditional flow matching.

    Maps (z, t, y) tuples to velocity predictions where z is latent state,
    t is time, and y is class label. Supports separate embed_label() and
    forward_from_embed() methods for use inside vmap(jacrev(...)) scopes
    without capturing problematic tensors.

    Architecture (mirrors VelocityMLP):
      class embedding [B] -> [B, embed_dim]
      input concatenation [z, t, c_emb] [B, latent_dim + 1 + embed_dim]
        -> backbone: 4 hidden layers, each Linear -> LayerNorm -> GELU
        -> v_head: Linear(hidden_dim, latent_dim)
      output: velocity [B, latent_dim]
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        embed_dim: int = 16,
        n_layers: int = 4,
    ) -> None:
        """
        Initialize ClassCondVelocityMLP.

        Args:
            latent_dim: dimensionality D of latent space.
            num_classes: number of classes K in range 0..K-1.
            hidden_dim: width of hidden layers (default 512).
            embed_dim: dimension of class embeddings (default 16).
            n_layers: number of hidden Linear -> LayerNorm -> GELU blocks (default 4).

        Behavior:
          1. Build self.embed = nn.Embedding(num_classes, embed_dim).
          2. Build self.backbone as n_layers blocks of Linear -> LayerNorm -> GELU,
             first block ingesting latent_dim + 1 + embed_dim, subsequent blocks hidden_dim -> hidden_dim.
          3. Build self.v_head = nn.Linear(hidden_dim, latent_dim).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.embed = nn.Embedding(num_classes, embed_dim)

        layers: list[nn.Module] = []
        prev = latent_dim + 1 + embed_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            prev = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.v_head = nn.Linear(hidden_dim, latent_dim)

    def embed_label(self, y: Tensor) -> Tensor:
        """
        Embed class labels.

        Args:
            y: class indices [B], dtype long, range 0..num_classes-1.

        Returns:
            c_emb: embedded class labels [B, embed_dim], dtype float.

        Behavior: return self.embed(y).

        Purpose: decouple embedding step from forward pass to enable
        pre-embedding outside vmap(jacrev(...)) scopes.
        """
        return self.embed(y)

    def forward_from_embed(self, z: Tensor, t: Tensor, c_emb: Tensor) -> Tensor:
        """
        Predict velocity from latent state, time, and pre-embedded class label.

        Args:
            z: latent state [B, latent_dim], dtype float.
            t: time [B], dtype float, range [0, 1].
            c_emb: pre-embedded class labels [B, embed_dim], dtype float.

        Returns:
            v: velocity predictions [B, latent_dim], dtype float.

        Procedure:
          1. Unsqueeze t to [B, 1].
          2. Concatenate [z, t, c_emb] along dim -1 -> [B, latent_dim+1+embed_dim].
          3. Pass through backbone -> [B, hidden_dim].
          4. Apply v_head -> [B, latent_dim].
          5. Return v.

        Purpose: entry point for differentiation inside vmap(jacrev(...)).
        Takes pre-embedded float tensor c_emb to avoid long tensor issues
        in autodiff graphs.
        """
        t_unsq = t.unsqueeze(1)  # [B] -> [B, 1]
        h = torch.cat([z, t_unsq, c_emb], dim=-1)  # [B, latent_dim + 1 + embed_dim]
        h = self.backbone(h)  # [B, hidden_dim]
        v = self.v_head(h)  # [B, latent_dim]
        return v

    def forward(self, z: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """
        Predict velocity from latent state, time, and class label.

        Args:
            z: latent state [B, latent_dim], dtype float.
            t: time [B], dtype float, range [0, 1].
            y: class indices [B], dtype long, range 0..num_classes-1.

        Returns:
            v: velocity predictions [B, latent_dim], dtype float.

        Procedure:
          1. Compute c_emb = self.embed_label(y).
          2. Return self.forward_from_embed(z, t, c_emb).

        Purpose: user-facing API for training and sampling.
        """
        c_emb = self.embed_label(y)
        return self.forward_from_embed(z, t, c_emb)
