import torch
import torch.nn as nn
from torch import Tensor


class VelocityMLP(nn.Module):
    """
    Velocity field MLP for flow matching.

    Maps (latent state, time) pairs to velocity predictions.
    Architecture: MLP with fixed hidden dims [512, 512, 512, 512],
    LayerNorm + GELU activations, input dim = latent_dim + 1,
    output dim = latent_dim.
    """

    def __init__(self, latent_dim: int, hidden_dims: list[int] | None = None) -> None:
        """
        Initialize VelocityMLP.

        Args:
            latent_dim: dimension of latent space. input to forward is
                        [batch_size, latent_dim] and output is [batch_size, latent_dim].
            hidden_dims: list of hidden layer dimensions. if None, defaults to
                         [512, 512, 512, 512]. first layer inputs latent_dim + 1
                         (latent_dim + time).

        Returns:
            None. initializes self.mlp as nn.Sequential.
        """
        super().__init__()
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [512, 512, 512, 512]

        layers = []
        prev_dim = latent_dim + 1

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_t: Tensor, t: Tensor) -> Tensor:
        """
        Predict velocity field given latent state and time.

        Args:
            z_t: latent state tensor, shape [batch_size, latent_dim].
            t: time tensor. shape can be scalar, [batch_size], or [batch_size, 1].
               will be unsqueezed/expanded to [batch_size, 1] if needed.

        Returns:
            velocity predictions, shape [batch_size, latent_dim].
            represents predicted velocity d/dt z(t) at given (z_t, t).
        """
        # normalize t to [batch_size, 1]
        if t.dim() == 0:
            t = t.unsqueeze(0)  # scalar -> [1]
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [batch_size] -> [batch_size, 1]

        # expand t to match batch size if needed
        if t.shape[0] == 1 and z_t.shape[0] > 1:
            t = t.expand(z_t.shape[0], 1)  # [1, 1] -> [batch_size, 1]

        # concatenate z_t and t: [batch_size, latent_dim + 1]
        z_and_t = torch.cat([z_t, t], dim=1)

        # predict velocity: [batch_size, latent_dim + 1] -> [batch_size, latent_dim]
        v = self.mlp(z_and_t)

        return v
