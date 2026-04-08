import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTVAE(nn.Module):
    """
    Convolutional VAE for MNIST.
    encode() -> reparameterize() -> decode() with combined KL+reconstruction loss.
    """

    def __init__(self, latent_dim: int = 14, beta: float = 1.0) -> None:
        """
        latent_dim: dimensionality of latent space
        beta: weighting for KL divergence in loss
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # encoder: 28x28 -> 14x14 -> 7x7 -> flatten -> fc
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(3136, 1024),
            nn.LeakyReLU(0.2),
        )

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # decoder: latent -> fc -> reshape -> deconv
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 3136),
            nn.LeakyReLU(0.2),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, 1, 28, 28]
        returns: (mu [B, latent_dim], logvar [B, latent_dim])
        """
        h = self.encoder(x)  # [B, 1024]
        mu = self.fc_mu(h)  # [B, latent_dim]
        logvar = self.fc_logvar(h)  # [B, latent_dim]
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        mu: [B, latent_dim]
        logvar: [B, latent_dim]
        returns: z [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)  # [B, latent_dim]
        eps = torch.randn_like(std)  # [B, latent_dim]
        z = mu + std * eps  # [B, latent_dim]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, latent_dim]
        returns: x_recon [B, 1, 28, 28]
        """
        h = self.decoder(z)  # [B, 3136]
        h = h.view(-1, 64, 7, 7)  # [B, 64, 7, 7]
        x_recon = self.decoder_conv(h)  # [B, 1, 28, 28]
        return x_recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, 1, 28, 28]
        returns: (x_recon [B, 1, 28, 28], mu [B, latent_dim], logvar [B, latent_dim])
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, 1, 28, 28] original
        x_recon: [B, 1, 28, 28] reconstructed
        mu: [B, latent_dim]
        logvar: [B, latent_dim]
        returns: scalar loss
        """
        eps = 1e-8

        # reconstruction loss: manual BCE
        bce = -torch.sum(
            x * torch.log(x_recon + eps) + (1 - x) * torch.log(1 - x_recon + eps),
            dim=(1, 2, 3),
        )  # [B]
        bce_loss = bce.mean()

        # kl divergence
        kl = -0.5 * torch.sum(
            1 + logvar - mu ** 2 - torch.exp(logvar),
            dim=-1,
        )  # [B]
        kl_loss = kl.mean()

        return bce_loss + self.beta * kl_loss
