"""VAE module initialization, re-exporting key components for public API."""

from src.models.vae.mnist_vae import MNISTVAE
from src.models.vae.train import train_vae

__all__ = ["MNISTVAE", "train_vae"]
