import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from src.models.vae.mnist_vae import MNISTVAE


def train_vae(
    vae: nn.Module,
    loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    ckpt_path: str | None = None,
) -> nn.Module:
    """train or load a vae.

    if ckpt_path exists, load from checkpoint and return in eval mode.
    otherwise, train vae on loader for epochs using AdamW with weight decay.
    save to checkpoint if ckpt_path provided. set eval mode before return.

    Args:
        vae: nn.Module VAE instance (mutated in place)
        loader: DataLoader with (x, _) tuples
        epochs: number of training epochs
        lr: learning rate for AdamW
        device: device string ('cuda' or 'cpu')
        ckpt_path: optional path to checkpoint file

    Returns:
        trained vae in eval mode
    """
    # load from checkpoint if available
    if ckpt_path is not None and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location='cpu')
        vae.load_state_dict(state)
        vae.to(device)
        vae.eval()
        return vae

    # train the vae
    vae.to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        for batch in loader:
            x, _ = batch
            x = x.to(device)  # [batch_size, ...]

            x_recon, mu, logvar = vae(x)
            loss = vae.loss(x, x_recon, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

    # finalize and save
    vae.eval()
    if ckpt_path is not None:
        torch.save(vae.state_dict(), ckpt_path)

    return vae
