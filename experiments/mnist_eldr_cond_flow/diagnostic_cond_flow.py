"""diagnostic plots for the class-conditional pretrained models.

produces a single figure `cond_flow_diagnostic.png` with four panels:
  A. VAE reconstructions on random MNIST test images
  B. per-class cond-flow samples decoded via global VAE (10 x 10 grid)
  C. PCA of cond-flow latent samples colored by class y
  D. per-class NLL on the MNIST test set (classes of poor fit stand out)

usage: python -m experiments.mnist_eldr_cond_flow.diagnostic_cond_flow [--config PATH]
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from src.models.vae import MNISTVAE
from src.models.flow import (
    ClassCondVelocityMLP,
    sample_class_cond_flow,
    log_prob_class_cond,
)
from experiments.utils.mnist_imbalance import get_mnist_dataset


def parse_args():
    p = argparse.ArgumentParser(description="class-conditional flow diagnostics")
    p.add_argument("--config",
                   default="experiments/mnist_eldr_cond_flow/config.yaml",
                   help="path to config yaml")
    p.add_argument("--n-recon", type=int, default=10,
                   help="number of VAE reconstruction pairs to show")
    p.add_argument("--n-samples-per-class", type=int, default=10,
                   help="cond-flow decoded samples per class (panel B)")
    p.add_argument("--n-pca-per-class", type=int, default=200,
                   help="cond-flow latent samples per class for PCA (panel C)")
    p.add_argument("--n-nll-per-class", type=int, default=100,
                   help="test-set samples per class for NLL (panel D)")
    p.add_argument("--sample-steps", type=int, default=100,
                   help="Euler steps for cond-flow sampling")
    p.add_argument("--log-prob-steps", type=int, default=100,
                   help="Euler steps for log_prob (smaller than production for speed)")
    return p.parse_args()


def load_models(config, device):
    """load trained global VAE and class-conditional flow. eval mode, on device."""
    ckpt_dir = config["ckpt_dir"]
    vae = MNISTVAE(latent_dim=config["latent_dim"])
    vae.load_state_dict(torch.load(f"{ckpt_dir}/vae_global.pt", map_location="cpu"))
    vae.to(device).eval()

    flow = ClassCondVelocityMLP(
        latent_dim=config["latent_dim"],
        num_classes=10,
        hidden_dim=config["cond_flow_hidden_dim"],
    )
    flow.load_state_dict(torch.load(f"{ckpt_dir}/cond_flow.pt", map_location="cpu"))
    flow.to(device).eval()
    return vae, flow


def panel_vae_recon(ax, vae, device, n):
    """panel A: top row original, bottom row VAE reconstruction, n images."""
    dataset = get_mnist_dataset(root="./data", train=False, download=True)
    idx = np.random.RandomState(0).choice(len(dataset), size=n, replace=False)
    imgs = torch.stack([dataset[i][0] for i in idx], dim=0).to(device)  # [n, 1, 28, 28]
    with torch.no_grad():
        mu, _ = vae.encode(imgs)
        recon = vae.decode(mu)
    grid = np.zeros((2 * 28, n * 28), dtype=np.float32)
    for i in range(n):
        grid[:28, i*28:(i+1)*28] = imgs[i, 0].cpu().numpy()
        grid[28:, i*28:(i+1)*28] = recon[i, 0].cpu().numpy()
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"A. VAE recon (top=original, bottom=recon, n={n})")
    ax.set_xticks([]); ax.set_yticks([])


def panel_class_samples(ax, vae, flow, device, n_per_class, steps):
    """panel B: 10x10 grid of decoded cond-flow samples, row=class, col=sample."""
    D = flow.latent_dim
    grid = np.zeros((10 * 28, n_per_class * 28), dtype=np.float32)
    with torch.no_grad():
        for y in range(10):
            z = sample_class_cond_flow(flow, y, n_per_class, D,
                                       device=device, steps=steps)
            imgs = vae.decode(z)  # [n_per_class, 1, 28, 28]
            for i in range(n_per_class):
                grid[y*28:(y+1)*28, i*28:(i+1)*28] = imgs[i, 0].cpu().numpy()
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"B. cond-flow samples decoded (row=y, {n_per_class}/class)")
    ax.set_xticks([]); ax.set_yticks([])


def panel_pca(ax, flow, device, n_per_class, steps):
    """panel C: PCA of latent samples colored by class."""
    D = flow.latent_dim
    zs, ys = [], []
    with torch.no_grad():
        for y in range(10):
            z = sample_class_cond_flow(flow, y, n_per_class, D,
                                       device=device, steps=steps).cpu().numpy()
            zs.append(z)
            ys.append(np.full(n_per_class, y))
    zs = np.concatenate(zs, axis=0)
    ys = np.concatenate(ys, axis=0)
    pca = PCA(n_components=2).fit_transform(zs)
    cmap = plt.get_cmap("tab10")
    for y in range(10):
        mask = ys == y
        ax.scatter(pca[mask, 0], pca[mask, 1], s=6, color=cmap(y),
                   label=str(y), alpha=0.6)
    ax.set_title(f"C. PCA of p_phi(z|y) samples (n={n_per_class}/class)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=8, ncol=2, loc="best", markerscale=1.5)


def panel_nll(ax, vae, flow, device, n_per_class, log_prob_steps):
    """panel D: mean NLL per class on MNIST test set, encoded via global VAE."""
    dataset = get_mnist_dataset(root="./data", train=False, download=True)
    labels = (dataset.targets.numpy()
              if isinstance(dataset.targets, torch.Tensor) else dataset.targets)
    nll_per_class = np.full(10, np.nan, dtype=np.float32)
    for y in range(10):
        idxs = np.where(labels == y)[0][:n_per_class]
        if len(idxs) == 0:
            continue
        imgs = torch.stack([dataset[i][0] for i in idxs], dim=0).to(device)
        with torch.no_grad():
            mu, _ = vae.encode(imgs)  # [m, D]
        y_t = torch.full((len(idxs),), y, dtype=torch.long, device=device)
        lp = log_prob_class_cond(flow, mu, y_t, steps=log_prob_steps,
                                 device=device, chunk_size=200)
        nll_per_class[y] = -lp.mean().item()
    ax.bar(np.arange(10), nll_per_class, color="steelblue")
    ax.set_xticks(np.arange(10))
    ax.set_xlabel("class y")
    ax.set_ylabel("mean NLL (nats)")
    mean_nll = np.nanmean(nll_per_class)
    ax.axhline(mean_nll, color="black", linestyle="--", linewidth=0.8,
               label=f"mean={mean_nll:.2f}")
    ax.legend(fontsize=9, loc="best")
    ax.set_title(f"D. test-set NLL per class (n={n_per_class}/class)")


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.get("seed", 0))
    np.random.seed(config.get("seed", 0))

    vae, flow = load_models(config, device)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panel_vae_recon(axes[0, 0], vae, device, args.n_recon)
    panel_class_samples(axes[0, 1], vae, flow, device,
                        args.n_samples_per_class, args.sample_steps)
    panel_pca(axes[1, 0], flow, device,
              args.n_pca_per_class, args.sample_steps)
    panel_nll(axes[1, 1], vae, flow, device,
              args.n_nll_per_class, args.log_prob_steps)

    fig.tight_layout()
    fig_dir = Path(config["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "cond_flow_diagnostic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
