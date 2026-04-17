"""quick diagnostic for pretrained VAEs and flows.

for each alpha, picks a couple of pairs and produces:
  - VAE reconstruction of real MNIST images (per-side)
  - VAE prior samples (z ~ N(0,I) -> decode)
  - flow samples (noise -> ODE -> VAE decode)
saves a single grid figure to figures/pretrain_diagnostic.png.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.models.vae import MNISTVAE
from src.models.flow import VelocityMLP, sample_flow
from experiments.utils.mnist_imbalance import (
    sample_dirichlet_weights,
    subsample_mnist,
    get_mnist_dataset,
)


def load_vae(ckpt_path: str, latent_dim: int, device: str) -> MNISTVAE:
    vae = MNISTVAE(latent_dim=latent_dim)
    vae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    vae.to(device).eval()
    return vae


def load_flow(ckpt_path: str, latent_dim: int, device: str) -> VelocityMLP:
    flow = VelocityMLP(latent_dim=latent_dim)
    flow.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    flow.to(device).eval()
    return flow


def get_real_images(dataset, weights, n: int, seed: int):
    """get n real images from the imbalanced subset."""
    indices = subsample_mnist(dataset, weights, min_per_class=10)
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(indices), size=n, replace=False)
    imgs = torch.stack([dataset[indices[i]][0] for i in chosen])  # [n, 1, 28, 28]
    return imgs


def vae_reconstruct(vae, imgs, device):
    """encode + decode through vae."""
    with torch.no_grad():
        x = imgs.to(device)
        mu, logvar = vae.encode(x)
        recon = vae.decode(mu)  # deterministic reconstruction
    return recon.cpu()


def vae_sample(vae, n, latent_dim, device):
    """sample from N(0,I) prior and decode."""
    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        imgs = vae.decode(z)
    return imgs.cpu()


def flow_sample_and_decode(flow, vae, n, latent_dim, device):
    """sample latents from flow, decode through vae."""
    with torch.no_grad():
        z = sample_flow(flow, n, latent_dim, device=device, steps=100)
        imgs = vae.decode(z)
    return imgs.cpu()


def img_grid(tensors, nrow):
    """tensors: [n, 1, 28, 28] -> numpy grid for imshow."""
    n = tensors.shape[0]
    ncol = nrow
    rows = (n + ncol - 1) // ncol
    grid = np.ones((rows * 28, ncol * 28))
    for i in range(n):
        r, c = divmod(i, ncol)
        grid[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = tensors[i, 0].numpy()
    return grid


def main():
    config = yaml.safe_load(
        open("experiments/mnist_eldr_estimation/config.yaml")
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = config["latent_dim"]
    ckpt_dir = config["ckpt_dir"]
    alphas = config["alphas"]
    seed = config["seed"]

    # pairs to inspect per alpha
    pairs_to_show = list(range(config["num_pairs_per_alpha"]))
    n_imgs = 8  # images per row

    dataset = get_mnist_dataset(root="./data", train=True)

    # layout: one column-group per alpha, one row-group per pair
    # rows within each pair-group: original_s0, recon_s0, original_s1, recon_s1,
    #                               vae_sample_s0, vae_sample_s1,
    #                               flow_sample_s0, flow_sample_s1
    n_alphas = len(alphas)
    n_pairs = len(pairs_to_show)

    # 8 rows per (alpha, pair): orig_s0, recon_s0, orig_s1, recon_s1,
    #                            vae_prior_s0, vae_prior_s1, flow_s0, flow_s1
    row_labels = [
        "orig s0", "recon s0", "orig s1", "recon s1",
        "prior s0", "prior s1", "flow s0", "flow s1",
    ]
    n_row_types = len(row_labels)

    total_rows = n_pairs * n_row_types
    total_cols = n_alphas

    fig, axes = plt.subplots(
        total_rows, total_cols,
        figsize=(n_alphas * 3.5, n_pairs * n_row_types * 1.2),
        squeeze=False,
    )

    for ai, alpha in enumerate(alphas):
        for pi, pair_idx in enumerate(pairs_to_show):
            alpha_idx = ai
            pair_seed = seed + alpha_idx * 1000 + pair_idx

            # load weights
            w_path = f"{ckpt_dir}/weights_alpha_{alpha_idx}_pair_{pair_idx}.pt"
            if not Path(w_path).exists():
                print(f"skip alpha={alpha} pair={pair_idx}: no weights")
                continue
            w_ckpt = torch.load(w_path, map_location="cpu")
            w0 = w_ckpt["w0"].numpy()
            w1 = w_ckpt["w1"].numpy()

            # load models
            vae0 = load_vae(
                f"{ckpt_dir}/vae_alpha_{alpha_idx}_pair_{pair_idx}_side0.pt",
                latent_dim, device,
            )
            vae1 = load_vae(
                f"{ckpt_dir}/vae_alpha_{alpha_idx}_pair_{pair_idx}_side1.pt",
                latent_dim, device,
            )
            flow0 = load_flow(
                f"{ckpt_dir}/flow_alpha_{alpha_idx}_pair_{pair_idx}_side0.pt",
                latent_dim, device,
            )
            flow1 = load_flow(
                f"{ckpt_dir}/flow_alpha_{alpha_idx}_pair_{pair_idx}_side1.pt",
                latent_dim, device,
            )

            # get real images from each side's distribution
            orig0 = get_real_images(dataset, w0, n_imgs, seed=pair_seed)
            orig1 = get_real_images(dataset, w1, n_imgs, seed=pair_seed + 1)

            # reconstructions
            recon0 = vae_reconstruct(vae0, orig0, device)
            recon1 = vae_reconstruct(vae1, orig1, device)

            # vae prior samples
            prior0 = vae_sample(vae0, n_imgs, latent_dim, device)
            prior1 = vae_sample(vae1, n_imgs, latent_dim, device)

            # flow samples
            fsamp0 = flow_sample_and_decode(flow0, vae0, n_imgs, latent_dim, device)
            fsamp1 = flow_sample_and_decode(flow1, vae1, n_imgs, latent_dim, device)

            grids = [
                img_grid(orig0, n_imgs),
                img_grid(recon0, n_imgs),
                img_grid(orig1, n_imgs),
                img_grid(recon1, n_imgs),
                img_grid(prior0, n_imgs),
                img_grid(prior1, n_imgs),
                img_grid(fsamp0, n_imgs),
                img_grid(fsamp1, n_imgs),
            ]

            for ri, grid in enumerate(grids):
                row = pi * n_row_types + ri
                ax = axes[row, ai]
                ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])

                if ai == 0:
                    ax.set_ylabel(
                        f"p{pi} {row_labels[ri]}",
                        fontsize=7, rotation=0, ha="right", va="center",
                    )
                if pi == 0 and ri == 0:
                    ax.set_title(f"alpha={alpha}", fontsize=9)

            # clean up gpu
            del vae0, vae1, flow0, flow1
            torch.cuda.empty_cache()

            print(f"done alpha={alpha} pair={pair_idx}")

    plt.tight_layout()
    fig_dir = Path(config["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "pretrain_diagnostic.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
