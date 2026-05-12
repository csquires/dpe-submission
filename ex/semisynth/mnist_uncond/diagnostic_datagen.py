"""datagen diagnostic: weight distributions, ldr histograms, pca, kl estimation.

lightweight mode (default): reads hdf5 data, produces datagen_diagnostic.png
heavy mode (--compute-kl): loads models, computes latent kl, produces datagen_kl_diagnostic.png
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.stats
import torch
import yaml

from ex.utils.mnist_imbalance import get_mnist_dataset, subsample_mnist
from ex.utils.diagnostics import (
    load_all_pairs, plot_ldr_histograms, plot_pca, plot_kl_scatter,
    plot_ldr_stats, plot_qq, plot_lightweight_figure, compute_hardness,
    print_hardness_table, plot_hardness_figure
)


def parse_args(args=None):
    """parse command-line arguments.

    args:
        args: list of strings to parse (default: None, uses sys.argv)

    returns:
        argparse.Namespace with compute_kl (bool)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-kl", action="store_true",
                        help="heavy mode: load flows, compute latent-space KL")
    parser.add_argument("--config",
                        default="ex/semisynth/mnist_uncond/config.yaml",
                        help="path to config yaml")
    return parser.parse_args(args)


def plot_weight_bars(ax, w0, w1, alpha, pair_idx):
    """plot weight distributions as overlaid bar chart.

    args:
        ax: matplotlib axes
        w0: [10] class weights for p0
        w1: [10] class weights for p1
        alpha: alpha value for title
        pair_idx: pair index for title
    """
    x = np.arange(10)
    width = 0.35
    ax.bar(x - width/2, w0, width, label="p0", color="tab:blue", alpha=0.8)
    ax.bar(x + width/2, w1, width, label="p1", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("digit")
    ax.set_ylabel("weight")
    ax.set_title(f"alpha={alpha}, pair {pair_idx}")
    ax.legend(fontsize=7)


def plot_class_counts(ax, dataset, w0, w1, alpha, pair_idx):
    """plot actual sample counts from subsampled MNIST.

    args:
        ax: matplotlib axes
        dataset: MNIST dataset object
        w0: [10] target weights for p0
        w1: [10] target weights for p1
        alpha: alpha value for title
        pair_idx: pair index for title
    """
    idx0 = subsample_mnist(dataset, w0, min_per_class=10)
    idx1 = subsample_mnist(dataset, w1, min_per_class=10)
    counts0 = Counter(dataset.targets[idx0].numpy())
    counts1 = Counter(dataset.targets[idx1].numpy())
    x = np.arange(10)
    width = 0.35
    ax.bar(x - width/2, [counts0.get(i, 0) for i in range(10)], width,
           label="p0", color="tab:blue", alpha=0.8)
    ax.bar(x + width/2, [counts1.get(i, 0) for i in range(10)], width,
           label="p1", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("digit")
    ax.set_ylabel("count")
    ax.set_title(f"alpha={alpha}, pair {pair_idx}")
    ax.legend(fontsize=7)


def compute_ldr_at_points(eval_points, vae_0, vae_1, flow_0, flow_1, vae_global, config, device):
    """compute log(p0(z)/p1(z)) at eval_points in global latent space.

    same formula as step1's true_ldrs but at arbitrary evaluation points.

    args:
        eval_points: [N, D] tensor in global latent space
        vae_0, vae_1: per-pair VAEs for encoding
        flow_0, flow_1: per-pair flows
        vae_global: global VAE for decoding to image space
        config: config dict with log_prob_steps and device
        device: torch device

    returns:
        tuple of ([N] log density ratios, dict of flow logprobs)
    """
    from src.models.flow import log_prob
    from ex.semisynth.mnist_uncond.step1_create_data import compute_log_jacobian

    batch_enc = 1000
    batch_lp = 500
    device_str = str(device)

    with torch.no_grad():
        # encode eval_points through each per-pair VAE
        z_in_0_list, z_in_1_list = [], []
        for i in range(0, len(eval_points), batch_enc):
            batch = eval_points[i:i+batch_enc].to(device)
            imgs = vae_global.decode(batch)
            mu0, _ = vae_0.encode(imgs)
            mu1, _ = vae_1.encode(imgs)
            z_in_0_list.append(mu0.cpu())
            z_in_1_list.append(mu1.cpu())
        z_in_0 = torch.cat(z_in_0_list)  # [N, 14]
        z_in_1 = torch.cat(z_in_1_list)  # [N, 14]

        # flow log-probs
        log_p0_list, log_p1_list = [], []
        for i in range(0, len(z_in_0), batch_lp):
            z0b = z_in_0[i:i+batch_lp].to(device)
            z1b = z_in_1[i:i+batch_lp].to(device)
            log_p0_list.append(log_prob(flow_0, z0b, steps=config["log_prob_steps"], device=device_str).cpu())
            log_p1_list.append(log_prob(flow_1, z1b, steps=config["log_prob_steps"], device=device_str).cpu())
        log_p0 = torch.cat(log_p0_list)  # [N]
        log_p1 = torch.cat(log_p1_list)  # [N]

    # jacobian corrections
    log_jac_0 = compute_log_jacobian(vae_0, vae_global, eval_points, device)
    log_jac_1 = compute_log_jacobian(vae_1, vae_global, eval_points, device)

    ldrs = (log_p0 + log_jac_0) - (log_p1 + log_jac_1)  # [N]
    flow_logprobs = {"log_p0": log_p0, "log_p1": log_p1}
    return ldrs, flow_logprobs


def load_models(ckpt_dir, latent_dim, ai, pi, device):
    """load vae_global + per-pair vae/flow models.

    returns: (vae_global, vae_0, vae_1, flow_0, flow_1)
    """
    from src.models.vae import MNISTVAE
    from src.models.flow import VelocityMLP

    vae_global = MNISTVAE(latent_dim=latent_dim)
    vae_global.load_state_dict(torch.load(f"{ckpt_dir}/vae_global.pt", map_location="cpu"))
    vae_global.to(device).eval()

    vae_0 = MNISTVAE(latent_dim=latent_dim)
    vae_0.load_state_dict(torch.load(
        f"{ckpt_dir}/vae_alpha_{ai}_pair_{pi}_side0.pt", map_location="cpu"))
    vae_0.to(device).eval()

    vae_1 = MNISTVAE(latent_dim=latent_dim)
    vae_1.load_state_dict(torch.load(
        f"{ckpt_dir}/vae_alpha_{ai}_pair_{pi}_side1.pt", map_location="cpu"))
    vae_1.to(device).eval()

    flow_0 = VelocityMLP(latent_dim=latent_dim)
    flow_0.load_state_dict(torch.load(
        f"{ckpt_dir}/flow_alpha_{ai}_pair_{pi}_side0.pt", map_location="cpu"))
    flow_0.to(device).eval()

    flow_1 = VelocityMLP(latent_dim=latent_dim)
    flow_1.load_state_dict(torch.load(
        f"{ckpt_dir}/flow_alpha_{ai}_pair_{pi}_side1.pt", map_location="cpu"))
    flow_1.to(device).eval()

    return vae_global, vae_0, vae_1, flow_0, flow_1


def plot_kl_figure(data, config, alphas, heavy_stats):
    """create heavy kl diagnostic figure: qq plots + correlation panels.

    uses precomputed latent KLs from heavy_stats. only reloads models
    for qq plots (needs flow log-probs at pstar).

    args:
        data: nested dict from load_all_pairs
        config: config dict
        alphas: list of alpha values
        heavy_stats: dict with kl_p0_p1, kl_p1_p0 arrays [n_alphas, num_pairs]
    """
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = config["ckpt_dir"]
    latent_dim = config["latent_dim"]
    num_pairs = config["num_pairs_per_alpha"]
    n_alphas = len(alphas)

    total_rows = num_pairs + 1  # qq rows + correlation row
    fig2 = plt.figure(figsize=(4 * n_alphas, 2.5 * total_rows))
    gs2 = gridspec.GridSpec(total_rows, n_alphas, figure=fig2, hspace=0.4, wspace=0.3)

    n_eval = 2000

    for pi in range(num_pairs):
        for ai, alpha in enumerate(alphas):
            print(f"qq plot: alpha_idx={ai} pair_idx={pi}")
            models = load_models(ckpt_dir, latent_dim, ai, pi, device)
            vae_global, vae_0, vae_1, flow_0, flow_1 = models

            rng = np.random.RandomState(42 + ai * 1000 + pi)
            pstar_full = torch.from_numpy(data[ai][pi]["pstar"]).float()
            idx = rng.choice(len(pstar_full), n_eval, replace=False)

            _, logprobs = compute_ldr_at_points(
                pstar_full[idx], vae_0, vae_1, flow_0, flow_1, vae_global, config, device)
            ax_qq = fig2.add_subplot(gs2[pi, ai])
            plot_qq(ax_qq, logprobs["log_p0"], logprobs["log_p1"], alpha, pi)

            del vae_global, vae_0, vae_1, flow_0, flow_1
            torch.cuda.empty_cache()

    # correlation panels from precomputed heavy_stats
    kl_fwd = heavy_stats["kl_p0_p1"]   # [n_alphas, num_pairs]
    kl_rev = heavy_stats["kl_p1_p0"]

    ax_corr = fig2.add_subplot(gs2[num_pairs, 0:n_alphas // 2])
    colors = [plt.cm.viridis(ai / (n_alphas - 1)) for ai in range(n_alphas)]
    for ai, alpha in enumerate(alphas):
        for pi in range(num_pairs):
            ax_corr.scatter(data[ai][pi]["kl_weights"], kl_fwd[ai, pi],
                            s=15, alpha=0.7, color=colors[ai])
    all_cat = [data[ai][pi]["kl_weights"] for ai in range(n_alphas) for pi in range(num_pairs)]
    lo = min(min(all_cat), kl_fwd.min())
    hi = max(max(all_cat), kl_fwd.max())
    ax_corr.plot([lo, hi], [lo, hi], "k--", alpha=0.3)
    ax_corr.set_xlabel("KL(w0 || w1)")
    ax_corr.set_ylabel("KL(p0 || p1) latent")
    ax_corr.set_title("categorical vs latent KL")

    ax_sym = fig2.add_subplot(gs2[num_pairs, n_alphas // 2:n_alphas])
    for ai in range(n_alphas):
        for pi in range(num_pairs):
            ax_sym.scatter(kl_fwd[ai, pi], kl_rev[ai, pi],
                           s=15, alpha=0.7, color=colors[ai])
    lo_s = min(kl_fwd.min(), kl_rev.min())
    hi_s = max(kl_fwd.max(), kl_rev.max())
    ax_sym.plot([lo_s, hi_s], [lo_s, hi_s], "k--", alpha=0.3)
    ax_sym.set_xlabel("KL(p0 || p1)")
    ax_sym.set_ylabel("KL(p1 || p0)")
    ax_sym.set_title("KL symmetry check")

    fig_dir = Path(config["figures_dir"])
    fig2.savefig(fig_dir / "datagen_kl_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved datagen_kl_diagnostic.png to {fig_dir}")

    # summary table
    print("\nalpha  pair  KL(w0||w1)  KL(p0||p1)  KL(p1||p0)  E_p*[LDR]")
    for ai, alpha in enumerate(alphas):
        for pi in range(num_pairs):
            mean_ldr = np.mean(data[ai][pi]["true_ldrs"])
            print(f"{alpha:<5.1f}  {pi:<4d}  "
                  f"{data[ai][pi]['kl_weights']:<10.2f}  {kl_fwd[ai, pi]:<10.2f}  "
                  f"{kl_rev[ai, pi]:<10.2f}  {mean_ldr:<10.4f}")


def compute_heavy_stats(data, config, alphas, num_pairs):
    """compute per-pair flow velocity norms and latent-space KL estimates (heavy).

    for each (alpha, pair):
      - loads flow models, evaluates ||v(z, t=0)||_2 at pstar samples
      - loads all models, computes KL(p0||p1) and KL(p1||p0) in latent space

    returns:
        dict with keys: v0_norm, v1_norm, kl_p0_p1, kl_p1_p0
        each [n_alphas, num_pairs]
    """
    from src.models.flow import VelocityMLP

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = config["ckpt_dir"]
    latent_dim = config["latent_dim"]
    n_eval = 2000

    out = {
        "v0_norm": np.zeros((len(alphas), num_pairs)),
        "v1_norm": np.zeros((len(alphas), num_pairs)),
        "kl_p0_p1": np.zeros((len(alphas), num_pairs)),
        "kl_p1_p0": np.zeros((len(alphas), num_pairs)),
    }

    for ai in range(len(alphas)):
        for pi in range(num_pairs):
            rng = np.random.RandomState(42 + ai * 1000 + pi)
            pstar = torch.from_numpy(data[ai][pi]["pstar"]).float()
            p0_full = torch.from_numpy(data[ai][pi]["p0"]).float()
            p1_full = torch.from_numpy(data[ai][pi]["p1"]).float()
            idx = rng.choice(len(pstar), n_eval, replace=False)

            # flow velocities (only needs flows)
            flow_0 = VelocityMLP(latent_dim=latent_dim)
            flow_0.load_state_dict(torch.load(
                f"{ckpt_dir}/flow_alpha_{ai}_pair_{pi}_side0.pt", map_location="cpu"))
            flow_0.to(device).eval()

            flow_1 = VelocityMLP(latent_dim=latent_dim)
            flow_1.load_state_dict(torch.load(
                f"{ckpt_dir}/flow_alpha_{ai}_pair_{pi}_side1.pt", map_location="cpu"))
            flow_1.to(device).eval()

            z = pstar[idx].to(device)
            t_zero = torch.zeros(len(z), device=device)
            with torch.no_grad():
                v0 = flow_0(z, t_zero)
                v1 = flow_1(z, t_zero)
                out["v0_norm"][ai, pi] = v0.norm(dim=1).mean().item()
                out["v1_norm"][ai, pi] = v1.norm(dim=1).mean().item()

            # latent KL (needs full model stack)
            models = load_models(ckpt_dir, latent_dim, ai, pi, device)
            vae_global, vae_0, vae_1, _, _ = models

            idx_kl = rng.choice(len(p0_full), n_eval, replace=False)
            ldrs_p0, _ = compute_ldr_at_points(
                p0_full[idx_kl], vae_0, vae_1, flow_0, flow_1, vae_global, config, device)
            ldrs_p1, _ = compute_ldr_at_points(
                p1_full[idx_kl], vae_0, vae_1, flow_0, flow_1, vae_global, config, device)

            out["kl_p0_p1"][ai, pi] = ldrs_p0.mean().item()
            out["kl_p1_p0"][ai, pi] = -ldrs_p1.mean().item()

            del flow_0, flow_1, vae_global, vae_0, vae_1
            torch.cuda.empty_cache()

            print(f"heavy stats: alpha_idx={ai} pair_idx={pi}  "
                  f"||v0||={out['v0_norm'][ai, pi]:.3f}  "
                  f"||v1||={out['v1_norm'][ai, pi]:.3f}  "
                  f"KL(p0||p1)={out['kl_p0_p1'][ai, pi]:.3f}  "
                  f"KL(p1||p0)={out['kl_p1_p0'][ai, pi]:.3f}")

    return out


def main():
    """main entry point."""
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    alphas = config["alphas"]
    num_pairs = config["num_pairs_per_alpha"]
    data_dir = config["data_dir"]

    # load all hdf5 data
    data = load_all_pairs(data_dir, alphas, num_pairs)

    # load mnist for class counts
    dataset = get_mnist_dataset(root="./data", train=True)

    # figure 1: lightweight diagnostic
    plot_lightweight_figure(data, dataset, alphas, config,
                            plot_weight_bars_fn=plot_weight_bars,
                            plot_class_counts_fn=plot_class_counts)

    # hardness variance analysis
    stats = compute_hardness(data, alphas, num_pairs)
    heavy_stats = None
    if args.compute_kl:
        heavy_stats = compute_heavy_stats(data, config, alphas, num_pairs)
    print_hardness_table(stats, alphas, heavy_stats)
    plot_hardness_figure(stats, alphas, config, heavy_stats, K=10)

    # figure 2: heavy kl diagnostic (qq plots + correlation)
    if args.compute_kl:
        plot_kl_figure(data, config, alphas, heavy_stats)


if __name__ == "__main__":
    main()
