"""datagen diagnostic: weight distributions, ldr histograms, pca, kl estimation.

lightweight mode (default): reads hdf5 data, produces datagen_diagnostic.png
heavy mode (--compute-kl): loads models, computes latent kl, produces datagen_kl_diagnostic.png
"""
import argparse
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.stats
import torch
import yaml
from sklearn.decomposition import PCA

from experiments.utils.mnist_imbalance import get_mnist_dataset, subsample_mnist


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
    return parser.parse_args(args)


def load_all_pairs(data_dir, alphas, num_pairs):
    """load all HDF5 pairs into nested dict.

    for each (alpha_idx, pair_idx), open HDF5 and read all keys.

    args:
        data_dir: directory containing alpha_*_pair_*.h5 files
        alphas: list of alpha values
        num_pairs: number of pairs per alpha

    returns:
        nested dict: data[alpha_idx][pair_idx] = {
            "w0": [10], "w1": [10], "kl_weights": float,
            "true_ldrs": [N], "pstar": [N, 14], "p0": [N, 14], "p1": [N, 14]
        }
    """
    data = {}
    for ai in range(len(alphas)):
        data[ai] = {}
        for pi in range(num_pairs):
            path = f"{data_dir}/alpha_{ai}_pair_{pi}.h5"
            with h5py.File(path, 'r') as f:
                data[ai][pi] = {
                    "w0": f["w0"][:],
                    "w1": f["w1"][:],
                    "kl_weights": float(f["kl_weights"][()]),
                    "true_ldrs": f["true_ldrs"][:],
                    "pstar": f["pstar_samples"][:],
                    "p0": f["p0_samples"][:],
                    "p1": f["p1_samples"][:],
                }
            print(f"loaded alpha_idx={ai}, pair_idx={pi}")
    return data


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


def plot_ldr_histograms(ax, pair_data, alpha):
    """plot overlaid true_ldrs histograms for all pairs at one alpha.

    args:
        ax: matplotlib axes
        pair_data: dict mapping pair_idx -> {"true_ldrs": [N]}
        alpha: alpha value for title
    """
    for pi, pd in pair_data.items():
        ldrs = pd["true_ldrs"]
        ax.hist(ldrs, bins=100, density=True, alpha=0.2, color="tab:blue")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlabel("log(p0/p1)")
    ax.set_ylabel("density")
    ax.set_title(f"alpha={alpha}")


def plot_pca(ax, pstar, p0, p1, alpha, pair_idx, n_plot=2000):
    """fit pca on pstar and scatter all three distributions.

    args:
        ax: matplotlib axes
        pstar: [N, 14] balanced reference samples
        p0: [N, 14] p0 samples
        p1: [N, 14] p1 samples
        alpha: alpha value for title
        pair_idx: pair index for title
        n_plot: max samples to plot per distribution
    """
    pca = PCA(n_components=2)
    pca.fit(pstar)
    rng = np.random.RandomState(42)
    idx = lambda n: rng.choice(n, min(n_plot, n), replace=False)
    ps = pca.transform(pstar[idx(len(pstar))])
    p0t = pca.transform(p0[idx(len(p0))])
    p1t = pca.transform(p1[idx(len(p1))])
    ax.scatter(ps[:, 0], ps[:, 1], s=1, alpha=0.2, c="gray", label="p*")
    ax.scatter(p0t[:, 0], p0t[:, 1], s=1, alpha=0.2, c="tab:blue", label="p0")
    ax.scatter(p1t[:, 0], p1t[:, 1], s=1, alpha=0.2, c="tab:orange", label="p1")
    ax.set_title(f"alpha={alpha}, pair {pair_idx}")
    ax.legend(fontsize=6, markerscale=5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def plot_kl_scatter(ax, data, alphas):
    """scatter plot of KL(w0||w1) vs alpha.

    args:
        ax: matplotlib axes
        data: nested dict from load_all_pairs
        alphas: list of alpha values
    """
    for ai, alpha in enumerate(alphas):
        kls = [data[ai][pi]["kl_weights"] for pi in data[ai]]
        ax.scatter([alpha] * len(kls), kls, s=15, alpha=0.6, color="tab:red")
        ax.scatter([alpha], [np.mean(kls)], s=50, marker="D", color="black", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("KL(w0 || w1)")
    ax.set_title("categorical KL vs alpha")


def plot_ldr_stats(ax, data, alphas):
    """scatter plot of true_ldrs mean and std vs alpha.

    args:
        ax: matplotlib axes
        data: nested dict from load_all_pairs
        alphas: list of alpha values
    """
    for ai, alpha in enumerate(alphas):
        pair_means = [np.mean(data[ai][pi]["true_ldrs"]) for pi in data[ai]]
        pair_stds = [np.std(data[ai][pi]["true_ldrs"]) for pi in data[ai]]
        ax.scatter([alpha] * len(pair_means), pair_means, s=10, alpha=0.4, color="tab:blue", label="mean LDR" if ai == 0 else None)
        ax.scatter([alpha] * len(pair_stds), pair_stds, s=10, alpha=0.4, color="tab:green", label="std LDR" if ai == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("value")
    ax.set_title("E_p*[log(p0/p1)] and std")
    ax.legend(fontsize=7)


def plot_qq(ax, log_p0, log_p1, alpha, pair_idx):
    """qq plot of flow log-probs for two flows.

    args:
        ax: matplotlib axes
        log_p0: [N] flow log-probs for flow_0
        log_p1: [N] flow log-probs for flow_1
        alpha: alpha value for title
        pair_idx: pair index for title
    """
    scipy.stats.probplot(log_p0.numpy(), dist="norm", plot=ax)
    (osm, osr), _ = scipy.stats.probplot(log_p1.numpy(), dist="norm")
    ax.plot(osm, osr, ".", color="tab:orange", markersize=1, alpha=0.5)
    ax.set_title(f"alpha={alpha}, pair {pair_idx}")
    ax.set_xlabel("theoretical quantiles")
    ax.set_ylabel("flow log-prob")


def plot_lightweight_figure(data, dataset, alphas, config):
    """create lightweight diagnostic figure covering all pairs.

    layout: 4 sections stacked vertically, each num_pairs rows x 4 cols,
    plus 1 row for ldr histograms and 1 row for kl summary.

    section order: weight bars, class counts, ldr histograms (1 row),
    pca scatter, kl summary (1 row).

    args:
        data: nested dict from load_all_pairs
        dataset: MNIST dataset for class counts
        alphas: list of alpha values
        config: config dict
    """
    num_pairs = config["num_pairs_per_alpha"]
    n_alphas = len(alphas)
    # rows: weights(num_pairs) + counts(num_pairs) + ldr(1) + pca(num_pairs) + summary(1)
    total_rows = 3 * num_pairs + 2
    fig = plt.figure(figsize=(4 * n_alphas, 2 * total_rows))
    gs = gridspec.GridSpec(total_rows, n_alphas, figure=fig, hspace=0.4, wspace=0.3)

    r = 0  # current row cursor

    # weight bars: num_pairs rows
    for pi in range(num_pairs):
        for ai, alpha in enumerate(alphas):
            ax = fig.add_subplot(gs[r, ai])
            plot_weight_bars(ax, data[ai][pi]["w0"], data[ai][pi]["w1"], alpha, pi)
        r += 1

    # class counts: num_pairs rows
    for pi in range(num_pairs):
        for ai, alpha in enumerate(alphas):
            ax = fig.add_subplot(gs[r, ai])
            plot_class_counts(ax, dataset, data[ai][pi]["w0"], data[ai][pi]["w1"], alpha, pi)
        r += 1

    # ldr histograms: 1 row (all pairs overlaid per alpha)
    for ai, alpha in enumerate(alphas):
        ax = fig.add_subplot(gs[r, ai])
        plot_ldr_histograms(ax, data[ai], alpha)
    r += 1

    # pca scatter: num_pairs rows
    for pi in range(num_pairs):
        for ai, alpha in enumerate(alphas):
            ax = fig.add_subplot(gs[r, ai])
            plot_pca(ax, data[ai][pi]["pstar"], data[ai][pi]["p0"],
                     data[ai][pi]["p1"], alpha, pi, n_plot=2000)
        r += 1

    # kl summary: 1 row, 2 wide panels
    ax_kl = fig.add_subplot(gs[r, 0:n_alphas // 2])
    plot_kl_scatter(ax_kl, data, alphas)
    ax_ldr = fig.add_subplot(gs[r, n_alphas // 2:n_alphas])
    plot_ldr_stats(ax_ldr, data, alphas)

    fig_dir = Path(config["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "datagen_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved datagen_diagnostic.png to {fig_dir}")


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
    from experiments.mnist_eldr_estimation.step1_create_data import compute_log_jacobian

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


def plot_kl_figure(data, config, alphas):
    """create heavy kl diagnostic figure covering all pairs.

    loads models per (alpha, pair), computes latent kl.
    qq plots: num_pairs rows x n_alphas cols.
    correlation panels: 1 row x 2 wide panels.

    args:
        data: nested dict from load_all_pairs
        config: config dict
        alphas: list of alpha values
    """
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = config["ckpt_dir"]
    latent_dim = config["latent_dim"]
    num_pairs = config["num_pairs_per_alpha"]
    n_alphas = len(alphas)

    # collect kl estimates across all (alpha, pair)
    kls_latent_p0_p1 = []
    kls_latent_p1_p0 = []
    kls_categorical = []
    alphas_for_plot = []
    pairs_for_plot = []

    total_rows = num_pairs + 1  # qq rows + correlation row
    fig2 = plt.figure(figsize=(4 * n_alphas, 2.5 * total_rows))
    gs2 = gridspec.GridSpec(total_rows, n_alphas, figure=fig2, hspace=0.4, wspace=0.3)

    n_eval = 2000  # subsample for speed across 40 pairs

    for pi in range(num_pairs):
        for ai, alpha in enumerate(alphas):
            print(f"heavy kl: alpha_idx={ai} pair_idx={pi}")
            models = load_models(ckpt_dir, latent_dim, ai, pi, device)
            vae_global, vae_0, vae_1, flow_0, flow_1 = models

            rng = np.random.RandomState(42 + ai * 1000 + pi)
            pstar_full = torch.from_numpy(data[ai][pi]["pstar"]).float()
            p0_full = torch.from_numpy(data[ai][pi]["p0"]).float()
            p1_full = torch.from_numpy(data[ai][pi]["p1"]).float()
            idx = rng.choice(len(pstar_full), n_eval, replace=False)

            # qq plot from pstar log-probs
            _, logprobs = compute_ldr_at_points(
                pstar_full[idx], vae_0, vae_1, flow_0, flow_1, vae_global, config, device)
            ax_qq = fig2.add_subplot(gs2[pi, ai])
            plot_qq(ax_qq, logprobs["log_p0"], logprobs["log_p1"], alpha, pi)

            # kl estimates from p0/p1 log-density-ratios
            ldrs_p0, _ = compute_ldr_at_points(
                p0_full[idx], vae_0, vae_1, flow_0, flow_1, vae_global, config, device)
            ldrs_p1, _ = compute_ldr_at_points(
                p1_full[idx], vae_0, vae_1, flow_0, flow_1, vae_global, config, device)

            kls_latent_p0_p1.append(ldrs_p0.mean().item())
            kls_latent_p1_p0.append(-ldrs_p1.mean().item())
            kls_categorical.append(data[ai][pi]["kl_weights"])
            alphas_for_plot.append(alpha)
            pairs_for_plot.append(pi)

            del vae_global, vae_0, vae_1, flow_0, flow_1
            torch.cuda.empty_cache()

    # correlation panels (bottom row, 2 wide panels)
    ax_corr = fig2.add_subplot(gs2[num_pairs, 0:n_alphas // 2])
    colors = [plt.cm.viridis(ai / (n_alphas - 1)) for ai, _ in enumerate(alphas)]
    for i in range(len(kls_categorical)):
        ai = alphas.index(alphas_for_plot[i])
        ax_corr.scatter(kls_categorical[i], kls_latent_p0_p1[i], s=15, alpha=0.7, color=colors[ai])
    lo = min(kls_categorical + kls_latent_p0_p1)
    hi = max(kls_categorical + kls_latent_p0_p1)
    ax_corr.plot([lo, hi], [lo, hi], "k--", alpha=0.3)
    ax_corr.set_xlabel("KL(w0 || w1)")
    ax_corr.set_ylabel("KL(p0 || p1) latent")
    ax_corr.set_title("categorical vs latent KL")

    ax_sym = fig2.add_subplot(gs2[num_pairs, n_alphas // 2:n_alphas])
    for i in range(len(kls_latent_p0_p1)):
        ai = alphas.index(alphas_for_plot[i])
        ax_sym.scatter(kls_latent_p0_p1[i], kls_latent_p1_p0[i], s=15, alpha=0.7, color=colors[ai])
    lo_s = min(kls_latent_p0_p1 + kls_latent_p1_p0)
    hi_s = max(kls_latent_p0_p1 + kls_latent_p1_p0)
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
    for i in range(len(kls_categorical)):
        ai = alphas.index(alphas_for_plot[i])
        mean_ldr = np.mean(data[ai][pairs_for_plot[i]]["true_ldrs"])
        print(f"{alphas_for_plot[i]:<5.1f}  {pairs_for_plot[i]:<4d}  "
              f"{kls_categorical[i]:<10.2f}  {kls_latent_p0_p1[i]:<10.2f}  "
              f"{kls_latent_p1_p0[i]:<10.2f}  {mean_ldr:<10.4f}")


def compute_hardness(data, alphas, num_pairs):
    """compute per-pair hardness metrics from saved data.

    metrics per (alpha_idx, pair_idx):
      - w_dist: ||w0 - w1||_2
      - kl_cat: KL(w0 || w1)
      - ldr_std: std(true_ldrs)
      - ldr_abs_mean: |mean(true_ldrs)|
      - ldr_abs_median: median(|true_ldrs|)
      - latent_mean_dist: ||mean(p0) - mean(p1)||_2

    returns:
        dict mapping metric_name -> [n_alphas, num_pairs] numpy array
    """
    metric_names = ["w_dist", "kl_cat", "ldr_std", "ldr_abs_mean",
                    "ldr_abs_median", "latent_mean_dist"]
    stats = {m: np.zeros((len(alphas), num_pairs)) for m in metric_names}

    for ai in range(len(alphas)):
        for pi in range(num_pairs):
            d = data[ai][pi]
            stats["w_dist"][ai, pi] = np.linalg.norm(d["w0"] - d["w1"])
            stats["kl_cat"][ai, pi] = d["kl_weights"]
            stats["ldr_std"][ai, pi] = np.std(d["true_ldrs"])
            stats["ldr_abs_mean"][ai, pi] = np.abs(np.mean(d["true_ldrs"]))
            stats["ldr_abs_median"][ai, pi] = np.median(np.abs(d["true_ldrs"]))
            stats["latent_mean_dist"][ai, pi] = np.linalg.norm(
                d["p0"].mean(axis=0) - d["p1"].mean(axis=0))
    return stats


def compute_flow_velocities(data, config, alphas, num_pairs):
    """compute mean initial flow velocity norms per pair (heavy).

    evaluates ||v(z, t=0)||_2 for z ~ pstar (subsampled), both flows.

    returns:
        dict with "v0_norm" and "v1_norm", each [n_alphas, num_pairs]
    """
    from src.models.flow import VelocityMLP

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = config["ckpt_dir"]
    latent_dim = config["latent_dim"]
    n_eval = 2000

    v_stats = {
        "v0_norm": np.zeros((len(alphas), num_pairs)),
        "v1_norm": np.zeros((len(alphas), num_pairs)),
    }

    for ai in range(len(alphas)):
        for pi in range(num_pairs):
            # load flows only (no vae needed)
            flow_0 = VelocityMLP(latent_dim=latent_dim)
            flow_0.load_state_dict(torch.load(
                f"{ckpt_dir}/flow_alpha_{ai}_pair_{pi}_side0.pt", map_location="cpu"))
            flow_0.to(device).eval()

            flow_1 = VelocityMLP(latent_dim=latent_dim)
            flow_1.load_state_dict(torch.load(
                f"{ckpt_dir}/flow_alpha_{ai}_pair_{pi}_side1.pt", map_location="cpu"))
            flow_1.to(device).eval()

            rng = np.random.RandomState(42 + ai * 1000 + pi)
            pstar = torch.from_numpy(data[ai][pi]["pstar"]).float()
            idx = rng.choice(len(pstar), n_eval, replace=False)
            z = pstar[idx].to(device)
            t_zero = torch.zeros(len(z), device=device)

            with torch.no_grad():
                v0 = flow_0(z, t_zero)  # [n_eval, latent_dim]
                v1 = flow_1(z, t_zero)
                v_stats["v0_norm"][ai, pi] = v0.norm(dim=1).mean().item()
                v_stats["v1_norm"][ai, pi] = v1.norm(dim=1).mean().item()

            del flow_0, flow_1
            torch.cuda.empty_cache()
            print(f"flow velocity: alpha_idx={ai} pair_idx={pi}  "
                  f"||v0||={v_stats['v0_norm'][ai, pi]:.3f}  "
                  f"||v1||={v_stats['v1_norm'][ai, pi]:.3f}")

    return v_stats


def print_hardness_table(stats, alphas, v_stats=None):
    """print per-alpha summary of hardness metrics.

    for each metric, shows median, iqr, mean, std across pairs.
    """
    metrics = list(stats.keys())
    if v_stats is not None:
        metrics += list(v_stats.keys())

    print("\n" + "=" * 80)
    print("HARDNESS VARIANCE SUMMARY (per alpha, across pairs)")
    print("=" * 80)

    for name in metrics:
        vals = stats[name] if name in stats else v_stats[name]
        print(f"\n--- {name} ---")
        print(f"{'alpha':<8} {'median':>8} {'IQR':>8} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
        for ai, alpha in enumerate(alphas):
            row = vals[ai]
            q25, q50, q75 = np.percentile(row, [25, 50, 75])
            print(f"{alpha:<8.1f} {q50:>8.3f} {q75-q25:>8.3f} {np.mean(row):>8.3f} "
                  f"{np.std(row):>8.3f} {np.min(row):>8.3f} {np.max(row):>8.3f}")


def plot_hardness_figure(stats, alphas, config, v_stats=None):
    """box plot grid of hardness metrics per alpha.

    saves to figures_dir/datagen_variance.png
    """
    all_metrics = dict(stats)
    if v_stats is not None:
        all_metrics.update(v_stats)

    names = list(all_metrics.keys())
    n = len(names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, name in enumerate(names):
        ax = axes[i // ncols, i % ncols]
        vals = all_metrics[name]  # [n_alphas, num_pairs]
        # box plot: one box per alpha
        bp = ax.boxplot([vals[ai] for ai in range(len(alphas))],
                        labels=[str(a) for a in alphas],
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("tab:blue")
            patch.set_alpha(0.4)
        # overlay individual points
        for ai in range(len(alphas)):
            jitter = np.random.RandomState(42).uniform(-0.1, 0.1, len(vals[ai]))
            ax.scatter(np.full(len(vals[ai]), ai + 1) + jitter, vals[ai],
                       s=12, alpha=0.6, color="tab:red", zorder=5)
        ax.set_xlabel("alpha")
        ax.set_ylabel(name)
        ax.set_title(name)

    # hide unused subplots
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.tight_layout()
    fig_dir = Path(config["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "datagen_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved datagen_variance.png to {fig_dir}")


def main():
    """main entry point."""
    args = parse_args()
    config = yaml.safe_load(open("experiments/mnist_eldr_estimation/config.yaml"))
    alphas = config["alphas"]
    num_pairs = config["num_pairs_per_alpha"]
    data_dir = config["data_dir"]

    # load all hdf5 data
    data = load_all_pairs(data_dir, alphas, num_pairs)

    # load mnist for class counts
    dataset = get_mnist_dataset(root="./data", train=True)

    # figure 1: lightweight diagnostic
    plot_lightweight_figure(data, dataset, alphas, config)

    # hardness variance analysis
    stats = compute_hardness(data, alphas, num_pairs)
    v_stats = None
    if args.compute_kl:
        v_stats = compute_flow_velocities(data, config, alphas, num_pairs)
    print_hardness_table(stats, alphas, v_stats)
    plot_hardness_figure(stats, alphas, config, v_stats)

    # figure 2: heavy kl diagnostic
    if args.compute_kl:
        plot_kl_figure(data, config, alphas)


if __name__ == "__main__":
    main()
