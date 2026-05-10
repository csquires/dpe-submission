"""datagen diagnostic for the dbpedia cond_flow experiment.

mirrors the mnist cond-flow datagen diagnostic, but:
  - replaces the per-image class-count panel with a K=14 dbpedia label histogram.
  - reads log_p_y from the flowhash-keyed cache (not the unkeyed mnist version).
  - skips any per-image visualization (sbert is a one-way encoder; no decoder).

lightweight mode (default): weight bars, label-target bars, ldr histograms,
  pca, hardness boxplots. emits datagen_diagnostic.png and datagen_variance.png.

heavy mode (--compute-kl): per-pair latent KL via global cond_flow at p0/p1
  samples and cached log_p_y at pstar. emits datagen_kl_diagnostic.png with
  QQ-plot grid + KL_cat-vs-KL_latent scatter + KL symmetry check.

reuses encoder-agnostic helpers from
  experiments.mnist_eldr_estimation.diagnostic_datagen and overrides only the
  modality-specific class-count panel + the lightweight figure assembly.
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from experiments.mnist_eldr_estimation.diagnostic_datagen import (
    load_all_pairs,
    plot_ldr_histograms,
    plot_pca,
    plot_kl_scatter,
    plot_ldr_stats,
    plot_qq,
    compute_hardness,
    print_hardness_table,
    plot_hardness_figure,
)
from experiments.utils.dbpedia_imbalance import (
    DBPEDIA_LABEL_NAMES,
    flow_state_hash,
)
from src.models.flow import ClassCondVelocityMLP, log_prob_class_cond


K = 14


def parse_args(args=None):
    """parse cli args.

    args:
        args: optional list of strings (default: sys.argv).
    returns:
        argparse.Namespace with config (str), compute_kl (bool),
        n_eval (int), log_prob_steps (int).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="experiments/dbpedia_eldr_cond_flow/config.yaml")
    p.add_argument("--compute-kl", action="store_true",
                   help="heavy mode: compute latent KL via cond flow")
    p.add_argument("--n-eval", type=int, default=500,
                   help="samples per pair-side for KL estimation (heavy mode)")
    p.add_argument("--log-prob-steps", type=int, default=100,
                   help="ODE steps for heavy-mode log_p_y (default 100)")
    return p.parse_args(args)


def expand_paths(config):
    """expand env var tokens in any string config value."""
    import os
    for k, v in list(config.items()):
        if isinstance(v, str) and "$" in v:
            config[k] = os.path.expandvars(v)
    return config


def plot_label_targets(ax, w0, w1, alpha, pair_idx):
    """plot K=14 target weights as overlaid bar chart.

    args:
        ax: matplotlib axes
        w0, w1: [14] target weights for p0, p1
        alpha: alpha value for title
        pair_idx: pair index for title

    rationale:
        for mnist this plot showed actual subsample counts; for dbpedia
        the per-pair samples are flow draws (not real text picks), so the
        relevant quantity is the prescribed target distribution itself.
        we render it once as a labeled bar chart so the entity-type names
        are visible.
    """
    x = np.arange(K)
    width = 0.35
    ax.bar(x - width/2, w0, width, label="p0", color="tab:blue", alpha=0.8)
    ax.bar(x + width/2, w1, width, label="p1", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:6] for n in DBPEDIA_LABEL_NAMES],
                       rotation=60, fontsize=6, ha="right")
    ax.set_ylabel("target weight")
    ax.set_title(f"alpha={alpha}, pair {pair_idx}")
    ax.legend(fontsize=7)


def plot_lightweight_figure(data, alphas, config):
    """create lightweight diagnostic figure covering all pairs.

    layout (per alpha column):
        weight bars (num_pairs rows)
        label targets (num_pairs rows)        # dbpedia-specific (K=14, named)
        ldr histograms (1 row, all pairs overlaid)
        pca scatter (num_pairs rows)
        kl + ldr summary (1 row, two wide panels)

    args:
        data: nested dict from load_all_pairs (pstar/p0/p1 are 64-dim here).
        alphas: list of alpha values.
        config: config dict with figures_dir, num_pairs_per_alpha.
    """
    num_pairs = config["num_pairs_per_alpha"]
    n_alphas = len(alphas)
    total_rows = 2 * num_pairs + 2
    fig = plt.figure(figsize=(4 * n_alphas, 2 * total_rows))
    gs = gridspec.GridSpec(total_rows, n_alphas, figure=fig,
                           hspace=0.55, wspace=0.3)

    r = 0
    for pi in range(num_pairs):
        for ai, alpha in enumerate(alphas):
            ax = fig.add_subplot(gs[r, ai])
            plot_label_targets(ax, data[ai][pi]["w0"], data[ai][pi]["w1"],
                               alpha, pi)
        r += 1

    for ai, alpha in enumerate(alphas):
        ax = fig.add_subplot(gs[r, ai])
        plot_ldr_histograms(ax, data[ai], alpha)
    r += 1

    for pi in range(num_pairs):
        for ai, alpha in enumerate(alphas):
            ax = fig.add_subplot(gs[r, ai])
            plot_pca(ax, data[ai][pi]["pstar"], data[ai][pi]["p0"],
                     data[ai][pi]["p1"], alpha, pi, n_plot=2000)
        r += 1

    ax_kl = fig.add_subplot(gs[r, 0:max(1, n_alphas // 2)])
    plot_kl_scatter(ax_kl, data, alphas)
    ax_ldr = fig.add_subplot(gs[r, max(1, n_alphas // 2):n_alphas])
    plot_ldr_stats(ax_ldr, data, alphas)

    fig_dir = Path(config["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "datagen_diagnostic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def stack_log_p_y(flow, points, steps, device, chunk_size=250):
    """compute log p(z | y=k) at points for k=0..K-1 via cond_flow backward ODE.

    args:
        flow: ClassCondVelocityMLP on device, eval mode.
        points: [N, D] tensor.
        steps: ODE steps for log_prob_class_cond.
        device: torch device.
        chunk_size: vmap chunk size for divergence (smaller than mnist's
                    500 because the 64-d jacobian is bigger).

    returns:
        [N, K] cpu tensor of log densities.
    """
    points = points.to(device)
    n = points.shape[0]
    out = torch.zeros(n, K)
    for k in range(K):
        y_k = torch.full((n,), k, dtype=torch.long, device=device)
        out[:, k] = log_prob_class_cond(
            flow, points, y_k,
            steps=steps, device=str(device), chunk_size=chunk_size,
        ).cpu()
    return out


def compute_heavy(data, config, alphas, num_pairs, n_eval, steps):
    """compute per-pair latent KL and pstar log-densities for QQ.

    inputs:
        data: nested dict from load_all_pairs.
        config: experiment config dict (data_dir/ckpt_dir already expanded).
        alphas, num_pairs: experimental grid sizes.
        n_eval: subsample size per pair-side for KL estimation.
        steps: ODE steps for log_prob_class_cond.

    returns:
        dict with kl_p0_p1, kl_p1_p0 [n_alphas, num_pairs] arrays and
        log_p0_pstar, log_p1_pstar dict[(ai,pi)] -> [n_eval] tensor for QQ.

    procedure:
      1. load global cond_flow + pstar log_p_y at the current flowhash.
      2. subsample log_p_y_pstar to n_eval (deterministic seed 42).
      3. for each pair, subsample p0_samples and p1_samples to n_eval and
         stack across pairs into one batch each.
      4. run cond_flow log_prob over the K classes on each stacked batch.
      5. mix per pair via logsumexp(log w_a + log p_y) to get log p_a at
         p0/p1 samples; estimate KL(p_a || p_b) by mean log-ratio.
      6. mix log_p_y_pstar with log w_0/log w_1 per pair for QQ panel.
    """
    device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("warning: cuda not available, falling back to cpu")
        device_str = "cpu"
    device = torch.device(device_str)
    ckpt_dir = config["ckpt_dir"]
    data_dir = config["data_dir"]

    flow = ClassCondVelocityMLP(
        latent_dim=config["latent_dim"],
        num_classes=K,
        hidden_dim=config["cond_flow_hidden_dim"],
    )
    ckpt_path = f"{ckpt_dir}/cond_flow.pt"
    flow.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    flow.to(device).eval()

    fh = flow_state_hash(ckpt_path)
    log_p_y_path = f"{data_dir}/log_p_y.{fh}.pt"
    if not Path(log_p_y_path).exists():
        raise FileNotFoundError(
            f"missing {log_p_y_path}; run step0 --mode log_p_y first")
    log_p_y_pstar_full = torch.load(log_p_y_path, map_location="cpu", weights_only=False)  # [N, K]
    rng_pstar = np.random.RandomState(42)
    pstar_idx = rng_pstar.choice(log_p_y_pstar_full.shape[0],
                                 n_eval, replace=False)
    log_p_y_pstar = log_p_y_pstar_full[pstar_idx]  # [n_eval, K]

    p0_chunks, p1_chunks = [], []
    for ai in range(len(alphas)):
        for pi in range(num_pairs):
            d = data[ai][pi]
            r = np.random.RandomState(42 + ai * 1000 + pi)
            i0 = r.choice(len(d["p0"]), n_eval, replace=False)
            i1 = r.choice(len(d["p1"]), n_eval, replace=False)
            p0_chunks.append(torch.from_numpy(d["p0"][i0]).float())
            p1_chunks.append(torch.from_numpy(d["p1"][i1]).float())
    p0_stack = torch.cat(p0_chunks, dim=0)
    p1_stack = torch.cat(p1_chunks, dim=0)

    print(f"computing log_p_y at p0_samples (N={len(p0_stack)})...")
    log_p_y_p0 = stack_log_p_y(flow, p0_stack, steps, device)
    print(f"computing log_p_y at p1_samples (N={len(p1_stack)})...")
    log_p_y_p1 = stack_log_p_y(flow, p1_stack, steps, device)

    n_a = len(alphas)
    kl_fwd = np.zeros((n_a, num_pairs))
    kl_rev = np.zeros((n_a, num_pairs))
    log_p0_pstar, log_p1_pstar = {}, {}

    for ai, alpha in enumerate(alphas):
        for pi in range(num_pairs):
            d = data[ai][pi]
            log_w0 = torch.log(torch.clamp(
                torch.from_numpy(d["w0"]).float(), min=1e-10))
            log_w1 = torch.log(torch.clamp(
                torch.from_numpy(d["w1"]).float(), min=1e-10))

            flat = ai * num_pairs + pi
            sl = slice(flat * n_eval, (flat + 1) * n_eval)

            lp_y_at_p0 = log_p_y_p0[sl]
            lp0_at_p0 = torch.logsumexp(log_w0.unsqueeze(0) + lp_y_at_p0, dim=1)
            lp1_at_p0 = torch.logsumexp(log_w1.unsqueeze(0) + lp_y_at_p0, dim=1)

            lp_y_at_p1 = log_p_y_p1[sl]
            lp0_at_p1 = torch.logsumexp(log_w0.unsqueeze(0) + lp_y_at_p1, dim=1)
            lp1_at_p1 = torch.logsumexp(log_w1.unsqueeze(0) + lp_y_at_p1, dim=1)

            kl_fwd[ai, pi] = (lp0_at_p0 - lp1_at_p0).mean().item()
            kl_rev[ai, pi] = (lp1_at_p1 - lp0_at_p1).mean().item()

            log_p0_pstar[(ai, pi)] = torch.logsumexp(
                log_w0.unsqueeze(0) + log_p_y_pstar, dim=1)
            log_p1_pstar[(ai, pi)] = torch.logsumexp(
                log_w1.unsqueeze(0) + log_p_y_pstar, dim=1)

            print(f"heavy: alpha={alpha} pair={pi}  "
                  f"KL(p0||p1)={kl_fwd[ai, pi]:.3f}  "
                  f"KL(p1||p0)={kl_rev[ai, pi]:.3f}")

    return {
        "kl_p0_p1": kl_fwd,
        "kl_p1_p0": kl_rev,
        "log_p0_pstar": log_p0_pstar,
        "log_p1_pstar": log_p1_pstar,
    }


def plot_kl_figure(data, config, alphas, heavy):
    """heavy-mode figure: QQ grid + KL_cat-vs-latent + KL symmetry.

    layout matches the mnist sibling: rows 0..num_pairs-1 are QQ plots,
    last row is two wide summary panels.
    """
    num_pairs = config["num_pairs_per_alpha"]
    n_a = len(alphas)
    rows = num_pairs + 1
    fig = plt.figure(figsize=(4 * n_a, 2.5 * rows))
    gs = gridspec.GridSpec(rows, n_a, figure=fig, hspace=0.4, wspace=0.3)

    for pi in range(num_pairs):
        for ai, alpha in enumerate(alphas):
            ax = fig.add_subplot(gs[pi, ai])
            plot_qq(ax,
                    heavy["log_p0_pstar"][(ai, pi)],
                    heavy["log_p1_pstar"][(ai, pi)],
                    alpha, pi)

    fwd = heavy["kl_p0_p1"]
    rev = heavy["kl_p1_p0"]
    colors = [plt.cm.viridis(ai / max(1, n_a - 1)) for ai in range(n_a)]

    ax_corr = fig.add_subplot(gs[num_pairs, 0:max(1, n_a // 2)])
    for ai in range(n_a):
        for pi in range(num_pairs):
            ax_corr.scatter(data[ai][pi]["kl_weights"], fwd[ai, pi],
                            s=15, alpha=0.7, color=colors[ai])
    cat_all = [data[ai][pi]["kl_weights"]
               for ai in range(n_a) for pi in range(num_pairs)]
    lo = min(min(cat_all), fwd.min())
    hi = max(max(cat_all), fwd.max())
    ax_corr.plot([lo, hi], [lo, hi], "k--", alpha=0.3)
    ax_corr.set_xlabel("KL(w0 || w1)")
    ax_corr.set_ylabel("KL(p0 || p1) latent")
    ax_corr.set_title("categorical vs latent KL")

    ax_sym = fig.add_subplot(gs[num_pairs, max(1, n_a // 2):n_a])
    for ai in range(n_a):
        for pi in range(num_pairs):
            ax_sym.scatter(fwd[ai, pi], rev[ai, pi],
                           s=15, alpha=0.7, color=colors[ai])
    lo_s = min(fwd.min(), rev.min())
    hi_s = max(fwd.max(), rev.max())
    ax_sym.plot([lo_s, hi_s], [lo_s, hi_s], "k--", alpha=0.3)
    ax_sym.set_xlabel("KL(p0 || p1)")
    ax_sym.set_ylabel("KL(p1 || p0)")
    ax_sym.set_title("KL symmetry check")

    fig_dir = Path(config["figures_dir"])
    out = fig_dir / "datagen_kl_diagnostic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    print("\nalpha  pair  KL(w0||w1)  KL(p0||p1)  KL(p1||p0)  E_p*[LDR]")
    for ai, alpha in enumerate(alphas):
        for pi in range(config["num_pairs_per_alpha"]):
            mean_ldr = float(np.mean(data[ai][pi]["true_ldrs"]))
            print(f"{alpha:<5.2f}  {pi:<4d}  "
                  f"{data[ai][pi]['kl_weights']:<10.2f}  "
                  f"{fwd[ai, pi]:<10.2f}  {rev[ai, pi]:<10.2f}  "
                  f"{mean_ldr:<10.4f}")


def main():
    """top-level: load data, run lightweight + hardness, optionally heavy KL."""
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    config = expand_paths(config)
    alphas = config["alphas"]
    num_pairs = config["num_pairs_per_alpha"]
    data_dir = config["data_dir"]

    data = load_all_pairs(data_dir, alphas, num_pairs)

    plot_lightweight_figure(data, alphas, config)

    stats = compute_hardness(data, alphas, num_pairs)
    heavy_stats = None
    if args.compute_kl:
        heavy_stats = compute_heavy(
            data, config, alphas, num_pairs,
            n_eval=args.n_eval, steps=args.log_prob_steps,
        )

    aug = None
    if heavy_stats is not None:
        aug = {
            "kl_p0_p1": heavy_stats["kl_p0_p1"],
            "kl_p1_p0": heavy_stats["kl_p1_p0"],
        }
    print_hardness_table(stats, alphas, aug)
    plot_hardness_figure(stats, alphas, config, aug, K=K)

    if heavy_stats is not None:
        plot_kl_figure(data, config, alphas, heavy_stats)


if __name__ == "__main__":
    main()
