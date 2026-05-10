"""datagen diagnostic for the cond_flow experiment.

lightweight mode (default): weight bars, ldr histograms, pca, hardness boxplots.
  emits datagen_diagnostic.png and datagen_variance.png.
heavy mode (--compute-kl): cond-flow-specific latent KL via global cond_flow
  and cached log_p_y at pstar. emits datagen_kl_diagnostic.png with
  QQ-plot grid + KL_cat-vs-KL_latent scatter + KL symmetry check.

reuses lightweight + hardness functions from the old per-pair-flow sibling
(experiments.mnist_eldr_estimation.diagnostic_datagen). only the heavy mode
is rewritten because pstar/p0/p1 here live in the global VAE's latent space,
so per-pair VAE encoding is not needed.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml

from experiments.mnist_eldr_estimation.diagnostic_datagen import (
    load_all_pairs,
    plot_lightweight_figure,
    compute_hardness,
    print_hardness_table,
    plot_hardness_figure,
    plot_qq,
)
from experiments.utils.mnist_imbalance import get_mnist_dataset
from src.models.flow import ClassCondVelocityMLP, log_prob_class_cond


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
                   default="experiments/mnist_eldr_cond_flow/config.yaml")
    p.add_argument("--compute-kl", action="store_true",
                   help="heavy mode: compute latent KL via cond flow")
    p.add_argument("--n-eval", type=int, default=500,
                   help="samples per pair-side for KL estimation (heavy mode)")
    p.add_argument("--log-prob-steps", type=int, default=100,
                   help="ODE steps for heavy-mode log_p_y (default 100)")
    return p.parse_args(args)


def stack_log_p_y(flow, points, steps, device, chunk_size=500):
    """compute log p(z | y=k) at points for k=0..9 via cond_flow backward ODE.

    args:
        flow: ClassCondVelocityMLP on device, eval mode.
        points: [N, D] tensor.
        steps: ODE steps for log_prob_class_cond.
        device: torch device.
        chunk_size: vmap chunk size for divergence.

    returns:
        [N, 10] cpu tensor of log densities.

    procedure: loop over k=0..9, fill column k with log_prob_class_cond(z, y=k).
    """
    points = points.to(device)
    n = points.shape[0]
    out = torch.zeros(n, 10)
    for k in range(10):
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
        config: experiment config dict.
        alphas, num_pairs: experimental grid sizes.
        n_eval: subsample size per pair-side for KL estimation.
        steps: ODE steps for log_prob_class_cond.

    returns:
        dict with:
          kl_p0_p1, kl_p1_p0: [n_alphas, num_pairs] numpy arrays.
          log_p0_pstar, log_p1_pstar: dict[(ai,pi)] -> [n_eval] tensor for QQ.

    procedure:
      1. load global cond_flow.
      2. load cached log_p_y at pstar; subsample to n_eval (deterministic seed).
      3. for each pair, subsample p0_samples and p1_samples to n_eval.
         stack across pairs into one big batch each.
      4. compute log p(z|y=k) for k=0..9 on the stacked p0 batch and the
         stacked p1 batch. these are the only two cond-flow ODE passes.
      5. for each pair: mix with log w0, log w1 via logsumexp to get
         log p_0, log p_1 at p0_samples and p1_samples. compute KL via
         monte carlo: KL(p0||p1) = mean(log p_0(z) - log p_1(z)) over p0_samples.
      6. for QQ: mix cached log_p_y at pstar with log w0, log w1 per pair.
    """
    device = torch.device(config.get("device",
                                     "cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = config["ckpt_dir"]

    flow = ClassCondVelocityMLP(
        latent_dim=config["latent_dim"],
        num_classes=10,
        hidden_dim=config["cond_flow_hidden_dim"],
    )
    flow.load_state_dict(torch.load(f"{ckpt_dir}/cond_flow.pt", map_location="cpu"))
    flow.to(device).eval()

    log_p_y_pstar_full = torch.load(
        f"{config['data_dir']}/log_p_y.pt", map_location="cpu"
    )  # [N_full, 10]
    rng_pstar = np.random.RandomState(42)
    pstar_idx = rng_pstar.choice(log_p_y_pstar_full.shape[0], n_eval, replace=False)
    log_p_y_pstar = log_p_y_pstar_full[pstar_idx]  # [n_eval, 10]

    # subsample p0/p1 per pair, stack across pairs
    p0_chunks, p1_chunks = [], []
    for ai in range(len(alphas)):
        for pi in range(num_pairs):
            d = data[ai][pi]
            r = np.random.RandomState(42 + ai * 1000 + pi)
            i0 = r.choice(len(d["p0"]), n_eval, replace=False)
            i1 = r.choice(len(d["p1"]), n_eval, replace=False)
            p0_chunks.append(torch.from_numpy(d["p0"][i0]).float())
            p1_chunks.append(torch.from_numpy(d["p1"][i1]).float())
    p0_stack = torch.cat(p0_chunks, dim=0)  # [n_alphas*num_pairs*n_eval, D]
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
                torch.from_numpy(d["w0"]).float(), min=1e-10))  # [10]
            log_w1 = torch.log(torch.clamp(
                torch.from_numpy(d["w1"]).float(), min=1e-10))

            flat = ai * num_pairs + pi
            sl = slice(flat * n_eval, (flat + 1) * n_eval)

            lp_y_at_p0 = log_p_y_p0[sl]  # [n_eval, 10]
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

    inputs:
        data: nested dict from load_all_pairs.
        config: experiment config dict.
        alphas: list of alpha values.
        heavy: dict from compute_heavy.

    layout (matches old expt's datagen_kl_diagnostic.png):
        rows 0..num_pairs-1: QQ plots, one per (alpha, pair).
        row num_pairs:
            left half:  KL(w0||w1) (x) vs KL(p0||p1) latent (y) with y=x ref.
            right half: KL(p0||p1) (x) vs KL(p1||p0) (y) with y=x ref.
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

    ax_corr = fig.add_subplot(gs[num_pairs, 0:n_a // 2])
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

    ax_sym = fig.add_subplot(gs[num_pairs, n_a // 2:n_a])
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
            print(f"{alpha:<5.1f}  {pi:<4d}  "
                  f"{data[ai][pi]['kl_weights']:<10.2f}  "
                  f"{fwd[ai, pi]:<10.2f}  {rev[ai, pi]:<10.2f}  "
                  f"{mean_ldr:<10.4f}")


def main():
    """top-level: load data, run lightweight + hardness, optionally heavy KL."""
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    alphas = config["alphas"]
    num_pairs = config["num_pairs_per_alpha"]
    data_dir = config["data_dir"]

    data = load_all_pairs(data_dir, alphas, num_pairs)
    dataset = get_mnist_dataset(root="./data", train=True)

    plot_lightweight_figure(data, dataset, alphas, config)

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
    plot_hardness_figure(stats, alphas, config, aug, K=10)

    if heavy_stats is not None:
        plot_kl_figure(data, config, alphas, heavy_stats)


if __name__ == "__main__":
    main()
