"""datagen diagnostic for DBpedia ELDR. Lightweight mode (default): weight bars, label-target bars, ldr histograms,
  pca, hardness boxplots. emits datagen_diagnostic.png and datagen_variance.png.

heavy mode (--compute-kl): per-pair latent KL via global cond_flow at p0/p1
  samples and cached log_p_y at pstar. emits datagen_kl_diagnostic.png with
  QQ-plot grid + KL_cat-vs-KL_latent scatter + KL symmetry check.
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ex.utils.diagnostics import (
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
from ex.utils.dbpedia_imbalance import (
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
                   default="ex/semisynth/dbpedia/config.yaml")
    p.add_argument("--compute-kl", action="store_true",
                   help="heavy mode: compute latent KL via cond flow")
    p.add_argument("--n-eval", type=int, default=500,
                   help="samples per pair-side for KL estimation (heavy mode)")
    p.add_argument("--log-prob-steps", type=int, default=100,
                   help="ODE steps for heavy-mode log_p_y (default 100)")
    p.add_argument("--skip-card", action="store_true",
                   help="skip the per-stratum data-card table + figure")
    p.add_argument("--skip-render", action="store_true",
                   help="skip the ground-truth mixture-weight sheet")
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
    """emit the lightweight diagnostics as per-section, pair-chunked files.

    replaces the old single mega-grid: label targets and pca become
    datagen_diagnostic_{section}_p{lo}-{hi}.png chunks; aggregate panels go
    to datagen_diagnostic_summary.png (see ex.utils.diagnostics helpers).

    args:
        data: nested dict from load_all_pairs (pstar/p0/p1 are 64-dim here).
        alphas: list of alpha values.
        config: config dict with figures_dir, num_pairs_per_alpha.
    """
    from ex.utils.diagnostics import plot_pair_section, plot_summary_figure
    num_pairs = config["num_pairs_per_alpha"]
    fig_dir = Path(config["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_pair_section(
        data, alphas, fig_dir, "weights",
        lambda ax, ai, alpha, pi: plot_label_targets(
            ax, data[ai][pi]["w0"], data[ai][pi]["w1"], alpha, pi),
        num_pairs)
    plot_pair_section(
        data, alphas, fig_dir, "pca",
        lambda ax, ai, alpha, pi: plot_pca(
            ax, data[ai][pi]["pstar"], data[ai][pi]["p0"], data[ai][pi]["p1"],
            alpha, pi, n_plot=2000),
        num_pairs)
    plot_summary_figure(data, alphas, fig_dir)


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
    from ex.utils.diagnostics import plot_pair_section
    num_pairs = config["num_pairs_per_alpha"]
    n_a = len(alphas)
    fig_dir = Path(config["figures_dir"])

    # qq grid: pair-chunked files instead of one mega-grid
    plot_pair_section(
        data, alphas, fig_dir, "qq",
        lambda ax, ai, alpha, pi: plot_qq(
            ax, heavy["log_p0_pstar"][(ai, pi)],
            heavy["log_p1_pstar"][(ai, pi)], alpha, pi),
        num_pairs, row_h=2.5, stem="datagen_kl")

    fwd = heavy["kl_p0_p1"]
    rev = heavy["kl_p1_p0"]
    colors = [plt.cm.viridis(ai / max(1, n_a - 1)) for ai in range(n_a)]

    fig = plt.figure(figsize=(4 * n_a, 3))
    gs = gridspec.GridSpec(1, n_a, figure=fig, wspace=0.3)
    ax_corr = fig.add_subplot(gs[0, 0:max(1, n_a // 2)])
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

    ax_sym = fig.add_subplot(gs[0, max(1, n_a // 2):n_a])
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

    out = fig_dir / "datagen_kl_summary.png"
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


def run_data_card(data, alphas, num_pairs, config):
    """per-stratum data card: dimensionality / multimodality / irregularity.

    dbpedia adds embedding-space metrics (anisotropy, hubness) on the 64-d
    SBERT-PCA codes. emits data_card.{md,tex,png,pdf} into figures_dir.
    """
    from ex.utils import data_card as dc
    names = ['twonn_id', 'part_ratio', 'mean_cos', 'hubness', 'gmm_modes',
             'eff_modes_w0', 'eff_modes_w1', 'lip_q90', 'hill_tail']
    vals = {m: [[] for _ in alphas] for m in names}
    for ai in range(len(alphas)):
        for pi in range(num_pairs):
            d = data[ai][pi]
            X, ldr = d['pstar'], d['true_ldrs']
            vals['twonn_id'][ai].append(dc.twonn_id(X))
            vals['part_ratio'][ai].append(dc.participation_ratio(X))
            vals['mean_cos'][ai].append(dc.mean_cos(X))
            vals['hubness'][ai].append(dc.hubness(X))
            vals['gmm_modes'][ai].append(dc.gmm_modes(X))
            vals['eff_modes_w0'][ai].append(dc.eff_modes(d['w0']))
            vals['eff_modes_w1'][ai].append(dc.eff_modes(d['w1']))
            vals['lip_q90'][ai].append(dc.lip_q(X, ldr))
            vals['hill_tail'][ai].append(dc.hill_tail(ldr))
    fig_dir = config['figures_dir']
    dc.write_card(str(Path(fig_dir) / 'data_card'),
                  [f'alpha={a:g}' for a in alphas], vals,
                  title='dbpedia data card (pstar SBERT-PCA codes) -- med [q1, q3] over pairs')
    dc.plot_metric_boxes(vals, alphas, sweep_name='alpha',
                         out_dir=fig_dir, prefix='data_card')


def run_weight_render(data, alphas, config, n_pairs=2):
    """ground-truth rendering: exact class-mixture weights (w0 vs w1) per cell.

    dbpedia samples are sentence embeddings with no decoder, so the honest
    human-viewable ground truth is the mixture composition itself.
    """
    K = len(DBPEDIA_LABEL_NAMES)
    x = np.arange(K)
    out = Path(config['figures_dir'])
    # one file per alpha so figures stay paper-sized and composable
    for ai, alpha in enumerate(alphas):
        fig, axes = plt.subplots(n_pairs, 1, figsize=(7.5, 1.9 * n_pairs),
                                 sharex=True, squeeze=False)
        for pi in range(n_pairs):
            ax = axes[pi, 0]
            d = data[ai][pi]
            ax.bar(x - 0.2, d['w0'], width=0.4, label='w0 (p0)', color='#4878d0')
            ax.bar(x + 0.2, d['w1'], width=0.4, label='w1 (p1)', color='#d65f5f')
            ax.set_ylabel(f'#{pi}', fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)
            if pi == 0:
                ax.legend(fontsize=9, ncol=2)
        axes[-1, 0].set_xticks(x)
        axes[-1, 0].set_xticklabels([n[:10] for n in DBPEDIA_LABEL_NAMES],
                                    rotation=40, ha='right', fontsize=9)
        fig.tight_layout()
        tag = f'alpha_{alpha:g}'.replace('.', 'p')
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'datagen_gt_weights_{tag}.{ext}', dpi=150,
                        bbox_inches='tight')
        plt.close(fig)
        print(f'saved datagen_gt_weights_{tag}.{{pdf,png}}')


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
    plot_hardness_figure(stats, alphas, config, aug, K=14)

    if not args.skip_card:
        run_data_card(data, alphas, num_pairs, config)
    if not args.skip_render:
        run_weight_render(data, alphas, config)

    if heavy_stats is not None:
        plot_kl_figure(data, config, alphas, heavy_stats)


if __name__ == "__main__":
    main()
