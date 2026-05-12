"""shared diagnostic utilities for datagen analysis: plotting, hardness metrics, data loading."""
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.stats
import torch
from pathlib import Path
from sklearn.decomposition import PCA

from experiments.utils.mnist_imbalance import get_mnist_dataset, subsample_mnist


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


def plot_lightweight_figure(data, dataset, alphas, config, plot_weight_bars_fn, plot_class_counts_fn):
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
        plot_weight_bars_fn: callable(ax, w0, w1, alpha, pair_idx) for weight bars
        plot_class_counts_fn: callable(ax, dataset, w0, w1, alpha, pair_idx) for class counts
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
            plot_weight_bars_fn(ax, data[ai][pi]["w0"], data[ai][pi]["w1"], alpha, pi)
        r += 1

    # class counts: num_pairs rows
    for pi in range(num_pairs):
        for ai, alpha in enumerate(alphas):
            ax = fig.add_subplot(gs[r, ai])
            plot_class_counts_fn(ax, dataset, data[ai][pi]["w0"], data[ai][pi]["w1"], alpha, pi)
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


def print_hardness_table(stats, alphas, heavy_stats=None):
    """print per-alpha summary of hardness metrics.

    for each metric, shows median, iqr, mean, std across pairs.
    """
    metrics = list(stats.keys())
    if heavy_stats is not None:
        metrics += list(heavy_stats.keys())

    print("\n" + "=" * 80)
    print("HARDNESS VARIANCE SUMMARY (per alpha, across pairs)")
    print("=" * 80)

    for name in metrics:
        vals = stats[name] if name in stats else heavy_stats[name]
        print(f"\n--- {name} ---")
        print(f"{'alpha':<8} {'median':>8} {'IQR':>8} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
        for ai, alpha in enumerate(alphas):
            row = vals[ai]
            q25, q50, q75 = np.percentile(row, [25, 50, 75])
            print(f"{alpha:<8.1f} {q50:>8.3f} {q75-q25:>8.3f} {np.mean(row):>8.3f} "
                  f"{np.std(row):>8.3f} {np.min(row):>8.3f} {np.max(row):>8.3f}")


def plot_hardness_figure(stats, alphas, config, heavy_stats=None, K=None):
    """box plot grid of hardness metrics per alpha.

    saves to figures_dir/datagen_variance.png. when K is provided, the kl_cat
    panel additionally shows analytical references derived in
    notes/semisynth_appendix.tex (appendix:weight-kl-bounds): the Dirichlet-mean
    pointwise lower bound E[ell(w)] (always valid, all alpha > 0), and the
    outer-Jensen upper bound on E[KL] (valid only for alpha > 1).

    args:
        stats: dict[name -> [n_alphas, num_pairs]] from compute_hardness.
        alphas: list of alpha values.
        config: experiment config dict (for figures_dir).
        heavy_stats: optional dict[name -> [n_alphas, num_pairs]] of extra
                     metrics (e.g., latent KL).
        K: number of classes (10 for MNIST, 14 for DBpedia). when None, no
           analytical overlay is drawn.
    """
    from experiments.utils.mnist_imbalance import bound_moments, expected_kl_jensen_ub

    all_metrics = dict(stats)
    if heavy_stats is not None:
        all_metrics.update(heavy_stats)

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
                        tick_labels=[str(a) for a in alphas],
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

        # analytical overlay on the kl_cat panel (closed-form Dirichlet bounds)
        if name == "kl_cat" and K is not None:
            lb_label_drawn = False
            ub_label_drawn = False
            for ai, alpha in enumerate(alphas):
                mom = bound_moments(alpha, K)
                ub_jensen = expected_kl_jensen_ub(alpha, K)
                xc = ai + 1
                ax.hlines(mom["E_ell"], xc - 0.35, xc + 0.35,
                          colors="tab:green", linestyles="--", linewidth=2,
                          zorder=6,
                          label="E[ell] (LB)" if not lb_label_drawn else None)
                lb_label_drawn = True
                if np.isfinite(ub_jensen):
                    ax.hlines(ub_jensen, xc - 0.35, xc + 0.35,
                              colors="tab:purple", linestyles="--", linewidth=2,
                              zorder=6,
                              label="Jensen UB on E[KL]"
                                    if not ub_label_drawn else None)
                    ub_label_drawn = True
            if lb_label_drawn or ub_label_drawn:
                ax.legend(fontsize=7, loc="best")

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
