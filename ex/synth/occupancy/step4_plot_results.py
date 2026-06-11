"""
Step 4: Plot Results for SMODICE ELDR Estimation

Line plots of MAE / ELDR error vs K1 for each beta panel, with translucent
fill_between error bands. One line figure per METHOD_GROUPS family,
styled via ex.utils.plot_style

Box plots (unchanged) are also produced per metric.
"""
import argparse
import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

matplotlib.use("Agg")

from ex.utils.plot_style import (
    apply as apply_style,
    style_for,
    METHOD_GROUPS,
    GROUP_LABEL,
    ERROR_BAND_ALPHA,
)

FONT_SIZE = 12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="ex/synth/occupancy/config.yaml")
    p.add_argument("--winners", default="scratch/gold_winners/winners.smodice_eldr_estimation.yaml")
    p.add_argument("--metric", default=None, choices=["pointwise_mae", "eldr_err"],
                   help="metric to plot; omit to plot both")
    return p.parse_args()


def load_summary(h5_path, metric):
    """Return dict: method -> (mean[n_k1, n_beta], se[n_k1, n_beta])."""
    results = {}
    prefix = f"{metric}_"
    suffix = "_mean"
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            if not (key.startswith(prefix) and key.endswith(suffix)):
                continue
            method = key[len(prefix):-len(suffix)]
            se_key = f"{prefix}{method}_se"
            if se_key not in f:
                continue
            results[method] = (f[key][:], f[se_key][:])
    return results


def plot_metric(results, k1_values, beta_values, metric, figures_dir):
    """one figure per METHOD_GROUPS family; within each, n_beta sharey subplots
    plotting metric mean vs K1 with +/- SE bands. shared y-range across all
    family figures so they are visually comparable side-by-side.
    """
    n_beta = len(beta_values)
    apply_style()

    # compute shared y_max across all methods present, all betas, mean + SE.
    y_max = 0.0
    for m, (mean, se) in results.items():
        upper = np.nanmax(mean + se)
        if np.isfinite(upper):
            y_max = max(y_max, float(upper))
    y_max *= 1.05 if y_max > 0 else 1.0

    metric_label = "Pointwise LDR MAE" if metric == "pointwise_mae" else "ELDR Error"
    os.makedirs(figures_dir, exist_ok=True)

    for group, members in METHOD_GROUPS.items():
        present = [m for m in members if m in results]
        if not present:
            continue
        fig, axes = plt.subplots(
            1, max(1, n_beta),
            figsize=(5 * max(1, n_beta), 4),
            sharey=True,
        )
        if n_beta == 1:
            axes = [axes]
        for beta_idx, ax in enumerate(axes):
            for m in present:
                mean, se = results[m]
                y = mean[:, beta_idx]
                lo = y - se[:, beta_idx]
                hi = y + se[:, beta_idx]
                kw = style_for(m)
                ax.plot(k1_values, y, label=m, **kw)
                ax.fill_between(k1_values, lo, hi,
                                color=kw["color"], alpha=ERROR_BAND_ALPHA, linewidth=0)
            ax.set_xlabel("K1")
            ax.set_title(f"beta = {beta_values[beta_idx]:.2f}")
            ax.set_xticks(k1_values)
            ax.set_xticklabels([f"{k:.1f}" for k in k1_values])
            ax.set_ylim(0, y_max)
            if beta_idx == 0:
                ax.set_ylabel(metric_label)
            ax.legend(loc="best")
        fig.suptitle(f"{GROUP_LABEL[group]} — {metric_label}")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = os.path.join(figures_dir, f"smodice_{metric}_{group}.{ext}")
            fig.savefig(path)
            print(f"Saved: {path}")
        plt.close(fig)


COLOR_NON_TRI = "#4878d0"   # steel blue
COLOR_TRI     = "#ee854a"   # coral orange

# Each family: (base_method_or_None, [triangular_variants])
# Families are shown as grouped adjacent boxes; base=blue, variants=orange.
METHOD_FAMILIES = [
    ("BDRE",           []),
    ("CTSM",           ["TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3"]),
    ("FMDRE",          ["TriangularFMDRE"]),
    ("FMDRE_S2",       []),
    ("MDRE_15",        ["TriangularMDRE"]),
    ("MultiHeadTDRE",  ["MultiHeadTriangularTDRE"]),
    ("TSM",            ["TriangularTSM"]),
    ("VFM",            ["TriangularVFM_V1", "TriangularVFM_V2", "TriangularVFM_V3"]),
]


def plot_boxplot(h5_path, metric, figures_dir):
    """Superimposed box plot: triangular variant(s) drawn on top of base method
    at the same x-position. Base = wide blue box; variants = narrower orange box(es).
    Families with no triangular variant shown as blue box only.
    """
    with h5py.File(h5_path, "r") as f:
        data = {
            k[len(metric) + 1:-len("_seed_values")]: f[k][:]
            for k in f.keys()
            if k.startswith(f"{metric}_") and k.endswith("_seed_values")
        }

    # distinct colors for V1/V2/V3 — orange, green, red (maximally separable)
    TRI_COLORS = ["#ff7f0e", "#2ca02c", "#d62728"]

    style_path = "full-width.mplstyle"
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        sns.set_style("whitegrid")
        matplotlib.rcParams["font.size"] = FONT_SIZE

    # count families that have at least one member in data
    valid_families = [(b, vs) for b, vs in METHOD_FAMILIES
                      if (b and b in data) or any(v in data for v in vs)]
    n_fam = len(valid_families)

    fig, ax = plt.subplots(figsize=(max(10, n_fam * 1.4), 5),
                           layout="constrained")


    xticks, xlabels = [], []

    def _draw_box(values, pos, width, color, alpha, zorder):
        bp = ax.boxplot(
            values,
            positions=[pos],
            widths=width,
            patch_artist=True,
            notch=False,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.2,
                            markerfacecolor=color, markeredgecolor=color,
                            linestyle="none"),
            medianprops=dict(color="black", linewidth=1.5, zorder=zorder + 1),
            whiskerprops=dict(linewidth=1.0, zorder=zorder),
            capprops=dict(linewidth=1.0, zorder=zorder),
            manage_ticks=False,
            zorder=zorder,
        )
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(alpha)
        bp["boxes"][0].set_zorder(zorder)

    for fam_idx, (base, variants) in enumerate(valid_families):
        pos = fam_idx + 1
        label = (base or variants[0]).replace("Triangular", "Tri").replace("MultiHead", "MH")
        xticks.append(pos)
        xlabels.append(label)

        # base box: wide, blue, behind
        if base and base in data:
            _draw_box(data[base], pos, width=0.65, color=COLOR_NON_TRI, alpha=0.65, zorder=2)

        # triangular variant(s): narrower, orange shade(s), on top
        avail = [v for v in variants if v in data]
        for vi, v in enumerate(avail):
            color = TRI_COLORS[vi % len(TRI_COLORS)]
            _draw_box(data[v], pos, width=0.38, color=color, alpha=0.75, zorder=3 + vi)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=10)
    ax.set_xlim(0.4, n_fam + 0.6)
    ylabel = "Pointwise LDR MAE" if metric == "pointwise_mae" else "ELDR Error"
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLOR_NON_TRI, alpha=0.65, label="Base (non-triangular)"),
        Patch(facecolor=TRI_COLORS[0], alpha=0.75, label="Triangular V1 (orange)"),
        Patch(facecolor=TRI_COLORS[1], alpha=0.75, label="Triangular V2 (green)"),
        Patch(facecolor=TRI_COLORS[2], alpha=0.75, label="Triangular V3 (red)"),
    ]
    # place legend outside top-right to avoid covering boxes
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0, framealpha=0.9)

    title = "Pointwise LDR MAE" if metric == "pointwise_mae" else "ELDR Error"
    ax.set_title(
        f"SMODICE — {title}: triangular variants superimposed on base method",
        fontsize=12,
    )

    os.makedirs(figures_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(figures_dir, f"smodice_{metric}_boxplot.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    k1_values = config["kl_targets"]["k1_values"]
    beta_values = config["kl_targets"]["beta_values"]
    encoding = config["encoding"]["type"]
    sigma = config["encoding"]["sigma"]
    processed_dir = config["processed_results_dir"]
    figures_dir = config["figures_dir"]

    h5_path = os.path.join(processed_dir, encoding, f"sigma_{sigma:.3f}", "summary.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"summary.h5 not found: {h5_path}\nRun step3 first.")

    metrics = ["pointwise_mae", "eldr_err"] if args.metric is None else [args.metric]
    for metric in metrics:
        results = load_summary(h5_path, metric)
        if not results:
            print(f"No results for {metric}; skipping.")
            continue
        print(f"Plotting line plot for {metric} ({len(results)} methods)...")
        plot_metric(results, k1_values, beta_values, metric, figures_dir)
        print(f"Plotting box plot for {metric}...")
        plot_boxplot(h5_path, metric, figures_dir)

    print(f"Done. Figures in: {figures_dir}")


if __name__ == "__main__":
    main()
