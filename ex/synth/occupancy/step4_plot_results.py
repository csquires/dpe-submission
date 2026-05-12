"""
Step 4: Plot Results for SMODICE ELDR Estimation

Line plots of MAE / ELDR error vs K1 for each K2 panel, with translucent
fill_between error bands (±1 SE). Colors match mnist_eldr_estimation step4.
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

# Colors matching mnist_eldr_estimation/step4_plot_results.py for shared methods.
# New smodice-only methods get distinct colors from the extended palette.
METHOD_COLORS = {
    # shared with mnist
    "TriangularMDRE":           "blue",
    "MultiHeadTriangularTDRE":  "orange",
    "VFM":                      "green",
    "TSM":                      "red",
    "CTSM":                     "purple",
    "TriangularCTSM_V1":        "brown",
    "BDRE":                     "cyan",
    "MDRE_15":                  "magenta",
    "TriangularTSM":            "gray",
    # smodice-specific
    "MultiHeadTDRE":            "#f7b6d2",  # light pink
    "FMDRE":                    "#ff9896",  # light red
    "FMDRE_S2":                 "#aec7e8",  # light blue
    "TriangularFMDRE":          "#c5b0d5",  # light purple
    "TriangularCTSM_V2":        "#c7c7c7",  # light gray
    "TriangularCTSM_V3":        "#9edae5",  # light cyan
    "TriangularVFM_V1":         "#bcbd22",  # olive
    "TriangularVFM_V2":         "#393b79",  # dark blue
    "TriangularVFM_V3":         "#637939",  # dark green
}

ERROR_BAND_ALPHA = 0.2
FONT_SIZE = 12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="ex/synth/occupancy/config.yaml")
    p.add_argument("--winners", default="scratch/gold_winners/winners.smodice_eldr_estimation.yaml")
    p.add_argument("--metric", default=None, choices=["pointwise_mae", "eldr_err"],
                   help="metric to plot; omit to plot both")
    return p.parse_args()


def load_summary(h5_path, metric):
    """Return dict: method -> (mean[n_k1, n_k2], se[n_k1, n_k2])."""
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


def plot_metric(results, k1_values, k2_values, metric, figures_dir):
    n_k2 = len(k2_values)
    methods = sorted(results.keys())

    style_path = "full-width.mplstyle"
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        sns.set_style("whitegrid")
        matplotlib.rcParams["font.size"] = FONT_SIZE

    fig, axes = plt.subplots(
        1, n_k2,
        figsize=(5 * n_k2, 4),
        constrained_layout=False,
        sharey=True,
    )
    fig.subplots_adjust(bottom=0.32, wspace=0.08)

    if n_k2 == 1:
        axes = [axes]

    for k2_idx, ax in enumerate(axes):
        k2_val = k2_values[k2_idx]
        for method in methods:
            mean, se = results[method]
            y = mean[:, k2_idx]
            y_lo = y - se[:, k2_idx]
            y_hi = y + se[:, k2_idx]
            color = METHOD_COLORS.get(method, "#888888")
            ax.plot(k1_values, y, label=method, color=color,
                    linewidth=1.5, marker="o", markersize=4)
            ax.fill_between(k1_values, y_lo, y_hi, color=color, alpha=ERROR_BAND_ALPHA)

        ax.set_xlabel("K1", fontsize=FONT_SIZE)
        ax.set_title(f"K2 = {k2_val:.1f}", fontsize=FONT_SIZE)
        ax.set_xticks(k1_values)
        ax.set_xticklabels([f"{k:.1f}" for k in k1_values])
        ax.grid(True, alpha=0.3)
        if k2_idx == 0:
            ylabel = "Pointwise LDR MAE" if metric == "pointwise_mae" else "ELDR Error"
            ax.set_ylabel(ylabel, fontsize=FONT_SIZE)

    # shared legend below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=6,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
        framealpha=0.9,
    )

    title = "Pointwise LDR MAE" if metric == "pointwise_mae" else "ELDR Error"
    fig.suptitle(f"SMODICE ELDR Estimation — {title}", fontsize=13, fontweight="bold")

    os.makedirs(figures_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(figures_dir, f"smodice_{metric}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
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
    k2_values = config["kl_targets"]["k2_values"]
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
        plot_metric(results, k1_values, k2_values, metric, figures_dir)
        print(f"Plotting box plot for {metric}...")
        plot_boxplot(h5_path, metric, figures_dir)

    print(f"Done. Figures in: {figures_dir}")


if __name__ == "__main__":
    main()
