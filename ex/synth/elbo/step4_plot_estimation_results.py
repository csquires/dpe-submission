"""
Step 4: Plot Results for ELBO Estimation

Line plots of MAE vs design_eig_percentage for each alpha panel,
with translucent fill_between error bands (±1 SE).
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

METHOD_COLORS = {
    "BDRE":                    "#1f77b4",   # blue
    "MDRE_15":                 "#2ca02c",   # green
    "MultiHeadTDRE":           "#ff7f0e",   # orange
    "MultiHeadTriangularTDRE": "#17becf",   # cyan
    "TSM":                     "#d62728",   # red
    "VFM":                     "#9467bd",   # purple
    "TriangularMDRE":          "#aec7e8",   # light blue
    "TriangularTSM":           "#ff9897",   # light red
    "CTSM":                    "#e377c2",   # pink
    "TriangularCTSM_V1":       "#e377c2",   # pink
    "TriangularCTSM_V2":       "#f7b6d2",   # light pink
    "TriangularCTSM_V3":       "#c5b0d5",   # light purple
    "FMDRE":                   "#7f7f7f",   # gray
    "FMDRE_S2":                "#c7c7c7",   # light gray
    "TriangularFMDRE":         "#b5a8c0",   # muted purple-gray
    "TriangularVFM_V1":        "#9467bd",   # purple
    "TriangularVFM_V2":        "#c5b0d5",   # light purple
    "TriangularVFM_V3":        "#bcbd22",   # olive
}

LEGEND_ORDER = [
    "BDRE", "MDRE_15", "MultiHeadTDRE", "MultiHeadTriangularTDRE",
    "TSM", "TriangularTSM",
    "VFM", "TriangularVFM_V1", "TriangularVFM_V2", "TriangularVFM_V3",
    "TriangularMDRE",
    "CTSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
    "FMDRE", "FMDRE_S2", "TriangularFMDRE",
]

ERROR_BAND_ALPHA = 0.15
FONT_SIZE = 11


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",  default="experiments/elbo_estimation/config1.yaml")
    p.add_argument("--winners", default="scratch/gold_winners/winners.elbo_estimation.yaml")
    return p.parse_args()


def main():
    args = parse_args()

    from src.utils.io import _load_config
    config = _load_config(args.config)

    processed_dir = config["processed_results_dir"]
    figures_dir   = config["figures_dir"]
    summary_path  = os.path.join(processed_dir, "summary.h5")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"summary.h5 not found: {summary_path}\nRun step3 first."
        )

    with open(args.winners) as f:
        winners = yaml.safe_load(f)
    present_methods = set(winners["methods"].keys())

    # load data
    with h5py.File(summary_path, "r") as f:
        alphas = f["alphas"][:]
        deps   = f["design_eig_percentages"][:]
        methods_in_file = [
            k[len("mae_"):-len("_mean")]
            for k in f.keys() if k.endswith("_mean")
        ]
        mean_by_m = {m: f[f"mae_{m}_mean"][:] for m in methods_in_file}
        se_by_m   = {m: f[f"mae_{m}_se"][:]   for m in methods_in_file}

    methods = [m for m in methods_in_file if m in present_methods]
    methods_ordered = sorted(
        methods,
        key=lambda n: LEGEND_ORDER.index(n) if n in LEGEND_ORDER else len(LEGEND_ORDER),
    )

    n_alpha = len(alphas)
    n_dep   = len(deps)

    # style
    style_path = "full-width.mplstyle"
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        sns.set_style("whitegrid")
        matplotlib.rcParams["font.size"] = FONT_SIZE

    fig, axes = plt.subplots(
        1, n_alpha,
        figsize=(4.5 * n_alpha, 4),
        constrained_layout=False,
        sharey=True,
    )
    fig.subplots_adjust(bottom=0.30, wspace=0.06)
    if n_alpha == 1:
        axes = [axes]

    for alpha_idx, ax in enumerate(axes):
        for m in methods_ordered:
            mean = mean_by_m[m][:, alpha_idx]   # (n_dep,)
            se   = se_by_m[m][:, alpha_idx]
            color = METHOD_COLORS.get(m, "#888888")
            ax.plot(deps, mean, label=m, color=color,
                    linewidth=1.5, marker="o", markersize=3.5)
            ax.fill_between(deps, mean - se, mean + se,
                            color=color, alpha=ERROR_BAND_ALPHA)

        ax.set_xlabel(r"$\beta$ (Design EIG %)", fontsize=FONT_SIZE)
        ax.set_title(fr"$\alpha = {alphas[alpha_idx]:.2g}$", fontsize=FONT_SIZE)
        ax.set_xticks(deps)
        ax.set_xticklabels([f"{d:.2f}" for d in deps], rotation=30, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, alpha=0.3)
        if alpha_idx == 0:
            ax.set_ylabel("ELBO Estimation Error (MAE)", fontsize=FONT_SIZE)

    # shared legend below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=6,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
        framealpha=0.9,
    )

    fig.suptitle(
        r"ELBO Estimation — MAE by Design Optimality ($\beta$) and Mixing Weight ($\alpha$)",
        fontsize=12, fontweight="bold",
    )

    os.makedirs(figures_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(figures_dir, f"elbo_estimation_mae.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    print(f"\nDone. Figures in: {figures_dir}")
    print(f"Methods plotted: {len(methods_ordered)}")


if __name__ == "__main__":
    main()
