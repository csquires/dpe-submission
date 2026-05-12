"""
Generate MAE-vs-n_pstar plots from processed pstar sample-complexity metrics.

Data flow:
- Input: processed_results/metrics.h5 from step3_process_results.py
- Output: figures/mae_vs_nsamples_pstar.pdf
"""

import os

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np
import yaml
import seaborn as sns


config = yaml.load(
    open("ex/pstar_sample_complexity/config.yaml", "r"),
    Loader=yaml.FullLoader,
)

PROCESSED_RESULTS_DIR = config["processed_results_dir"]
FIGURES_DIR = config["figures_dir"]
ALGORITHMS = config.get("algorithms", [])

processed_results_filename = f"{PROCESSED_RESULTS_DIR}/metrics.h5"

if not os.path.exists(processed_results_filename):
    print(f"Error: {processed_results_filename} not found.")
    print("Please run step3_process_results.py first.")
    raise SystemExit(1)


def build_style_maps(algorithms: list[str]) -> tuple[dict[str, tuple], dict[str, str]]:
    """Assign a stable color to every configured algorithm."""
    palette = sns.color_palette("colorblind", n_colors=max(len(algorithms), 1))
    colors = {alg: palette[idx] for idx, alg in enumerate(algorithms)}
    return colors, {}


def display_name(alg: str) -> str:
    """Compact, plot-friendly method names."""
    labels = {
        "MultiHeadTriangularTDRE": "MH-TDRE",
        "TriangularCTSM": "CTSM",
        "TriangularCTSM2D": "CTSM-2D",
        "TriangularFMDRE": "FMDRE",
        "TriangularMDRE": "MDRE",
        "TriangularTSM": "TSM",
        "TriangularVFM": "VFM",
        "TriangularVFM2D": "VFM-2D",
    }
    return labels.get(alg, alg)


with h5py.File(processed_results_filename, "r") as f:
    nsamples_pstar_values = f["nsamples_pstar_values"][:]

    mae_by_alg = {}
    mae_std_by_alg = {}

    for alg in ALGORITHMS:
        mae_key = f"mae_{alg}"
        mae_std_key = f"mae_std_{alg}"

        if mae_key not in f:
            print(f"Warning: {mae_key} not found, skipping {alg}")
            continue

        mae_by_alg[alg] = f[mae_key][:]
        mae_std_by_alg[alg] = f[mae_std_key][:] if mae_std_key in f else np.zeros_like(mae_by_alg[alg])


if not mae_by_alg:
    print("Error: No algorithms loaded successfully from metrics.h5.")
    raise SystemExit(1)

for alg in mae_by_alg:
    assert len(mae_by_alg[alg]) == len(nsamples_pstar_values), (
        f"Algorithm {alg}: mae length {len(mae_by_alg[alg])} != "
        f"nsamples_pstar length {len(nsamples_pstar_values)}"
    )
    assert len(mae_std_by_alg[alg]) == len(nsamples_pstar_values), (
        f"Algorithm {alg}: mae_std length {len(mae_std_by_alg[alg])} != "
        f"nsamples_pstar length {len(nsamples_pstar_values)}"
    )


def plot_mae_vs_nsamples_pstar(
    nsamples: np.ndarray,
    mae_by_alg: dict[str, np.ndarray],
    mae_std_by_alg: dict[str, np.ndarray],
    output_dir: str,
    output_name: str,
) -> None:
    """Plot MAE curves for all available algorithms."""
    plot_algs = [alg for alg in ALGORITHMS if alg in mae_by_alg]
    colors, _ = build_style_maps(plot_algs)

    LINEWIDTH = 1.8
    MARKERSIZE = 4.5

    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (6.8, 4.4)
    plt.rcParams["font.size"] = 10.5

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    plotted_any = False
    for alg in plot_algs:
        mae = mae_by_alg[alg]

        if np.all(np.isnan(mae)):
            print(f"Warning: {alg} has all-NaN MAE values, skipping")
            continue

        plotted_any = True

        ax.plot(
            nsamples,
            mae,
            label=display_name(alg),
            color=colors[alg],
            linewidth=LINEWIDTH,
            marker="o",
            markersize=MARKERSIZE,
        )

    if not plotted_any:
        print("Error: all loaded MAE curves are NaN; no figure was generated.")
        raise SystemExit(1)

    finite_maes = np.concatenate(
        [arr[np.isfinite(arr)] for arr in mae_by_alg.values() if np.any(np.isfinite(arr))]
    )
    ymin = max(0.5, 10 ** np.floor(np.log10(finite_maes.min())))
    ymax = 10 ** np.ceil(np.log10(finite_maes.max()))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"Number of $p^*$ Training Samples ($n_{p^*}$)")
    ax.set_ylabel("Mean Absolute Error")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(0.02, 0.62),
        frameon=True,
        ncol=2,
        fontsize=8.5,
    )
    sns.despine()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/{output_name}.pdf"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {filepath}")


plot_mae_vs_nsamples_pstar(
    nsamples=nsamples_pstar_values,
    mae_by_alg=mae_by_alg,
    mae_std_by_alg=mae_std_by_alg,
    output_dir=FIGURES_DIR,
    output_name="mae_vs_nsamples_pstar",
)

print("\n" + "=" * 80)
print("Sample complexity figure saved:")
print(f"{FIGURES_DIR}/mae_vs_nsamples_pstar.pdf")
print("=" * 80)
