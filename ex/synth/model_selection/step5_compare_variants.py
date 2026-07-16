"""plot ELDR error vs nsamples_test across variants (log-log).

separate evaluation MC noise (falls ~1/sqrt(n_test)) from estimator bias (plateaus).
read variant artifacts from git; tolerate missing variants; emit reference guide line.

usage:
  python -m ex.synth.model_selection.step5_compare_variants
"""
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ex.synth.model_selection.variants import VARIANTS
from ex.utils.faceted_lines import plot_legend, order_methods
from ex.utils.plot_style import apply as apply_style, style_for, ERROR_BAND_ALPHA

# constants
FACETS = [
    ("p0", "p*=p_0"),
    ("p1", "p*=p_1"),
    ("mid", "midpoint"),
    ("dist", "distant"),
]

OUT_DIR = "ex/synth/model_selection/figures/compare"

THIN_LW = 1.0
MARKER_SIZE = 3.0


def load_variant_data(tag: str) -> tuple[dict, bool]:
    """load ELDR error means and SEs for variant tag.

    args:
      tag: variant tag (e.g., "tr8192_te2048").

    returns:
      (data_dict, success) where:
      - data_dict: {method: {"mean": arr, "se": arr}} with shapes (n_test_sets,).
      - success: True if file exists, False if missing.

    procedure:
      1. construct path: ex/synth/model_selection/processed_results/<tag>/new_pstar.h5
      2. if file doesn't exist: print notice, return ({}, False).
      3. open h5; iterate over keys matching "eldr_err_*_mean" (extract method name).
      4. for each method, load eldr_err_{method}_mean and eldr_err_{method}_se.
      5. average over KL dimension (axis 0) to get per-test-set error: shape (n_test_sets,).
      6. return {method: {"mean": mean_arr, "se": se_arr}, ...}, True.
    """
    path = f"ex/synth/model_selection/processed_results/{tag}/new_pstar.h5"
    if not os.path.exists(path):
        print(f"  skip variant {tag}: {path} not found")
        return {}, False

    data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key.startswith("eldr_err_") and key.endswith("_mean"):
                method = key.replace("eldr_err_", "").replace("_mean", "")
                mean_arr = f[key][:]  # (n_kl, n_test_sets)
                se_arr = f[f"eldr_err_{method}_se"][:]  # (n_kl, n_test_sets)

                # average over KL dimension to get shape (n_test_sets,)
                mean = mean_arr.mean(axis=0)
                se = se_arr.mean(axis=0)

                data[method] = {"mean": mean, "se": se}

    return data, True


def aggregate_variants() -> tuple[dict, list[str], bool]:
    """load all variant data; aggregate into per-method curves.

    returns:
      (aggregated, present_tags, all_ok) where:
      - aggregated: {method: {"mean": array([n_var, n_test_sets]), "se": array(...)}}
      - present_tags: list of tags that were successfully loaded (in order).
      - all_ok: False if any variant missing (non-fatal); True if all present.

    procedure:
      1. for each tag in sorted(VARIANTS.keys()):
         - load via load_variant_data(tag).
         - collect per-method results.
      2. merge into {method: {"mean": (n_variants, n_test_sets), "se": (n_variants, n_test_sets)}}.
      3. return aggregated dict, present_tags list, bool (True iff all 4 variants loaded).

    fail condition:
      if ZERO variants present: raise FileNotFoundError with diagnostic.
    """
    aggregated = {}
    present_tags = []

    for tag in sorted(VARIANTS.keys()):
        data, ok = load_variant_data(tag)
        if not ok:
            continue
        present_tags.append(tag)

        for method, arr_dict in data.items():
            if method not in aggregated:
                aggregated[method] = {"mean": [], "se": []}
            aggregated[method]["mean"].append(arr_dict["mean"])
            aggregated[method]["se"].append(arr_dict["se"])

    if not present_tags:
        raise FileNotFoundError(
            f"no variant processed_results found; checked:\n" +
            "\n".join(
                f"  ex/synth/model_selection/processed_results/{tag}/new_pstar.h5"
                for tag in sorted(VARIANTS.keys())
            )
        )

    # convert lists to arrays
    for method in aggregated:
        aggregated[method]["mean"] = np.array(aggregated[method]["mean"])  # (n_present, n_test_sets)
        aggregated[method]["se"] = np.array(aggregated[method]["se"])

    all_ok = len(present_tags) == len(VARIANTS)
    return aggregated, present_tags, all_ok


def prepare_plot_data(aggregated: dict, present_tags: list[str]) -> tuple[np.ndarray, dict, dict]:
    """reshape aggregated data into per-test-set format for plotting.

    args:
      aggregated: {method: {"mean": (n_variants, n_test_sets), "se": (n_variants, n_test_sets)}}
      present_tags: tags in order (determines x-axis order).

    returns:
      (x, mean_by_facet, se_by_facet) where:
      - x: array of nsamples_test values in order of present_tags, shape (n_variants,).
      - mean_by_facet: {method: array([n_variants, n_facets])} where column i is test-set i.
      - se_by_facet: {method: array([n_variants, n_facets])} same structure.
    """
    # x-axis: nsamples_test for present variants
    x = np.array([VARIANTS[tag][1] for tag in present_tags], dtype=float)

    mean_by_facet = {}
    se_by_facet = {}
    for method, arr_dict in aggregated.items():
        mean_by_facet[method] = arr_dict["mean"]  # (n_variants, n_test_sets)
        se_by_facet[method] = arr_dict["se"]  # (n_variants, n_test_sets)

    return x, mean_by_facet, se_by_facet


def plot_reference_noise_line(ax, x, label: str = "1/sqrt(n_test)") -> None:
    """overlay reference line y ~ 1/sqrt(x) to guide eye on noise vs bias.

    args:
      ax: matplotlib axis.
      x: x-axis values (nsamples_test).
      label: legend label.

    procedure:
      1. compute y = k / sqrt(x) where k is chosen so reference touches top of plot.
      2. plot as thin dashed gray line.
      3. add to legend.
    """
    # scale so reference is visible but not dominant
    k = x.max() / np.sqrt(x.min())
    y_ref = k / np.sqrt(x)
    ax.loglog(x, y_ref, "k--", linewidth=0.8, alpha=0.4, label=label)


def plot_panels_custom(x, mean_by_facet, se_by_facet, methods):
    """create faceted plots with reference lines.

    args:
      x: x-axis values (nsamples_test), shape (n_variants,).
      mean_by_facet: {method: (n_variants, n_facets)}.
      se_by_facet: {method: (n_variants, n_facets)}.
      methods: ordered list of method names.

    creates one figure per facet, saves to ex/synth/model_selection/figures/compare/.
    """
    apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    for fi, (fkey, flabel) in enumerate(FACETS):
        fig, ax = plt.subplots(figsize=(5, 4))

        any_method = False
        for method in methods:
            y = np.asarray(mean_by_facet[method])[:, fi]
            if not np.isfinite(y).any():
                continue
            any_method = True

            e = np.asarray(se_by_facet[method])[:, fi]
            kw = style_for(method)

            # plot data with label
            ax.loglog(x, y, label=method, linewidth=THIN_LW, markersize=MARKER_SIZE, **kw)

            # add error band
            band_lo = y - e
            band_hi = y + e
            band_lo = np.maximum(band_lo, 1e-4)  # clamp to avoid log errors
            ax.fill_between(x, band_lo, band_hi, color=kw["color"], alpha=ERROR_BAND_ALPHA, linewidth=0)

        if not any_method:
            plt.close(fig)
            print(f"  skip compare_variants_{fkey}: no finite data")
            continue

        # add reference line
        plot_reference_noise_line(ax, x)

        ax.set_xlabel("nsamples_test")
        ax.set_ylabel("mean ELDR error")
        ax.set_title(f"test set: {flabel}")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"compare_variants_{fkey}.{ext}"), dpi=150)
        plt.close(fig)
        print(f"  saved compare_variants_{fkey}.{{pdf,png}}")


def main() -> None:
    """orchestrate load, aggregate, and plot across variants."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # (1) load and aggregate
    print("\n=== Loading variant data ===")
    aggregated, present_tags, all_ok = aggregate_variants()
    if not all_ok:
        print(f"WARNING: only {len(present_tags)}/{len(VARIANTS)} variants present")

    methods = order_methods(aggregated.keys())
    print(f"Loaded {len(aggregated)} methods from {len(present_tags)} variants\n")

    # (2) prepare plot data
    x, mean_by_facet, se_by_facet = prepare_plot_data(aggregated, present_tags)

    # (3) plot panels with reference lines
    print(f"=== Plotting (log-log, {len(present_tags)} points per method) ===")
    plot_panels_custom(x, mean_by_facet, se_by_facet, methods)

    # (4) emit legend
    plot_legend(methods, OUT_DIR, "compare_variants")

    # (5) summary
    print(f"\n=== Summary ===")
    print(f"Variants: {', '.join(present_tags)}")
    print(f"X values (nsamples_test): {x}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Figures: {OUT_DIR}/")
    print(f"\nInterpretation:")
    print(f"  - If error falls ~1/sqrt(n): MC sampling noise dominates (improvement possible with larger n_test).")
    print(f"  - If error plateaus: estimator bias dominates (method choice or hyperparams needed).")
    print(f"  - distant test set often shows bias; p0/p1/midpoint more sensitive to MC noise.")


if __name__ == "__main__":
    main()
