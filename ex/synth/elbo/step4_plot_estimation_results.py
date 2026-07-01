"""
Step 4: Plot Results for ELBO Estimation

eig-/occupancy-style line plots: one figure per alpha, every method on the same
axes (thin lines + translucent +/- SE band), colors/markers from ex.utils.plot_style.
the shared legend is emitted as its own figure.

plots ABSOLUTE ELDR error (mae_{m} from step3): relative error is degenerate for
ELBO because the true ELDR is identically 0 at alpha=1 (division by zero).
"""
import argparse
import os

import h5py
import numpy as np
import yaml

from ex.utils.faceted_lines import plot_panels, plot_legend


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",  default="ex/synth/elbo/config1.yaml")
    p.add_argument("--winners", default="scratch/gold_winners/winners.elbo.yaml")
    return p.parse_args()


def main():
    args = parse_args()

    from src.utils.io import _load_config
    config = _load_config(args.config)

    processed_dir = config["processed_results_dir"]
    figures_dir   = config["figures_dir"]
    summary_path  = os.path.join(processed_dir, "summary.h5")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.h5 not found: {summary_path}\nRun step3 first.")

    with open(args.winners) as f:
        winners = yaml.safe_load(f)
    present_methods = set(winners["methods"].keys())

    # mae_{m}_mean / _se are grids of shape (n_dep, n_alpha) = absolute ELDR error.
    with h5py.File(summary_path, "r") as f:
        alphas = f["alphas"][:]
        deps   = f["design_eig_percentages"][:]
        methods = [k[len("mae_"):-len("_mean")] for k in f.keys() if k.endswith("_mean")]
        methods = [m for m in methods if m in present_methods]
        mean = {m: f[f"mae_{m}_mean"][:] for m in methods}   # (n_dep, n_alpha)
        se   = {m: f[f"mae_{m}_se"][:]   for m in methods}

    # facets = alphas; x = design_eig_percentage (beta). grids are already
    # [n_dep, n_alpha] == [len(x), n_facets], so they feed plot_panels directly.
    facets = [(f"alpha_{a:.2g}".replace(".", "p"), fr"$\alpha = {a:.2g}$")
              for a in alphas]

    plotted = plot_panels(
        deps, facets, mean, se,
        xlabel=r"$\beta$ (Design EIG %)",
        ylabel="ELDR Error (abs)",
        out_dir=figures_dir, prefix="elbo_eldr_err",
        xscale="linear", yscale="log",
    )
    plot_legend(plotted, figures_dir, prefix="elbo_eldr_err")

    print(f"\nDone. Figures in: {figures_dir}")
    print(f"Methods plotted: {len(plotted)}")


if __name__ == "__main__":
    main()
