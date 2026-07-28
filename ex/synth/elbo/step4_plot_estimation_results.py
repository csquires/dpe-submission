"""
Step 4: Plot Results for ELBO Estimation

one figure per (metric, alpha): a single row of method-group panels
(vfm_fmdre / tsm_ctsm / cls) concatenated left to right via ex.utils.group_panels,
with a shared y-range across alphas for comparability. sibling {stem}.md/.tex
tables carry the plotted values (one section per alpha). metrics:
  regret   -- per-cell normalized ELDR regret, MoM point + bootstrap IQR band
  eldr_err -- absolute ELDR error (mae_{m} from step3), mean +/- SE band
pointwise LDR MAE is not available for elbo: the raw campaign stored only the
integrated est_eldrs per cell, not per-sample LDR estimates.
"""
import argparse
import os

import h5py
import numpy as np
import yaml

from ex.utils.group_panels import plot_group_row
from ex.utils.tables import fmt_pm, fmt_iqr, write_tables


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",  default="ex/synth/elbo/config1.yaml")
    p.add_argument("--winners", default="scratch/gold_winners/winners.elbo.yaml")
    return p.parse_args()


def load_grids(f, pattern, methods):
    """dict method -> (n_dep, n_alpha) for '{pattern}' with {m} substituted."""
    out = {}
    for m in methods:
        key = pattern.format(m=m)
        if key in f:
            out[m] = f[key][:]
    return out


def shared_ylim(lo, hi, yscale):
    """global (lo, hi) across all methods/alphas so per-alpha figures compare."""
    los = [np.nanmin(v) for v in lo.values() if np.isfinite(v).any()]
    his = [np.nanmax(v) for v in hi.values() if np.isfinite(v).any()]
    y_lo, y_hi = min(los), max(his)
    if yscale == "log":
        return (max(y_lo, 1e-4) * 0.8, y_hi * 1.25)
    if yscale == "symlog":
        # 0 at the bottom (linear region), headroom above the pack for the legend
        return (0.0, y_hi * 3.0)
    return (min(0.0, y_lo), y_hi * 1.08)


def plot_metric(deps, alphas, mean, lo, hi, *, ylabel, prefix, yscale,
                cell_fn, table_title, figures_dir, linthresh=None):
    """per-alpha group-row figures + one table file with a section per alpha."""
    ylim = shared_ylim(lo, hi, yscale)
    sections = []
    for ai, a in enumerate(alphas):
        tag = f"alpha_{a:.2g}".replace(".", "p")
        col = lambda d, m: d[m][:, ai]
        drawn = plot_group_row(
            deps,
            {m: col(mean, m) for m in mean},
            {m: col(lo, m) for m in mean},
            {m: col(hi, m) for m in mean},
            xlabel=r"$\beta$ (Design EIG %)", ylabel=ylabel,
            out_dir=figures_dir, prefix=f"{prefix}_{tag}",
            yscale=yscale, ylim=ylim, linthresh=linthresh,
        )
        if drawn:
            header = ["Method"] + [f"beta={d:g}" for d in deps]
            rows = [[m] + [cell_fn(m, di, ai) for di in range(len(deps))]
                    for m in drawn]
            sections.append((fr"{table_title} -- alpha = {a:.2g}", header, rows))
    if sections:
        write_tables(os.path.join(figures_dir, f"{prefix}_table"), sections)


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
    present = set(winners["methods"].keys())

    with h5py.File(summary_path, "r") as f:
        alphas = f["alphas"][:]
        deps   = f["design_eig_percentages"][:]
        methods = sorted({k[len("mae_"):-len("_mean")] for k in f.keys()
                          if k.startswith("mae_") and k.endswith("_mean")} & present)
        mae    = load_grids(f, "mae_{m}_mean", methods)
        mae_se = load_grids(f, "mae_{m}_se", methods)
        reg    = load_grids(f, "regret_{m}_mom", methods)
        reg_lo = load_grids(f, "regret_{m}_lo", methods)
        reg_hi = load_grids(f, "regret_{m}_hi", methods)

    os.makedirs(figures_dir, exist_ok=True)

    if reg:
        plot_metric(
            deps, alphas, reg, reg_lo, reg_hi,
            ylabel="Rel. ELDR regret (MoM, IQR band)", prefix="elbo_regret_mom",
            yscale="symlog", linthresh=1e-3,
            cell_fn=lambda m, di, ai: fmt_iqr(reg[m][di, ai], reg_lo[m][di, ai], reg_hi[m][di, ai]),
            table_title="ELDR regret MoM [bootstrap IQR]", figures_dir=figures_dir,
        )

    mae_lo = {m: mae[m] - mae_se[m] for m in mae}
    mae_hi = {m: mae[m] + mae_se[m] for m in mae}
    plot_metric(
        deps, alphas, mae, mae_lo, mae_hi,
        ylabel="ELDR error (abs)", prefix="elbo_eldr_err", yscale="log",
        cell_fn=lambda m, di, ai: fmt_pm(mae[m][di, ai], mae_se[m][di, ai]),
        table_title="Absolute ELDR error, mean +/- SE", figures_dir=figures_dir,
    )

    print("note: pointwise LDR MAE unavailable for elbo (raw results hold integrated "
          "est_eldrs only); plotted regret + eldr_err.")
    print(f"\nDone. Figures in: {figures_dir}")


if __name__ == "__main__":
    main()
