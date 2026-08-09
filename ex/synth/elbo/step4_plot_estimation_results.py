"""
Step 4: Plot Results for ELBO Estimation

one figure per (metric, alpha): a single row of method-group panels
(vfm_fmdre / tsm_ctsm / cls) concatenated left to right via ex.utils.group_panels,
with a shared y-range across alphas for comparability. sibling {stem}.md/.tex
tables carry the plotted values (one section per alpha). each metric also gets an
alpha-pooled figure ({prefix}_alpha_pooled) + {prefix}_pooled_table, aggregating
every alpha into one column on the same y-range as the per-alpha set. metrics:
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
from ex.utils.plot_style import display_name
from ex.utils.tables import fmt_pm, fmt_iqr, write_tables


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",  default="ex/synth/elbo/config1.yaml")
    p.add_argument("--winners", default="scratch/gold_winners/winners.elbo.yaml")
    return p.parse_args()


def load_grids(f, pattern, methods):
    """dict method -> (n_dep, n_col) for '{pattern}' with {m} substituted."""
    out = {}
    for m in methods:
        key = pattern.format(m=m)
        if key in f:
            out[m] = f[key][:]
    return out


def load_variant(f, methods, infix=""):
    """grid bundle for one column layout.

    infix '' reads the per-alpha grids (n_dep, n_alpha); 'pooled_' reads the
    alpha-pooled grids (n_dep, 1) that step3 writes alongside them.
    """
    stat = lambda kind, s: load_grids(f, f"{kind}_{{m}}_{infix}{s}", methods)
    return {
        "mae":     stat("mae", "mean"),   "mae_se":   stat("mae", "se"),
        "mae_med": stat("mae", "med"),    "mae_q1":   stat("mae", "q1"),
        "mae_q3":  stat("mae", "q3"),
        "reg":     stat("regret", "mom"), "reg_lo":   stat("regret", "lo"),
        "reg_hi":  stat("regret", "hi"),  "reg_bstd": stat("regret", "bstd"),
    }


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


def plot_metric(deps, cols, mean, lo, hi, *, ylabel, prefix, yscale,
                cell_fn, table_title, figures_dir, table_stem=None, ylim=None,
                linthresh=None, extra=None):
    """one group-row figure per column + one table file with a section per column.

    Args:
      cols: list of (tag, label), one per column of the (n_dep, n_col) grids.
        tag suffixes the figure filename, label names the table section; the
        per-alpha call passes one entry per alpha, the pooled call passes one.
      ylim: shared y-range, computed from the data when None. pass the per-alpha
        range into the pooled call so both figures read on the same scale.
      extra: optional list of (title, cell_fn) appending further table sections
        per column (e.g. a median [q1, q3] companion to a mean +/- SE primary).
    Returns:
      the (lo, hi) y-range used.
    """
    if ylim is None:
        ylim = shared_ylim(lo, hi, yscale)
    sections = []
    for ci, (tag, label) in enumerate(cols):
        col = lambda d, m: d[m][:, ci]
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
            mk = lambda cf: [[display_name(m)] + [cf(m, di, ci) for di in range(len(deps))]
                             for m in drawn]
            sections.append((f"{table_title} -- {label}", header, mk(cell_fn)))
            for etitle, ecf in (extra or []):
                sections.append((f"{etitle} -- {label}", header, mk(ecf)))
    if sections:
        write_tables(os.path.join(figures_dir, table_stem or f"{prefix}_table"), sections)
    return ylim


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
        by_alpha = load_variant(f, methods)
        pooled   = load_variant(f, methods, "pooled_")

    os.makedirs(figures_dir, exist_ok=True)

    def draw(v, cols, table_suffix, ylim):
        """both metrics for one column layout; returns the y-ranges used."""
        used = {}
        reg, reg_lo, reg_hi, reg_bstd = v["reg"], v["reg_lo"], v["reg_hi"], v["reg_bstd"]
        if reg:
            reg_extra = []
            if reg_bstd:
                reg_extra.append((
                    "ELDR regret MoM +/- bootstrap std",
                    lambda m, di, ci: fmt_pm(reg[m][di, ci], reg_bstd[m][di, ci]),
                ))
            used["reg"] = plot_metric(
                deps, cols, reg, reg_lo, reg_hi,
                ylabel="Rel. ELDR regret", prefix="elbo_regret_mom",
                yscale="symlog", linthresh=1e-3, ylim=ylim.get("reg"),
                cell_fn=lambda m, di, ci: fmt_iqr(reg[m][di, ci], reg_lo[m][di, ci], reg_hi[m][di, ci]),
                table_title="ELDR regret MoM [bootstrap IQR]", figures_dir=figures_dir,
                table_stem=f"elbo_regret_mom{table_suffix}_table", extra=reg_extra,
            )

        mae, mae_se = v["mae"], v["mae_se"]
        mae_med, mae_q1, mae_q3 = v["mae_med"], v["mae_q1"], v["mae_q3"]
        if mae:
            mae_lo = {m: mae[m] - mae_se[m] for m in mae}
            mae_hi = {m: mae[m] + mae_se[m] for m in mae}
            mae_extra = []
            if mae_med:
                mae_extra.append((
                    "Absolute ELDR error, median [q1, q3]",
                    lambda m, di, ci: fmt_iqr(mae_med[m][di, ci], mae_q1[m][di, ci], mae_q3[m][di, ci]),
                ))
            used["mae"] = plot_metric(
                deps, cols, mae, mae_lo, mae_hi,
                ylabel="ELDR error (abs)", prefix="elbo_eldr_err", yscale="log",
                ylim=ylim.get("mae"),
                cell_fn=lambda m, di, ci: fmt_pm(mae[m][di, ci], mae_se[m][di, ci]),
                table_title="Absolute ELDR error, mean +/- SE", figures_dir=figures_dir,
                table_stem=f"elbo_eldr_err{table_suffix}_table", extra=mae_extra,
            )
        return used

    # per-alpha first, then the pooled variant on the same y-ranges so the two
    # sets of figures are directly comparable.
    cols_alpha = [(f"alpha_{a:.2g}".replace(".", "p"), f"alpha = {a:.2g}") for a in alphas]
    ylim = draw(by_alpha, cols_alpha, "", {})
    draw(pooled, [("alpha_pooled", "all alphas pooled")], "_pooled", ylim)

    print("note: pointwise LDR MAE unavailable for elbo (raw results hold integrated "
          "est_eldrs only); plotted regret + eldr_err.")
    print(f"\nDone. Figures in: {figures_dir}")


if __name__ == "__main__":
    main()
