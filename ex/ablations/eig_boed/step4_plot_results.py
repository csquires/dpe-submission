"""render eig_boed results: oracle shortfall, anytime regret, bias scatter, failure rate.

all rendering; no metric computation. reads processed_results.h5 (step3 output).
four figures: oracle shortfall (headline), regret_by_round (R
panels), bias scatter (3 family panels), failure rate (table). all output: pdf+png
+md+tex. timestamped backup of figures before regen. canonical config keys:
data_dim, n_rounds, n_trials, figures_dir, processed_results_dir.
"""
import argparse
import os
import shutil
import time
from datetime import datetime

import h5py
import numpy as np
import yaml
from scipy.stats import theilslopes, pearsonr

from ex.utils.plot_style import apply as apply_style, METHOD_GROUPS, style_for
from ex.utils.group_panels import plot_group_row
from ex.utils.faceted_lines import plot_panels
from ex.utils.tables import fmt_iqr, fmt_pm, write_tables

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(cfg_path):
    """parse config.yaml; extract and validate required canonical keys.

    args:
      cfg_path: path to config.yaml
    returns:
      dict with data_dim, n_rounds, n_trials, figures_dir, processed_results_dir
    """
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    required = ["data_dim", "n_rounds", "n_trials", "figures_dir", "processed_results_dir"]
    for key in required:
        if key not in cfg:
            raise KeyError(f"missing required config key: {key}")

    return cfg


def load_results(processed_results_path):
    """read processed_results.h5 (step3 output); return canonicalized (method, geometry) dict.

    geometry is a NAME SUFFIX (_diag, _rot); read each separately and combine.
    datasets per (m, g):
      scalars: oracle_shortfall, oracle_shortfall_lo, oracle_shortfall_hi,
               info_gained, info_gained_lo, info_gained_hi, rank_tau, rank_slope,
               failure_rate
      arrays (R,T): regret_by_round, regret_by_round_lo, regret_by_round_hi
      arrays (R,): per_round_failures
      arrays (K,): startup_est, startup_true (variable K per stratum)

    returns:
      dict[(method, geometry)] -> {key -> array/scalar}
    """
    # canonical metrics from step3 (in order of length, longest first, to avoid prefix conflicts)
    metrics = [
        "regret_by_round_lo", "regret_by_round_hi", "regret_by_round",
        "oracle_shortfall_lo", "oracle_shortfall_hi", "oracle_shortfall",
        "info_gained_lo", "info_gained_hi", "info_gained",
        "per_round_failures", "rank_slope", "rank_tau",
        "failure_rate", "startup_est", "startup_true",
    ]

    with h5py.File(processed_results_path, 'r') as f:
        keys = list(f.keys())

    # extract all (method, geometry) pairs by matching metric prefix + geom suffix
    pairs = set()
    for key in keys:
        for geom in ["diag", "rot"]:
            if key.endswith(f"_{geom}"):
                prefix = key[:-len(f"_{geom}")]
                # try each metric in order (longest first to avoid partial matches)
                for metric in metrics:
                    if prefix.startswith(metric + "_"):
                        method = prefix[len(metric) + 1:]
                        if method:  # ensure method is non-empty
                            pairs.add((method, geom))
                        break

    # group datasets by (method, geometry)
    results = {}
    with h5py.File(processed_results_path, 'r') as f:
        for method, geom in sorted(pairs):
            results[(method, geom)] = {}

            # collect all datasets for this (method, geometry)
            for key in f.keys():
                if key.endswith(f"_{method}_{geom}"):
                    # strip suffix to get the metric name
                    metric = key[:-len(f"_{method}_{geom}")]
                    # [()] reads both scalar (0-d) and array datasets; [:] fails on scalars
                    results[(method, geom)][metric] = f[key][()]

    return results


def backup_figures(figures_dir):
    """create timestamped backup of figures_dir if it exists and has pdf/png files.

    creates {figures_dir}_bak_{YYYYMMDD_HHMMSS} via shutil.copytree.
    prints confirmation; creates figures_dir if missing.
    """
    if os.path.isdir(figures_dir):
        has_figures = any(
            f.endswith(('.pdf', '.png'))
            for f in os.listdir(figures_dir)
        )
        if has_figures:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{figures_dir}_bak_{timestamp}"
            shutil.copytree(figures_dir, backup_path)
            print(f"backed up figures to {backup_path}")

    os.makedirs(figures_dir, exist_ok=True)


def plot_oracle_shortfall(results, cfg):
    """oracle shortfall headline figure: geometry on shared axes, symlog y.

    extract oracle_shortfall_<m>_<g> scalars + lo/hi bounds. plot diag and rot
    side-by-side for each method. emit pdf+png + md+tex table.
    """
    # group data: {(method, geom): scalar}
    data = {}
    for (m, g), d in results.items():
        if "oracle_shortfall" in d:
            data[(m, g)] = d["oracle_shortfall"]

    # extract unique methods
    methods = sorted(set(m for m, g in data.keys()))

    # collect lo/hi
    lo_data = {}
    hi_data = {}
    for (m, g), d in results.items():
        if "oracle_shortfall_lo" in d:
            lo_data[(m, g)] = d["oracle_shortfall_lo"]
        if "oracle_shortfall_hi" in d:
            hi_data[(m, g)] = d["oracle_shortfall_hi"]

    # check if there's any finite data
    if not any(np.isfinite(v) for v in data.values()):
        print("  skip oracle_shortfall: no finite data")
        return

    os.makedirs(cfg["figures_dir"], exist_ok=True)
    apply_style()

    # plot: one point per method x geometry pair
    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = 0
    x_labels = []
    x_ticks = []

    for method in methods:
        # get diag and rot if present
        pairs = [(g, f"{method}_{g}") for g in ["diag", "rot"] if (method, g) in data]

        for geom, label in pairs:
            val = data[(method, geom)]
            lo_val = lo_data.get((method, geom), val)
            hi_val = hi_data.get((method, geom), val)

            if np.isfinite(val):
                # plot point with error bar
                err_lo = val - lo_val
                err_hi = hi_val - val
                ax.errorbar(x_pos, val, yerr=[[err_lo], [err_hi]], fmt='o', markersize=6,
                           capsize=4, alpha=0.7)

            x_labels.append(label)
            x_ticks.append(x_pos)
            x_pos += 1

        # add gap between methods
        x_pos += 0.5

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Oracle shortfall (nats)")
    ax.set_yscale("symlog")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(cfg["figures_dir"], f"oracle_shortfall.{ext}"), dpi=300)
    plt.close(fig)
    print("  saved oracle_shortfall.{pdf,png}")

    # emit table: rows = methods, columns = diag/rot
    table_rows = []
    for method in methods:
        cells = [method]
        for geom in ["diag", "rot"]:
            if (method, geom) in data and np.isfinite(data[(method, geom)]):
                cells.append(fmt_iqr(
                    data[(method, geom)],
                    lo_data.get((method, geom), data[(method, geom)]),
                    hi_data.get((method, geom), data[(method, geom)])
                ))
            else:
                cells.append("--")
        table_rows.append(cells)

    write_tables(os.path.join(cfg["figures_dir"], "oracle_shortfall_table"), [(
        "Oracle shortfall (nats) [bootstrap IQR]",
        ["Method", "Diag", "Rot"],
        table_rows
    )])


def plot_regret_by_round(results, cfg):
    """regret by round: R panels via faceted_lines, all methods on same axes, symlog y.

    extract regret_by_round_<m>_<g> (R,T) + lo/hi. pool geometries (average).
    reshape to (T, R) for plot_panels; emit pdf+png per round + md+tex table.
    """
    # extract regret data
    regret_data = {}
    for (m, g), d in results.items():
        if "regret_by_round" in d:
            # only create geometry entries that carry data (a corrupt/missing
            # geometry must not leave an empty dict that ["mean"] would KeyError on)
            regret_data.setdefault(m, {})[g] = {
                "mean": d["regret_by_round"],
                "lo": d.get("regret_by_round_lo", d["regret_by_round"]),
                "hi": d.get("regret_by_round_hi", d["regret_by_round"]),
            }

    # pool geometries: average across diag and rot
    R = cfg["n_rounds"]
    T = cfg["n_trials"]

    pooled_mean = {}
    pooled_lo = {}
    pooled_hi = {}

    for m in regret_data:
        arrs = [regret_data[m][g]["mean"] for g in ["diag", "rot"] if g in regret_data[m]]
        if arrs:
            pooled_mean[m] = np.nanmean(arrs, axis=0)
            arrs_lo = [regret_data[m][g]["lo"] for g in ["diag", "rot"] if g in regret_data[m]]
            arrs_hi = [regret_data[m][g]["hi"] for g in ["diag", "rot"] if g in regret_data[m]]
            pooled_lo[m] = np.nanmean(arrs_lo, axis=0)
            pooled_hi[m] = np.nanmean(arrs_hi, axis=0)

    # reshape (R, T) -> (T, R) for plot_panels
    reshaped_mean = {m: pooled_mean[m].T for m in pooled_mean}
    reshaped_lo = {m: pooled_lo[m].T for m in pooled_lo}
    reshaped_hi = {m: pooled_hi[m].T for m in pooled_hi}

    # convert lo/hi to symmetric se: se = (hi - lo) / 2
    reshaped_se = {m: (reshaped_hi[m] - reshaped_lo[m]) / 2.0 for m in reshaped_mean}

    # x-axis: trial indices
    x = np.arange(T)

    # facets: one per round
    facets = [(f"r{r}", f"Round {r+1}") for r in range(R)]

    drawn = plot_panels(
        x, facets,
        reshaped_mean, reshaped_se,
        xlabel="Trial",
        ylabel="Regret (nats)",
        out_dir=cfg["figures_dir"],
        prefix="regret_by_round",
        yscale="symlog",
        shared_y=True
    )

    # emit table: rows = methods, columns = per-round medians
    table_cols = [f"Round {r+1}" for r in range(R)]
    table_rows = []
    for m in sorted(drawn):
        cells = [m]
        for r in range(R):
            # take median across trials in this round
            med = np.nanmedian(pooled_mean[m][r, :])
            cells.append(f"{med:.3g}")
        table_rows.append(cells)

    write_tables(os.path.join(cfg["figures_dir"], "regret_by_round_table"), [(
        "Anytime regret [bootstrap IQR] per round",
        ["Method"] + table_cols,
        table_rows
    )])


def plot_bias_scatter(results, cfg):
    """bias scatter: 3 panels (vfm_fmdre, tsm_ctsm, cls); y=x ref + Theil-Sen overlay.

    extract startup_est_<m>_<g> and startup_true_<m>_<g> arrays. pool geometries
    (concatenate). per family: scatter, y=x line, Theil-Sen slope. emit pdf+png
    per family + md+tex table per family.
    """
    # extract startup arrays
    startup_data = {}
    for (m, g), d in results.items():
        if "startup_est" in d and "startup_true" in d:
            # only create geometry entries that carry data (see plot_regret_by_round)
            startup_data.setdefault(m, {})[g] = {
                "est": d["startup_est"],
                "true": d["startup_true"],
            }

    # pool geometries (concatenate)
    pooled_est = {}
    pooled_true = {}
    for m in startup_data:
        est_arrs = [startup_data[m][g]["est"] for g in ["diag", "rot"] if g in startup_data[m]]
        true_arrs = [startup_data[m][g]["true"] for g in ["diag", "rot"] if g in startup_data[m]]
        if est_arrs:
            pooled_est[m] = np.concatenate(est_arrs)
            pooled_true[m] = np.concatenate(true_arrs)

    # organize by family
    families = {
        "vfm_fmdre": METHOD_GROUPS.get("vfm_fmdre", []),
        "tsm_ctsm": METHOD_GROUPS.get("tsm_ctsm", []),
        "cls": METHOD_GROUPS.get("cls", []),
    }

    apply_style()
    os.makedirs(cfg["figures_dir"], exist_ok=True)

    # emit one figure per family
    for family_key, family_methods in families.items():
        fig, ax = plt.subplots(figsize=(5.5, 5))

        # collect data for this family
        x_all = []
        y_all = []
        has_data = False

        for m in family_methods:
            if m in pooled_true:
                # filter finite pairs
                finite = np.isfinite(pooled_true[m]) & np.isfinite(pooled_est[m])
                if np.any(finite):
                    x_m = pooled_true[m][finite]
                    y_m = pooled_est[m][finite]
                    x_all.extend(x_m)
                    y_all.extend(y_m)
                    has_data = True

                    # scatter plot
                    kw = style_for(m)
                    ax.scatter(x_m, y_m, label=_disp(m), alpha=0.5, s=20, **kw)

        if not has_data:
            plt.close(fig)
            print(f"  skip bias_scatter_{family_key}: no finite data")
            continue

        # y=x reference line
        lim_min = min(np.min(x_all) if x_all else 0, np.min(y_all) if y_all else 0)
        lim_max = max(np.max(x_all) if x_all else 1, np.max(y_all) if y_all else 1)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3, linewidth=1, label="y=x")

        # Theil-Sen regression
        if len(x_all) > 1:
            slope, intercept, _, _ = theilslopes(y_all, x_all, alpha=0.68)
            x_line = np.array([lim_min, lim_max])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="gray", alpha=0.6, linewidth=1.5, label=f"Theil-Sen (m={slope:.2f})")

        ax.set_xlabel("True EIG")
        ax.set_ylabel("Estimated EIG")
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(cfg["figures_dir"], f"bias_scatter_{family_key}.{ext}"), dpi=300)
        plt.close(fig)
        print(f"  saved bias_scatter_{family_key}.{{pdf,png}}")

        # emit table per family
        table_rows = []
        for m in family_methods:
            if m in pooled_true:
                finite = np.isfinite(pooled_true[m]) & np.isfinite(pooled_est[m])
                if np.any(finite):
                    x_m = pooled_true[m][finite]
                    y_m = pooled_est[m][finite]
                    n_pairs = len(x_m)
                    if n_pairs > 1:
                        slope, intercept, _, _ = theilslopes(y_m, x_m, alpha=0.68)
                        r, _ = pearsonr(x_m, y_m)
                    else:
                        slope, intercept, r = np.nan, np.nan, np.nan
                    table_rows.append([
                        m,
                        f"{n_pairs}",
                        fmt_pm(slope, None),
                        fmt_pm(intercept, None),
                        fmt_pm(r, None)
                    ])

        if table_rows:
            write_tables(os.path.join(cfg["figures_dir"], f"bias_scatter_{family_key}_table"), [(
                f"Bias scatter {family_key}: Theil-Sen regression",
                ["Method", "N (pairs)", "Slope", "Intercept", "Pearson r"],
                table_rows
            )])


def plot_failure_rate(results, cfg):
    """failure rate: table of per-method rates (pooled geometries).

    extract failure_rate_<m>_<g>; pool geometries (mean). sort by rate descending.
    emit md+tex table only.
    """
    # extract failure rates
    fr_data = {}
    for (m, g), d in results.items():
        if "failure_rate" in d:
            if m not in fr_data:
                fr_data[m] = []
            fr_data[m].append(d["failure_rate"])

    # pool geometries (mean)
    pooled_fr = {m: np.nanmean(fr_data[m]) for m in fr_data if fr_data[m]}

    # sort by rate descending
    sorted_methods = sorted(pooled_fr.keys(), key=lambda m: pooled_fr[m], reverse=True)

    # emit table
    table_rows = [
        [m, f"{100 * pooled_fr[m]:.1f}%"]
        for m in sorted_methods
    ]

    write_tables(os.path.join(cfg["figures_dir"], "failure_rate_table"), [(
        "Failure rate (%) per method",
        ["Method", "Failure rate"],
        table_rows
    )])


_REGRET_FAMILIES = [
    ("bdre", ["BDRE"]),
    ("mdre", ["MDRE", "TriangularMDRE"]),
    ("ctsm", ["CTSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3"]),
    ("tsm", ["TSM", "TriangularTSM"]),
    ("fmdre", ["FMDRE", "FMDRE_S2", "TriangularFMDRE"]),
    ("vfm", ["VFM", "TriangularVFM_V1", "TriangularVFM_V2", "TriangularVFM_V3"]),
    ("mhtdre", ["MultiHeadTDRE", "MultiHeadTriangularTDRE"]),
]


_GEOMS = ["diag", "rot"]
_ORACLE_REFS = ["oracle_exact", "oracle_tilt_lo", "oracle_tilt_hi"]
_ALL_METHODS = [m for _, ms in _REGRET_FAMILIES for m in ms]
_FAM_DISPLAY = {"bdre": "BDRE", "mdre": "MDRE", "ctsm": "CTSM", "tsm": "TSM",
                "fmdre": "FMDRE", "vfm": "VFM", "mhtdre": "TDRE"}


def _disp(m):
    """method label for legends/tables (shared rule: see plot_style.display_name)."""
    from ex.utils.plot_style import display_name
    return display_name(m)


def _g(data, metric, method, geom, band=""):
    """fetch '<metric>_<method>_<geom>[_band]' from the flat h5 dict, or None."""
    return data.get(f"{metric}_{method}_{geom}" + (f"_{band}" if band else ""))


def _fmt(med, lo, hi):
    """median [q25, q75], or median, or '--' when missing/nan."""
    if med is None or not np.isfinite(float(med)):
        return "--"
    m = float(med)
    if lo is not None and hi is not None and np.isfinite(float(lo)) and np.isfinite(float(hi)):
        return f"{m:.3f} [{float(lo):.3f}, {float(hi):.3f}]"
    return f"{m:.3f}"


def _fmt_trials(med, lo, hi):
    """format a trial-index stat (integers; inf where the threshold is never met)."""
    def one(v):
        if v is None:
            return "--"
        v = float(v)
        return "inf" if not np.isfinite(v) else f"{v:.0f}"
    if med is None:
        return "--"
    if lo is not None and hi is not None:
        return f"{one(med)} [{one(lo)}, {one(hi)}]"
    return one(med)


def plot_regret_unfurled(data, cfg):
    """one figure PER (family, geometry): ANYTIME design regret (median line +
    IQR band) unfurled across all rounds. x = 0..R*T-1, dotted vlines + labels at
    round boundaries (each a fresh ceiling + fresh BO). baselines solid, their
    Triangular variants dashed. geometries are NOT pooled -- one plot each."""
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    R = int(cfg["n_rounds"])
    T = int(cfg["n_trials"])
    x = np.arange(R * T)
    bounds = [r * T for r in range(1, R)]
    figdir = cfg["figures_dir"]

    for geom in _GEOMS:
        for fam, methods in _REGRET_FAMILIES:
            present = []
            for m in methods:
                med = _g(data, "regret_by_round", m, geom)
                if med is not None and np.asarray(med).ndim == 2 and np.asarray(med).size:
                    present.append((m, np.asarray(med, dtype=float)))
            if not present:
                continue
            fig, ax = plt.subplots(figsize=(9.0, 3.2))
            for m, med in present:
                y = med.reshape(-1)
                color = style_for(m).get("color")
                ax.plot(x, y, color=color, lw=1.7, label=_disp(m), alpha=0.95)
                lo = _g(data, "regret_by_round", m, geom, "lo")
                hi = _g(data, "regret_by_round", m, geom, "hi")
                if lo is not None and hi is not None:
                    lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
                    if lo.shape == med.shape and hi.shape == med.shape:
                        ax.fill_between(x, lo.reshape(-1), hi.reshape(-1),
                                        color=color, alpha=0.15, lw=0)
            for b in bounds:
                ax.axvline(b, ls=":", color="0.55", lw=0.9)
            ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            for r in range(R):
                ax.text(r * T + T / 2.0, 0.97, f"round {r}", transform=trans,
                        ha="center", va="top", fontsize=7.5, color="0.4")
            ax.set_xlim(0, R * T - 1)
            ax.set_xlabel("Trials")
            ax.set_ylabel("anytime design regret (nats)")
            ax.set_title(f"{_FAM_DISPLAY.get(fam, fam.upper())}  [{geom}]")
            # legend OUTSIDE the axes (right) so it can't overlap the round labels
            ax.legend(fontsize=8, ncol=1, frameon=False,
                      loc="upper left", bbox_to_anchor=(1.01, 1.0))
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(figdir, f"regret_unfurled_{fam}_{geom}.{ext}"),
                            dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved regret_unfurled_{fam}_{geom}.{{pdf,png}}")


def write_scalar_metric_table(data, cfg, metric, title, fname):
    """per-geometry table (Method | Diag | Rot) of a scalar metric as median [IQR]."""
    rows = []
    for m in _ORACLE_REFS + _ALL_METHODS:
        cells = [_disp(m)]
        for geom in _GEOMS:
            cells.append(_fmt(_g(data, metric, m, geom),
                              _g(data, metric, m, geom, "lo"),
                              _g(data, metric, m, geom, "hi")))
        if cells[1] != "--" or cells[2] != "--":
            rows.append(cells)
    write_tables(os.path.join(cfg["figures_dir"], fname),
                 [(title, ["Method", "Diag", "Rot"], rows)])
    print(f"  wrote {fname}.{{md,tex}}")


def write_regret_tables(data, cfg):
    """per-geometry table of the numbers BEHIND the anytime regret plots: the
    terminal (end-of-round) anytime design regret for each round, median [IQR].
    kept SEPARATE from the trials-to-best-25%-regret table below."""
    R = int(cfg["n_rounds"])
    for geom in _GEOMS:
        rows = []
        for m in _ORACLE_REFS + _ALL_METHODS:
            med = _g(data, "regret_by_round", m, geom)
            if med is None or np.asarray(med).ndim != 2 or not np.asarray(med).size:
                continue
            med = np.asarray(med, dtype=float)
            lo = _g(data, "regret_by_round", m, geom, "lo")
            hi = _g(data, "regret_by_round", m, geom, "hi")
            lo = np.asarray(lo, dtype=float) if lo is not None else None
            hi = np.asarray(hi, dtype=float) if hi is not None else None
            cells = [_disp(m)]
            for r in range(R):
                l = lo[r, -1] if (lo is not None and lo.shape == med.shape) else None
                h = hi[r, -1] if (hi is not None and hi.shape == med.shape) else None
                cells.append(_fmt(med[r, -1], l, h))
            rows.append(cells)
        headers = ["Method"] + [f"r{r} (end)" for r in range(R)]
        write_tables(os.path.join(cfg["figures_dir"], f"regret_table_{geom}"),
                     [(f"Anytime design regret at end of each round [{geom}], median [IQR]",
                       headers, rows)])
        print(f"  wrote regret_table_{geom}.{{md,tex}}")


def write_trials_to_q25_tables(data, cfg):
    """SEPARATE per-geometry table: per-round best-25% regret threshold (bottom
    quartile of regret pooled over trials x cells), plus trials-to-reach-it
    (earliest within-round trial per rollout hitting that round's threshold,
    inf-imputed, pooled across rounds+rollouts) as median [IQR]."""
    R = int(cfg["n_rounds"])
    for geom in _GEOMS:
        rows = []
        for m in _ORACLE_REFS + _ALL_METHODS:
            q25 = _g(data, "regret_q25", m, geom)
            if q25 is None or not np.asarray(q25).size:
                continue
            q25 = np.asarray(q25, dtype=float)
            cells = [_disp(m)]
            for r in range(R):
                cells.append("--" if (r >= q25.size or not np.isfinite(q25[r])) else f"{q25[r]:.3f}")
            cells.append(_fmt_trials(_g(data, "trials_to_q25", m, geom),
                                     _g(data, "trials_to_q25", m, geom, "lo"),
                                     _g(data, "trials_to_q25", m, geom, "hi")))
            rows.append(cells)
        headers = (["Method"] + [f"best-25% regret r{r}" for r in range(R)]
                   + ["Trials to reach (median [IQR])"])
        write_tables(os.path.join(cfg["figures_dir"], f"regret_q25_table_{geom}"),
                     [(f"Best-25% regret per round & trials-to-reach [{geom}]", headers, rows)])
        print(f"  wrote regret_q25_table_{geom}.{{md,tex}}")


def main():
    """load config, backup figures, load results, emit all figures and tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='ex/ablations/eig_boed/config.yaml')
    args = parser.parse_args()

    # load config
    cfg = load_config(args.config)

    # construct processed_results path
    processed_results_path = os.path.join(cfg["processed_results_dir"], "processed_results.h5")

    # backup figures
    backup_figures(cfg["figures_dir"])

    # clear existing figures/tables (backed up above) so only the current set remains
    for fn in os.listdir(cfg["figures_dir"]):
        if fn.endswith((".pdf", ".png", ".md", ".tex")):
            os.remove(os.path.join(cfg["figures_dir"], fn))

    # apply style
    apply_style()

    # load the flat h5 dict directly (metric values + _lo/_hi IQR bands)
    with h5py.File(processed_results_path, "r") as f:
        data = {k: f[k][()] for k in f.keys()}

    # ONLY: anytime trial-wise regret plot per geometry + per-geometry tables
    plot_regret_unfurled(data, cfg)
    write_regret_tables(data, cfg)               # numbers behind the plots
    write_trials_to_q25_tables(data, cfg)        # SEPARATE trials-to-best-25%-regret
    write_scalar_metric_table(data, cfg, "rank_tau",
                              "Startup rank tau_b (median [IQR])", "rank_tau_table")
    write_scalar_metric_table(data, cfg, "oracle_shortfall",
                              "Oracle shortfall (nats), median [IQR]", "oracle_shortfall_table")

    print(f"done. figures + tables in: {cfg['figures_dir']}")


if __name__ == '__main__':
    main()
