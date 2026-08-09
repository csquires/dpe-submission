"""render elbo_boed results: per-round eig-oracle (design) shortfall (primary,
running-true-posterior oracle), the anytime posterior KL curves, LOCAL
(per-round tempering) and GLOBAL (vs the running true posterior), and per-round
design-mislocation (angle + true-EIG gap), plus summary tables.

all rendering; no metric computation. reads processed_results.h5 (step3
output, flat schema: keys = "<metric>_<method>_<geom>[_lo|_hi]"). sibling of
ex/ablations/eig_elbo_boed/step4_plot_results.py: same styling idiom (solid
per-method lines, legend outside the axes, per-round R-point plots), reusing
its _REGRET_FAMILIES/_FAM_DISPLAY/_ALL_METHODS/_disp/_g/_fmt helpers directly.
elbo_boed's design channel is analytic (top-eigenvector, method-independent)
-> no regret_A or rank_tau here, since regret_A==0 trivially and there are no
DRE-EIG estimates to rank. timestamped
backup of figures before regen. canonical config keys: data_dim, n_rounds,
n_trials, figures_dir, processed_results_dir.

output: one shortfall / post_kl / mislocation plot (pdf+png) per (family,
geometry) pair that has data, plus five tables (md+tex): shortfall_table,
post_kl_table, mislocation_table, alpha_bias_table (per geometry), and
fallback_rate_table (scalar, Diag/Rot columns).
"""
import argparse
import os
import shutil
from datetime import datetime

import h5py
import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ex.utils.plot_style import apply as apply_style, style_for
from ex.utils.tables import write_tables
from ex.ablations.eig_elbo_boed.step4_plot_results import (
    _REGRET_FAMILIES, _FAM_DISPLAY, _ALL_METHODS, _disp, _g, _fmt, _plot_unfurled,
)


_GEOMS = ["diag", "rot"]


def load_config(cfg_path):
    """parse config.yaml; extract and validate required canonical keys.

    args:
      cfg_path: path to config.yaml
    returns:
      dict with data_dim, n_rounds, n_trials, figures_dir, processed_results_dir
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    required = ["data_dim", "n_rounds", "n_trials", "figures_dir", "processed_results_dir"]
    for key in required:
        if key not in cfg:
            raise KeyError(f"missing required config key: {key}")

    return cfg


def backup_figures(figures_dir):
    """timestamped backup of figures_dir (if it holds pdf/png), then ensure it exists.

    creates {figures_dir}_bak_{YYYYMMDD_HHMMSS} via shutil.copytree when the
    dir already has rendered figures; always leaves figures_dir present after.
    """
    if os.path.isdir(figures_dir):
        has_figures = any(f.endswith((".pdf", ".png")) for f in os.listdir(figures_dir))
        if has_figures:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{figures_dir}_bak_{timestamp}"
            shutil.copytree(figures_dir, backup_path)
            print(f"backed up figures to {backup_path}")

    os.makedirs(figures_dir, exist_ok=True)


def _plot_per_round(data, cfg, metric, prefix, ylabel):
    """shared body for the three per-round (R points) figures: exact-oracle
    shortfall, posterior-approx KL, and design-mislocation eig-gap. one
    figure per (family, geometry): median line + IQR band over round index
    0..R-1.
    """
    R = int(cfg["n_rounds"])
    x = np.arange(R)
    figdir = cfg["figures_dir"]

    for geom in _GEOMS:
        for fam, methods in _REGRET_FAMILIES:
            present = []
            for m in methods:
                med = _g(data, metric, m, geom)
                if med is not None and np.asarray(med).ndim == 1 and np.asarray(med).size:
                    present.append((m, np.asarray(med, dtype=float)))
            if not present:
                continue

            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            for m, med in present:
                color = style_for(m).get("color")
                ax.plot(x, med, color=color, lw=1.7, alpha=0.95, label=_disp(m))
                lo = _g(data, metric, m, geom, "lo")
                hi = _g(data, metric, m, geom, "hi")
                if lo is not None and hi is not None:
                    lo = np.asarray(lo, dtype=float)
                    hi = np.asarray(hi, dtype=float)
                    if lo.shape == med.shape and hi.shape == med.shape:
                        ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
            ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels([f"r{r}" for r in range(R)])
            ax.set_xlabel("Round")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{_FAM_DISPLAY[fam]} [{geom}]")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=1, frameon=False,
                      loc="upper left", bbox_to_anchor=(1.01, 1.0))
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(figdir, f"{prefix}_{fam}_{geom}.{ext}"),
                            dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {prefix}_{fam}_{geom}.{{pdf,png}}")


def plot_shortfall_per_round(data, cfg):
    """eig-oracle (design) cumulative shortfall per round (primary; running-true-
    posterior oracle); see _plot_per_round.
    """
    _plot_per_round(data, cfg, "shortfall", "shortfall",
                     "eig-oracle (design) shortfall (nats)")


def plot_post_kl_anytime(data, cfg):
    """anytime posterior KL unfurled across every round's alpha-BO trials: LOCAL
    (per-round tempering vs the method's own alpha=1 update) and GLOBAL (vs the
    running true posterior). see eig_elbo_boed.step4_plot_results._plot_unfurled.
    """
    _plot_unfurled(data, cfg, "post_kl_local_anytime", "post_kl_local_anytime",
                    "local tempering KL(q_a||q_1) (nats)", "n_trials_alpha")
    _plot_unfurled(data, cfg, "post_kl_global_anytime", "post_kl_global_anytime",
                    "belief KL vs true posterior (nats)", "n_trials_alpha")


def plot_mislocation_per_round(data, cfg):
    """design-mislocation true-EIG gap per round (lower=better); see _plot_per_round."""
    _plot_per_round(data, cfg, "mislocation_eiggap", "mislocation",
                     "design-mislocation EIG gap (nats)")


def _per_round_rows(data, cfg, metric, geom):
    """rows = [_disp(m), r0, r1, ...] of median [IQR] for a (R,)-shaped metric,
    one row per method in _ALL_METHODS that has data for (metric, geom).
    """
    R = int(cfg["n_rounds"])
    rows = []
    for m in _ALL_METHODS:
        med = _g(data, metric, m, geom)
        if med is None or not isinstance(med, np.ndarray) or med.ndim != 1 or med.size == 0:
            continue
        med = np.asarray(med, dtype=float)
        lo = _g(data, metric, m, geom, "lo")
        hi = _g(data, metric, m, geom, "hi")
        lo = np.asarray(lo, dtype=float) if lo is not None else None
        hi = np.asarray(hi, dtype=float) if hi is not None else None
        cells = [_disp(m)]
        for r in range(R):
            if r < med.size:
                l = lo[r] if (lo is not None and r < lo.size) else None
                h = hi[r] if (hi is not None and r < hi.size) else None
                cells.append(_fmt(med[r], l, h))
            else:
                cells.append("--")
        rows.append(cells)
    return rows


def write_shortfall_tables(data, cfg):
    """per-geometry table of exact-oracle shortfall per round, median [IQR]."""
    R = int(cfg["n_rounds"])
    for geom in _GEOMS:
        headers = ["Method"] + [f"r{r}" for r in range(R)]
        write_tables(os.path.join(cfg["figures_dir"], f"shortfall_table_{geom}"),
                     [(f"Eig-oracle (design) shortfall per round [{geom}], median [IQR]",
                       headers, _per_round_rows(data, cfg, "shortfall", geom))])
        print(f"  wrote shortfall_table_{geom}.{{md,tex}}")


def write_post_kl_tables(data, cfg):
    """per-geometry posterior-KL table: LOCAL per-round tempering gap and GLOBAL
    KL vs the running true posterior, as two sections (median [IQR]).
    """
    R = int(cfg["n_rounds"])
    headers = ["Method"] + [f"r{r}" for r in range(R)]
    for geom in _GEOMS:
        sections = [
            (f"Local per-round tempering KL(q_a || q_1) [{geom}], median [IQR]",
             headers, _per_round_rows(data, cfg, "post_kl", geom)),
            (f"Global KL vs true posterior [{geom}], median [IQR]",
             headers, _per_round_rows(data, cfg, "post_kl_global", geom)),
        ]
        write_tables(os.path.join(cfg["figures_dir"], f"post_kl_table_{geom}"), sections)
        print(f"  wrote post_kl_table_{geom}.{{md,tex}}")


def write_mislocation_tables(data, cfg):
    """per-geometry table of design-mislocation per round: the location angle
    (|cos(xi_r, xi_r*)|, from mislocation_angle) and the true-EIG gap (from
    mislocation_eiggap), as two sections of the same file (median [IQR]).
    """
    R = int(cfg["n_rounds"])
    headers = ["Method"] + [f"r{r}" for r in range(R)]
    for geom in _GEOMS:
        sections = [
            (f"Mislocation angle |cos(xi_r, xi_r*)| per round [{geom}], median [IQR]",
             headers, _per_round_rows(data, cfg, "mislocation_angle", geom)),
            (f"Mislocation true-EIG gap per round [{geom}], median [IQR]",
             headers, _per_round_rows(data, cfg, "mislocation_eiggap", geom)),
        ]
        write_tables(os.path.join(cfg["figures_dir"], f"mislocation_table_{geom}"), sections)
        print(f"  wrote mislocation_table_{geom}.{{md,tex}}")


def write_alpha_bias_tables(data, cfg):
    """per-geometry table of alpha-tempering error per round: signed bias
    (alpha_hat - 1, from alpha_bias) and absolute error (|alpha_hat - 1|,
    from alpha_abs_err), as two sections of the same file (median [IQR]).
    """
    R = int(cfg["n_rounds"])
    headers = ["Method"] + [f"r{r}" for r in range(R)]
    for geom in _GEOMS:
        sections = [
            (f"Alpha bias (alpha_hat - 1) per round [{geom}], median [IQR]",
             headers, _per_round_rows(data, cfg, "alpha_bias", geom)),
            (f"Alpha absolute error |alpha_hat - 1| per round [{geom}], median [IQR]",
             headers, _per_round_rows(data, cfg, "alpha_abs_err", geom)),
        ]
        write_tables(os.path.join(cfg["figures_dir"], f"alpha_bias_table_{geom}"), sections)
        print(f"  wrote alpha_bias_table_{geom}.{{md,tex}}")


def write_scalar_metric_table(data, cfg, metric, title, fname):
    """per-geometry table (Method | Diag | Rot) of a scalar metric, median [IQR]."""
    rows = []
    for m in _ALL_METHODS:
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


def main():
    """load config, backup figures, load results, emit all figures and tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ex/ablations/elbo_boed/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_results_path = os.path.join(cfg["processed_results_dir"], "processed_results.h5")

    backup_figures(cfg["figures_dir"])

    # clear old figures/tables (backed up above) so only the new set remains
    for fn in os.listdir(cfg["figures_dir"]):
        if fn.endswith((".pdf", ".png", ".md", ".tex")):
            os.remove(os.path.join(cfg["figures_dir"], fn))

    apply_style()

    # load the flat h5 dict directly (metric values + _lo/_hi IQR bands)
    with h5py.File(processed_results_path, "r") as f:
        data = {k: f[k][()] for k in f.keys()}

    plot_shortfall_per_round(data, cfg)
    plot_post_kl_anytime(data, cfg)
    plot_mislocation_per_round(data, cfg)

    write_shortfall_tables(data, cfg)
    write_post_kl_tables(data, cfg)
    write_mislocation_tables(data, cfg)
    write_alpha_bias_tables(data, cfg)
    write_scalar_metric_table(data, cfg, "fallback_rate",
                              "Alpha fallback rate (median [IQR])", "fallback_rate_table")

    print(f"done. figures + tables in: {cfg['figures_dir']}")


if __name__ == "__main__":
    main()
