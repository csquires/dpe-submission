"""datagen diagnostic for pendulum trajectory ELDR experiment.

second cell axis is beta (a fixed mixture weight), NOT a prescribed K2.
all panels label this axis as 'beta' and treat it as a knob (no y=x
reference between beta and KL2_realized).

modes:
  default:     read cells under data_dir/k1_{i}_beta_{j}_seed_{s}.h5,
               produce datagen_diagnostic.png + datagen_variance.png.
  --show-grid: read traj_kl_cache/traj_grid_{hash}.h5 and produce
               grid_diagnostic.png. works with no per-cell HDF5 yet.

panels (default mode):
  - K1 prescribed vs realized scatter (y=x reference)
  - beta (set) vs K2_realized scatter (no reference line)
  - (alpha*, beta) coverage in 2-d
  - Bellman residual scatter (q_O across cells, q_E baseline)
  - LDR histograms grid (per (k1_idx, beta_idx); seeds overlaid)
  - phase-space (theta, theta_dot) at t=0 and t=T
  - PCA of flat trajectories (first available cell)
  - hardness boxplot grid + summary table
"""
import argparse
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ex.semisynth.pendulum.step1_create_data import (
    _build_env_and_q_cfg,
    _resolve_reward,
)
from ex.utils.alpha_grid import make_alphas
from ex.utils.diagnostic_primitives import (
    collect_cells,
    load_kl_grid,
    plot_pca_panel,
)
from ex.utils.prescribed_kls import hash_pendulum_cfg


# raw attr names live in the per-cell hdf5; remap to standard names
# used by the plotters in this file.
KEY_MAP = {
    "k1_pre": "K1_prescribed",
    "k1_real": "K1_realized",
    "k2_real": "K2_realized",
    "beta": "beta",
    "alpha": "alpha_star",
    "integrated_eldr": "integrated_eldr",
}


def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="ex/semisynth/pendulum/config.yaml")
    p.add_argument("--show-grid", action="store_true",
                   help="plot the cached traj_kl grid (no per-cell HDF5 needed)")
    return p.parse_args(args)


# ----------------------------------------------------------------------
# config / path resolution
# ----------------------------------------------------------------------

def build_grid_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """reconstruct cfg dict consumed by hash_pendulum_cfg.

    must mirror the dict assembled inside step1_create_data.per_cell so
    the resulting hash matches the cached grid file.
    """
    env_cfg, q_cfg = _build_env_and_q_cfg(config)
    alphas = make_alphas(config["traj_kl_grid"]).tolist()
    betas = np.linspace(0, 1, config["traj_kl_grid"]["G_beta"]).tolist()
    return {
        "env_cfg": env_cfg,
        "q_cfg": q_cfg,
        "T": int(config["trajectory"]["T"]),
        "sigma_pi": float(config["pendulum"]["sigma_pi"]),
        "M": int(config["traj_kl_grid"]["M"]),
        "r_E_name": config["pendulum"]["r_E_name"],
        "r_anti_name": config["pendulum"]["r_anti_name"],
        "alphas": alphas,
        "betas": betas,
    }


def find_grid_cache(config: Dict[str, Any]) -> str:
    """resolve the cached traj_kl grid path; raises FileNotFoundError if missing."""
    cfg = build_grid_cfg(config)
    h = hash_pendulum_cfg(cfg)
    cache_dir = config["traj_kl_grid"]["cache_dir"]
    path = Path(cache_dir) / f"traj_grid_{h}.h5"
    if not path.exists():
        raise FileNotFoundError(
            f"no cached grid at {path}. run "
            f"`python -m ex.semisynth.pendulum.step1_create_data --smoke` "
            f"to build it first."
        )
    return str(path)


def enumerate_cell_paths(config: Dict[str, Any]
                         ) -> Dict[Tuple[int, int], List[Tuple[int, str]]]:
    """walk data_dir for k1_{i}_beta_{j}_seed_{s}.h5; group by (k1_idx, beta_idx).

    returns dict[(k1_idx, beta_idx)] -> list of (seed, path); only existing files included.
    """
    data_dir = Path(config["data_dir"])
    k1_values = [float(v) for v in config["kl_targets"]["k1_values"]]
    beta_values = [float(v) for v in config["kl_targets"]["beta_values"]]
    seeds_default = int(config["kl_targets"]["seeds_default"])

    out: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
    for k1_idx, beta_idx in product(range(len(k1_values)), range(len(beta_values))):
        seeds = []
        for seed in range(seeds_default):
            path = data_dir / f"k1_{k1_idx}_beta_{beta_idx}_seed_{seed}.h5"
            if path.exists():
                seeds.append((seed, str(path)))
        if seeds:
            out[(k1_idx, beta_idx)] = seeds
    return out


# ----------------------------------------------------------------------
# KL-grid plots (--show-grid mode)
# ----------------------------------------------------------------------

def plot_kl1_curve(ax, grid: Dict[str, Any], k1_targets: List[float]) -> None:
    """line plot of KL1[alpha] with optional SE band; K1 targets as horizontal lines."""
    ax.plot(grid["alphas"], grid["KL1"], "-o", markersize=3, color="tab:blue")
    if "KL1_se" in grid:
        lo = grid["KL1"] - grid["KL1_se"]
        hi = grid["KL1"] + grid["KL1_se"]
        ax.fill_between(grid["alphas"], lo, hi, alpha=0.2, color="tab:blue")
    for k1 in k1_targets:
        ax.axhline(k1, color="tab:red", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$KL_1 = KL(d_E \| d_O)$")
    ax.set_title(r"$KL_1[\alpha]$ (red dashes: $K_1$ targets)")


def plot_kl2_heatmap(ax, grid: Dict[str, Any],
                     k1_targets: List[float], beta_targets: List[float]) -> None:
    """heatmap of KL2[alpha, beta]; overlay (alpha*, beta_set) for each (K1, beta) cell.

    alpha* is obtained by inverting KL1 at K1*; beta_set comes straight
    from config (no inversion). matches what step1_create_data does.
    """
    KL2 = np.asarray(grid["KL2"])
    alphas = np.asarray(grid["alphas"])
    betas = np.asarray(grid["betas"])
    im = ax.imshow(
        KL2.T, origin="lower", aspect="auto",
        extent=[alphas[0], alphas[-1], betas[0], betas[-1]],
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, fraction=0.046)
    KL1 = np.asarray(grid["KL1"])
    if np.all(np.diff(KL1) > 0):
        for k1 in k1_targets:
            if KL1[0] <= k1 <= KL1[-1]:
                a_star = float(np.interp(k1, KL1, alphas))
                for b_set in beta_targets:
                    if betas[0] <= b_set <= betas[-1]:
                        ax.plot([a_star], [b_set], marker="x",
                                markersize=8, color="white",
                                markeredgewidth=2)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"$KL_2[\alpha, \beta]$ (white x: $(\alpha^*, \beta_{set})$ cells)")


def plot_kl_se_heatmap(ax, grid: Dict[str, Any]) -> None:
    """heatmap of KL2 standard error (pendulum-only)."""
    if "KL2_se" not in grid:
        ax.set_visible(False)
        return
    KL2_se = np.asarray(grid["KL2_se"])
    alphas = np.asarray(grid["alphas"])
    betas = np.asarray(grid["betas"])
    im = ax.imshow(
        KL2_se.T, origin="lower", aspect="auto",
        extent=[alphas[0], alphas[-1], betas[0], betas[-1]],
        cmap="magma",
    )
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"$KL_2$ standard error (MC)")


def plot_monotone_panel(ax, grid: Dict[str, Any]) -> None:
    """stacked-bar panel: KL1 alpha-monotonicity + per-alpha KL2 beta-monotonicity."""
    monotone_alpha = bool(grid.get("attrs", {}).get("monotone_alpha", True))
    mb = np.asarray(grid.get("monotone_beta_per_alpha", [])).astype(bool)
    alphas = np.asarray(grid["alphas"])
    if mb.size == 0:
        mb = np.ones(len(alphas), dtype=bool)
    ax.bar(alphas, mb.astype(float), width=(alphas[1] - alphas[0]) * 0.8,
           color=["tab:green" if m else "tab:red" for m in mb], alpha=0.7)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$KL_2$ monotone in $\beta$")
    ax.set_title(f"monotonicity ($KL_1$ alpha-monotone: {monotone_alpha})")


def plot_grid_figure(grid: Dict[str, Any],
                     k1_targets: List[float], beta_targets: List[float],
                     fig_path: str, has_se: bool = False) -> None:
    """assemble KL-grid panels into one figure.

    layout:
      row 0: KL1 curve | KL2 heatmap
      row 1: monotonicity | KL2 SE (if has_se) or empty
    """
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    plot_kl1_curve(fig.add_subplot(gs[0, 0]), grid, k1_targets)
    plot_kl2_heatmap(fig.add_subplot(gs[0, 1]), grid, k1_targets, beta_targets)
    plot_monotone_panel(fig.add_subplot(gs[1, 0]), grid)
    if has_se:
        plot_kl_se_heatmap(fig.add_subplot(gs[1, 1]), grid)
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_path}")


# ----------------------------------------------------------------------
# per-cell aggregate plots
# ----------------------------------------------------------------------

def plot_k1_inversion_check(ax,
                            cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                            k1_values: List[float]) -> None:
    """scatter (K1 prescribed, K1 realized) across all (cell, seed) with y=x.

    color encodes k1_idx (viridis).
    """
    n1 = len(k1_values)
    pts = []
    for (ai, _), recs in cells.items():
        col = plt.cm.viridis(ai / max(1, n1 - 1))
        for r in recs:
            a = r["attrs"]
            if "k1_pre" in a and "k1_real" in a:
                ax.scatter(a["k1_pre"], a["k1_real"], s=15, alpha=0.6, color=col)
                pts.append(float(a["k1_pre"]))
                pts.append(float(a["k1_real"]))
    if pts:
        lo, hi = min(pts), max(pts)
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3)
    ax.set_xlabel(r"$K_1$ prescribed")
    ax.set_ylabel(r"$K_1$ realized")
    ax.set_title(r"$K_1$: prescribed vs realized (hue: $k_1$ idx)")


def plot_k1_vs_k2_realized(ax,
                           cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                           k1_values: List[float],
                           beta_values: List[float]) -> None:
    r"""$K_1$ prescribed vs $K_2$ realized; one line per $\beta$ value.

    each marker is the median across seeds at one (K1, $\beta$) cell. with
    singleton $\beta$ this is a single monotone line; with swept $\beta$ this
    is one line per $\beta$, color-coded.
    """
    n2 = len(beta_values)
    by_beta: Dict[int, List[Tuple[float, float]]] = {}
    for (ai, bi), recs in cells.items():
        vals = [r["attrs"]["k2_real"] for r in recs
                if "k2_real" in r["attrs"]]
        if not vals:
            continue
        by_beta.setdefault(bi, []).append(
            (float(k1_values[ai]), float(np.median(vals))))
    for bi, pts in sorted(by_beta.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        col = plt.cm.plasma(bi / max(1, n2 - 1))
        ax.plot(xs, ys, "-o", color=col, markersize=5,
                label=rf"$\beta$={beta_values[bi]}")
    ax.set_xlabel(r"$K_1$ prescribed")
    ax.set_ylabel("$K_2$ realized\n(median over seeds)")
    ax.set_title(r"$K_1$ vs $K_2$ realized (one line per $\beta$)")
    if n2 > 1:
        ax.legend(fontsize=7)


def plot_ldr_histograms_grid(
    fig, gs_slice,
    cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
    k1_values: List[float], beta_values: List[float],
    extract_ldrs: Callable[[Dict[str, Any]], Optional[np.ndarray]],
    title_prefix: str = "log p0 / p1",
) -> None:
    """grid of ldr histograms, axes_ij = (k1_idx, beta_idx). seeds overlaid.

    args:
      fig: matplotlib figure.
      gs_slice: a SubplotSpec sliced to the desired sub-region.
      cells: per-cell records.
      k1_values, beta_values: sweep axis values (used for titles only).
      extract_ldrs: callable taking a single cell record -> ldrs np.ndarray.
                    None return is skipped.
      title_prefix: title prefix for each subplot.
    """
    n1 = len(k1_values)
    n2 = len(beta_values)
    sub_gs = gridspec.GridSpecFromSubplotSpec(
        n1, n2, subplot_spec=gs_slice, hspace=0.5, wspace=0.4)
    for ai in range(n1):
        for bi in range(n2):
            ax = fig.add_subplot(sub_gs[ai, bi])
            recs = cells.get((ai, bi), [])
            for r in recs:
                ldrs = extract_ldrs(r)
                if ldrs is None or len(ldrs) == 0:
                    continue
                ax.hist(ldrs, bins=50, density=True, alpha=0.25,
                        color="tab:blue")
            ax.axvline(0, color="black", linestyle="--", linewidth=0.4)
            ax.set_yscale("log")
            ax.set_title(
                rf"$K_1$={k1_values[ai]}, $\beta$={beta_values[bi]}",
                fontsize=7)
            ax.tick_params(labelsize=6)
    pos = gs_slice.get_position(fig)
    fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + 0.02,
             f"{title_prefix} histograms (rows: $k_1$ idx, cols: $\\beta$ idx)",
             ha="center", va="bottom", fontsize=10)


# ----------------------------------------------------------------------
# hardness aggregation
# ----------------------------------------------------------------------

def compute_hardness(
    cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
    k1_values: List[float], beta_values: List[float],
    extra_metrics: Optional[Dict[str, Callable[[Dict[str, Any]], float]]] = None,
) -> Dict[str, np.ndarray]:
    """compute per-cell hardness statistics.

    standard metrics:
      integrated_eldr, k1_err = |k1_real - k1_pre|, alpha_star, beta_set,
      k2_realized.

    note: no k2_err. beta is a knob (not prescribed), so |k2_real - beta|
    is a category error and was retired during the K2 -> beta refactor.

    args:
      cells: per-cell records.
      k1_values, beta_values: define the (n1, n2) shape.
      extra_metrics: dict[name -> callable(cell_record) -> float], experiment-specific.

    returns:
      dict[name -> array of shape [n1, n2, max_seeds]] padded with NaN where missing.
    """
    n1 = len(k1_values)
    n2 = len(beta_values)
    max_seeds = max((len(v) for v in cells.values()), default=0)

    standard = {
        "integrated_eldr": lambda r: r["attrs"].get("integrated_eldr", np.nan),
        "k1_err": lambda r: abs(r["attrs"].get("k1_real", np.nan)
                                 - r["attrs"].get("k1_pre", np.nan)),
        "alpha_star": lambda r: r["attrs"].get("alpha", np.nan),
        "beta_set": lambda r: r["attrs"].get("beta", np.nan),
        "k2_realized": lambda r: r["attrs"].get("k2_real", np.nan),
    }
    metrics = {**standard, **(extra_metrics or {})}

    out = {name: np.full((n1, n2, max_seeds), np.nan) for name in metrics}
    for (ai, bi), recs in cells.items():
        for si, r in enumerate(recs):
            for name, fn in metrics.items():
                try:
                    out[name][ai, bi, si] = float(fn(r))
                except Exception:
                    out[name][ai, bi, si] = np.nan
    return out


def print_hardness_table(hardness: Dict[str, np.ndarray],
                         k1_values: List[float],
                         beta_values: List[float]) -> None:
    """print median/iqr/mean/std summary across seeds, per cell, per metric."""
    print("\n" + "=" * 80)
    print("HARDNESS / VARIANCE SUMMARY (per (k1_idx, beta_idx), across seeds)")
    print("=" * 80)
    for name, arr in hardness.items():
        print(f"\n--- {name} ---")
        print(f"{'k1':>6} {'beta':>6} {'median':>9} {'IQR':>8} "
              f"{'mean':>9} {'std':>8} {'n_seeds':>8}")
        for ai, k1 in enumerate(k1_values):
            for bi, b in enumerate(beta_values):
                row = arr[ai, bi]
                row = row[~np.isnan(row)]
                if len(row) == 0:
                    continue
                q25, q50, q75 = np.percentile(row, [25, 50, 75])
                print(f"{k1:>6.2f} {b:>6.2f} {q50:>9.3f} "
                      f"{q75 - q25:>8.3f} {np.mean(row):>9.3f} "
                      f"{np.std(row):>8.3f} {len(row):>8d}")


def plot_hardness_boxplots(hardness: Dict[str, np.ndarray],
                           k1_values: List[float],
                           beta_values: List[float],
                           fig_path: str) -> None:
    """boxplot grid: one panel per metric, boxes per k1_idx (collapsing beta+seed).

    each box aggregates all (beta_idx, seed) at that k1_idx. richer view
    is the print_hardness_table output.
    """
    del beta_values  # used by table, not by boxplot
    names = list(hardness.keys())
    n = len(names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    n1 = len(k1_values)
    for i, name in enumerate(names):
        ax = axes[i // ncols, i % ncols]
        arr = hardness[name]
        per_k1 = []
        for ai in range(n1):
            row = arr[ai].ravel()
            row = row[~np.isnan(row)]
            per_k1.append(row if len(row) > 0 else np.array([np.nan]))
        bp = ax.boxplot(per_k1, tick_labels=[f"{k:.2f}" for k in k1_values],
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("tab:blue")
            patch.set_alpha(0.4)
        ax.set_xlabel(r"$K_1$ prescribed")
        ax.set_ylabel(name)
        ax.set_title(name)
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)
    fig.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_path}")


# ----------------------------------------------------------------------
# pendulum-specific extraction + panels
# ----------------------------------------------------------------------

def extract_pendulum_ldrs(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    r"""log $p_O$ - log $p_E$ at pstar samples = log($p_0/p_1$)."""
    log_p = rec.get("log_p_pstar")
    if log_p is None:
        return None
    return log_p[:, 1] - log_p[:, 2]


def plot_phase_space(ax, samples: np.ndarray, T: int, title: str,
                     t_idx: int) -> None:
    r"""scatter ($\theta$, $\dot\theta$) marginal at timestep t_idx.

    args:
      ax: axes.
      samples: [N, (T+1)*3] flat, columns = ($\theta$, $\dot\theta$, action) per step.
      T: trajectory length T (so samples reshape to [N, T+1, 3]).
      title: subplot title.
      t_idx: which timestep index in [0, T] to plot.
    """
    N = samples.shape[0]
    arr = samples.reshape(N, T + 1, 3)
    theta = arr[:, t_idx, 0]
    theta_dot = arr[:, t_idx, 1]
    ax.scatter(theta, theta_dot, s=3, alpha=0.3, color="tab:blue")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot\theta$")
    ax.set_title(title)


def plot_phase_space_grid(fig, gs_slice,
                          cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                          k1_values: List[float],
                          beta_values: List[float],
                          T: int) -> None:
    """grid of phase-space scatters at t=0 (left) and t=T (right) for pstar samples.

    one row per (k1_idx, beta_idx) cell; first seed only.
    """
    items = sorted(cells.keys())
    n_rows = len(items)
    if n_rows == 0:
        return
    sub_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, 2, subplot_spec=gs_slice, hspace=0.5, wspace=0.3)
    for ri, (ai, bi) in enumerate(items):
        rec = cells[(ai, bi)][0]
        if "samples_pstar" not in rec:
            continue
        ax0 = fig.add_subplot(sub_gs[ri, 0])
        ax_T = fig.add_subplot(sub_gs[ri, 1])
        # column titles on top row only; row info as left-col ylabel
        col_t0 = "initial (t=0)" if ri == 0 else ""
        col_tT = f"terminal (t={T})" if ri == 0 else ""
        plot_phase_space(ax0, rec["samples_pstar"], T, col_t0, t_idx=0)
        plot_phase_space(ax_T, rec["samples_pstar"], T, col_tT, t_idx=T)
        ax0.set_ylabel(
            rf"$K_1$={k1_values[ai]}, $\beta$={beta_values[bi]}"
            "\n" r"$\dot\theta$", fontsize=8)


def plot_bellman_residuals(ax,
                           cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                           k1_values: List[float]) -> None:
    """scatter q_O_residual vs K1 (log-log); q_E_residual as horizontal line."""
    q_E = None
    for (ai, _), recs in cells.items():
        for r in recs:
            attrs = r["attrs"]
            if q_E is None and "q_E_residual" in attrs:
                q_E = float(attrs["q_E_residual"])
            if "q_O_residual" in attrs:
                ax.scatter(k1_values[ai], float(attrs["q_O_residual"]),
                           s=15, alpha=0.6, color="tab:orange")
    if q_E is not None:
        ax.axhline(q_E, color="tab:blue", linestyle="--", linewidth=1,
                   label=rf"$q_E$ residual = {q_E:.2e}")
        ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$K_1$ prescribed")
    ax.set_ylabel("Bellman residual")
    ax.set_title(r"FQI Bellman residuals ($q_O$ across cells, $q_E$ baseline)")


# ----------------------------------------------------------------------
# lightweight figure assembly + main
# ----------------------------------------------------------------------

def plot_lightweight_figure(cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                            config: Dict[str, Any]) -> None:
    """assemble lightweight figure to figures_dir/datagen_diagnostic.png."""
    k1_values = config["kl_targets"]["k1_values"]
    beta_values = config["kl_targets"]["beta_values"]
    T = int(config["trajectory"]["T"])
    n1 = len(k1_values)
    n2 = len(beta_values)
    n_cells = len(cells)

    # rows: (0) summary 1x4, (1) ldr histogram block n1*n2,
    #       (2) phase space block n_cells rows, (3) pca scatter 1 row
    fig = plt.figure(figsize=(4 * max(n1, n2, 4), 4 + 2 * n1 + 2 * n_cells + 4))
    gs = gridspec.GridSpec(4, 1, figure=fig,
                           height_ratios=[1, n1 * 1.2, max(1, n_cells), 2],
                           hspace=0.5)

    sub0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0],
                                             wspace=0.4)
    plot_k1_inversion_check(fig.add_subplot(sub0[0, 0]), cells, k1_values)
    plot_k1_vs_k2_realized(fig.add_subplot(sub0[0, 1]), cells,
                           k1_values, beta_values)
    plot_bellman_residuals(fig.add_subplot(sub0[0, 2]), cells, k1_values)

    plot_ldr_histograms_grid(fig, gs[1], cells, k1_values, beta_values,
                             extract_pendulum_ldrs,
                             title_prefix=r"$\log p_O - \log p_E$ at $p^*$")

    plot_phase_space_grid(fig, gs[2], cells, k1_values, beta_values, T)

    pca_cell = next(iter(cells.values()), None)
    if pca_cell is not None and "samples_pstar" in pca_cell[0]:
        sub3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[3])
        ax_pca = fig.add_subplot(sub3[0, 0])
        rec = pca_cell[0]
        plot_pca_panel(ax_pca, rec["samples_pstar"], rec["samples_p0"],
                       rec["samples_p1"],
                       title=f"PCA, first available cell, seed={rec['seed']}")

    fig_dir = Path(config["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "datagen_diagnostic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))

    if args.show_grid:
        path = find_grid_cache(config)
        grid = load_kl_grid(path, expected=("KL1", "KL2", "alphas", "betas"))
        fig_dir = Path(config["figures_dir"])
        plot_grid_figure(grid,
                         k1_targets=config["kl_targets"]["k1_values"],
                         beta_targets=config["kl_targets"]["beta_values"],
                         fig_path=str(fig_dir / "grid_diagnostic.png"),
                         has_se=("KL2_se" in grid))
        return

    paths_by_idx = enumerate_cell_paths(config)
    if not paths_by_idx:
        print(f"no per-cell HDF5 files found under {config['data_dir']}. "
              f"run step1_create_data.py first, or use --show-grid.")
        return

    cells = collect_cells(paths_by_idx, KEY_MAP)
    plot_lightweight_figure(cells, config)

    hardness = compute_hardness(
        cells,
        config["kl_targets"]["k1_values"],
        config["kl_targets"]["beta_values"],
        extra_metrics={
            "KL_O_E": lambda r: r["attrs"].get("KL_O_E", np.nan),
            "KL_E_mix": lambda r: r["attrs"].get("KL_E_mix", np.nan),
            "mc_se": lambda r: r["attrs"].get("mc_se", np.nan),
            "q_O_residual": lambda r: r["attrs"].get("q_O_residual", np.nan),
        },
    )
    print_hardness_table(hardness,
                         config["kl_targets"]["k1_values"],
                         config["kl_targets"]["beta_values"])
    plot_hardness_boxplots(hardness,
                           config["kl_targets"]["k1_values"],
                           config["kl_targets"]["beta_values"],
                           str(Path(config["figures_dir"]) / "datagen_variance.png"))


if __name__ == "__main__":
    main()
