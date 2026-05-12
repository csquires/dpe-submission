"""shared lightweight diagnostics for 2-d (k1, k2) prescribed-kl ex.

both pendulum and smodice prescribe (k1, k2) targets via inversion of a 2-d
KL grid (KL1[alpha], KL2[alpha, beta]). both store per-cell HDF5s with
analytic log-densities/LDRs. this module hosts the shared plot helpers.

scope:
  - cell HDF5 loader with attr-name normalization
  - cached KL-grid loader and plot helpers (KL1, KL2, monotone, SE)
  - prescribed-vs-realized scatter
  - (alpha*, beta*) coverage in 2-d
  - LDR histogram grid (cells x seeds)
  - hardness boxplot grid + summary table

per-experiment files own:
  - HDF5 path resolution (smodice fans out by encoding/sigma)
  - sample-shape-specific panels (trajectory phase, encoding-aware PCA)
"""
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


STD_ATTR_KEYS = ("k1_pre", "k2_pre", "k1_real", "k2_real",
                 "alpha", "beta", "integrated_eldr")


def _read_attrs(h: h5py.File, key_map: Dict[str, str]) -> Dict[str, Any]:
    """copy attrs out of an hdf5 handle, renaming via key_map.

    args:
      h: open h5py file handle.
      key_map: standard_name -> attr_name_in_file. unmatched keys are skipped.

    returns:
      dict containing standard names that were present, plus all raw attrs
      (under their original names) for downstream experiment-specific use.
    """
    out = {k: v for k, v in h.attrs.items()}
    for std, raw in key_map.items():
        if raw in h.attrs:
            out[std] = float(h.attrs[raw]) if np.isscalar(h.attrs[raw]) else h.attrs[raw]
    return out


def load_cell(path: str, key_map: Dict[str, str],
              datasets: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """load datasets and (normalized) attrs from one per-cell hdf5.

    args:
      path: cell hdf5 path.
      key_map: standard_name -> attr_name in this experiment's hdf5.
      datasets: optional explicit dataset whitelist. if None, load all.

    returns:
      dict with keys:
        attrs: dict of attrs (standard names + raw).
        <dataset_name>: numpy array, for each requested dataset that exists.
    """
    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        out["attrs"] = _read_attrs(f, key_map)
        ds_names = datasets if datasets is not None else list(f.keys())
        for name in ds_names:
            if name in f:
                out[name] = f[name][:]
    return out


def collect_cells(
    paths_by_idx: Dict[Tuple[int, int], List[Tuple[int, str]]],
    key_map: Dict[str, str],
    datasets: Optional[Iterable[str]] = None,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """load all cells found in paths_by_idx into a nested dict.

    args:
      paths_by_idx: dict[(k1_idx, k2_idx)] -> list of (seed, path) tuples.
                    only cells whose hdf5 exists should be passed in.
      key_map: standard_name -> attr_name (passed to load_cell).
      datasets: optional dataset whitelist passed to load_cell.

    returns:
      dict[(k1_idx, k2_idx)] -> list of cell dicts (one per seed, in input order).
      cells with zero seeds are omitted.
    """
    out: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for idx, seeds in paths_by_idx.items():
        if not seeds:
            continue
        records = []
        for seed, path in seeds:
            rec = load_cell(path, key_map, datasets)
            rec["seed"] = seed
            records.append(rec)
        out[idx] = records
    return out


# ============================================================
# cached kl-grid plotting (--show-grid mode)
# ============================================================

def load_kl_grid(cache_path: str, expected: Iterable[str]) -> Dict[str, Any]:
    """load a cached kl-grid hdf5; return all datasets and attrs.

    args:
      cache_path: path to the grid hdf5 (smodice grid_{hash}.h5 or
                  pendulum traj_grid_{hash}.h5).
      expected: dataset names that must be present; missing keys raise.

    returns:
      dict of arrays + sub-dict 'attrs'.
    """
    out: Dict[str, Any] = {}
    with h5py.File(cache_path, "r") as f:
        for name in expected:
            if name not in f:
                raise KeyError(f"{cache_path} missing dataset {name}")
            out[name] = f[name][:]
        for name in f.keys():
            if name not in out:
                out[name] = f[name][:]
        out["attrs"] = {k: v for k, v in f.attrs.items()}
    return out


def plot_kl1_curve(ax, grid: Dict[str, Any], k1_targets: List[float]) -> None:
    """line plot of KL1[alpha] with optional SE band, K1 targets overlaid as
    horizontal lines.
    """
    ax.plot(grid["alphas"], grid["KL1"], "-o", markersize=3, color="tab:blue")
    if "KL1_se" in grid:
        lo = grid["KL1"] - grid["KL1_se"]
        hi = grid["KL1"] + grid["KL1_se"]
        ax.fill_between(grid["alphas"], lo, hi, alpha=0.2, color="tab:blue")
    for k1 in k1_targets:
        ax.axhline(k1, color="tab:red", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("alpha")
    ax.set_ylabel("KL1 = KL(d_E || d_O)")
    ax.set_title("KL1[alpha] (red dashes: K1 targets)")


def plot_kl2_heatmap(ax, grid: Dict[str, Any],
                     k1_targets: List[float], k2_targets: List[float]) -> None:
    """heatmap of KL2[alpha, beta]. overlays achievable (K1, K2) targets:
    for each (k1*, k2*), find approximate (alpha, beta) cell.
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

    # locate (alpha, beta) for each (K1, K2) target via the same idea as
    # prescribe(): invert KL1 first to get alpha*, then KL2 at alpha* for beta*.
    KL1 = np.asarray(grid["KL1"])
    if np.all(np.diff(KL1) > 0):
        for k1 in k1_targets:
            if KL1[0] <= k1 <= KL1[-1]:
                a_star = float(np.interp(k1, KL1, alphas))
                # nearest alpha row
                ai = int(np.argmin(np.abs(alphas - a_star)))
                slice_ = KL2[ai]
                # decreasing in beta
                if np.all(np.diff(slice_) <= 1e-9 * max(1.0, np.abs(slice_).max())):
                    for k2 in k2_targets:
                        if slice_[-1] <= k2 <= slice_[0]:
                            b_star = float(np.interp(k2, slice_[::-1], betas[::-1]))
                            ax.plot([a_star], [b_star], marker="x",
                                    markersize=8, color="white", markeredgewidth=2)
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title("KL2[alpha, beta] (white x: (K1, K2) targets)")


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
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title("KL2 standard error (MC)")


def plot_monotone_panel(ax, grid: Dict[str, Any]) -> None:
    """stacked-bar panel showing KL1 monotonicity and per-alpha KL2 monotonicity."""
    monotone_alpha = bool(grid.get("attrs", {}).get("monotone_alpha", True))
    mb = np.asarray(grid.get("monotone_beta_per_alpha", [])).astype(bool)
    alphas = np.asarray(grid["alphas"])
    if mb.size == 0:
        # smodice grid: build_kl_grid asserts monotonicity so always passes
        mb = np.ones(len(alphas), dtype=bool)
    ax.bar(alphas, mb.astype(float), width=(alphas[1] - alphas[0]) * 0.8,
           color=["tab:green" if m else "tab:red" for m in mb], alpha=0.7)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("alpha")
    ax.set_ylabel("KL2 monotone in beta")
    ax.set_title(f"monotonicity (KL1 alpha-monotone: {monotone_alpha})")


def plot_grid_figure(grid: Dict[str, Any],
                     k1_targets: List[float], k2_targets: List[float],
                     fig_path: str, has_se: bool = False) -> None:
    """assemble grid panels into one figure.

    layout:
      row 0: KL1 curve | KL2 heatmap
      row 1: monotonicity | KL2 SE (pendulum only) or empty
    """
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    plot_kl1_curve(fig.add_subplot(gs[0, 0]), grid, k1_targets)
    plot_kl2_heatmap(fig.add_subplot(gs[0, 1]), grid, k1_targets, k2_targets)
    plot_monotone_panel(fig.add_subplot(gs[1, 0]), grid)
    if has_se:
        plot_kl_se_heatmap(fig.add_subplot(gs[1, 1]), grid)
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_path}")


# ============================================================
# per-cell aggregate plots
# ============================================================

def _flat_attr(cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
               key: str) -> List[Tuple[int, int, int, float]]:
    """yield (k1_idx, k2_idx, seed, value) for every cell with attr[key]."""
    out = []
    for (ai, bi), recs in cells.items():
        for r in recs:
            v = r["attrs"].get(key, None)
            if v is None:
                continue
            out.append((ai, bi, r["seed"], float(v)))
    return out


def plot_prescribed_vs_realized(ax_k1, ax_k2,
                                cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                                k1_values: List[float], k2_values: List[float]) -> None:
    """scatter (prescribed, realized) for K1 and K2 across all (cell, seed)."""
    n1 = len(k1_values)
    n2 = len(k2_values)
    for (ai, bi), recs in cells.items():
        col = plt.cm.viridis(ai / max(1, n1 - 1))
        for r in recs:
            a = r["attrs"]
            if "k1_pre" in a and "k1_real" in a:
                ax_k1.scatter(a["k1_pre"], a["k1_real"], s=15, alpha=0.6, color=col)
            if "k2_pre" in a and "k2_real" in a:
                col2 = plt.cm.plasma(bi / max(1, n2 - 1))
                ax_k2.scatter(a["k2_pre"], a["k2_real"], s=15, alpha=0.6, color=col2)
    pts1 = _flat_attr(cells, "k1_pre") + _flat_attr(cells, "k1_real")
    pts2 = _flat_attr(cells, "k2_pre") + _flat_attr(cells, "k2_real")
    if pts1:
        lo = min(p[3] for p in pts1)
        hi = max(p[3] for p in pts1)
        ax_k1.plot([lo, hi], [lo, hi], "k--", alpha=0.3)
    if pts2:
        lo = min(p[3] for p in pts2)
        hi = max(p[3] for p in pts2)
        ax_k2.plot([lo, hi], [lo, hi], "k--", alpha=0.3)
    ax_k1.set_xlabel("K1 prescribed")
    ax_k1.set_ylabel("K1 realized")
    ax_k1.set_title("K1: prescribed vs realized (color = k1_idx)")
    ax_k2.set_xlabel("K2 prescribed")
    ax_k2.set_ylabel("K2 realized")
    ax_k2.set_title("K2: prescribed vs realized (color = k2_idx)")


def plot_alpha_beta_coverage(ax,
                             cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                             k1_values: List[float], k2_values: List[float]) -> None:
    """scatter of (alpha*, beta*) across all (cell, seed).

    color encodes k1_idx (hue) and k2_idx (marker size).
    """
    n1 = max(1, len(k1_values) - 1)
    for (ai, bi), recs in cells.items():
        col = plt.cm.viridis(ai / n1)
        size = 20 + 30 * (bi / max(1, len(k2_values) - 1))
        for r in recs:
            a = r["attrs"]
            if "alpha" in a and "beta" in a:
                ax.scatter(a["alpha"], a["beta"], s=size, alpha=0.6,
                           color=col, edgecolors="black", linewidths=0.3)
    ax.set_xlabel("alpha*")
    ax.set_ylabel("beta*")
    ax.set_title("(alpha*, beta*) coverage  (hue: k1_idx, size: k2_idx)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)


def plot_ldr_histograms_grid(
    fig, gs_slice,
    cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
    k1_values: List[float], k2_values: List[float],
    extract_ldrs: Callable[[Dict[str, Any]], np.ndarray],
    title_prefix: str = "log p0 / p1",
) -> None:
    """grid of ldr histograms, axes_ij = (k1_idx, k2_idx). seeds overlaid.

    args:
      fig: matplotlib figure.
      gs_slice: a SubplotSpec sliced to the desired sub-region.
      cells: per-cell records.
      k1_values, k2_values: sweep axis values (used for titles only).
      extract_ldrs: callable taking a single cell record -> ldrs np.ndarray.
                    None return is skipped.
      title_prefix: title prefix for each subplot.
    """
    n1 = len(k1_values)
    n2 = len(k2_values)
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
            ax.set_title(
                f"K1={k1_values[ai]}, K2={k2_values[bi]}", fontsize=7)
            ax.tick_params(labelsize=6)
    # add single super-row label via the first axes
    fig.text(0.5, 1.0 - 0.005,
             f"{title_prefix} histograms (rows: k1_idx, cols: k2_idx)",
             ha="center", fontsize=10)


def compute_hardness(
    cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
    k1_values: List[float], k2_values: List[float],
    extra_metrics: Optional[Dict[str, Callable[[Dict[str, Any]], float]]] = None,
) -> Dict[str, np.ndarray]:
    """compute per-cell hardness statistics.

    standard metrics:
      integrated_eldr, k1_err = |k1_real - k1_pre|, k2_err, alpha_star, beta_star.

    extra_metrics: dict[name -> callable(cell_record) -> float], experiment-specific.

    returns:
      dict[name -> array of shape [n1, n2, max_seeds]] padded with NaN where missing.
    """
    n1 = len(k1_values)
    n2 = len(k2_values)
    max_seeds = max((len(v) for v in cells.values()), default=0)

    standard = {
        "integrated_eldr": lambda r: r["attrs"].get("integrated_eldr", np.nan),
        "k1_err": lambda r: abs(r["attrs"].get("k1_real", np.nan)
                                 - r["attrs"].get("k1_pre", np.nan)),
        "k2_err": lambda r: abs(r["attrs"].get("k2_real", np.nan)
                                 - r["attrs"].get("k2_pre", np.nan)),
        "alpha_star": lambda r: r["attrs"].get("alpha", np.nan),
        "beta_star": lambda r: r["attrs"].get("beta", np.nan),
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
                         k1_values: List[float], k2_values: List[float]) -> None:
    """print median/iqr/mean/std summary across seeds, per cell, per metric."""
    print("\n" + "=" * 80)
    print("HARDNESS / VARIANCE SUMMARY (per (k1_idx, k2_idx), across seeds)")
    print("=" * 80)
    for name, arr in hardness.items():
        print(f"\n--- {name} ---")
        print(f"{'k1':>6} {'k2':>6} {'median':>9} {'IQR':>8} "
              f"{'mean':>9} {'std':>8} {'n_seeds':>8}")
        for ai, k1 in enumerate(k1_values):
            for bi, k2 in enumerate(k2_values):
                row = arr[ai, bi]
                row = row[~np.isnan(row)]
                if len(row) == 0:
                    continue
                q25, q50, q75 = np.percentile(row, [25, 50, 75])
                print(f"{k1:>6.2f} {k2:>6.2f} {q50:>9.3f} "
                      f"{q75 - q25:>8.3f} {np.mean(row):>9.3f} "
                      f"{np.std(row):>8.3f} {len(row):>8d}")


def plot_hardness_boxplots(hardness: Dict[str, np.ndarray],
                           k1_values: List[float], k2_values: List[float],
                           fig_path: str) -> None:
    """boxplot grid: one panel per metric. each panel shows boxes per (k1_idx).

    each box aggregates across all (k2_idx, seed) at that k1_idx. the cell
    structure is collapsed to keep panels readable; richer view is the
    print_hardness_table output.
    """
    names = [n for n in hardness.keys()]
    n = len(names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    n1 = len(k1_values)
    for i, name in enumerate(names):
        ax = axes[i // ncols, i % ncols]
        arr = hardness[name]  # [n1, n2, max_seeds]
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
        ax.set_xlabel("K1 prescribed")
        ax.set_ylabel(name)
        ax.set_title(name)
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)
    fig.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_path}")


# ============================================================
# pca helper (encoding-agnostic, used by both wrappers)
# ============================================================

def plot_pca_panel(ax, pstar: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                   title: str, n_plot: int = 1500) -> None:
    """fit pca on pstar; scatter all three sets in 2-d.

    args:
      ax: matplotlib axes.
      pstar, p0, p1: [N, D] arrays (flat).
      title: subplot title.
      n_plot: per-set max points.
    """
    from sklearn.decomposition import PCA
    if pstar.ndim != 2 or p0.ndim != 2 or p1.ndim != 2:
        raise ValueError("pstar/p0/p1 must be 2-d [N, D]")
    pca = PCA(n_components=2)
    pca.fit(pstar)
    rng = np.random.RandomState(0)
    pick = lambda x: x[rng.choice(len(x), min(n_plot, len(x)), replace=False)]
    ps = pca.transform(pick(pstar))
    p0t = pca.transform(pick(p0))
    p1t = pca.transform(pick(p1))
    ax.scatter(ps[:, 0], ps[:, 1], s=2, alpha=0.2, c="gray", label="p*")
    ax.scatter(p0t[:, 0], p0t[:, 1], s=2, alpha=0.2, c="tab:blue", label="p0")
    ax.scatter(p1t[:, 0], p1t[:, 1], s=2, alpha=0.2, c="tab:orange", label="p1")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize=6, markerscale=4)
