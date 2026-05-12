"""datagen diagnostic for pendulum trajectory ELDR experiment.

mirrors the lightweight half of mnist_eldr_estimation/diagnostic_datagen.py.
no model loading is needed: per-cell HDF5s already store analytic
log-densities under three policies (pi^beta*, pi_O, pi_E).

modes:
  default:     read cells under data_dir/k1_{i}_k2_{j}_seed_{s}.h5,
               produce datagen_diagnostic.png + datagen_variance.png.
  --show-grid: read traj_kl_cache/traj_grid_{hash}.h5 and produce
               grid_diagnostic.png. works with no per-cell HDF5 yet.

panels:
  - prescribed-vs-realized scatter for K1 and K2
  - (alpha*, beta*) coverage in 2-d
  - LDR histograms grid (per (k1_idx, k2_idx); seeds overlaid)
  - PCA of flat trajectories
  - phase-space (theta, theta_dot) at t=0 and t=T
  - Bellman residual scatter (q_E, q_O) across cells
  - hardness boxplot grid + summary table
"""
import argparse
import os
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yaml

from experiments.pendulum.step1_create_data import (
    _build_env_and_q_cfg,
    _resolve_reward,
)
from experiments.utils.alpha_grid import make_alphas
from experiments.utils.diagnostic_kl_grid import (
    collect_cells,
    compute_hardness,
    load_kl_grid,
    plot_alpha_beta_coverage,
    plot_grid_figure,
    plot_hardness_boxplots,
    plot_ldr_histograms_grid,
    plot_pca_panel,
    plot_prescribed_vs_realized,
    print_hardness_table,
)
from experiments.utils.prescribed_kls import hash_pendulum_cfg


KEY_MAP = {
    "k1_pre": "K1_prescribed",
    "k2_pre": "K2_prescribed",
    "k1_real": "K1_realized",
    "k2_real": "K2_realized",
    "alpha": "alpha_star",
    "beta": "beta_star",
    "integrated_eldr": "integrated_eldr",
}


def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="experiments/pendulum/config.yaml")
    p.add_argument("--show-grid", action="store_true",
                   help="plot the cached traj_kl grid (no per-cell HDF5 needed)")
    return p.parse_args(args)


def build_grid_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """reconstruct the cfg dict consumed by hash_pendulum_cfg.

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
            f"`python -m experiments.pendulum.step1_create_data --smoke` "
            f"to build it first."
        )
    return str(path)


def enumerate_cell_paths(config: Dict[str, Any]) -> Dict[Tuple[int, int], List[Tuple[int, str]]]:
    """walk data_dir for k1_{i}_k2_{j}_seed_{s}.h5; group by (k1_idx, k2_idx).

    returns dict[(k1_idx, k2_idx)] -> list of (seed, path); only existing files included.
    """
    data_dir = Path(config["data_dir"])
    k1_values = [float(v) for v in config["kl_targets"]["k1_values"]]
    k2_values = [float(v) for v in config["kl_targets"]["k2_values"]]
    # cast threshold defensively: yaml parses "1.0e9" as str under YAML 1.1.
    hard_threshold = float(config["kl_targets"]["hard_corner_threshold"])
    seeds_default = int(config["kl_targets"]["seeds_default"])
    seeds_hard = int(config["kl_targets"]["seeds_hard"])

    out: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
    for k1_idx, k2_idx in product(range(len(k1_values)), range(len(k2_values))):
        K1 = k1_values[k1_idx]
        K2 = k2_values[k2_idx]
        n_seeds = (seeds_hard if (K1 >= hard_threshold and K2 >= hard_threshold)
                   else seeds_default)
        seeds = []
        for seed in range(n_seeds):
            path = data_dir / f"k1_{k1_idx}_k2_{k2_idx}_seed_{seed}.h5"
            if path.exists():
                seeds.append((seed, str(path)))
        if seeds:
            out[(k1_idx, k2_idx)] = seeds
    return out


def extract_pendulum_ldrs(rec: Dict[str, Any]) -> np.ndarray:
    """log p_O - log p_E at pstar samples = log(p0/p1)."""
    log_p = rec.get("log_p_pstar")
    if log_p is None:
        return None
    return log_p[:, 1] - log_p[:, 2]


def plot_phase_space(ax, samples: np.ndarray, T: int, title: str,
                     t_idx: int) -> None:
    """scatter (theta, theta_dot) marginal at timestep t_idx of the trajectory.

    args:
      ax: axes.
      samples: [N, (T+1)*3] flat, columns = (theta, theta_dot, action) per timestep.
      T: trajectory length T (so samples reshape to [N, T+1, 3]).
      title: subplot title.
      t_idx: which timestep index in [0, T] to plot.
    """
    N = samples.shape[0]
    arr = samples.reshape(N, T + 1, 3)
    theta = arr[:, t_idx, 0]
    theta_dot = arr[:, t_idx, 1]
    ax.scatter(theta, theta_dot, s=3, alpha=0.3, color="tab:blue")
    ax.set_xlabel("theta")
    ax.set_ylabel("theta_dot")
    ax.set_title(title)


def plot_phase_space_grid(fig, gs_slice,
                          cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                          k1_values: List[float], k2_values: List[float],
                          T: int) -> None:
    """grid of phase-space scatters at t=0 (left) and t=T (right) for pstar samples.

    one row per (k1_idx, k2_idx) cell; first seed only (overlay would be unreadable).
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
        title_pre = f"K1={k1_values[ai]}, K2={k2_values[bi]}"
        plot_phase_space(ax0, rec["samples_pstar"], T,
                         f"{title_pre}, t=0", t_idx=0)
        plot_phase_space(ax_T, rec["samples_pstar"], T,
                         f"{title_pre}, t=T", t_idx=T)


def plot_bellman_residuals(ax,
                           cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                           k1_values: List[float]) -> None:
    """scatter of q_O_residual vs k1_idx; horizontal line for q_E_residual."""
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
                   label=f"q_E residual = {q_E:.2e}")
        ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("K1 prescribed")
    ax.set_ylabel("Bellman residual")
    ax.set_title("FQI Bellman residuals (q_O across cells, q_E baseline)")


def plot_lightweight_figure(cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                            config: Dict[str, Any]) -> None:
    """assemble lightweight figure to figures_dir/datagen_diagnostic.png."""
    k1_values = config["kl_targets"]["k1_values"]
    k2_values = config["kl_targets"]["k2_values"]
    T = int(config["trajectory"]["T"])
    n1 = len(k1_values)
    n2 = len(k2_values)
    n_cells = len(cells)

    # rows: (1) summary 1x4, (2) ldr histogram block n1*n2,
    #       (3) phase space block n_cells rows, (4) pca scatter 1 row
    fig = plt.figure(figsize=(4 * max(n1, n2, 4), 4 + 2 * n1 + 2 * n_cells + 4))
    gs = gridspec.GridSpec(4, 1, figure=fig,
                           height_ratios=[1, n1 * 1.2, max(1, n_cells), 2],
                           hspace=0.5)

    # row 0: summary panels
    sub0 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0],
                                             wspace=0.4)
    ax_k1 = fig.add_subplot(sub0[0, 0])
    ax_k2 = fig.add_subplot(sub0[0, 1])
    ax_ab = fig.add_subplot(sub0[0, 2])
    ax_br = fig.add_subplot(sub0[0, 3])
    plot_prescribed_vs_realized(ax_k1, ax_k2, cells, k1_values, k2_values)
    plot_alpha_beta_coverage(ax_ab, cells, k1_values, k2_values)
    plot_bellman_residuals(ax_br, cells, k1_values)

    # row 1: ldr histograms
    plot_ldr_histograms_grid(fig, gs[1], cells, k1_values, k2_values,
                             extract_pendulum_ldrs,
                             title_prefix="log p_O - log p_E at pstar")

    # row 2: phase space at t=0, T
    plot_phase_space_grid(fig, gs[2], cells, k1_values, k2_values, T)

    # row 3: pca on flat trajectories - use first cell with full samples
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
                         k2_targets=config["kl_targets"]["k2_values"],
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
        config["kl_targets"]["k2_values"],
        extra_metrics={
            "KL_O_E": lambda r: r["attrs"].get("KL_O_E", np.nan),
            "KL_E_mix": lambda r: r["attrs"].get("KL_E_mix", np.nan),
            "mc_se": lambda r: r["attrs"].get("mc_se", np.nan),
            "q_O_residual": lambda r: r["attrs"].get("q_O_residual", np.nan),
        },
    )
    print_hardness_table(hardness,
                         config["kl_targets"]["k1_values"],
                         config["kl_targets"]["k2_values"])
    plot_hardness_boxplots(hardness,
                           config["kl_targets"]["k1_values"],
                           config["kl_targets"]["k2_values"],
                           str(Path(config["figures_dir"]) / "datagen_variance.png"))


if __name__ == "__main__":
    main()
