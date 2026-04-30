"""datagen diagnostic for SMODICE occupancy ELDR experiment.

mirrors the lightweight half of mnist_eldr_estimation/diagnostic_datagen.py.
no model loading needed: per-cell HDF5s already store analytic
discrete + smoothed LDRs.

modes:
  default:     read cells under data_dir/{encoding_type}/{sigma_dir}/
               kl1_{i}_kl2_{j}_seed_{s}.h5,
               produce datagen_diagnostic.png + datagen_variance.png.
  --show-grid: read grid_cache/grid_{hash}.h5 and produce
               grid_diagnostic.png. works with no per-cell HDF5 yet.

panels:
  - prescribed-vs-realized scatter for K1 and K2
  - (alpha*, beta*) coverage in 2-d
  - LDR histograms grid (per (k1_idx, k2_idx); seeds overlaid; smoothed)
  - PCA of samples (skipped for onehot encodings)
  - discrete vs smoothed LDR scatter (skipped for onehot encodings)
  - hardness boxplot grid + summary table

cli:
  --encoding-type:  override config["encoding"]["type"] when fanning out
                    over data_dir subdirs.
  --sigma:          override config["encoding"]["sigma"].
"""
import argparse
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yaml

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
from experiments.utils.prescribed_kls import hash_mdp_config


KEY_MAP = {
    "k1_pre": "prescribed_K1",
    "k2_pre": "prescribed_K2",
    "k1_real": "realized_K1",
    "k2_real": "realized_K2",
    "alpha": "alpha_star",
    "beta": "beta_star",
    "integrated_eldr": "integrated_eldr",
}


def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="experiments/smodice_eldr_estimation/config.yaml")
    p.add_argument("--show-grid", action="store_true",
                   help="plot the cached kl grid (no per-cell HDF5 needed)")
    p.add_argument("--encoding-type", default=None,
                   help="override config encoding.type for cell discovery")
    p.add_argument("--sigma", type=float, default=None,
                   help="override config encoding.sigma for cell discovery")
    return p.parse_args(args)


def build_mdp_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """rebuild the mdp_cfg consumed by hash_mdp_config; must match step1's."""
    gw = config["gridworld"]
    return {
        "L": gw["L"],
        "p_slip": gw["p_slip"],
        "gamma": gw["gamma"],
        "terminals": gw["terminals"],
        "expert_goal": gw["expert_goal"],
        "anti_goal": gw["anti_goal"],
        "tau": gw["tau"],
        "G_alpha": config["kl_grid"]["G_alpha"],
        "G_beta": config["kl_grid"]["G_beta"],
        "mu0_kind": gw.get("mu0_kind", "uniform"),
        "mu0_centers": gw.get("mu0_centers", []),
        "reward_kind": gw.get("reward_kind", "sparse"),
        "reward_sigma": float(gw.get("reward_sigma", 1.0)),
    }


def find_grid_cache(config: Dict[str, Any]) -> str:
    """resolve cached kl-grid path; raise if missing."""
    h = hash_mdp_config(build_mdp_cfg(config))
    cache_dir = config["kl_grid"]["cache_dir"]
    path = Path(cache_dir) / f"grid_{h}.h5"
    if not path.exists():
        raise FileNotFoundError(
            f"no cached grid at {path}. run "
            f"`python -m experiments.smodice_eldr_estimation.step1_create_data --smoke` "
            f"to build it first."
        )
    return str(path)


def resolve_data_subdir(config: Dict[str, Any],
                        encoding_type: str = None,
                        sigma: float = None) -> Path:
    """compute data subdir for a given encoding/sigma combination.

    matches step1_create_data.per_cell's path layout exactly.
    """
    enc = encoding_type or config["encoding"]["type"]
    if enc.startswith("onehot"):
        sigma_dir = "sigma_na"
    else:
        sg = sigma if sigma is not None else config["encoding"]["sigma"]
        sigma_dir = f"sigma_{sg:.3f}"
    return Path(config["data_dir"]) / enc / sigma_dir


def enumerate_cell_paths(config: Dict[str, Any], data_subdir: Path
                         ) -> Dict[Tuple[int, int], List[Tuple[int, str]]]:
    """walk data_subdir for kl1_{i}_kl2_{j}_seed_{s}.h5; group by (k1_idx, k2_idx)."""
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
            path = data_subdir / f"kl1_{k1_idx}_kl2_{k2_idx}_seed_{seed}.h5"
            if path.exists():
                seeds.append((seed, str(path)))
        if seeds:
            out[(k1_idx, k2_idx)] = seeds
    return out


def extract_smoothed_ldrs(rec: Dict[str, Any]) -> np.ndarray:
    return rec.get("true_ldrs_smoothed")


def plot_discrete_vs_smoothed(ax,
                              cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                              n_per_cell: int = 200) -> None:
    """scatter discrete vs smoothed LDR samples across cells.

    args:
      ax: axes.
      cells: per-cell records.
      n_per_cell: subsample to keep plot manageable.
    """
    rng = np.random.RandomState(0)
    for (ai, bi), recs in cells.items():
        for r in recs:
            disc = r.get("true_ldrs_discrete")
            smo = r.get("true_ldrs_smoothed")
            if disc is None or smo is None:
                continue
            n = min(n_per_cell, len(disc))
            idx = rng.choice(len(disc), n, replace=False)
            ax.scatter(disc[idx], smo[idx], s=2, alpha=0.15, color="tab:blue")
    ax.plot(ax.get_xlim(), ax.get_xlim(), "k--", alpha=0.4)
    ax.set_xlabel("discrete LDR")
    ax.set_ylabel("smoothed LDR")
    ax.set_title("discrete vs smoothed LDR (encoding-dependent smoothing)")


def plot_lightweight_figure(cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                            config: Dict[str, Any],
                            encoding_type: str) -> None:
    """assemble lightweight figure to figures_dir/datagen_diagnostic_{enc}.png."""
    k1_values = config["kl_targets"]["k1_values"]
    k2_values = config["kl_targets"]["k2_values"]
    n1 = len(k1_values)
    is_onehot = encoding_type.startswith("onehot")

    fig = plt.figure(figsize=(4 * max(n1, 4), 4 + 2 * n1 + 4))
    gs = gridspec.GridSpec(3, 1, figure=fig,
                           height_ratios=[1, n1 * 1.2, 2],
                           hspace=0.5)

    # row 0: summary panels (4 wide)
    sub0 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0],
                                             wspace=0.4)
    plot_prescribed_vs_realized(fig.add_subplot(sub0[0, 0]),
                                fig.add_subplot(sub0[0, 1]),
                                cells, k1_values, k2_values)
    plot_alpha_beta_coverage(fig.add_subplot(sub0[0, 2]),
                             cells, k1_values, k2_values)
    if not is_onehot:
        plot_discrete_vs_smoothed(fig.add_subplot(sub0[0, 3]), cells)
    else:
        ax = fig.add_subplot(sub0[0, 3])
        ax.set_visible(False)

    # row 1: ldr histograms (smoothed)
    plot_ldr_histograms_grid(fig, gs[1], cells, k1_values, k2_values,
                             extract_smoothed_ldrs,
                             title_prefix="log d_O / d_E at pstar (smoothed)")

    # row 2: pca for non-onehot
    if not is_onehot:
        first = next(iter(cells.values()))[0]
        if "pstar_samples" in first:
            sub2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[2])
            ax_pca = fig.add_subplot(sub2[0, 0])
            plot_pca_panel(ax_pca, first["pstar_samples"], first["p0_samples"],
                           first["p1_samples"],
                           title=f"PCA, first cell, seed={first['seed']}, enc={encoding_type}")

    fig_dir = Path(config["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"datagen_diagnostic_{encoding_type}.png"
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
                         has_se=False)
        return

    encoding_type = args.encoding_type or config["encoding"]["type"]
    data_subdir = resolve_data_subdir(config, encoding_type, args.sigma)
    paths_by_idx = enumerate_cell_paths(config, data_subdir)
    if not paths_by_idx:
        print(f"no per-cell HDF5 files found under {data_subdir}. "
              f"run step1_create_data.py first, or use --show-grid.")
        return

    cells = collect_cells(paths_by_idx, KEY_MAP)
    plot_lightweight_figure(cells, config, encoding_type)

    hardness = compute_hardness(
        cells,
        config["kl_targets"]["k1_values"],
        config["kl_targets"]["k2_values"],
        extra_metrics={
            "inv_kl_O_E": lambda r: r["attrs"].get("inv_kl_O_E", np.nan),
            "ldr_std": lambda r: float(np.std(r["true_ldrs_smoothed"]))
                if "true_ldrs_smoothed" in r else np.nan,
            "latent_mean_dist": lambda r: float(np.linalg.norm(
                np.asarray(r["p0_samples"]).mean(axis=0)
                - np.asarray(r["p1_samples"]).mean(axis=0)))
                if "p0_samples" in r and "p1_samples" in r else np.nan,
        },
    )
    print_hardness_table(hardness,
                         config["kl_targets"]["k1_values"],
                         config["kl_targets"]["k2_values"])
    plot_hardness_boxplots(hardness,
                           config["kl_targets"]["k1_values"],
                           config["kl_targets"]["k2_values"],
                           str(Path(config["figures_dir"])
                               / f"datagen_variance_{encoding_type}.png"))


if __name__ == "__main__":
    main()
