r"""ground-truth marginal LDR colormap on PCA-projected pendulum trajectories.

intended as the background layer for plugin_dre ablation visualization on
the real pendulum data.

for each (K1, beta) cell:
  1. fit PCA on samples_pstar (flat 18-d trajectories -> 2-d).
  2. project samples_p0 and samples_p1 with the same PCA.
  3. 2D Gaussian KDE on each projected set.
  4. evaluate log(p_{Y_0} / p_{Y_1}) on a regular grid covering both clouds.
  5. pcolormesh with a diverging cmap centered at 0.

note: this is the well-defined target for plugin_dre on PCA-projected data
(the MARGINAL log-density ratio in projection space), NOT a Jacobian-adjusted
plug-in of the original-space LDR. those two quantities only coincide when
the projection is square invertible.
"""
import argparse
from itertools import product
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from ex.semisynth.pendulum.diagnostic_datagen import (
    KEY_MAP,
    enumerate_cell_paths,
)
from ex.utils.diagnostic_primitives import collect_cells


def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="ex/semisynth/pendulum/config.yaml")
    p.add_argument("--grid", type=int, default=120,
                   help="KDE evaluation grid resolution per axis")
    p.add_argument("--margin", type=float, default=0.1,
                   help="fractional margin around the union of projected clouds")
    p.add_argument("--cmap-clip", type=float, default=5.0,
                   help="symmetric clip on log-ratio for the colormap")
    p.add_argument("--min-n", type=int, default=100,
                   help="minimum pstar sample count to accept a cell")
    p.add_argument("--out-name", default="pca_ldr_gt.png",
                   help="output filename under figures_dir")
    return p.parse_args(args)


def kde_ldr_on_grid(p0_pca: np.ndarray, p1_pca: np.ndarray,
                    grid_res: int, margin: float, clamp: float
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fit gaussian KDE on each projected sample set, eval log-ratio on a 2D grid.

    args:
      p0_pca, p1_pca: [N, 2] arrays.
      grid_res: grid points per axis.
      margin: fractional margin on each side.
      clamp: symmetric clip on log-ratio for plotting.

    returns:
      X, Y: meshgrid arrays (grid_res, grid_res).
      LDR: log(p0/p1) on the grid, clipped to [-clamp, clamp].
    """
    pts = np.concatenate([p0_pca, p1_pca], axis=0)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    dx, dy = x_max - x_min, y_max - y_min
    x_min, x_max = x_min - margin * dx, x_max + margin * dx
    y_min, y_max = y_min - margin * dy, y_max + margin * dy
    xs = np.linspace(x_min, x_max, grid_res)
    ys = np.linspace(y_min, y_max, grid_res)
    X, Y = np.meshgrid(xs, ys)
    flat = np.stack([X.ravel(), Y.ravel()], axis=0)
    log_p0 = gaussian_kde(p0_pca.T).logpdf(flat).reshape(grid_res, grid_res)
    log_p1 = gaussian_kde(p1_pca.T).logpdf(flat).reshape(grid_res, grid_res)
    return X, Y, np.clip(log_p0 - log_p1, -clamp, clamp)


def plot_cell_background(ax, rec, grid_res, margin, clamp, title):
    """fit PCA(pstar), project p0/p1, KDE log-ratio, pcolormesh background.

    returns the QuadMesh handle (for sharing one colorbar across cells).
    """
    pstar = np.asarray(rec["samples_pstar"])
    pca = PCA(n_components=2)
    pca.fit(pstar)
    p0_pca = pca.transform(np.asarray(rec["samples_p0"]))
    p1_pca = pca.transform(np.asarray(rec["samples_p1"]))
    X, Y, LDR = kde_ldr_on_grid(p0_pca, p1_pca, grid_res, margin, clamp)
    im = ax.pcolormesh(X, Y, LDR, cmap="RdBu_r",
                       vmin=-clamp, vmax=clamp, shading="auto")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=9)
    return im


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    paths_by_idx = enumerate_cell_paths(config)
    if not paths_by_idx:
        print(f"no per-cell HDF5 files found under {config['data_dir']}.")
        return
    cells = collect_cells(paths_by_idx, KEY_MAP)

    k1_values = config["kl_targets"]["k1_values"]
    beta_values = config["kl_targets"]["beta_values"]
    n1 = len(k1_values)
    n2 = len(beta_values)

    fig, axes = plt.subplots(n1, n2, figsize=(5 * n2, 4 * n1), squeeze=False)
    last_im = None
    for ai, bi in product(range(n1), range(n2)):
        ax = axes[ai, bi]
        recs = cells.get((ai, bi))
        rec = None
        if recs:
            for r in recs:
                pstar = r.get("samples_pstar")
                if pstar is not None and pstar.shape[0] >= args.min_n:
                    rec = r
                    break
        if rec is None:
            ax.text(0.5, 0.5, "no full cell",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        title = (rf"$K_1$={k1_values[ai]}, $\beta$={beta_values[bi]}"
                 rf", seed={rec['seed']}")
        last_im = plot_cell_background(ax, rec, args.grid, args.margin,
                                       args.cmap_clip, title)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(),
                     shrink=0.6, label=r"$\log p_{Y_0} / p_{Y_1}$ (clipped)")
    fig.suptitle("Marginal LDR (KDE) on PCA($p^*$)-projected pendulum trajectories",
                 fontsize=11)
    out = Path(config["figures_dir"]) / args.out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
