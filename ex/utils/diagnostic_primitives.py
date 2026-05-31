"""generic primitives shared across per-experiment diagnostic_datagen scripts.

scope is intentionally narrow: only things with no experiment-specific
semantics live here. labeled plots (axes named 'K1', 'beta', etc.),
hardness metrics, KL-grid plots and any cell-aggregate composition belong
in the per-experiment diagnostic_datagen file.

contents:
  - _read_attrs / load_cell / collect_cells: HDF5 per-cell loader with
    optional standard-name remap.
  - load_kl_grid: generic HDF5 grid loader returning all datasets + attrs.
  - plot_pca_panel: encoding-agnostic 2-d PCA scatter of pstar, p0, p1.
"""
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


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
      paths_by_idx: dict[(i_idx, j_idx)] -> list of (seed, path) tuples.
                    only cells whose hdf5 exists should be passed in.
      key_map: standard_name -> attr_name (passed to load_cell).
      datasets: optional dataset whitelist passed to load_cell.

    returns:
      dict[(i_idx, j_idx)] -> list of cell dicts (one per seed, in input order).
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


def load_kl_grid(cache_path: str, expected: Iterable[str]) -> Dict[str, Any]:
    """load a cached kl-grid hdf5; return all datasets and attrs.

    args:
      cache_path: path to the grid hdf5.
      expected: dataset names that must be present; missing keys raise KeyError.

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
