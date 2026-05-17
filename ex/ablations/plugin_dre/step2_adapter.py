"""step2_runner adapter for plugin_dre.

cell axis: row index in dataset.h5 (kl_idx * num_instances_per_kl + local_idx).
bucket axis: f"kl_idx_{kl_idx}".
per-cell output: shape (num_grid_points,) — predicted ldr at the grid points.

uses ex.utils.hpo.method_specs.METHOD_SPECS whose canonical builders take
(input_dim, device, num_waypoints, **hp).
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from src.utils.io import _load_config
from ex.utils.hpo.method_specs import METHOD_SPECS


def load_config(path: str) -> dict:
    config = _load_config(path)
    for k in ("data_dir", "results_dir", "kl_divergences", "num_instances_per_kl",
              "data_dim", "device", "seed"):
        if k not in config:
            raise ValueError(f"config missing key: {k}")
    return config


def _open_dataset(config: dict) -> h5py.File:
    return h5py.File(os.path.join(config["data_dir"], "dataset.h5"), "r")


def list_cells(config: dict) -> list[int]:
    return list(range(len(config["kl_divergences"]) * config["num_instances_per_kl"]))


def bucket_for_cell(cell_idx: int, config: dict) -> str:
    return f"kl_idx_{cell_idx // config['num_instances_per_kl']}"


def _build_estimator(method: str, hp: dict, config: dict, device: str):
    """build an estimator via the canonical METHOD_SPECS builder.

    kwargs are merged into a single dict with hp last, so gold-yaml hp always
    overrides METHOD_SPECS defaults on collision (e.g. num_waypoints).
    """
    if method not in METHOD_SPECS:
        raise KeyError(f"method {method!r} not in METHOD_SPECS")
    spec = METHOD_SPECS[method]
    nwp = spec.get("num_waypoints", None)
    kwargs = {
        "input_dim": config["data_dim"],
        "device": device,
        "num_waypoints": nwp if nwp is not None else 0,
        **hp,
    }
    return spec["builder"](**kwargs)


def fit_and_eval(method: str, hp: dict, cell_idx: int, config: dict,
                 device: str) -> dict:
    seed_val = hash((method, cell_idx)) % (2**32)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    estimator = _build_estimator(method, hp, config, device)

    with _open_dataset(config) as f:
        p0 = torch.from_numpy(f["samples_p0_arr"][cell_idx]).to(device)
        p1 = torch.from_numpy(f["samples_p1_arr"][cell_idx]).to(device)
        grid = torch.from_numpy(f["grid_points_arr"][cell_idx]).to(device)

    estimator.fit(p0, p1)
    with torch.no_grad():
        est = estimator.predict_ldr(grid)
    return {"est_ldrs": est.detach().cpu().numpy().astype(np.float32)}


_FAST = {"BDRE", "MDRE_15", "TDRE_5", "MultiHeadTDRE", "MultiHeadTriangularTDRE",
         "TriangularMDRE"}
_MEDIUM = {"CTSM", "TSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
           "TriangularTSM"}
_SLOW = {"VFM", "VFMOrthros", "FMDRE", "FMDRE_S2", "TriangularVFM_V1", "TriangularVFM_V2",
         "TriangularVFM_V3", "TriangularFMDRE"}


def walltime_per_cell_seconds(method: str, config: dict) -> int:
    if method in _FAST: return 30
    if method in _MEDIUM: return 60
    if method in _SLOW: return 120
    return 60


def resources_for_method(method: str) -> str:
    if method in _SLOW:
        return "--gpus=1 --cpus-per-task=4 --mem=16G"
    return "--gpus=1 --cpus-per-task=2 --mem=8G"


def is_cpu_eligible(method: str) -> bool:
    return method in _FAST | _MEDIUM


def method_label(method: str) -> str:
    return method


def gather_dataset_name(method: str, config: dict) -> str:
    return f"est_ldrs_grid_{method}"


def gather_output_path(config: dict) -> str:
    out_dir = config.get("results_dir", "ex/ablations/plugin_dre/results")
    return os.path.join(out_dir, "raw_results.h5")
