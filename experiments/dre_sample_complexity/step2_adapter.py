"""step2_runner adapter for dre_sample_complexity.

cell axis: flat int over (row, ntrain_idx). row = kl_idx * num_instances_per_kl +
local_idx. ntrain_idx = index into config['nsamples_train_values'].
bucket axis: f"kl_idx_{kl_idx}" (independent of ntrain_idx).

per-cell output: shape (nsamples_test,) — the predicted ldrs at samples_pstar.
gather: stacks into (nrows, n_nsamples_train, nsamples_test) matching the
original step2 output shape via per-(row, ntrain) cell layout.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from src.utils.io import _load_config
from experiments.dre_sample_complexity.hpo_search_spaces import SEARCH_SPACES
from experiments.utils.hpo.method_specs import METHOD_SPECS


def _decode(flat_idx: int, config: dict) -> tuple[int, int]:
    n_nsamples = len(config["nsamples_train_values"])
    return flat_idx // n_nsamples, flat_idx % n_nsamples   # (row, ntrain_idx)


def load_config(path: str) -> dict:
    config = _load_config(path)
    for k in ("data_dir", "raw_results_dir", "kl_divergences", "num_instances_per_kl",
              "nsamples_train_values", "nsamples_test", "data_dim", "device", "seed"):
        if k not in config:
            raise ValueError(f"config missing key: {k}")
    return config


def _open_dataset(config: dict) -> h5py.File:
    return h5py.File(os.path.join(config["data_dir"], "dataset.h5"), "r")


def list_cells(config: dict) -> list[int]:
    nrows = len(config["kl_divergences"]) * config["num_instances_per_kl"]
    return list(range(nrows * len(config["nsamples_train_values"])))


def bucket_for_cell(cell_idx: int, config: dict) -> str:
    row, _ = _decode(cell_idx, config)
    return f"kl_idx_{row // config['num_instances_per_kl']}"


def _build_estimator(method: str, hp: dict, config: dict, device: str):
    """SEARCH_SPACES first (config-aware builders); else fall back to METHOD_SPECS
    (canonical builders that take num_waypoints, no config)."""
    if method in SEARCH_SPACES:
        return SEARCH_SPACES[method]["builder"](
            input_dim=config["data_dim"], device=device, config=config, **hp,
        )
    if method not in METHOD_SPECS:
        raise KeyError(f"method {method!r} in neither SEARCH_SPACES nor METHOD_SPECS")
    spec = METHOD_SPECS[method]
    nwp = spec.get("num_waypoints", None)
    return spec["builder"](
        input_dim=config["data_dim"], device=device,
        num_waypoints=nwp if nwp is not None else 0, **hp,
    )


def fit_and_eval(method: str, hp: dict, cell_idx: int, config: dict,
                 device: str) -> dict:
    row, ntrain_idx = _decode(cell_idx, config)
    nsamples_train = config["nsamples_train_values"][ntrain_idx]

    seed_val = hash((method, cell_idx)) % (2**32)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    estimator = _build_estimator(method, hp, config, device)

    with _open_dataset(config) as f:
        p0 = torch.from_numpy(f["samples_p0_arr"][row][:nsamples_train]).to(device)
        p1 = torch.from_numpy(f["samples_p1_arr"][row][:nsamples_train]).to(device)
        pstar = torch.from_numpy(f["samples_pstar_arr"][row]).to(device)

    estimator.fit(p0, p1)
    with torch.no_grad():
        est = estimator.predict_ldr(pstar)
    return {"est_ldrs": est.detach().cpu().numpy().astype(np.float32)}


_FAST = {"BDRE", "MDRE_15", "TDRE_5", "MultiHeadTDRE", "MultiHeadTriangularTDRE",
         "TriangularMDRE"}
_MEDIUM = {"CTSM", "TSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
           "TriangularTSM"}
_SLOW = {"VFM", "FMDRE", "FMDRE_S2", "TriangularVFM_V1", "TriangularVFM_V2",
         "TriangularVFM_V3", "TriangularFMDRE"}


def walltime_per_cell_seconds(method: str, config: dict) -> int:
    if method in _FAST: return 60
    if method in _MEDIUM: return 90
    if method in _SLOW: return 180
    return 90


def resources_for_method(method: str) -> str:
    if method in _SLOW:
        return "--gpus=1 --cpus-per-task=4 --mem=16G"
    return "--gpus=1 --cpus-per-task=2 --mem=8G"


def is_cpu_eligible(method: str) -> bool:
    return method in _FAST | _MEDIUM


def method_label(method: str) -> str:
    return method


def gather_dataset_name(method: str, config: dict) -> str:
    return f"est_ldrs_arr_{method}"


def gather_output_path(config: dict) -> str:
    out_dir = config.get("raw_results_dir", "experiments/dre_sample_complexity/raw_results")
    return os.path.join(out_dir, "results.h5")
