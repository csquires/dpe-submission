"""step2_runner adapter for pstar_sample_complexity.

cell axis: flat int over (instance_idx, pstar_idx). 20 instances x N pstar values.
bucket axis: f"pstar_idx_{pstar_idx}".

note: in 200broad analysis, pstar_sample_complexity has only triangular methods
(no non-triangular HPO data) — there is no winners.<this exp>.uniform_200broad.yaml
in scratch/200broad/winners_pinned/. to run this experiment via step2_runner you
must pass --winners pointing at a winners.yaml that has at least one method with
a hyperparams entry. legacy schema-B winners.json files (used by the original
step2) are also supported by load_winners.

per-cell output: shape (nsamples_test,) — predictions on samples_test.
"""
from __future__ import annotations

import os
import time

import h5py
import numpy as np
import torch

from src.utils.io import _load_config
from ex.utils.hpo.method_specs import METHOD_SPECS as SEARCH_SPACES
from ex.utils.hpo.method_specs import METHOD_SPECS


def _decode(flat_idx: int, config: dict) -> tuple[int, int]:
    n_pstar = len(config["nsamples_pstar_values"])
    return flat_idx // n_pstar, flat_idx % n_pstar   # (instance_idx, pstar_idx)


def load_config(path: str) -> dict:
    config = _load_config(path)
    for k in ("data_dir", "raw_results_dir", "nsamples_pstar_values",
              "num_instances", "nsamples_test", "data_dim", "device", "seed"):
        if k not in config:
            raise ValueError(f"config missing key: {k}")
    return config


def _open_dataset(config: dict) -> h5py.File:
    return h5py.File(os.path.join(config["data_dir"], "dataset.h5"), "r")


def list_cells(config: dict) -> list[int]:
    return list(range(config["num_instances"] * len(config["nsamples_pstar_values"])))


def bucket_for_cell(cell_idx: int, config: dict) -> str:
    _, pstar_idx = _decode(cell_idx, config)
    return f"pstar_idx_{pstar_idx}"


def _build_estimator(method: str, hp: dict, config: dict, device: str):
    """SEARCH_SPACES first (config-aware builders); else fall back to METHOD_SPECS
    (canonical builders that take num_waypoints, no config). pstar's gold winners
    are exclusively triangular methods, so most go through the METHOD_SPECS path.

    kwargs are merged into a single dict with hp last, so gold-yaml hp always
    overrides config / METHOD_SPECS defaults on collision (e.g. num_waypoints).
    """
    if method in SEARCH_SPACES:
        kwargs = {"input_dim": config["data_dim"], "device": device, "config": config, **hp}
        return SEARCH_SPACES[method]["builder"](**kwargs)
    if method not in METHOD_SPECS:
        raise KeyError(f"method {method!r} in neither SEARCH_SPACES nor METHOD_SPECS")
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
    instance_idx, pstar_idx = _decode(cell_idx, config)
    nsamples_pstar = config["nsamples_pstar_values"][pstar_idx]

    seed_val = hash((method, cell_idx)) % (2**32)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    estimator = _build_estimator(method, hp, config, device)

    with _open_dataset(config) as f:
        p0 = torch.from_numpy(f["samples_p0_arr"][instance_idx]).to(device)
        p1 = torch.from_numpy(f["samples_p1_arr"][instance_idx]).to(device)
        pstar = torch.from_numpy(f["samples_pstar_arr"][instance_idx][:nsamples_pstar]).to(device)
        test = torch.from_numpy(f["samples_test_arr"][instance_idx]).to(device)

    estimator.fit(p0, p1, pstar)   # pstar_sample_complexity always needs pstar
    with torch.no_grad():
        est = estimator.predict_ldr(test)
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
    return f"est_ldrs_arr_{method}"


def gather_output_path(config: dict) -> str:
    out_dir = config.get("raw_results_dir", "ex/pstar_sample_complexity/raw_results")
    # original step2 wrote one h5 per (method, nsamples_pstar); my gather emits one
    # combined file with shape (n_cells, nsamples_test). step3 may need a small
    # update OR a custom gather override that splits into per-pstar h5 files.
    return os.path.join(out_dir, "results_all_cells.h5")
