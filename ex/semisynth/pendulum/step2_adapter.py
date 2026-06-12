"""step2_runner adapter for pendulum_eldr_estimation.

cell axis: flat int over (k1_idx, beta_idx, seed). 3 k1 x 1 beta x 40 seeds = 120 cells.
bucket axis: f"k1_idx_{k1_idx}".
input_dim: 18 (pendulum trajectory dimension; per-cell h5 has samples_p0 of shape
           (5000, 18)). configurable via config['data_dim'] if present.
per-cell input: <data_dir>/k1_<k1>_beta_<b>_seed_<s>.h5
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from src.utils.io import _load_config
from ex.utils.hpo.method_specs import METHOD_SPECS


def _decode(flat_idx: int, config: dict) -> tuple[int, int, int]:
    n_beta = len(config["kl_targets"]["beta_values"])
    seeds = config["kl_targets"].get("seeds_default", 1)
    seed = flat_idx % seeds
    rest = flat_idx // seeds
    return rest // n_beta, rest % n_beta, seed


def load_config(path: str) -> dict:
    config = _load_config(path)
    for k in ("data_dir", "raw_results_dir", "kl_targets", "num_waypoints", "device", "seed"):
        if k not in config:
            raise ValueError(f"config missing key: {k}")
    return config


def _input_dim(config: dict) -> int:
    # pendulum trajectory dim = horizon * state_dim; in practice 18 for default config.
    # detect from a single sample file if available, else fall back to 18.
    return config.get("data_dim", 18)


def list_cells(config: dict) -> list[int]:
    n_k1 = len(config["kl_targets"]["k1_values"])
    n_beta = len(config["kl_targets"]["beta_values"])
    seeds = config["kl_targets"].get("seeds_default", 1)
    return list(range(n_k1 * n_beta * seeds))


def bucket_for_cell(cell_idx: int, config: dict) -> str:
    k1, _, _ = _decode(cell_idx, config)
    return f"k1_idx_{k1}"


_METHOD_ALIAS = {p[0]: p[1] for p in __import__("ex.utils.hpo.method_specs",
                                                fromlist=["ALIAS_PAIRS"]).ALIAS_PAIRS}


def fit_and_eval(method: str, hp: dict, cell_idx: int, config: dict,
                 device: str) -> dict:
    canonical = _METHOD_ALIAS.get(method, method)
    if canonical not in METHOD_SPECS:
        raise KeyError(f"method {method!r} (canonical {canonical!r}) not in pendulum_eldr_estimation METHOD_SPECS")
    k1, beta, seed = _decode(cell_idx, config)
    data_path = os.path.join(config["data_dir"], f"k1_{k1}_beta_{beta}_seed_{seed}.h5")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"step1 data not found: {data_path}")

    spec = METHOD_SPECS[canonical]
    builder = spec["builder"]
    requires_pstar = spec.get("requires_pstar", False)

    seed_val = hash((method, cell_idx)) % (2**32)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    # merge kwargs into a single dict so hp can override config defaults
    # (avoids "got multiple values for keyword argument" when hp contains
    # num_waypoints, as it does for waypoint-based methods like MDRE_15,
    # MultiHeadTDRE, MultiHeadTriangularTDRE, TriangularMDRE).
    builder_kwargs = {
        "input_dim": _input_dim(config),
        "device": device,
        "num_waypoints": config["num_waypoints"],
        **hp,
    }
    estimator = builder(**builder_kwargs)

    with h5py.File(data_path, "r") as f:
        pstar = torch.from_numpy(f["samples_pstar"][()]).float().to(device)
        p0 = torch.from_numpy(f["samples_p0"][()]).float().to(device)
        p1 = torch.from_numpy(f["samples_p1"][()]).float().to(device)

    if requires_pstar:
        estimator.fit(p0, p1, pstar)
    else:
        estimator.fit(p0, p1)

    with torch.no_grad():
        est = estimator.predict_ldr(pstar)
    return {"est_ldrs": est.detach().cpu().numpy().astype(np.float32)}


_FAST = {"BDRE", "MDRE_15", "TDRE_5", "MultiHeadTDRE", "MultiHeadTriangularTDRE",
         "TriangularMDRE"}
_MEDIUM = {"CTSM", "TSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
           "TriangularTSM"}
_SLOW = {"VFM", "VFMOrthros", "FMDRE", "FMDRE_S2", "TriangularVFM_V1", "TriangularVFM_V2",
         "TriangularVFM_V3", "TriangularFMDRE"}


def walltime_per_cell_seconds(method: str, config: dict) -> int:
    if method in _FAST: return 60
    if method in _MEDIUM: return 120
    if method in _SLOW: return 240
    return 120


def resources_for_method(method: str) -> str:
    if method in _SLOW:
        return "--gpus=1 --cpus-per-task=4 --mem=24G"
    return "--gpus=1 --cpus-per-task=2 --mem=12G"


def is_cpu_eligible(method: str) -> bool:
    return method in _FAST | _MEDIUM


def method_label(method: str) -> str:
    return method


def gather_dataset_name(method: str, config: dict) -> str:
    return f"est_ldrs_{method}"


def gather_output_path(config: dict) -> str:
    return os.path.join(config.get("raw_results_dir",
        "ex/semisynth/pendulum/raw_results"), "results_all_cells.h5")
