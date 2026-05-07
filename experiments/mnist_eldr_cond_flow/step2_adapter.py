"""step2_runner adapter for mnist_eldr_cond_flow.

mirrors mnist_eldr_estimation: cell axis = flat (alpha_idx, pair_idx),
bucket = alpha_idx_<n>, per-cell <data_dir>/alpha_<a>_pair_<p>.h5.

note: my v2 winners file is named winners.mnist_cond_flow.uniform_200broad.yaml
(not mnist_eldr_cond_flow). pass that path explicitly to dispatch --winners.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from src.utils.io import _load_config
from experiments.mnist_eldr_cond_flow.hpo_search_spaces import SEARCH_SPACES


def _decode(flat_idx: int, config: dict) -> tuple[int, int]:
    n_pairs = config["num_pairs_per_alpha"]
    return flat_idx // n_pairs, flat_idx % n_pairs


def load_config(path: str) -> dict:
    config = _load_config(path)
    for k in ("data_dir", "raw_results_dir", "alphas", "num_pairs_per_alpha",
              "latent_dim", "num_waypoints", "device", "seed"):
        if k not in config:
            raise ValueError(f"config missing key: {k}")
    return config


def list_cells(config: dict) -> list[int]:
    return list(range(len(config["alphas"]) * config["num_pairs_per_alpha"]))


def bucket_for_cell(cell_idx: int, config: dict) -> str:
    a, _ = _decode(cell_idx, config)
    return f"alpha_idx_{a}"


def fit_and_eval(method: str, hp: dict, cell_idx: int, config: dict,
                 device: str) -> dict:
    if method not in SEARCH_SPACES:
        raise KeyError(f"method {method!r} not in mnist_eldr_cond_flow SEARCH_SPACES")
    a, p = _decode(cell_idx, config)
    data_path = os.path.join(config["data_dir"], f"alpha_{a}_pair_{p}.h5")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"step1 data not found: {data_path}")

    spec = SEARCH_SPACES[method]
    builder = spec["builder"]
    requires_pstar = spec.get("requires_pstar", False)

    seed_val = hash((method, cell_idx)) % (2**32)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    estimator = builder(
        input_dim=config["latent_dim"], device=device,
        num_waypoints=config["num_waypoints"], **hp,
    )

    with h5py.File(data_path, "r") as f:
        pstar = torch.from_numpy(f["pstar_samples"][()]).to(device)
        p0 = torch.from_numpy(f["p0_samples"][()]).to(device)
        p1 = torch.from_numpy(f["p1_samples"][()]).to(device)

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
_SLOW = {"VFM", "FMDRE", "FMDRE_S2", "TriangularVFM_V1", "TriangularVFM_V2",
         "TriangularVFM_V3", "TriangularFMDRE"}


def walltime_per_cell_seconds(method: str, config: dict) -> int:
    if method in _FAST: return 30
    if method in _MEDIUM: return 60
    if method in _SLOW: return 120
    return 60


def resources_for_method(method: str) -> str:
    if method in _SLOW:
        return "--gpus=1 --cpus-per-task=4 --mem=16G"
    return "--gpus=1 --cpus-per-task=2 --mem=16G"


def is_cpu_eligible(method: str) -> bool:
    return method in _FAST | _MEDIUM


def method_label(method: str) -> str:
    return method


def gather_dataset_name(method: str, config: dict) -> str:
    return f"est_ldrs_{method}"


def gather_output_path(config: dict) -> str:
    return os.path.join(config.get("raw_results_dir",
        "experiments/mnist_eldr_cond_flow/raw_results"), "results_all_cells.h5")
