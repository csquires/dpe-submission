"""step2_runner adapter for model_selection.

contract (consumed by ex.utils.step2_runner.{dispatch,worker,gather}):

  load_config(path) -> dict
  list_cells(config) -> Iterable[int]
  bucket_for_cell(cell_idx, config) -> str
  fit_and_eval(method, hp, cell_idx, config, device) -> dict with keys
      'est_ldrs' (ntest_sets, nsamples_test), 'mae_per_test_set' (ntest_sets,)
  walltime_per_cell_seconds(method, config) -> int
  resources_for_method(method) -> sbatch resources string
  is_cpu_eligible(method) -> bool
  method_label(method) -> str  # for cap_for / watchdog

model_selection has 70 cells = 7 KL distances x 10 instances; bucket axis is
kl_idx = cell_idx // num_instances_per_kl (10).
"""
from __future__ import annotations

from typing import Any
import os

import h5py
import numpy as np
import torch
import yaml

from src.utils.io import _load_config
# bypass the stale ex.synth.model_selection.hpo_search_spaces (references
# TDRE_5 which is no longer in the registry); use METHOD_SPECS directly. step2
# only needs the builder + requires_pstar at runtime, not the search range.
from ex.utils.hpo.method_specs import METHOD_SPECS


# -----------------------------------------------------------------------------
# config
# -----------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """load + validate model_selection config; expand DPE_DATA_ROOT in data_dir."""
    config = _load_config(path)
    required = [
        "data_dir", "raw_results_dir", "num_instances_per_kl", "kl_distances",
        "nsamples_train", "ntest_sets", "nsamples_test", "device", "seed",
    ]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    # expand env var if present (in case _load_config doesn't)
    config["data_dir"] = os.path.expandvars(str(config["data_dir"]))
    return config


# -----------------------------------------------------------------------------
# cell enumeration + bucket mapping
# -----------------------------------------------------------------------------

def _open_dataset(config: dict) -> h5py.File:
    path = os.path.join(config["data_dir"], "dataset_newpstar.h5")
    return h5py.File(path, "r")


def list_cells(config: dict) -> list[int]:
    """all 70 row indices (7 kl * 10 instances)."""
    with _open_dataset(config) as f:
        nrows = f["kl_distance_arr"].shape[0]
    return list(range(nrows))


def bucket_for_cell(cell_idx: int, config: dict) -> str:
    """kl_idx_<n> for cell at row idx (0-based)."""
    return f"kl_idx_{cell_idx // config['num_instances_per_kl']}"


# -----------------------------------------------------------------------------
# fit + eval
# -----------------------------------------------------------------------------

# methods that take pstar samples during fit
def _requires_pstar(method: str) -> bool:
    return METHOD_SPECS.get(method, {}).get("requires_pstar", False)


def fit_and_eval(method: str, hp: dict, cell_idx: int, config: dict,
                 device: str) -> dict:
    """fit estimator with hp on row=cell_idx; predict all ntest_sets test sets.

    returns:
        est_ldrs:           array (ntest_sets, nsamples_test)
        mae_per_test_set:   array (ntest_sets,)
        true_ldrs:          array (ntest_sets, nsamples_test)
    """
    if method not in METHOD_SPECS:
        raise KeyError(f"method {method!r} not registered in METHOD_SPECS")

    spec = METHOD_SPECS[method]
    builder = spec["builder"]
    # input_dim for these binary-classifier-based estimators is the data
    # dimensionality of a single sample (model_selection: data_dim=3),
    # NOT the number of training samples. config["data_dim"] gives this.
    input_dim = config["data_dim"]
    num_waypoints = spec.get("num_waypoints", None)

    # deterministic per-(method, cell)
    seed_val = hash((method, cell_idx)) % (2**32)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    # builders take (input_dim, device, num_waypoints) as required keyword args
    # plus **hp for the rest. hp from winners.yaml never includes these.
    # num_waypoints is required-but-ignored for non-waypoint methods (e.g. BDRE).
    builder_kwargs = {
        "input_dim": input_dim,
        "device": device,
        "num_waypoints": num_waypoints if num_waypoints is not None else 0,
        **hp,
    }
    estimator = builder(**builder_kwargs)

    with _open_dataset(config) as ds:
        samples_p0 = torch.from_numpy(ds["samples_p0_arr"][cell_idx]).to(device)
        samples_p1 = torch.from_numpy(ds["samples_p1_arr"][cell_idx]).to(device)
        if _requires_pstar(method):
            pstar_train = torch.from_numpy(ds["samples_pstar_train_arr"][cell_idx]).to(device)
            estimator.fit(samples_p0, samples_p1, pstar_train)
        else:
            estimator.fit(samples_p0, samples_p1)

        ntest = config["ntest_sets"]
        nsamp = config["nsamples_test"]
        est_ldrs = np.zeros((ntest, nsamp), dtype=np.float32)
        true_ldrs = np.zeros((ntest, nsamp), dtype=np.float32)
        mae_per = np.zeros(ntest, dtype=np.float32)
        for t in range(ntest):
            pstar_test = torch.from_numpy(ds["samples_pstar_arr"][cell_idx, t]).to(device)
            true_ldrs_t = torch.from_numpy(ds["true_ldrs_arr"][cell_idx, t]).to(device)
            with torch.no_grad():
                est = estimator.predict_ldr(pstar_test)
            est_ldrs[t] = est.detach().cpu().numpy().astype(np.float32)
            true_ldrs[t] = true_ldrs_t.cpu().numpy().astype(np.float32)
            mae_per[t] = float(torch.abs(est - true_ldrs_t).mean().item())

    return {
        "est_ldrs": est_ldrs,
        "mae_per_test_set": mae_per,
        "true_ldrs": true_ldrs,
    }


# -----------------------------------------------------------------------------
# walltime + resources (used by dispatch.py)
# -----------------------------------------------------------------------------

# rough per-cell seconds at dim=3, observed from variant_sweep timings
_FAST_METHODS = {"BDRE", "MDRE_15", "TDRE_5", "MultiHeadTDRE", "MultiHeadTriangularTDRE",
                 "TriangularMDRE"}
_MEDIUM_METHODS = {"CTSM", "TSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
                   "TriangularTSM"}
_SLOW_METHODS = {"VFM", "VFMOrthros", "FMDRE", "FMDRE_S2", "TriangularVFM_V1", "TriangularVFM_V2",
                 "TriangularVFM_V3", "TriangularFMDRE"}


def walltime_per_cell_seconds(method: str, config: dict) -> int:
    """rough estimate of per-cell fit-and-eval wallclock at dim=3."""
    if method in _FAST_METHODS:
        return 30
    if method in _MEDIUM_METHODS:
        return 60
    if method in _SLOW_METHODS:
        return 120
    return 90  # conservative default


def resources_for_method(method: str) -> str:
    """sbatch GPU/cpu/mem flags."""
    if method in _SLOW_METHODS:
        return "--gpus=1 --cpus-per-task=4 --mem=24G"
    return "--gpus=1 --cpus-per-task=2 --mem=12G"


def is_cpu_eligible(method: str) -> bool:
    """fast/medium MLP-DRE methods could run on cpu_array drains; SI methods stay GPU."""
    return method in _FAST_METHODS | _MEDIUM_METHODS


def method_label(method: str) -> str:
    """label used in queue file METHOD column (drives cap_for / cpu-eligibility decisions)."""
    return method
