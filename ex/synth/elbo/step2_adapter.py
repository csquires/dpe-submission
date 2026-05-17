"""step2_runner adapter for elbo.

cell axis: row index in design_arr of dataset_d=<D>,nsamples=<N>.h5 (24000 rows).
bucket axis: none — single HP set per (method, exp); per_bucket override unused.

per-cell output: scalar ELDR estimate (mean of est_ldrs over the 50000 pstar samples).
gather: stacks per-cell scalars into est_eldrs_arr_<method> of shape (nrows,) — matches
the original step2 output dataset name.

quirks:
- input_dim is data_dim + 1 (theta has data_dim coords, y has 1 coord, concatenated).
- TriangularMDRE takes 3-arg fit (p0, p1, pstar); other methods take 2-arg fit.
- elbo has no integration_steps in any HP; keep it that way.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from src.utils.io import _load_config
from ex.utils.hpo.method_specs import METHOD_SPECS


# methods that need pstar at fit time (forwarded from METHOD_SPECS.requires_pstar)
def _requires_pstar(method: str) -> bool:
    return METHOD_SPECS.get(method, {}).get("requires_pstar", False)


# -----------------------------------------------------------------------------
# config / dataset
# -----------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """load + validate elbo config; _load_config already expands env vars."""
    config = _load_config(path)
    required = ["data_dir", "raw_results_dir", "data_dim", "nsamples", "device", "seed"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    return config


def _dataset_path(config: dict) -> str:
    filename = config.get(
        "dataset_filename",
        f"dataset_d={config['data_dim']},nsamples={config['nsamples']}.h5",
    )
    return os.path.join(config["data_dir"], filename)


def _open_dataset(config: dict) -> h5py.File:
    return h5py.File(_dataset_path(config), "r")


# -----------------------------------------------------------------------------
# cell enumeration
# -----------------------------------------------------------------------------

def list_cells(config: dict) -> list[int]:
    """all design_arr row indices."""
    with _open_dataset(config) as f:
        nrows = f["design_arr"].shape[0]
    return list(range(nrows))


def bucket_for_cell(cell_idx: int, config: dict) -> None:
    """no bucket axis: every row uses the same HP. returns None so loader uses default."""
    return None


# -----------------------------------------------------------------------------
# fit + eval
# -----------------------------------------------------------------------------

def fit_and_eval(method: str, hp: dict, cell_idx: int, config: dict,
                 device: str) -> dict:
    """fit estimator with hp on row=cell_idx; predict on pstar samples; return scalar mean.

    output:
        est_ldrs:  array (1,)            (the ELDR scalar wrapped for h5)
        est_ldrs_full: array (nsamples,) (per-sample ldr predictions, optional)
    """
    if method not in METHOD_SPECS:
        raise KeyError(f"method {method!r} not registered in METHOD_SPECS")

    spec = METHOD_SPECS[method]
    builder = spec["builder"]
    # input_dim = data_dim + 1 because samples are theta (data_dim) concat y (1)
    input_dim = config["data_dim"] + 1
    num_waypoints = spec.get("num_waypoints", None)

    seed_val = hash((method, cell_idx)) % (2**32)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    builder_kwargs = {
        "input_dim": input_dim,
        "device": device,
        "num_waypoints": num_waypoints if num_waypoints is not None else 0,
        **hp,
    }
    estimator = builder(**builder_kwargs)

    with _open_dataset(config) as ds:
        theta_star = torch.from_numpy(ds["theta_star_samples_arr"][cell_idx]).float().to(device)
        y_star = torch.from_numpy(ds["y_star_samples_arr"][cell_idx]).float().to(device)
        samples_pstar = torch.cat([theta_star, y_star], dim=1)

        theta0 = torch.from_numpy(ds["theta0_samples_arr"][cell_idx]).float().to(device)
        y0 = torch.from_numpy(ds["y0_samples_arr"][cell_idx]).float().to(device)
        samples_p0 = torch.cat([theta0, y0], dim=1)

        theta1 = torch.from_numpy(ds["theta1_samples_arr"][cell_idx]).float().to(device)
        y1 = torch.from_numpy(ds["y1_samples_arr"][cell_idx]).float().to(device)
        samples_p1 = torch.cat([theta1, y1], dim=1)

    if _requires_pstar(method):
        estimator.fit(samples_p0, samples_p1, samples_pstar)
    else:
        estimator.fit(samples_p0, samples_p1)

    with torch.no_grad():
        eldr = float(estimator.predict_eldr(samples_pstar).item())

    return {
        # gather expects 'est_ldrs' as the per-cell array; we wrap the scalar in shape (1,)
        # so it stacks cleanly to (nrows, 1) and gather can squeeze if needed.
        "est_ldrs": np.array([eldr], dtype=np.float32),
    }


# -----------------------------------------------------------------------------
# walltime + resources
# -----------------------------------------------------------------------------

# rough per-cell seconds at data_dim=3, nsamples=50000
_FAST = {"BDRE", "MDRE_15", "TDRE_5", "MultiHeadTDRE", "MultiHeadTriangularTDRE", "TriangularMDRE"}
_MEDIUM = {"CTSM", "TSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3", "TriangularTSM"}
_SLOW = {"VFM", "VFMOrthros", "FMDRE", "FMDRE_S2", "TriangularVFM_V1", "TriangularVFM_V2",
         "TriangularVFM_V3", "TriangularFMDRE"}


def walltime_per_cell_seconds(method: str, config: dict) -> int:
    if method in _FAST: return 60
    if method in _MEDIUM: return 90
    if method in _SLOW: return 180
    return 120


def resources_for_method(method: str) -> str:
    if method in _SLOW:
        return "--gpus=1 --cpus-per-task=4 --mem=24G"
    return "--gpus=1 --cpus-per-task=2 --mem=16G"


def is_cpu_eligible(method: str) -> bool:
    return method in _FAST | _MEDIUM


def method_label(method: str) -> str:
    return method


# -----------------------------------------------------------------------------
# gather overrides (elbo writes 'est_eldrs_arr_<method>' not the default 'est_ldrs_arr_*')
# -----------------------------------------------------------------------------

def gather_dataset_name(method: str, config: dict) -> str:
    return f"est_eldrs_arr_{method}"


def gather_output_path(config: dict) -> str:
    out_dir = config.get("raw_results_dir", "ex/synth/elbo/raw_results")
    stem = config.get("dataset_filename", f"dataset_d={config['data_dim']},nsamples={config['nsamples']}.h5")
    stem = stem.replace("dataset_", "results_").removesuffix(".h5")
    return os.path.join(out_dir, f"{stem}.h5")
