"""step2_runner adapter for eig.

cell axis: row index in design_arr of dataset_d=<D>,nsamples=<N>.h5.
bucket axis: none — single HP set per (method, exp); per_bucket override unused.

per-cell output: scalar EIG estimate (one float per design).
gather: stacks per-cell scalars into est_eigs_arr_<method> of shape (nrows,) — matches
the original step2 output dataset name.

quirks:
- input_dim is data_dim + 1 (theta has data_dim coords, y has 1 coord, concatenated).
- per-cell builds (joint, shuffled) via joint_and_shuffled, fits the DRE on those,
  and reports the mean predict_ldr(joint) as the scalar EIG estimate.
- triangular DRE variants (requires_pstar=True) reuse joint as pstar at fit time.
- the original step2 also computed true_eigs_arr from Sigma_pi + design; this is a
  deterministic post-processing step that gather_postprocess() can run after the
  per-method gathers complete (true_eigs depends only on the dataset, not on any
  estimator).
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from src.utils.io import _load_config
from ex.utils.eig_ldr import joint_and_shuffled
from ex.utils.hpo.method_specs import METHOD_SPECS


def _requires_pstar(method: str) -> bool:
    return METHOD_SPECS.get(method, {}).get("requires_pstar", False)


# -----------------------------------------------------------------------------
# config / dataset
# -----------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """load + validate eig config; _load_config already expands env vars."""
    config = _load_config(path)
    required = ["data_dir", "raw_results_dir", "data_dim", "nsamples", "device", "seed"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    return config


def _dataset_path(config: dict) -> str:
    return os.path.join(
        config["data_dir"],
        f"dataset_d={config['data_dim']},nsamples={config['nsamples']}.h5",
    )


def _open_dataset(config: dict) -> h5py.File:
    return h5py.File(_dataset_path(config), "r")


# -----------------------------------------------------------------------------
# cell enumeration
# -----------------------------------------------------------------------------

def list_cells(config: dict) -> list[int]:
    """flat row indices reserved for step2.

    delegates to the hpo adapter's step2_pool() so the train/holdout/step2
    three-way stratified split is the single source of truth. eig keeps the
    32/8/all convention — step2_pool() returns cell_pool() (every design
    row), so this list_cells matches the historical full-pool behavior.
    """
    from ex.utils.hpo.adapters import get_adapter
    adapter = get_adapter("eig")
    return [int(cell[0]) for cell in adapter.step2_pool()]


def bucket_for_cell(cell_idx: int, config: dict) -> None:
    """no bucket axis: every row uses the same HP. returns None so loader uses default."""
    return None


# -----------------------------------------------------------------------------
# fit + eval
# -----------------------------------------------------------------------------

_METHOD_ALIAS = {p[0]: p[1] for p in __import__("ex.utils.hpo.method_specs",
                                                fromlist=["ALIAS_PAIRS"]).ALIAS_PAIRS}

# legacy holdout names -> current builder kwargs, applied PER METHOD because
# different builders use different conventions (MDRE uses learning_rate +
# latent_dim natively; VFM/CTSM use n_steps, not n_epochs).
_HP_KEY_ALIAS_BY_METHOD = {
    "VFM":           {"n_epochs": "n_steps"},
    "VFMOrthros":    {"n_epochs": "n_steps"},
    "CTSM":          {"n_epochs": "n_steps"},
    "FMDRE":         {"n_epochs": "n_steps"},
    "FMDRE_S2":      {"n_epochs": "n_steps"},
}


def _normalize_hp(method: str, hp: dict) -> dict:
    """rename legacy hp keys to what the current builder expects, per method."""
    alias = _HP_KEY_ALIAS_BY_METHOD.get(method, {})
    return {alias.get(k, k): v for k, v in hp.items()}


def fit_and_eval(method: str, hp: dict, cell_idx: int, config: dict,
                 device: str) -> dict:
    """build estimator with hp; fit on (joint, shuffled-marginals); return scalar
    EIG estimate = mean predict_ldr(joint).

    triangular methods (requires_pstar) reuse joint as pstar at fit time.
    returns:
        est_ldrs: array (1,) holding the scalar EIG estimate
    """
    canonical = _METHOD_ALIAS.get(method, method)
    if canonical not in METHOD_SPECS:
        raise KeyError(f"method {method!r} (canonical {canonical!r}) not registered in METHOD_SPECS")

    spec = METHOD_SPECS[canonical]
    builder = spec["builder"]
    input_dim = config["data_dim"] + 1   # theta (D) || y (1)
    num_waypoints = spec.get("num_waypoints", None)

    seed_val = hash((method, cell_idx)) % (2**32)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    builder_kwargs = {
        "input_dim": input_dim,
        "device": device,
        "num_waypoints": num_waypoints if num_waypoints is not None else 0,
        **_normalize_hp(method, hp),
    }
    dre = builder(**builder_kwargs)

    with _open_dataset(config) as ds:
        theta = torch.from_numpy(ds["theta_samples_arr"][cell_idx]).to(device)
        y = torch.from_numpy(ds["y_samples_arr"][cell_idx]).to(device)

    joint, shuffled = joint_and_shuffled(theta, y)
    if _requires_pstar(method):
        dre.fit(joint, shuffled, joint)
    else:
        dre.fit(joint, shuffled)
    with torch.no_grad():
        eig = float(dre.predict_eldr(joint).item())

    return {"est_ldrs": np.array([eig], dtype=np.float32)}


# -----------------------------------------------------------------------------
# walltime + resources
# -----------------------------------------------------------------------------

# rough per-cell seconds at data_dim=3, nsamples=50000 (one EIG estimation per cell)
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
# gather overrides (eig writes 'est_eigs_arr_<method>' not the default 'est_ldrs_arr_*')
# -----------------------------------------------------------------------------

def gather_dataset_name(method: str, config: dict) -> str:
    return f"est_eigs_arr_{method}"


def gather_output_path(config: dict) -> str:
    out_dir = config.get("raw_results_dir", "ex/synth/eig/raw_results")
    fname = f"results_d={config['data_dim']},nsamples={config['nsamples']}.h5"
    return os.path.join(out_dir, fname)


# -----------------------------------------------------------------------------
# post-gather: compute true_eigs from dataset (no estimator needed)
# -----------------------------------------------------------------------------

def gather_postprocess(config: dict, out_path: str) -> None:
    """append 'true_eigs_arr' to results.h5 if not already present.

    true_eigs is a deterministic function of (Sigma_pi, design); no per-cell fit
    needed. invoke after gather has run for all methods. safe to call repeatedly.
    """
    with h5py.File(out_path, "a") as f:
        if "true_eigs_arr" in f and not f.attrs.get("force_true_eigs", False):
            return
    with _open_dataset(config) as ds:
        nrows = ds["design_arr"].shape[0]
        true_eigs = np.zeros(nrows, dtype=np.float32)
        for idx in range(nrows):
            design = torch.from_numpy(ds["design_arr"][idx])
            sigma = torch.from_numpy(ds["prior_covariance_arr"][idx])
            quad = (design.T @ sigma @ design).squeeze()
            true_eigs[idx] = (0.5 * torch.log1p(quad / 1.0)).item()  # sigma2=1.0 default
    with h5py.File(out_path, "a") as f:
        if "true_eigs_arr" in f:
            del f["true_eigs_arr"]
        f.create_dataset("true_eigs_arr", data=true_eigs)
    print(f"  appended true_eigs_arr (shape={true_eigs.shape}) to {out_path}")
