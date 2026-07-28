"""step2_runner adapter for elbo.

cell axis: flat row index in dataset_d=3,n_p0p1=10000,n_pstar=5000.h5 (2048 rows).
  encoding: row = alpha_idx * n_flat_per_alpha + flat_idx_per_alpha,
  where flat_idx_per_alpha = prior_idx * n_beta * n_designs + beta_idx * n_designs + design_idx.
  only 1536 of 2048 rows are evaluated in step2 (the disjoint HPO adapter's step2 pool).

bucket axis: none — single HP set per (method, exp); per_bucket override unused.

per-cell output: scalar ELDR estimate (mean of est_ldrs over the ~5000 pstar samples).
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
    required = ["data_dir", "raw_results_dir", "data_dim", "device", "seed"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    return config


def _dataset_path(config: dict) -> str:
    filename = config.get(
        "dataset_filename",
        f"dataset_d={config['data_dim']},n_p0p1={config['n_p0p1']},n_pstar={config['n_pstar']}.h5",
    )
    return os.path.join(config["data_dir"], filename)


def _open_dataset(config: dict) -> h5py.File:
    return h5py.File(_dataset_path(config), "r")


# -----------------------------------------------------------------------------
# cell enumeration
# -----------------------------------------------------------------------------

def list_cells(config: dict) -> list[int]:
    """step2 pool: cells reserved for step2, NOT seen by HPO.

    delegates to the hpo adapter's step2_pool() so step2 runs only on
    the 1536 cells disjoint from train (2 cells per stratum) and holdout
    (2 cells per stratum) pools. 128 strata × 12 step2 cells per stratum = 1536.

    cell encoding MUST match the dataset row layout written by step1, whose loop
    order is prior -> beta -> design -> alpha with alpha INNERMOST:
        row = flat_idx * n_alphas + alpha_idx
    where flat_idx = (prior_idx * n_beta + beta_idx) * n_designs + design_idx.
    fit_and_eval indexes the h5 arrays directly by this value, so an alpha-major
    encoding would silently select the wrong rows.
    """
    from ex.utils.hpo.adapters import get_adapter
    adapter = get_adapter("elbo")
    n_alphas = len(config["alphas"])
    return sorted(
        flat_idx * n_alphas + alpha_idx
        for (alpha_idx, flat_idx) in adapter.step2_pool()
    )


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


# measured per-cell GPU cost of each method's WINNING hp (elbo HPO holdout,
# 2026-07-24), rounded up to ~2x. dispatch multiplies by a further 1.5 and adds
# 90s startup, so the effective per-cell budget is ~3x the measured cost.
#
# these are deliberately NOT derived from the _FAST/_MEDIUM/_SLOW tiers: those
# sets also drive is_cpu_eligible(), so re-tiering to fix a walltime would
# wrongly make a heavy gpu method cpu-eligible. cost and placement are
# independent axes -- e.g. VFM is 434s/cell while TriangularVFM_V2 is 38s,
# yet both must stay gpu-only.
#
# an earlier estimate scaled the old 50k-sample numbers down by data size
# (1/3). that is wrong for the flow/score methods: their cost is dominated by
# ODE integration steps, which do not shrink with fewer samples. it would have
# under-budgeted 7 of 19 methods and got their chunks CANCELLED DUE TO TIME
# LIMIT mid-run (the 2026-06-14 failure mode).
_WINNER_SECONDS: dict[str, int] = {
    "BDRE": 20, "MDRE_15": 20, "TriangularMDRE": 20,
    "TriangularCTSM_V2": 40, "CTSM": 40, "TriangularCTSM_V3": 50,
    "TriangularCTSM_V1": 70, "TSM": 70,
    "MultiHeadTriangularTDRE": 80, "TriangularVFM_V2": 80,
    "TriangularVFM_V1": 100, "TriangularTSM": 110, "MultiHeadTDRE": 120,
    "TriangularFMDRE": 170, "VFMOrthros": 240,
    "FMDRE": 300, "FMDRE_S2": 300,
    "TriangularVFM_V3": 420, "VFM": 900,
}


def walltime_per_cell_seconds(method: str, config: dict) -> int:
    """per-cell walltime budget, from measured winner-hp cost (see _WINNER_SECONDS).

    falls back to a conservative tier estimate for any method without a
    measurement, biased high: an over-estimate only delays backfill, whereas an
    under-estimate loses the whole chunk tail to a timeout cancellation.
    """
    if method in _WINNER_SECONDS:
        return _WINNER_SECONDS[method]
    if method in _FAST: return 90
    if method in _MEDIUM: return 300
    if method in _SLOW: return 900
    return 300


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


def gather_grid_size(config: dict) -> int:
    """full grid size (2048), NOT len(list_cells) (1536 step2 pool).

    gather iterates range(gather_grid_size) and reads cell_<row>.h5 by true row
    index; the 512 held-out HPO rows have no fragment and are NaN-filled, so the
    output aligns with step3's true_eldrs over the full prior*dep*design*alpha grid.
    """
    return (config["num_priors"] * len(config["design_eig_percentages"])
            * config["num_designs_per_setting"] * len(config["alphas"]))


def gather_output_path(config: dict) -> str:
    out_dir = config.get("raw_results_dir", "ex/synth/elbo/raw_results")
    # note: the fallback must not be an eager .get() default -- it would KeyError
    # on old configs that carry dataset_filename but no n_p0p1/n_pstar.
    stem = config.get("dataset_filename")
    if stem is None:
        stem = f"dataset_d={config['data_dim']},n_p0p1={config['n_p0p1']},n_pstar={config['n_pstar']}.h5"
    stem = stem.replace("dataset_", "results_").removesuffix(".h5")
    return os.path.join(out_dir, f"{stem}.h5")
