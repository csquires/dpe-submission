"""step2_runner adapter for dokls two-leg vs direct ablation.

contract (consumed by ex.utils.step2_runner.{dispatch,worker,gather}):

  load_config(path) -> dict
  list_cells(config) -> Iterable[int]
  bucket_for_cell(cell_idx, config) -> str
  fit_and_eval(method, hp, cell_idx, config, device) -> dict with keys
      'est_ldrs' (nsamples,), 'mae_per_test_set' (1,), 'true_ldrs' (nsamples,)
  gather_output_path(config) -> str
  gather_dataset_name(method, config) -> str
  walltime_per_cell_seconds(method, config) -> int
  resources_for_method(method) -> sbatch resources string
  is_cpu_eligible(method) -> bool
  method_label(method) -> str  # for cap_for / watchdog

dokls has 70 cells = 7 KL distances x 10 instances; bucket axis is
kl_idx = cell_idx // num_instances_per_kl (10).
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from src.utils.io import _load_config
from ex.ablations.dokls.factory import build
from ex.ablations.dokls import variants


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
  """load + validate dokls config; expand DPE_DATA_ROOT in data_dir."""
  config = _load_config(path)
  required = [
      "data_dir", "raw_results_dir", "device", "seed",
      "num_instances_per_kl", "kl_distances", "data_dim",
      "route", "pstar_idx", "nsamples",
  ]
  missing = [k for k in required if k not in config]
  if missing:
    raise ValueError(f"config missing keys: {missing}")

  # expand env var if present
  config["data_dir"] = os.path.expandvars(str(config["data_dir"]))
  config["raw_results_dir"] = os.path.expandvars(str(config["raw_results_dir"]))

  # validate dataset exists
  dataset_path = os.path.join(config["data_dir"], "dataset.h5")
  if not os.path.exists(dataset_path):
    import logging
    logging.warning(f"dataset.h5 not found at {dataset_path}; step1 may not have run yet")

  return config


def _variant_params(config: dict) -> tuple[str, int, int, int, str]:
  """resolve (route, pstar_idx, N, Nstar, tag) from config + DPE_DOKLS_* env.

  N (nsamples) is the p0/p1 budget; Nstar (nstar, default N) the p* budget. the
  tag gets the _ns{Nstar} suffix only when decoupled, matching variants.VARIANTS.
  """
  route = os.environ.get("DPE_DOKLS_ROUTE", config.get("route", "two_leg"))
  pstar_idx = int(os.environ.get("DPE_DOKLS_PSTAR_IDX", config.get("pstar_idx", 0)))
  N = int(os.environ.get("DPE_DOKLS_NSAMPLES", config.get("nsamples", 8192)))
  nstar = int(os.environ.get("DPE_DOKLS_NSTAR", config.get("nstar", N)))
  tag = f"q{pstar_idx}_N{N}" + ("" if nstar == N else f"_ns{nstar}")
  return route, pstar_idx, N, nstar, tag


def gather_output_path(config: dict) -> str:
  """path where gather writes the unified results; ensures route+tag distinctness.

  uses variants.experiment_name(tag, route) to build the output filename,
  ensuring direct/two_leg and decoupled-Nstar runs never clobber each other.
  """
  route, _pstar_idx, _N, _nstar, tag = _variant_params(config)
  exp_name = variants.experiment_name(tag, route)
  return os.path.join(config["raw_results_dir"], f"{exp_name}_results.h5")


def gather_dataset_name(method: str, config: dict) -> str:
  """dataset key template in results h5; unique per method within already-scoped file."""
  return f"est_ldrs_arr_{method}"


# ---------------------------------------------------------------------------
# cell enumeration + bucket mapping
# ---------------------------------------------------------------------------

def list_cells(config: dict) -> list[int]:
  """all 70 row indices (7 kl x 10 instances)."""
  dataset_path = os.path.join(config["data_dir"], "dataset.h5")
  with h5py.File(dataset_path, "r") as f:
    nrows = f["samples_p0_arr"].shape[0]
  return list(range(nrows))


def bucket_for_cell(cell_idx: int, config: dict) -> str:
  """kl_idx_<n> for cell at row idx (0-based); stratifies HPO by KL distance."""
  return f"kl_idx_{cell_idx // config['num_instances_per_kl']}"


# ---------------------------------------------------------------------------
# fit + eval
# ---------------------------------------------------------------------------

def fit_and_eval(method: str, hp: dict, cell_idx: int, config: dict,
                 device: str) -> dict:
  """fit estimator with hp on row=cell_idx; MC-average on pstar[:N] in-sample.

  returns:
      est_ldrs:           array (nsamples,), per-sample predicted LDR
      mae_per_test_set:   array (1,), in-sample MAE (scalar wrapped)
      true_ldrs:          array (nsamples,), ground truth per-sample LDR
  """
  # resolve variant parameters from config + env override. N = p0/p1 budget,
  # Nstar = p* budget (leg anchoring + MC average); equal on the diagonal.
  route, pstar_idx, nsamples, nstar, _tag = _variant_params(config)

  assert route in {"two_leg", "direct"}, f"invalid route: {route}"
  assert pstar_idx in {0, 1}, f"invalid pstar_idx: {pstar_idx}"
  assert nsamples in {1024, 2048, 4096, 8192}, f"invalid nsamples: {nsamples}"
  assert nstar in {1024, 2048, 4096, 8192}, f"invalid nstar: {nstar}"

  # deterministic seeding per (method, cell_idx) for reproducibility
  seed_val = hash((method, cell_idx)) % (2**32)
  torch.manual_seed(seed_val)
  np.random.seed(seed_val)

  # load data from dataset.h5
  dataset_path = os.path.join(config["data_dir"], "dataset.h5")
  with h5py.File(dataset_path, "r") as f:
    # p0/p1 use the N budget; p* (and its true ldrs) use the Nstar budget.
    p0 = torch.from_numpy(f["samples_p0_arr"][cell_idx, :nsamples, :]).to(device)
    p1 = torch.from_numpy(f["samples_p1_arr"][cell_idx, :nsamples, :]).to(device)
    pstar = torch.from_numpy(f["pstar_arr"][cell_idx, pstar_idx, :nstar, :]).to(device)
    true_ldrs = torch.from_numpy(f["true_ldrs_arr"][cell_idx, pstar_idx, :nstar]).to(device)
    # [cell, pstar] indexes to a 0-d numpy scalar, which torch.from_numpy
    # rejects (needs an ndarray); np.asarray keeps it a 0-d array.
    true_eldr = torch.from_numpy(np.asarray(f["true_eldr_arr"][cell_idx, pstar_idx])).to(device)

  # build estimator via dokls local factory
  estimator = build(method, route, input_dim=3, device=device, **hp)

  # fit (in-sample; pstar[:N] used both for training and evaluation)
  estimator.fit(
      p0, p1, pstar,
      step_cb=None,
      eval_data={"true_ldrs": true_ldrs},
      step_cb_interval=50
  )

  # predict and evaluate in-sample on pstar[:N]
  with torch.no_grad():
    est_logits = estimator.predict_ldr(pstar)  # shape [nsamples]

  est_ldrs_np = est_logits.detach().cpu().numpy().astype(np.float32)
  true_ldrs_np = true_ldrs.cpu().numpy().astype(np.float32)

  mae = np.abs(est_ldrs_np - true_ldrs_np).mean()

  return {
      "est_ldrs": est_ldrs_np,              # shape [nsamples]
      "mae_per_test_set": np.array([mae]),  # shape [1]
      "true_ldrs": true_ldrs_np,            # shape [nsamples]
  }


# ---------------------------------------------------------------------------
# walltime + resources (used by dispatch.py)
# ---------------------------------------------------------------------------

_FAST = {"BDRE", "BDRE_NWJ", "BDRE_DV", "MultiHeadTDRE", "MHT_NWJ", "MHT_DV"}
_MEDIUM = {"TSM", "CTSM"}
_SLOW = {"VFM", "FMDRE"}
_CPU_ONLY = {"TSM", "CTSM", "BDRE", "BDRE_NWJ", "BDRE_DV", "MultiHeadTDRE", "MHT_NWJ", "MHT_DV"}
_GPU_METHODS = {"VFM", "FMDRE"}


def walltime_per_cell_seconds(method: str, config: dict) -> int:
  """rough per-cell wallclock seconds at dim=3.

  two-leg methods cost ~2x their direct equivalents due to interleaved critics/nets.
  """
  route = os.environ.get("DPE_DOKLS_ROUTE", config.get("route", "two_leg"))

  if method in _FAST:
    return 60 if route == "two_leg" else 30
  elif method in _MEDIUM:
    return 120 if route == "two_leg" else 60
  elif method in _SLOW:
    return 240 if route == "two_leg" else 120
  else:
    return 120


def resources_for_method(method: str) -> str:
  """sbatch GPU/cpu/mem flags.

  GPU methods request 1 GPU + 4 cpus + 24G; CPU methods request 2 cpus + 12G.
  two_leg doubles network size, so memory is higher than model_selection.
  """
  if method in _GPU_METHODS:
    return "--gpus=1 --cpus-per-task=4 --mem=24G"
  elif method in _CPU_ONLY:
    return "--cpus-per-task=2 --mem=12G"
  else:
    return "--cpus-per-task=2 --mem=12G"


def is_cpu_eligible(method: str) -> bool:
  """whether this method can run on cpu_dispatcher lane (no GPU)."""
  return method in _CPU_ONLY


def method_label(method: str) -> str:
  """label used in queue file METHOD column."""
  return method
