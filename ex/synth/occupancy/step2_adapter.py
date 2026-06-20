"""step2_runner adapter for smodice_eldr_estimation.

cell axis: flat integer index over the cartesian product
    (k1_idx in [0..len(k1_values)-1])
    x (beta_idx in [0..len(beta_values)-1])
    x (seed in [0..seeds-1])
encoded as `flat_idx = ((k1_idx * len(beta_values)) + beta_idx) * seeds + seed`.

bucket axis: f"k1_idx_{k1_idx}" — the variant_sweep markdown stratifies smodice
findings by k1; per_bucket overrides in winners.yaml can target individual k1
strata if needed.

per-cell input: <data_dir>/<encoding_subdir>/kl1_<k1>_beta_<b>_seed_<s>.h5
per-cell output (fragment): <DPE_DATA_ROOT>/<exp>/step2_results/<method>/cell_<flat_idx>.h5
    contains 'est_ldrs' of shape (num_samples,) and attrs (encoding, k1_idx, beta_idx, seed).

quirks:
- encoding-aware path logic: encoding_type 'gaussian_blob'/'flow_pushforward' uses
  sigma subdir; 'onehot_*' uses 'sigma_na' subdir.
- input_dim depends on encoding: derive_input_dim() handles all four encodings.
- only methods listed in SUPPORTED_ENCODINGS for the active encoding are runnable.
- triangular methods take 3-arg fit (p0, p1, pstar); SmoothedTabularPluginDRE takes
  latent kwargs; everything else takes 2-arg fit.
- the original step2 wrote one h5 per (k1, beta, seed) cell containing all method
  est_ldrs as separate datasets. this adapter writes one h5 per (method, cell)
  fragment instead. step3 may need a small update OR a custom gather hook to
  re-aggregate fragments into the original per-cell h5 layout.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch
import yaml

from src.utils.io import _load_config
from ex.utils.hpo.method_specs import METHOD_SPECS
from src.sampling.frozen_flow import FrozenFlow


# encodings supported by each method (mirrors original step2's SUPPORTED_ENCODINGS)
SUPPORTED_ENCODINGS = {
    "TabularPluginDRE":          {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    "SmoothedTabularPluginDRE":  {"gaussian_blob", "flow_pushforward"},
    "BDRE":                      {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    "MDRE_15":                   {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    "MultiHeadTDRE":             {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    "TriangularMDRE":            {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    "MultiHeadTriangularTDRE":   {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    "TSM":                       {"gaussian_blob", "flow_pushforward"},
    "CTSM":                      {"gaussian_blob", "flow_pushforward"},
    "VFM":                       {"gaussian_blob", "flow_pushforward"},
    "VFMOrthros":                {"gaussian_blob", "flow_pushforward"},
    "FMDRE":                     {"gaussian_blob", "flow_pushforward"},
    "FMDRE_S2":                  {"gaussian_blob", "flow_pushforward"},
    "TriangularCTSM_V1":         {"gaussian_blob", "flow_pushforward"},
    "TriangularCTSM_V2":         {"gaussian_blob", "flow_pushforward"},
    "TriangularCTSM_V3":         {"gaussian_blob", "flow_pushforward"},
    "TriangularVFM_V1":          {"gaussian_blob", "flow_pushforward"},
    "TriangularVFM_V2":          {"gaussian_blob", "flow_pushforward"},
    "TriangularVFM_V3":          {"gaussian_blob", "flow_pushforward"},
    "TriangularTSM":             {"gaussian_blob", "flow_pushforward"},
    "TriangularFMDRE":           {"gaussian_blob", "flow_pushforward"},
}

NEEDS_LATENT = {"SmoothedTabularPluginDRE"}


def _requires_pstar(method: str) -> bool:
    return METHOD_SPECS.get(method, {}).get("requires_pstar", False)


# -----------------------------------------------------------------------------
# config + per-cell index encoding
# -----------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """load smodice config; install MDP shape into encoding for downstream use."""
    config = _load_config(path)
    required = ["data_dir", "raw_results_dir", "encoding", "kl_targets", "num_samples",
                "device", "seed", "gridworld"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    L = config["gridworld"]["L"]
    enc = dict(config["encoding"])
    enc["n_states"] = L * L
    enc["n_actions"] = 4
    enc["L"] = L
    config["encoding"] = enc
    return config


def _encoding_subdir(base: str, encoding_cfg: dict) -> str:
    encoding_type = encoding_cfg["type"]
    if encoding_type.startswith("onehot"):
        return os.path.join(base, encoding_type, "sigma_na")
    sigma = encoding_cfg["sigma"]
    return os.path.join(base, encoding_type, f"sigma_{sigma:.3f}")


def _decode_cell(flat_idx: int, config: dict) -> tuple[int, int, int]:
    """return (k1_idx, beta_idx, seed) from flat cell index."""
    n_beta = len(config["kl_targets"]["beta_values"])
    seeds = config["kl_targets"].get("seeds_default", 1)
    seed = flat_idx % seeds
    rest = flat_idx // seeds
    beta_idx = rest % n_beta
    k1_idx = rest // n_beta
    return k1_idx, beta_idx, seed


def list_cells(config: dict) -> list[int]:
    """flat ints for (k1_idx, beta_idx, seed) tuples reserved for step2.

    delegates to the hpo adapter's step2_pool() so step2 runs only on the
    cells NOT seen during hpo. encoding mirrors the existing convention:
    flat_idx = ((k1_idx * n_beta) + beta_idx) * seeds + seed.
    """
    from ex.utils.hpo.adapters import get_adapter
    adapter = get_adapter("occupancy")
    n_beta = len(config["kl_targets"]["beta_values"])
    seeds = config["kl_targets"].get("seeds_default", 1)
    return [
        ((k1_idx * n_beta) + beta_idx) * seeds + seed
        for (k1_idx, beta_idx, seed) in adapter.step2_pool()
    ]


def bucket_for_cell(cell_idx: int, config: dict) -> str:
    k1_idx, _, _ = _decode_cell(cell_idx, config)
    return f"k1_idx_{k1_idx}"


# -----------------------------------------------------------------------------
# fit + eval
# -----------------------------------------------------------------------------

def _derive_input_dim(encoding_cfg: dict) -> int:
    encoding_type = encoding_cfg["type"]
    n_states = encoding_cfg["n_states"]
    n_actions = encoding_cfg["n_actions"]
    if encoding_type == "onehot_joint":
        return n_states * n_actions
    if encoding_type == "onehot_concat":
        return n_states + n_actions
    if encoding_type in ("gaussian_blob", "flow_pushforward"):
        return encoding_cfg.get("embed_dim", 6)
    raise ValueError(f"unknown encoding type: {encoding_type}")


def _attach_flow_module(encoding_cfg: dict, base_seed: int) -> dict:
    """deep-copy + attach FrozenFlow if encoding_type == flow_pushforward."""
    if encoding_cfg["type"] != "flow_pushforward":
        return encoding_cfg
    enc = dict(encoding_cfg)
    flow_cfg = enc.get("flow", {})
    enc["flow_module"] = FrozenFlow(
        dim=enc["embed_dim"],
        n_layers=flow_cfg.get("layers", 4),
        seed=flow_cfg.get("seed", base_seed),
    )
    return enc


_METHOD_ALIAS = {p[0]: p[1] for p in __import__("ex.utils.hpo.method_specs",
                                                fromlist=["ALIAS_PAIRS"]).ALIAS_PAIRS}


def fit_and_eval(method: str, hp: dict, cell_idx: int, config: dict,
                 device: str) -> dict:
    """fit on (p0, p1) (and pstar if triangular); predict_ldr on pstar samples."""
    canonical = _METHOD_ALIAS.get(method, method)
    if canonical not in METHOD_SPECS:
        raise KeyError(f"method {method!r} (canonical {canonical!r}) not registered in METHOD_SPECS")
    encoding_cfg = config["encoding"]
    encoding_type = encoding_cfg["type"]
    if encoding_type not in SUPPORTED_ENCODINGS.get(canonical, set()):
        raise ValueError(f"method {method!r} (canonical {canonical!r}) does not support encoding {encoding_type!r}")

    spec = METHOD_SPECS[canonical]
    builder = spec["builder"]
    input_dim = _derive_input_dim(encoding_cfg)
    num_waypoints = spec.get("num_waypoints", None)

    seed_val = hash((method, cell_idx)) % (2**32)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    # tabular methods take encoding_cfg / mdp shape, not flat hp dict
    if method == "TabularPluginDRE":
        from src.methods import TabularPluginDRE
        decode = "argmax" if encoding_type.startswith("onehot") else "nn"
        estimator = TabularPluginDRE(
            n_states=encoding_cfg["n_states"],
            n_actions=encoding_cfg["n_actions"],
            encoding_cfg=_attach_flow_module(encoding_cfg, config["seed"]),
            decode=decode,
            device=device,
        )
    elif method == "SmoothedTabularPluginDRE":
        from src.methods import SmoothedTabularPluginDRE
        estimator = SmoothedTabularPluginDRE(
            n_states=encoding_cfg["n_states"],
            n_actions=encoding_cfg["n_actions"],
            encoding_cfg=_attach_flow_module(encoding_cfg, config["seed"]),
            device=device,
        )
    else:
        builder_kwargs = {
            "input_dim": input_dim,
            "device": device,
            "num_waypoints": num_waypoints if num_waypoints is not None else 0,
            **hp,
        }
        estimator = builder(**builder_kwargs)

    # load per-cell h5
    k1_idx, beta_idx, seed = _decode_cell(cell_idx, config)
    data_subdir = _encoding_subdir(config["data_dir"], encoding_cfg)
    data_path = os.path.join(data_subdir, f"kl1_{k1_idx}_beta_{beta_idx}_seed_{seed}.h5")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"step1 data not found: {data_path}")
    with h5py.File(data_path, "r") as f:
        pstar_samples = torch.from_numpy(f["pstar_samples"][...]).float().to(device)
        p0_samples = torch.from_numpy(f["p0_samples"][...]).float().to(device)
        p1_samples = torch.from_numpy(f["p1_samples"][...]).float().to(device)
        latent_p0 = torch.from_numpy(f["p0_latent"][...]).long().to(device)
        latent_p1 = torch.from_numpy(f["p1_latent"][...]).long().to(device)

    if _requires_pstar(method):
        estimator.fit(p0_samples, p1_samples, pstar_samples)
    elif method in NEEDS_LATENT:
        estimator.fit(p0_samples, p1_samples, latent_p0=latent_p0, latent_p1=latent_p1)
    else:
        estimator.fit(p0_samples, p1_samples)

    with torch.no_grad():
        est_ldrs = estimator.predict_ldr(pstar_samples)
    est_ldrs_np = est_ldrs.detach().cpu().numpy().astype(np.float32)

    return {"est_ldrs": est_ldrs_np}


# -----------------------------------------------------------------------------
# walltime + resources
# -----------------------------------------------------------------------------

# rough per-cell seconds for L=16 gridworld with num_samples=5000
_FAST = {"BDRE", "MDRE_15", "MultiHeadTDRE", "MultiHeadTriangularTDRE",
         "TriangularMDRE", "TabularPluginDRE", "SmoothedTabularPluginDRE"}
_MEDIUM = {"CTSM", "TSM", "TriangularCTSM_V1", "TriangularCTSM_V2",
           "TriangularCTSM_V3", "TriangularTSM"}
_SLOW = {"VFM", "VFMOrthros", "FMDRE", "FMDRE_S2", "TriangularVFM_V1", "TriangularVFM_V2",
         "TriangularVFM_V3", "TriangularFMDRE"}


def walltime_per_cell_seconds(method: str, config: dict) -> int:
    if method in _FAST: return 30
    if method in _MEDIUM: return 90
    if method in _SLOW: return 180
    return 120


def resources_for_method(method: str) -> str:
    if method in _SLOW:
        return "--gpus=1 --cpus-per-task=4 --mem=24G"
    return "--gpus=1 --cpus-per-task=2 --mem=12G"


def is_cpu_eligible(method: str) -> bool:
    return method in _FAST | _MEDIUM


def method_label(method: str) -> str:
    return method


# -----------------------------------------------------------------------------
# gather overrides (smodice naming convention: 'est_ldrs_<method>')
# -----------------------------------------------------------------------------

def gather_dataset_name(method: str, config: dict) -> str:
    """smodice's existing step3 reads 'est_ldrs_<method>' (no '_arr' suffix)."""
    return f"est_ldrs_{method}"


def gather_output_path(config: dict) -> str:
    """write a single combined results.h5 under raw_results_dir/<encoding>/<sigma>/.

    NOTE: the original smodice step2 wrote one h5 per (k1, beta, seed) cell with
    all methods inside. this gather emits ONE results.h5 with arrays of shape
    (n_cells, num_samples) per method instead. step3 may need a small update
    to read the per-cell slice for each (k1, beta, seed) tuple — or write a
    custom gather override that reproduces the original layout. for the
    handoff, the simpler shape is committed; teammate can adapt step3 if
    needed.
    """
    out_dir = _encoding_subdir(config["raw_results_dir"], config["encoding"])
    return os.path.join(out_dir, "results_all_cells.h5")
