"""factory function for step2_runner adapters.

extracts common 95% of logic (config loading, cell decoding, fit/eval,
tier-based dispatch) into make_adapter_module. each per-experiment adapter
calls this once and re-exports the 10 callable attributes.
"""
from __future__ import annotations

import os
from typing import Callable

import h5py
import numpy as np
import torch

from src.utils.io import _load_config
from ex.utils.step2_runner.adapter_specs import AdapterSpec


_FAST = {"BDRE", "MDRE_15", "TDRE_5", "MultiHeadTDRE", "MultiHeadTriangularTDRE",
         "TriangularMDRE"}
_MEDIUM = {"CTSM", "TSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
           "TriangularTSM"}
_SLOW = {"VFM", "FMDRE", "FMDRE_S2", "TriangularVFM_V1", "TriangularVFM_V2",
         "TriangularVFM_V3", "TriangularFMDRE"}

_REQUIRED_KEYS = ("data_dir", "raw_results_dir", "alphas", "num_pairs_per_alpha",
                  "num_waypoints", "device", "seed")


def _decode(flat_idx: int, config: dict) -> tuple[int, int]:
    """decode flat cell index to (alpha_idx, pair_idx)."""
    n_pairs = config["num_pairs_per_alpha"]
    return flat_idx // n_pairs, flat_idx % n_pairs


def make_adapter_module(spec: AdapterSpec, search_spaces: dict) -> dict[str, Callable]:
    """factory function to build a complete adapter module.

    given an AdapterSpec and search_spaces dict, returns a dict mapping
    10 callable names to closures that capture spec and search_spaces.
    each per-experiment adapter calls this once and re-exports the results.

    spec: AdapterSpec with fields name, input_dim_fn, walltimes, mem_slow,
          mem_default, default_output_dir.
    search_spaces: dict mapping method names to {'builder': ..., 'requires_pstar': ...}.

    returns: dict with keys load_config, list_cells, bucket_for_cell,
             fit_and_eval, walltime_per_cell_seconds, resources_for_method,
             is_cpu_eligible, method_label, gather_dataset_name, gather_output_path.
    """

    def load_config(path: str) -> dict:
        """load and validate config from yaml file."""
        config = _load_config(path)
        for k in _REQUIRED_KEYS:
            if k not in config:
                raise ValueError(f"config missing key: {k}")
        return config

    def list_cells(config: dict) -> list[int]:
        """enumerate all flat cell indices in this experiment."""
        n = len(config["alphas"]) * config["num_pairs_per_alpha"]
        return list(range(n))

    def bucket_for_cell(cell_idx: int, config: dict) -> str:
        """assign cell to alpha-indexed bucket for parallelization."""
        a, _ = _decode(cell_idx, config)
        return f"alpha_idx_{a}"

    def fit_and_eval(method: str, hp: dict, cell_idx: int, config: dict,
                     device: str) -> dict:
        """fit estimator and return ldr predictions.

        steps:
        1. check method in search_spaces (raise KeyError).
        2. decode flat_idx to (alpha, pair) indices.
        3. build data_path and check existence (raise FileNotFoundError).
        4. extract builder and requires_pstar from search_spaces[method].
        5. seed RNG reproducibly.
        6. build builder_kwargs with input_dim from spec.input_dim_fn(config).
        7. instantiate estimator.
        8. load h5 data (pstar, p0, p1).
        9. call fit (with or without pstar).
        10. predict ldr and return as dict with key "est_ldrs".
        """
        if method not in search_spaces:
            raise KeyError(f"method {method!r} not in {spec.name} SEARCH_SPACES")
        a, p = _decode(cell_idx, config)
        data_path = os.path.join(config["data_dir"], f"alpha_{a}_pair_{p}.h5")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"step1 data not found: {data_path}")

        spec_entry = search_spaces[method]
        builder = spec_entry["builder"]
        requires_pstar = spec_entry.get("requires_pstar", False)

        seed_val = hash((method, cell_idx)) % (2**32)
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)

        builder_kwargs = {
            "input_dim": spec.input_dim_fn(config),
            "device": device,
            "num_waypoints": config["num_waypoints"],
            **hp,
        }
        estimator = builder(**builder_kwargs)

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

    def walltime_per_cell_seconds(method: str, config: dict) -> int:
        """dispatch walltime estimate by method tier."""
        if method in _FAST:
            return spec.walltimes["fast"]
        if method in _MEDIUM:
            return spec.walltimes["medium"]
        if method in _SLOW:
            return spec.walltimes["slow"]
        return spec.walltimes["default"]

    def resources_for_method(method: str) -> str:
        """dispatch resource requirement by method tier."""
        if method in _SLOW:
            return spec.mem_slow
        return spec.mem_default

    def is_cpu_eligible(method: str) -> bool:
        """check if method can run on CPU."""
        return method in (_FAST | _MEDIUM)

    def method_label(method: str) -> str:
        """return human-readable label for method."""
        return method

    def gather_dataset_name(method: str, config: dict) -> str:
        """name for gathered results dataset for this method."""
        return f"est_ldrs_{method}"

    def gather_output_path(config: dict) -> str:
        """path to gathered results file."""
        return os.path.join(config.get("raw_results_dir", spec.default_output_dir),
                            "results_all_cells.h5")

    return {
        "load_config": load_config,
        "list_cells": list_cells,
        "bucket_for_cell": bucket_for_cell,
        "fit_and_eval": fit_and_eval,
        "walltime_per_cell_seconds": walltime_per_cell_seconds,
        "resources_for_method": resources_for_method,
        "is_cpu_eligible": is_cpu_eligible,
        "method_label": method_label,
        "gather_dataset_name": gather_dataset_name,
        "gather_output_path": gather_output_path,
    }
