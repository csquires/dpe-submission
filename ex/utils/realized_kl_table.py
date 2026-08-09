"""
realized K1 (KL divergence) measurement, stratum selection, and caching.

Computes KL(p^pi_E || p^pi_O) via direct Monte Carlo on flow-based policies.
Content-addressed yaml caching with atomic writes.

All policies are flows (no Gaussian base, no correction term).
Direction: KL(p_E || p_O), trajectories rolled under pi_E.
"""
import numpy as np
import yaml
import hashlib
import os
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

# import core sampling and dynamics
from src.sampling.pendulum_traj import traj_kl_mc
from src.utils.pendulum import F, sample_mu0, log_mu0
from src.utils.pendulum_policies import FlowPolicy
from src.models.flow.train_flow_policy import load_flow


def measure_realized_k1(
    ladder: list[dict],
    ckpt_E: str,
    env_cfg,
    T: int,
    M: int = 500,
    seed: int = 0,
) -> list[dict]:
    """
    Compute realized K1 for each ladder entry via direct Monte Carlo.

    KL(p || q) = E_tau ~ p[log p(tau) - log q(tau)], where tau are
    trajectories sampled under p_E.

    Procedure:
      (1) root_gen = np.random.default_rng(seed)
      (2) For each entry in ladder with keys {alpha, ckpt_O}:
          - Load pi_E and pi_O as FlowPolicy wrappers
          - sub_gen = root_gen.spawn(1)[0]  # CRN: fresh per alpha
          - Compute KL via traj_kl_mc (direct MC, no importance weighting)
          - If kl_hat < 0: warn, report raw value (no clipping)
          - Append {alpha, ckpt_O, kl_hat, kl_se} to output
      (3) Return measured entries in original ladder order

    Args:
      ladder: list of dicts with keys {alpha, ckpt_O}. Caller filters
              out gate-failed entries; this function does NOT check gate.
      ckpt_E: path to expert flow checkpoint (string).
      env_cfg: PendulumCfg instance.
      T: int, trajectory horizon.
      M: int, number of MC samples (trajectories to roll). default 500.
      seed: int, root RNG seed for reproducibility.

    Returns:
      list of dicts with keys {alpha, ckpt_O, kl_hat, kl_se}, order preserved.

    Raises:
      Any exception from load_flow or traj_kl_mc.
    """
    # root generator for CRN
    root_gen = np.random.default_rng(seed)

    # load expert policy once
    pi_E = FlowPolicy(load_flow(ckpt_E, device="cpu"))

    measured = []
    for entry in ladder:
        alpha = entry["alpha"]
        ckpt_O = entry["ckpt_O"]

        # fresh sub-generator per alpha (CRN discipline)
        sub_gen = root_gen.spawn(1)[0]

        # load on-policy flow
        pi_O = FlowPolicy(load_flow(ckpt_O, device="cpu"))

        # compute KL via direct MC on trajectories from p_E
        kl_dict = traj_kl_mc(
            pi_E.sample,
            pi_E.log_prob,
            pi_O.log_prob,
            F,
            sample_mu0,
            log_mu0,
            T,
            M,
            env_cfg,
            sub_gen,
        )

        kl_hat = kl_dict["kl_hat"]
        kl_se = kl_dict["kl_se"]

        # warn on negative KL (MC noise at small alpha), but report raw
        if kl_hat < 0:
            warnings.warn(
                f"realized K1 negative at alpha={alpha}, measured {kl_hat}",
                stacklevel=2,
            )

        # append to output, preserving order
        measured.append({
            "alpha": alpha,
            "ckpt_O": ckpt_O,
            "kl_hat": float(kl_hat),
            "kl_se": float(kl_se),
        })

    return measured


def write_alphas_chosen(data_dir, ckpt_E, strata, ladder):
    """write the CANONICAL alphas_chosen.yaml (single source of truth).

    schema (consumed by step1 per_cell, append_test_set, diagnostics):
      ckpt_E: str, flow_hash_E: sha16 of ckpt_E file bytes,
      strata: list indexed by k1_idx, each {alpha, ckpt_O, flow_hash_O,
        rl_hash_O, K1_realized_flow, K1_se, stratum_label},
      ladder: full measured table [{alpha, kl_hat, kl_se}, ...] (provenance).
    atomic write; returns the path.
    """
    from ex.utils.flow_gate import _compute_ckpt_hash
    doc = {
        "ckpt_E": str(ckpt_E),
        "flow_hash_E": _compute_ckpt_hash(str(ckpt_E)),
        "strata": strata,
        "ladder": ladder,
    }
    path = Path(data_dir) / "alphas_chosen.yaml"
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    os.replace(tmp, str(path))
    return str(path)


def load_alphas_chosen(data_dir):
    """load and validate the canonical doc; actionable error if absent."""
    path = Path(data_dir) / "alphas_chosen.yaml"
    if not path.exists():
        raise ValueError(
            f"alphas_chosen.yaml not found at {path}; "
            "run ex/semisynth/pendulum/step0d_stamp_strata.py first")
    with open(path) as f:
        doc = yaml.safe_load(f)
    for key in ("ckpt_E", "flow_hash_E", "strata"):
        if key not in doc:
            raise ValueError(f"alphas_chosen.yaml missing key: {key}")
    return doc
