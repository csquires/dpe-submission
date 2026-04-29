"""build one alpha row of the pendulum trajectory KL grid.

CLI entry that takes --alpha-idx i and computes:
  - q_O_i (via load_or_build_q; cached by reward+alpha hash)
  - KL1[i] = KL(p^pi_E || p^pi_O_i)
  - KL2[i, :] = KL(p^pi_mix(i, beta_j) || p^pi_E)  for j in range(G_beta)
  - q_O_residual_i

results are written to ${cache_dir}/traj_alpha_{i}_{hash}.h5 atomically.
the assembler (assemble_traj_grid.py) reads these files and stacks them
into the canonical traj_grid_{hash}.h5.

CRN: bit-identical to sequential build_traj_kl_grid. each per-alpha job
recreates the root generator `gen = default_rng(global_seed)`, advances
it `i+1` times via gen.integers() (matching the i-th iteration of the
sequential loop), and uses the result as alpha_seed. then the within-row
dependency chain (alpha_gen → traj_kl_mc(KL1) → alpha_gen.integers() →
beta_seed → traj_kl_mc(KL2)) follows the exact same code path as
sequential, so KL1[i] and KL2[i, :] are bit-equal to a sequential run.

Q_E concurrency: this script assumes Q_E is already cached (preheat job
should run first). it will still build Q_E if missing, which is wasteful
but not incorrect — load_or_build_q does atomic writes so concurrent
builds don't corrupt the cache.
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import yaml

from src.utils.io import _load_config, _write_hdf5_atomic
from src.utils.pendulum import F, sample_mu0, log_mu0
from src.utils.pendulum_policies import GaussPolicy, MixPolicy
from src.utils.pendulum_q import load_or_build_q
from src.sampling.pendulum_traj import traj_kl_mc

from experiments.pendulum_eldr_estimation.step1_create_data import (
    _build_env_and_q_cfg, _resolve_reward,
)
from experiments.utils.prescribed_kls import hash_pendulum_cfg


def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-idx", type=int, required=True,
                   help="grid index in [0, G_alpha) for this row")
    p.add_argument("--config",
                   default="experiments/pendulum_eldr_estimation/config.yaml")
    return p.parse_args(args)


def make_r_O(alpha_val: float, r_E_fn, r_anti_fn):
    """closure: r_O = (1 - alpha) * r_E + alpha * r_anti, defeats late-binding."""
    def r_O(s, a, s_next, cfg):
        return (1.0 - alpha_val) * r_E_fn(s, a, s_next, cfg) + alpha_val * r_anti_fn(s, a, s_next, cfg)
    return r_O


def build_grid_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """assemble the cfg dict consumed by hash_pendulum_cfg; must match step1."""
    env_cfg, q_cfg = _build_env_and_q_cfg(config)
    alphas = np.linspace(0, 1, config["traj_kl_grid"]["G_alpha"]).tolist()
    betas = np.linspace(0, 1, config["traj_kl_grid"]["G_beta"]).tolist()
    return {
        "env_cfg": env_cfg,
        "q_cfg": q_cfg,
        "T": int(config["trajectory"]["T"]),
        "sigma_pi": float(config["pendulum"]["sigma_pi"]),
        "M": int(config["traj_kl_grid"]["M"]),
        "r_E_name": config["pendulum"]["r_E_name"],
        "r_anti_name": config["pendulum"]["r_anti_name"],
        "alphas": alphas,
        "betas": betas,
    }


def main():
    args = parse_args()
    config = _load_config(args.config)

    grid_cfg = build_grid_cfg(config)
    cfg_hash = hash_pendulum_cfg(grid_cfg)
    cache_dir = Path(config["traj_kl_grid"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_path = cache_dir / f"traj_alpha_{args.alpha_idx}_{cfg_hash}.h5"
    if out_path.exists():
        print(f"alpha row already cached at {out_path}; nothing to do.")
        return

    env_cfg = grid_cfg["env_cfg"]
    q_cfg = grid_cfg["q_cfg"]
    alphas = np.array(grid_cfg["alphas"])
    betas = np.array(grid_cfg["betas"])
    G_alpha = len(alphas)
    G_beta = len(betas)

    if not (0 <= args.alpha_idx < G_alpha):
        sys.exit(f"--alpha-idx {args.alpha_idx} out of range [0, {G_alpha})")

    alpha_i = float(alphas[args.alpha_idx])
    r_E = _resolve_reward(grid_cfg["r_E_name"])
    r_anti = _resolve_reward(grid_cfg["r_anti_name"])
    sigma_pi = grid_cfg["sigma_pi"]
    T = grid_cfg["T"]
    M = grid_cfg["M"]
    kl_se_threshold = float(config["traj_kl_grid"].get("kl_se_warn_threshold", 0.1))

    # load expert Q (preheat job should have built this; otherwise we build it here)
    q_E_result = load_or_build_q(
        env_cfg, r_E, grid_cfg["r_E_name"], q_cfg, str(cache_dir),
        F=F, alpha=None, r_E_name=None, r_anti_name=None,
    )
    pi_E = GaussPolicy(q_E_result["Q"], sigma_pi, env_cfg, q_cfg)

    # build / load Q_O for this alpha
    r_O = make_r_O(alpha_i, r_E, r_anti)
    q_O_result = load_or_build_q(
        env_cfg, r_O, "r_O", q_cfg, str(cache_dir),
        F=F, alpha=alpha_i,
        r_E_name=grid_cfg["r_E_name"], r_anti_name=grid_cfg["r_anti_name"],
    )
    pi_O = GaussPolicy(q_O_result["Q"], sigma_pi, env_cfg, q_cfg)
    q_O_residual = float(q_O_result["bellman_residual"])

    # CRN: replicate sequential build_traj_kl_grid bit-exactly.
    # sequential: for i in range(G_alpha): alpha_seed = gen.integers(0, 2**63-1).
    # parallel: redo the gen.integers() chain up to and including this alpha_idx.
    global_seed = int(config.get("seed", 1729))
    gen = np.random.default_rng(global_seed)
    for _ in range(args.alpha_idx):
        gen.integers(0, 2**63 - 1)  # advance past prior alphas
    alpha_seed = int(gen.integers(0, 2**63 - 1))
    alpha_gen = np.random.default_rng(alpha_seed)

    # KL1[i] = KL(p^pi_E || p^pi_O_i) under rollouts from pi_E
    print(f"alpha_idx={args.alpha_idx}, alpha={alpha_i:.4f}: computing KL1...")
    res_KL1 = traj_kl_mc(
        pi_E.sample, pi_E.log_prob, pi_O.log_prob,
        F, sample_mu0, log_mu0, T, M, env_cfg, alpha_gen,
        kl_se_warn_threshold=kl_se_threshold,
    )
    kl1 = float(res_KL1["kl_hat"])
    kl1_se = float(res_KL1["kl_se"])
    print(f"  KL1[{args.alpha_idx}] = {kl1:.4f} +/- {kl1_se:.4f}")

    # KL2[i, j] = KL(p^pi_mix || p^pi_E) under rollouts from pi_mix
    kl2_row = np.zeros(G_beta)
    kl2_se_row = np.zeros(G_beta)
    for j, beta in enumerate(betas):
        pi_mix = MixPolicy(pi_O, pi_E, float(beta))
        # match sequential: beta_seed drawn from alpha_gen *after* traj_kl_mc consumed it.
        beta_seed = int(alpha_gen.integers(0, 2**63 - 1))
        beta_gen = np.random.default_rng(beta_seed)
        res_KL2 = traj_kl_mc(
            pi_mix.sample, pi_mix.log_prob, pi_E.log_prob,
            F, sample_mu0, log_mu0, T, M, env_cfg, beta_gen,
            kl_se_warn_threshold=kl_se_threshold,
        )
        kl2_row[j] = float(res_KL2["kl_hat"])
        kl2_se_row[j] = float(res_KL2["kl_se"])
        print(f"  KL2[{args.alpha_idx}, {j}] (beta={beta:.4f}) = "
              f"{kl2_row[j]:.4f} +/- {kl2_se_row[j]:.4f}")

    datasets = {
        "KL1": np.array([kl1]),
        "KL1_se": np.array([kl1_se]),
        "KL2_row": kl2_row,
        "KL2_se_row": kl2_se_row,
    }
    attrs = {
        "alpha_idx": int(args.alpha_idx),
        "alpha": float(alpha_i),
        "G_alpha": int(G_alpha),
        "G_beta": int(G_beta),
        "q_O_residual": q_O_residual,
        "cfg_hash": cfg_hash,
        "global_seed": int(global_seed),
    }
    _write_hdf5_atomic(str(out_path), datasets, attrs)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
