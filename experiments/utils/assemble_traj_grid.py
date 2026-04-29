"""assemble per-alpha pendulum trajectory KL files into the canonical grid hdf5.

reads ${cache_dir}/traj_alpha_{i}_{hash}.h5 for i in [0, G_alpha) and writes
${cache_dir}/traj_grid_{hash}.h5 in the same schema as build_traj_kl_grid.

usage:
  python -m experiments.utils.assemble_traj_grid [--config <path>] [--rebuild]

errors out if any per-alpha file is missing (assembler must run after the
sbatch array completes successfully).
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import warnings

from src.utils.io import _load_config, _write_hdf5_atomic
from src.utils.pendulum import F, sample_mu0, log_mu0
from src.utils.pendulum_q import load_or_build_q

from experiments.pendulum_eldr_estimation.step1_create_data import _resolve_reward
from experiments.utils.build_traj_alpha import build_grid_cfg
from experiments.utils.prescribed_kls import hash_pendulum_cfg, assert_monotone


def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="experiments/pendulum_eldr_estimation/config.yaml")
    p.add_argument("--rebuild", action="store_true",
                   help="rewrite traj_grid_{hash}.h5 even if present")
    return p.parse_args(args)


def main():
    args = parse_args()
    config = _load_config(args.config)

    grid_cfg = build_grid_cfg(config)
    cfg_hash = hash_pendulum_cfg(grid_cfg)
    cache_dir = Path(config["traj_kl_grid"]["cache_dir"])
    out_path = cache_dir / f"traj_grid_{cfg_hash}.h5"

    if out_path.exists() and not args.rebuild:
        print(f"traj_grid already exists at {out_path}. pass --rebuild to overwrite.")
        return

    env_cfg = grid_cfg["env_cfg"]
    q_cfg = grid_cfg["q_cfg"]
    alphas = np.array(grid_cfg["alphas"])
    betas = np.array(grid_cfg["betas"])
    G_alpha = len(alphas)
    G_beta = len(betas)

    # collect per-alpha files
    KL1 = np.zeros(G_alpha)
    KL1_se = np.zeros(G_alpha)
    KL2 = np.zeros((G_alpha, G_beta))
    KL2_se = np.zeros((G_alpha, G_beta))
    q_O_residuals = np.zeros(G_alpha)

    for i in range(G_alpha):
        path = cache_dir / f"traj_alpha_{i}_{cfg_hash}.h5"
        if not path.exists():
            raise FileNotFoundError(
                f"missing per-alpha file: {path}. "
                f"ensure the sbatch array job for alpha {i} completed successfully."
            )
        with h5py.File(path, "r") as f:
            KL1[i] = float(f["KL1"][0])
            KL1_se[i] = float(f["KL1_se"][0])
            KL2[i, :] = f["KL2_row"][:]
            KL2_se[i, :] = f["KL2_se_row"][:]
            q_O_residuals[i] = float(f.attrs["q_O_residual"])

    # load Q_E (cached by preheat) for q_E + q_E_residual + q_O_grid stacking
    r_E = _resolve_reward(grid_cfg["r_E_name"])
    r_anti = _resolve_reward(grid_cfg["r_anti_name"])
    q_E_result = load_or_build_q(
        env_cfg, r_E, grid_cfg["r_E_name"], q_cfg, str(cache_dir),
        F=F, alpha=None, r_E_name=None, r_anti_name=None,
    )
    q_E = q_E_result["Q"]
    q_E_residual = float(q_E_result["bellman_residual"])

    # rebuild Q_O grid by re-loading each alpha's Q_O cache (cheap; cache hit)
    N_theta, N_theta_dot, N_action = q_E.shape
    q_O_grid = np.zeros((G_alpha, N_theta, N_theta_dot, N_action))
    for i, alpha in enumerate(alphas):
        # closure must match the one in build_traj_alpha exactly so the
        # cache hash lines up. r_O itself is not invoked at load time.
        def make_r_O(a):
            def r_O(s, a_, s_next, cfg):
                return (1.0 - a) * r_E(s, a_, s_next, cfg) + a * r_anti(s, a_, s_next, cfg)
            return r_O
        q_O_result = load_or_build_q(
            env_cfg, make_r_O(float(alpha)), "r_O", q_cfg, str(cache_dir),
            F=F, alpha=float(alpha),
            r_E_name=grid_cfg["r_E_name"], r_anti_name=grid_cfg["r_anti_name"],
        )
        q_O_grid[i] = q_O_result["Q"]

    # monotonicity checks (mirror build_traj_kl_grid behavior)
    monotone_alpha = True
    try:
        assert_monotone(KL1, axis=0, kind="increasing")
    except (ValueError, AssertionError):
        monotone_alpha = False
        warnings.warn("KL1 non-monotone in alpha")

    monotone_beta_per_alpha = np.zeros(G_alpha, dtype=bool)
    for i in range(G_alpha):
        try:
            assert_monotone(KL2[i, :], axis=0, kind="decreasing")
            monotone_beta_per_alpha[i] = True
        except (ValueError, AssertionError):
            monotone_beta_per_alpha[i] = False
            warnings.warn(f"KL2[{i}, :] non-monotone in beta")

    datasets = {
        "KL1": KL1, "KL2": KL2,
        "KL1_se": KL1_se, "KL2_se": KL2_se,
        "alphas": alphas, "betas": betas,
        "q_E": q_E, "q_O_grid": q_O_grid,
        "q_O_residuals": q_O_residuals,
        "monotone_beta_per_alpha": monotone_beta_per_alpha.astype(np.int32),
    }
    attrs = {
        "hash": cfg_hash,
        "G_alpha": int(G_alpha),
        "G_beta": int(G_beta),
        "T": int(grid_cfg["T"]),
        "M": int(grid_cfg["M"]),
        "sigma_pi": float(grid_cfg["sigma_pi"]),
        "q_E_residual": q_E_residual,
        "monotone_alpha": int(monotone_alpha),
    }
    _write_hdf5_atomic(str(out_path), datasets, attrs)

    # report
    print(f"saved {out_path}")
    print(f"\nKL1 range: [{KL1.min():.4f}, {KL1.max():.4f}]")
    print(f"KL2 range (whole grid): [{KL2.min():.4f}, {KL2.max():.4f}]")
    print(f"monotone_alpha: {monotone_alpha}")
    print(f"monotone_beta_per_alpha: {monotone_beta_per_alpha.sum()}/{G_alpha} rows monotone")


if __name__ == "__main__":
    main()
