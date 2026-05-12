import os
import argparse
import numpy as np
from pathlib import Path
from itertools import product
from typing import Callable, Dict, Tuple, Any

from src.utils.io import _load_config, _set_seed, _hdf5_exists, _write_hdf5_atomic
from src.utils.pendulum import PendulumCfg, F, sample_mu0, log_mu0, r_upright, r_swingdown
from src.utils.pendulum_q import QGridCfg, load_or_build_q
from src.utils.pendulum_policies import GaussPolicy, MixPolicy
from src.sampling.pendulum_traj import rollout, log_density, pack
from ex.utils.prescribed_kls import load_or_build_traj_grid, prescribe_traj
from ex.utils.alpha_grid import make_alphas


def _resolve_reward(name: str) -> Callable:
    """
    map reward name string to reward function.

    args:
        name: one of {"upright", "swingdown"}

    returns:
        callable reward function r(s, a, s_next, cfg) -> float

    raises:
        ValueError if name not in the mapping
    """
    reward_map = {
        "upright": r_upright,
        "swingdown": r_swingdown,
    }
    if name not in reward_map:
        raise ValueError(f"unknown reward name: {name}. choose from {list(reward_map.keys())}")
    return reward_map[name]


def _build_env_and_q_cfg(config: Dict[str, Any]) -> Tuple[PendulumCfg, QGridCfg]:
    """
    extract dataclass fields from config dict and construct PendulumCfg and QGridCfg.

    validates ranges: g > 0, ell > 0, m > 0, dt > 0, action_clip > 0, theta_dot_clip > 0,
                      sigma_pi > 0, T > 0, N_theta > 0, N_theta_dot > 0, N_action > 0,
                      gamma in (0, 1], fqi_max_iter > 0, fqi_tol > 0.

    args:
        config: dict with keys
          config["pendulum"]: {g, ell, m, dt, action_clip, theta_dot_clip,
                               mu0: {theta_bounds, theta_dot_bounds}}
          config["q_grid"]: {N_theta, N_theta_dot, N_action, gamma, fqi_max_iter, fqi_tol}

    returns:
        tuple (env_cfg: PendulumCfg, q_cfg: QGridCfg)

    raises:
        ValueError if any required key missing or validation fails
        KeyError if any top-level section missing
    """
    pend = config["pendulum"]
    q_gr = config["q_grid"]

    # mu0_box: read nested theta + theta_dot bounds and coerce into nested tuples
    # so the frozen dataclass is hashable and hash_pendulum_cfg's json is stable.
    mu0_dict = pend["mu0"]
    mu0_box = (
        tuple(float(x) for x in mu0_dict["theta_bounds"]),
        tuple(float(x) for x in mu0_dict["theta_dot_bounds"]),
    )

    env_cfg = PendulumCfg(
        g=float(pend["g"]),
        ell=float(pend["ell"]),
        m=float(pend["m"]),
        dt=float(pend["dt"]),
        action_clip=float(pend["action_clip"]),
        theta_dot_clip=float(pend["theta_dot_clip"]),
        mu0_box=mu0_box,
    )

    q_cfg = QGridCfg(
        N_theta=int(q_gr["N_theta"]),
        N_theta_dot=int(q_gr["N_theta_dot"]),
        N_action=int(q_gr["N_action"]),
        gamma=float(q_gr["gamma"]),
        fqi_max_iter=int(q_gr["fqi_max_iter"]),
        fqi_tol=float(q_gr["fqi_tol"]),
    )

    # validation
    assert env_cfg.g > 0, f"g must be positive, got {env_cfg.g}"
    assert env_cfg.ell > 0, f"ell must be positive, got {env_cfg.ell}"
    assert env_cfg.m > 0, f"m must be positive, got {env_cfg.m}"
    assert env_cfg.dt > 0, f"dt must be positive, got {env_cfg.dt}"
    assert env_cfg.action_clip > 0, f"action_clip must be positive, got {env_cfg.action_clip}"
    assert env_cfg.theta_dot_clip > 0, f"theta_dot_clip must be positive, got {env_cfg.theta_dot_clip}"
    assert 0 < q_cfg.gamma <= 1, f"gamma must be in (0, 1], got {q_cfg.gamma}"
    assert q_cfg.fqi_max_iter > 0, f"fqi_max_iter must be positive, got {q_cfg.fqi_max_iter}"
    assert q_cfg.fqi_tol > 0, f"fqi_tol must be positive, got {q_cfg.fqi_tol}"

    return env_cfg, q_cfg


def per_cell(
    config: Dict[str, Any],
    k1_idx: int,
    k2_idx: int,
    seed: int,
    force: bool = False,
) -> bool:
    """
    generate trajectory data for a single (K₁, K₂, seed) cell.

    workflow:
      1. set seed to config["seed"] + seed
      2. build env_cfg and q_cfg from config
      3. resolve r_E and r_anti reward functions from config names
      4. extract trajectory length T, num_samples N, sigma_pi from config
      5. build grid_cfg dict from config traj_kl_grid section
      6. load or build cached trajectory-KL grid via load_or_build_traj_grid
      7. look up (K₁, K₂) targets from config["kl_targets"]
      8. prescribe (α*, β*) via prescribe_traj; check feasibility
      9. if infeasible: log reason, return False
      10. snap α* to nearest grid index i_snap; retrieve q_O = grid["q_O_grid"][i_snap]
      11. construct π_E, π_O, π^β* policies
      12. roll out N trajectories under each of three policies (π^β*, π_O, π_E)
      13. compute cross-densities: log_p_{pstar,p0,p1}[N, 3] (columns: π^β*, π_O, π_E)
      14. compute inverse-direction KLs and integrated ELDR via MC
      15. compute true_ldrs[N] = log p0(pstar) - log p1(pstar) at pstar samples
          (column 1 minus column 2 of log_p_pstar; p0 = π_O, p1 = π_E)
      16. write HDF5 atomically to {data_dir}/k1_{k1_idx}_k2_{k2_idx}_seed_{seed}.h5

    HDF5 schema written (per-cell):
      datasets:
        samples_pstar : float32, [N, (T+1)*3]
        samples_p0    : float32, [N, (T+1)*3]
        samples_p1    : float32, [N, (T+1)*3]
        log_p_pstar   : float32, [N, 3], columns = (π^β*, π_O, π_E)
        log_p_p0      : float32, [N, 3], columns = (π^β*, π_O, π_E)
        log_p_p1      : float32, [N, 3], columns = (π^β*, π_O, π_E)
        true_ldrs     : float32, [N], = log p0(pstar) - log p1(pstar)
                        (consumed by HPO/eval as the ground-truth per-sample LDR)
      attrs: alpha_star, beta_star, K1_*, K2_*, KL_*, integrated_eldr,
             mc_se, T, N, sigma_pi, i_snap, seed, q_E_residual, q_O_residual

    args:
      config: loaded yaml config dict
      k1_idx: index into config["kl_targets"]["k1_values"]
      k2_idx: index into config["kl_targets"]["k2_values"]
      seed: seed offset; actual numpy/torch seed = config["seed"] + seed
      force: if False, skip if output HDF5 exists

    returns:
      True if data written successfully; False if cell skipped (exists or infeasible).
    """

    actual_seed = config["seed"] + seed
    _set_seed(actual_seed)

    # guard empty kl_targets early -- single-cell CLI path does not protect against this.
    k1_values = config["kl_targets"]["k1_values"]
    k2_values = config["kl_targets"]["k2_values"]
    if len(k1_values) == 0 or len(k2_values) == 0:
        print(
            "kl_targets is empty; cannot prescribe a target. "
            "run with --smoke to print the feasible (K1, K2) region, "
            "then populate kl_targets in config.yaml and rerun."
        )
        return False

    env_cfg, q_cfg = _build_env_and_q_cfg(config)
    r_E = _resolve_reward(config["pendulum"]["r_E_name"])
    r_anti = _resolve_reward(config["pendulum"]["r_anti_name"])
    sigma_pi = float(config["pendulum"]["sigma_pi"])
    T = int(config["trajectory"]["T"])
    N = int(config["num_samples"])

    alphas = make_alphas(config["traj_kl_grid"])
    betas  = np.linspace(0, 1, config["traj_kl_grid"]["G_beta"])
    M      = int(config["traj_kl_grid"]["M"])

    grid_cfg = {
        "env_cfg": env_cfg,
        "q_cfg": q_cfg,
        "T": T,
        "sigma_pi": sigma_pi,
        "M": M,
        "r_E_name": config["pendulum"]["r_E_name"],
        "r_anti_name": config["pendulum"]["r_anti_name"],
        "alphas": alphas.tolist(),
        "betas": betas.tolist(),
        "diagnostic_grid": bool(config["traj_kl_grid"].get("diagnostic_grid", True)),
        "kl_se_warn_threshold": float(config["traj_kl_grid"].get("kl_se_warn_threshold", 0.1)),
    }

    grid = load_or_build_traj_grid(
        cfg=grid_cfg,
        F=F,
        sample_mu0=sample_mu0,
        log_mu0=log_mu0,
        r_E=r_E,
        r_anti=r_anti,
        cache_dir=config["traj_kl_grid"]["cache_dir"],
    )

    K1 = float(config["kl_targets"]["k1_values"][k1_idx])
    K2 = float(config["kl_targets"]["k2_values"][k2_idx])
    res = prescribe_traj(grid, K1, K2)

    if not res["feasible"]:
        print(f"skip (k1={K1}, k2={K2}): {res['reason']}")
        return False

    alpha_star = float(res["alpha_star"])
    beta_star = float(res["beta_star"])
    i_snap = int(res["i_snap"])

    # build q_O at the inverted alpha_star (cached on disk by alpha key);
    # avoids the grid-snap error that dominates k1_err in steep regions.
    def _make_r_O(a: float) -> Callable:
        def r_O(s, a_, s_next, cfg):
            return (1.0 - a) * r_E(s, a_, s_next, cfg) + a * r_anti(s, a_, s_next, cfg)
        return r_O

    q_O_result = load_or_build_q(
        env_cfg, _make_r_O(alpha_star), "r_O", q_cfg,
        config["traj_kl_grid"]["cache_dir"],
        F=F, alpha=alpha_star,
        r_E_name=config["pendulum"]["r_E_name"],
        r_anti_name=config["pendulum"]["r_anti_name"],
    )
    q_O = q_O_result["Q"]
    q_E = grid["q_E"]

    pi_E = GaussPolicy(q_E, sigma_pi, env_cfg, q_cfg)
    pi_O = GaussPolicy(q_O, sigma_pi, env_cfg, q_cfg)
    pi_mix = MixPolicy(pi_O, pi_E, beta_star)

    gen_roll = np.random.default_rng(actual_seed + 1)

    # [N, T+1, 2], [N, T+1, 1] respectively
    states_pstar, actions_pstar = rollout(pi_mix.sample, F, sample_mu0, T, N, env_cfg, gen_roll)
    states_p0,    actions_p0    = rollout(pi_O.sample,  F, sample_mu0, T, N, env_cfg, gen_roll)
    states_p1,    actions_p1    = rollout(pi_E.sample,  F, sample_mu0, T, N, env_cfg, gen_roll)

    def crossdens(states, actions):
        """compute cross-densities under three policies [π^β*, π_O, π_E]."""
        # [N]
        log_pmix = log_density(states, actions, pi_mix.log_prob, log_mu0, env_cfg)
        log_pO   = log_density(states, actions, pi_O.log_prob,   log_mu0, env_cfg)
        log_pE   = log_density(states, actions, pi_E.log_prob,   log_mu0, env_cfg)
        # [N, 3]
        return np.stack([log_pmix, log_pO, log_pE], axis=-1)

    # [N, 3]: columns are [π^β*, π_O, π_E]
    log_p_pstar = crossdens(states_pstar, actions_pstar)
    log_p_p0    = crossdens(states_p0,    actions_p0)
    log_p_p1    = crossdens(states_p1,    actions_p1)

    # inverse-direction KLs (realized at the prescribed point)
    KL_O_E   = (log_p_p0[:, 1]    - log_p_p0[:, 2]).mean()
    KL_E_mix = (log_p_p1[:, 2]    - log_p_p1[:, 0]).mean()
    KL_mix_E = (log_p_pstar[:, 0] - log_p_pstar[:, 2]).mean()
    KL_mix_O = (log_p_pstar[:, 0] - log_p_pstar[:, 1]).mean()
    integrated_eldr = KL_mix_E - KL_mix_O
    mc_se = float((log_p_pstar[:, 0] - log_p_pstar[:, 2]).std(ddof=1) / np.sqrt(N))

    # per-sample log density ratio of p0 over p1 at pstar samples.
    # columns of log_p_pstar are (π^β*, π_O, π_E); p0 = π_O (col 1), p1 = π_E (col 2).
    true_ldrs = log_p_pstar[:, 1] - log_p_pstar[:, 2]

    output_path = os.path.join(
        config["data_dir"],
        f"k1_{k1_idx}_k2_{k2_idx}_seed_{seed}.h5"
    )

    if _hdf5_exists(output_path) and not force:
        return False

    # pack returns [N, T+1, 3] float32 (no flatten). store flat [N, (T+1)*3] in HDF5
    # to match the [N, D] shape that downstream src/methods/* consumers expect.
    # step2 will reshape back to [N, T+1, 3] only if it needs the structured form.
    samples_pstar = pack(states_pstar, actions_pstar).reshape(N, -1)
    samples_p0    = pack(states_p0,    actions_p0).reshape(N, -1)
    samples_p1    = pack(states_p1,    actions_p1).reshape(N, -1)

    datasets = {
        "samples_pstar": samples_pstar,
        "samples_p0": samples_p0,
        "samples_p1": samples_p1,
        "log_p_pstar": log_p_pstar.astype(np.float32),
        "log_p_p0": log_p_p0.astype(np.float32),
        "log_p_p1": log_p_p1.astype(np.float32),
        "true_ldrs": true_ldrs.astype(np.float32),
    }

    attrs = {
        "alpha_star": alpha_star,
        "beta_star": beta_star,
        "K1_prescribed": K1,
        "K2_prescribed": K2,
        "K1_realized": float(res["realized_K1"]),
        "K2_realized": float(res["realized_K2"]),
        "KL_O_E": KL_O_E,
        "KL_E_mix": KL_E_mix,
        "KL_mix_E": KL_mix_E,
        "KL_mix_O": KL_mix_O,
        "integrated_eldr": integrated_eldr,
        "mc_se": mc_se,
        "T": T,
        "N": N,
        "sigma_pi": sigma_pi,
        "i_snap": i_snap,
        "seed": seed,
        "q_E_residual": float(grid.get("q_E_residual", np.nan)),
        "q_O_residual": float(q_O_result["bellman_residual"]),
    }

    _write_hdf5_atomic(output_path, datasets, attrs)
    print(f"saved {output_path}")
    return True


def main():
    """
    CLI dispatcher: parse arguments and dispatch to per_cell or sweep.

    flags:
      --smoke: run single smoke cell (first available k1_idx, k2_idx, seed=0); force=True
      --k1-idx K1_IDX: if set with k2-idx and seed, run single cell
      --k2-idx K2_IDX: if set with k1-idx and seed, run single cell
      --seed SEED: if set with k1-idx and k2-idx, run single cell
      --force: force recomputation (ignore existing HDF5 files)

    behaviors:
      1. load config from ex/semisynth/pendulum/config.yaml
      2. --smoke: pick first available (k1_idx=0, k2_idx=0, seed=0) and run per_cell.
         - if config["kl_targets"]["k1_values"] is empty: print message explaining that
           user should populate kl_targets after seeing feasible region, then call
           load_or_build_traj_grid directly to trigger a single grid build and print
           feasible region, then return (do not crash).
      3. single-cell (--k1-idx / --k2-idx / --seed all set): run per_cell once.
      4. default (sweep): iterate over (k1_idx, k2_idx) in product of ranges.
         - for each (K1, K2) pair, decide n_seeds via hard_corner_threshold policy:
           if K1 >= threshold and K2 >= threshold: n_seeds = seeds_hard
           else: n_seeds = seeds_default
         - call per_cell(config, k1_idx, k2_idx, seed, force) for seed in range(n_seeds)
         - track and print summary: processed, skipped, total_cells
    """

    parser = argparse.ArgumentParser(
        description=(
            "generate trajectory-ELDR data for pendulum. "
            "first run: --smoke with empty kl_targets builds the trajectory-KL grid "
            "(~10-30 min on one CPU core) and prints the feasible (K1, K2) region; "
            "no data cells are written. populate kl_targets in config.yaml, then rerun."
        )
    )
    parser.add_argument("--k1-idx", type=int, default=None, help="K1 grid index")
    parser.add_argument("--k2-idx", type=int, default=None, help="K2 grid index")
    parser.add_argument("--seed", type=int, default=None, help="seed offset")
    parser.add_argument("--force", action="store_true", help="force recomputation")
    parser.add_argument("--smoke", action="store_true", help="smoke test: 1 cell")
    args = parser.parse_args()

    config_path = "ex/semisynth/pendulum/config.yaml"
    config = _load_config(config_path)

    if args.smoke:
        k1_values = config["kl_targets"].get("k1_values", [])
        k2_values = config["kl_targets"].get("k2_values", [])

        if len(k1_values) == 0 or len(k2_values) == 0:
            print("kl_targets is empty. populating it after initial grid build...")
            print("building trajectory-KL grid to determine feasible region...")

            env_cfg, q_cfg = _build_env_and_q_cfg(config)
            r_E = _resolve_reward(config["pendulum"]["r_E_name"])
            r_anti = _resolve_reward(config["pendulum"]["r_anti_name"])
            sigma_pi = float(config["pendulum"]["sigma_pi"])
            T = int(config["trajectory"]["T"])

            alphas = make_alphas(config["traj_kl_grid"])
            betas  = np.linspace(0, 1, config["traj_kl_grid"]["G_beta"])
            M      = int(config["traj_kl_grid"]["M"])

            grid_cfg = {
                "env_cfg": env_cfg,
                "q_cfg": q_cfg,
                "T": T,
                "sigma_pi": sigma_pi,
                "M": M,
                "r_E_name": config["pendulum"]["r_E_name"],
                "r_anti_name": config["pendulum"]["r_anti_name"],
                "alphas": alphas.tolist(),
                "betas": betas.tolist(),
                "diagnostic_grid": bool(config["traj_kl_grid"].get("diagnostic_grid", True)),
                "kl_se_warn_threshold": float(config["traj_kl_grid"].get("kl_se_warn_threshold", 0.1)),
            }

            grid = load_or_build_traj_grid(
                cfg=grid_cfg,
                F=F,
                sample_mu0=sample_mu0,
                log_mu0=log_mu0,
                r_E=r_E,
                r_anti=r_anti,
                cache_dir=config["traj_kl_grid"]["cache_dir"],
            )

            print("grid build complete. feasible region visible in grid. update kl_targets in config.yaml.")
            return

        print(f"smoke: (k1_idx=0, k2_idx=0, seed=0)")
        per_cell(config, 0, 0, 0, force=True)
        print("smoke test complete")

    elif args.k1_idx is not None and args.k2_idx is not None and args.seed is not None:
        per_cell(config, args.k1_idx, args.k2_idx, args.seed, force=args.force)

    else:
        k1_values = config["kl_targets"]["k1_values"]
        k2_values = config["kl_targets"]["k2_values"]
        hard_threshold = config["kl_targets"]["hard_corner_threshold"]
        seeds_default = config["kl_targets"]["seeds_default"]
        seeds_hard = config["kl_targets"]["seeds_hard"]

        total_cells = len(k1_values) * len(k2_values)
        processed = 0
        skipped = 0

        for k1_idx, k2_idx in product(range(len(k1_values)), range(len(k2_values))):
            K1 = k1_values[k1_idx]
            K2 = k2_values[k2_idx]

            is_hard = (K1 >= hard_threshold and K2 >= hard_threshold)
            n_seeds = seeds_hard if is_hard else seeds_default

            for seed in range(n_seeds):
                result = per_cell(config, k1_idx, k2_idx, seed, force=args.force)
                if result:
                    processed += 1
                else:
                    skipped += 1

        print(f"\ncompletion summary:")
        print(f"  processed: {processed}")
        print(f"  skipped: {skipped}")
        print(f"  total cells: {total_cells}")


if __name__ == "__main__":
    main()
