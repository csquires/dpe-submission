import os
import argparse
import yaml
import numpy as np
import h5py
from itertools import product
from typing import Dict, Tuple, Any

from src.utils.io import _load_config, _set_seed
from src.utils.pendulum import PendulumCfg, F, sample_mu0, log_mu0
from src.utils.pendulum_policies import FlowPolicy, MixPolicy
from src.models.flow.train_flow_policy import load_flow
from src.sampling.pendulum_traj import rollout, log_density, pack


def _build_env_and_q_cfg(config: Dict[str, Any]) -> Tuple[PendulumCfg, None]:
    """extract dataclass fields from config dict and construct PendulumCfg.

    args:
        config: dict with keys
          config["pendulum"]: {g, ell, m, dt, action_clip, theta_dot_clip,
                               mu0: {theta_bounds, theta_dot_bounds}}
          config["q_grid"]: (unused; included for backward compat)

    returns:
        tuple (env_cfg: PendulumCfg, None)

    raises:
        ValueError if any required key missing or validation fails
        KeyError if any top-level section missing
    """
    pend = config["pendulum"]

    # mu0_box: read nested theta + theta_dot bounds
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

    # validation
    assert env_cfg.g > 0
    assert env_cfg.ell > 0
    assert env_cfg.m > 0
    assert env_cfg.dt > 0
    assert env_cfg.action_clip > 0
    assert env_cfg.theta_dot_clip > 0

    return env_cfg, None


def per_cell(
    config: Dict[str, Any],
    k1_idx: int,
    beta_idx: int,
    seed: int,
    n_test: int,
    env_cfg: PendulumCfg,
    alphas_chosen: Dict[str, str],
    rl_ckpts_base: str,
) -> bool:
    """append test set to a single cell h5 file.

    workflow:
      1. construct cell path and load existing h5
      2. read cell attrs: beta, flow_ckpt_hash_E, flow_ckpt_hash_O, T
      3. guard: skip if sigma_pi attr present (gaussian-era cell)
      4. resolve flow ckpt paths via alphas_chosen hash dict with validation
      5. load flows and build FlowPolicy objects
      6. set seed and build mix policy
      7. generate N_test fresh rollouts via default_rng(actual_seed + 1000)
      8. compute cross-densities at flow samples
      9. extract true_ldrs = log_pO - log_pE
      10. append to h5 (additive, delete if exists)
      11. sanity print: mean(test_ldr) vs mean(train_ldr) vs integrated_eldr
      12. return success flag

    args:
        config: loaded yaml config dict
        k1_idx: index into k1_values
        beta_idx: index into beta_values
        seed: seed offset; actual seed = config["seed"] + seed
        n_test: number of test samples
        env_cfg: environment config
        alphas_chosen: hash -> ckpt_path mapping from alphas_chosen.yaml
        rl_ckpts_base: base directory for RL checkpoints (unused; future flexibility)

    returns:
        True if appended successfully, False if cell does not exist or is gaussian-era
    """
    actual_seed = config["seed"] + seed
    _set_seed(actual_seed)

    # construct and load cell h5
    data_dir = config["data_dir"]
    cell_path = os.path.join(
        data_dir,
        f"k1_{k1_idx}_beta_{beta_idx}_seed_{seed}.h5"
    )

    if not os.path.exists(cell_path):
        return False

    # read cell attrs and train data
    with h5py.File(cell_path, 'r') as f:
        # gaussian-era guard: refuse to mix generations
        if 'sigma_pi' in f.attrs:
            print(f"SKIP k1={k1_idx} beta={beta_idx} seed={seed}: cell has sigma_pi attr (gaussian-era); refusing to mix")
            return False

        beta = float(f.attrs['beta'])
        flow_ckpt_hash_E = str(f.attrs['flow_ckpt_hash_E'])
        flow_ckpt_hash_O = str(f.attrs['flow_ckpt_hash_O'])
        T = int(f.attrs['T'])
        integrated_eldr = float(f.attrs['integrated_eldr'])
        true_ldrs_train = f['true_ldrs'][:]

    # resolve flow ckpt paths from hash -> path mapping (alphas_chosen.yaml)
    if flow_ckpt_hash_E not in alphas_chosen:
        raise ValueError(f"flow_ckpt_hash_E={flow_ckpt_hash_E} not found in alphas_chosen; "
                         f"ckpt for cell (k1={k1_idx}, beta={beta_idx}, seed={seed}) must exist; hashes must match exactly")
    if flow_ckpt_hash_O not in alphas_chosen:
        raise ValueError(f"flow_ckpt_hash_O={flow_ckpt_hash_O} not found in alphas_chosen; "
                         f"ckpt for cell (k1={k1_idx}, beta={beta_idx}, seed={seed}) must exist; hashes must match exactly")

    # load flows and build flow policies
    flow_E = load_flow(alphas_chosen[flow_ckpt_hash_E], device="cpu")
    flow_O = load_flow(alphas_chosen[flow_ckpt_hash_O], device="cpu")
    pi_E = FlowPolicy(flow_E, env_cfg)
    pi_O = FlowPolicy(flow_O, env_cfg)
    pi_mix = MixPolicy(pi_O, pi_E, beta)

    # generate fresh test rollouts + true ldr in chunks (5000 per chunk).
    # flows batch log_prob computation; chunking keeps memory bounded by model capacity.
    # only log(pi_O) - log(pi_E) needed (test points not stored downstream).
    gen_test = np.random.default_rng(actual_seed + 1000)
    chunk = 5000
    ldr_parts = []
    remaining = n_test
    while remaining > 0:
        m = min(chunk, remaining)
        s, a = rollout(pi_mix.sample, F, sample_mu0, T, m, env_cfg, gen_test)
        log_pO = log_density(s, a, pi_O.log_prob, log_mu0, env_cfg)   # [m]
        log_pE = log_density(s, a, pi_E.log_prob, log_mu0, env_cfg)   # [m]
        ldr_parts.append((log_pO - log_pE).astype(np.float32))
        remaining -= m
    samples_test_true_ldrs = np.concatenate(ldr_parts)  # [n_test] f32

    # append to h5 (additive, overwrite if exists)
    with h5py.File(cell_path, 'r+') as f:
        if 'samples_test' in f:
            del f['samples_test']
        if 'samples_test_true_ldrs' in f:
            del f['samples_test_true_ldrs']
        # only per-cell true ldrs are used downstream; drop the (unused) test points.
        f.create_dataset('samples_test_true_ldrs', data=samples_test_true_ldrs)

    # sanity check
    mean_test_ldr = np.mean(samples_test_true_ldrs)
    mean_train_ldr = np.mean(true_ldrs_train)
    print(f"k1={k1_idx} beta={beta_idx} seed={seed}: mean_test_ldr={mean_test_ldr:.4f}, mean_train_ldr={mean_train_ldr:.4f}, integrated_eldr={integrated_eldr:.4f}")

    return True


def main():
    """CLI dispatcher: parse arguments and dispatch to per_cell loop.

    arguments:
      --config: path to config yaml (default: ex/semisynth/pendulum/config.yaml)
      --n-test: number of test samples per cell (default: 100000)
      --seed: global seed offset (default: from config)
      --cell-range: optional SLURM array range, format "start:stop"
    """
    parser = argparse.ArgumentParser(
        description="append test set to pendulum cells"
    )
    parser.add_argument(
        "--config",
        default="ex/semisynth/pendulum/config.yaml",
        help="path to config yaml"
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=100000,
        help="number of test samples per cell"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="global seed offset"
    )
    parser.add_argument(
        "--cell-range",
        type=str,
        default=None,
        help="SLURM array range format start:stop"
    )
    args = parser.parse_args()

    # load config and flow ckpt mapping
    config = _load_config(args.config)
    config["data_dir"] = os.path.expandvars(config["data_dir"])

    # canonical doc -> hash-to-path mapping (provenance guard uses hashes)
    from ex.utils.realized_kl_table import load_alphas_chosen
    doc = load_alphas_chosen(config["data_dir"])
    alphas_chosen = {doc["flow_hash_E"]: doc["ckpt_E"]}
    for stratum in doc["strata"]:
        alphas_chosen[stratum["flow_hash_O"]] = stratum["ckpt_O"]

    if args.seed is not None:
        config["seed"] = args.seed

    env_cfg, _ = _build_env_and_q_cfg(config)
    rl_ckpts_base = os.path.expandvars(config.get("rl_ckpts_dir", "ex/semisynth/pendulum/rl_ckpts"))

    # extract cell grid: strata from canonical doc, single beta, campaign seeds
    n_strata = len(doc["strata"])
    seeds_default = config["campaign"]["seeds_default"]

    # parse cell range
    # flatten to (k1, beta, seed) triples so --cell-range gives fine array chunks.
    all_cells = list(product(range(n_strata), range(1), range(seeds_default)))
    if args.cell_range:
        start, stop = map(int, args.cell_range.split(':'))
        all_cells = all_cells[start:stop]

    # loop over cells
    processed = 0
    skipped = 0

    for k1_idx, beta_idx, seed in all_cells:
        result = per_cell(config, k1_idx, beta_idx, seed, args.n_test,
                          env_cfg, alphas_chosen, rl_ckpts_base)
        if result:
            processed += 1
        else:
            skipped += 1

    print(f"\nappend test set summary:")
    print(f"  processed: {processed}")
    print(f"  skipped: {skipped}")
    print(f"  total cells: {len(all_cells)}")


if __name__ == "__main__":
    main()
