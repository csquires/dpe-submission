import os
import argparse
import yaml
import numpy as np
from pathlib import Path
from itertools import product
from typing import Callable, Dict, Tuple, Any, List

from src.utils.io import _load_config, _set_seed, _hdf5_exists, _write_hdf5_atomic
from src.utils.pendulum import PendulumCfg, F, sample_mu0, log_mu0
from src.utils.pendulum_policies import FlowPolicy, MixPolicy
from src.sampling.pendulum_traj import rollout, log_density, pack
from src.models.flow.train_flow_policy import load_flow
from ex.utils.flow_gate import assert_gate


def _read_hdf5_attrs(path):
    """read the attrs dict of an h5 file (guard helper; io.py has no reader)."""
    import h5py
    with h5py.File(path, "r") as f:
        return dict(f.attrs)


def load_chosen(config):
    """load the canonical alphas_chosen doc (see realized_kl_table)."""
    from ex.utils.realized_kl_table import load_alphas_chosen
    return load_alphas_chosen(config["data_dir"])

def per_cell(
    config: Dict[str, Any],
    k1_idx: int,
    beta_idx: int,
    seed: int,
    force: bool = False,
) -> bool:
    """
    generate trajectory data for a single (alpha, seed) cell via flow policies.

    workflow:
      1. set seed to config["seed"] + seed
      2. read alphas_chosen.yaml via load_chosen(config)
      3. lookup stratum at k1_idx; extract {alpha, ckpt_E, ckpt_O, flow_hash_E/O, rl_hash_O}
      4. assert_gate on both flow checkpoints before compute
      5. load FlowPolicy from ckpt_E and ckpt_O; construct MixPolicy(pi_O, pi_E, beta=0.5)
      6. extract trajectory length T, num_samples N from config
      7. roll out N trajectories under each of three policies (pi^{beta*}, pi_O, pi_E)
      8. compute cross-densities: log_p_{pstar,p0,p1}[N, 3] (columns: pi^{beta*}, pi_O, pi_E)
      9. compute inverse-direction KLs and integrated ELDR via MC
      10. compute true_ldrs[N] = log p0(pstar) - log p1(pstar) at pstar samples
      11. write HDF5 atomically to {data_dir}/k1_{k1_idx}_beta_{beta_idx}_seed_{seed}.h5

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
      attrs: alpha_chosen, beta, k1_stratum_label, K1_realized_flow, K1_realized_flow_se,
             K2_realized, KL_*, integrated_eldr, mc_se, T, N, seed,
             flow_ckpt_hash_E, flow_ckpt_hash_O, rl_ckpt_hash_O

    args:
      config: loaded yaml config dict
      k1_idx: index into load_chosen(config) list
      beta_idx: unused (beta fixed to 0.5); kept for CLI compatibility
      seed: seed offset; actual numpy/torch seed = config["seed"] + seed
      force: if False, skip if output HDF5 exists

    returns:
      True if data written successfully; False if cell skipped (exists).
    """

    actual_seed = config["seed"] + seed
    _set_seed(actual_seed)

    doc = load_chosen(config)
    strata = doc["strata"]
    if k1_idx >= len(strata):
        raise IndexError(f"k1_idx {k1_idx} >= len(strata) {len(strata)}")

    stratum = strata[k1_idx]
    ckpt_E = doc["ckpt_E"]
    ckpt_O = stratum["ckpt_O"]

    # gate guard: assert gate reports exist and pass before any compute
    assert_gate(config["gate"]["out_dir"], ckpt_E)
    assert_gate(config["gate"]["out_dir"], ckpt_O)

    # construct flow-based policies
    pi_E = FlowPolicy(load_flow(ckpt_E, device="cpu"))
    pi_O = FlowPolicy(load_flow(ckpt_O, device="cpu"))
    pi_mix = MixPolicy(pi_O, pi_E, beta=0.5)

    T = int(config["trajectory"]["T"])
    N = int(config["num_samples"])

    # build env_cfg from config (needed for rollout; env_cfg not used elsewhere)
    pend_cfg = config["pendulum"]
    mu0_dict = pend_cfg["mu0"]
    mu0_box = (
        tuple(float(x) for x in mu0_dict["theta_bounds"]),
        tuple(float(x) for x in mu0_dict["theta_dot_bounds"]),
    )
    env_cfg = PendulumCfg(
        g=float(pend_cfg["g"]),
        ell=float(pend_cfg["ell"]),
        m=float(pend_cfg["m"]),
        dt=float(pend_cfg["dt"]),
        action_clip=float(pend_cfg["action_clip"]),
        theta_dot_clip=float(pend_cfg["theta_dot_clip"]),
        mu0_box=mu0_box,
    )

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
        f"k1_{k1_idx}_beta_{beta_idx}_seed_{seed}.h5"
    )

    # guard: block gaussian-era data reuse (check existing h5 for sigma_pi attr)
    if _hdf5_exists(output_path):
        existing_attrs = _read_hdf5_attrs(output_path)
        if "sigma_pi" in existing_attrs:
            raise RuntimeError(
                f"target {output_path} contains Gaussian-era mixing data (sigma_pi attr). "
                "config data_dir must point to a new versioned path."
            )
        if not force:
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
        "alpha_chosen": float(stratum["alpha"]),
        "beta": 0.5,
        "k1_stratum_label": int(stratum["stratum_label"]),
        "K1_realized_flow": float(stratum["K1_realized_flow"]),
        "K1_realized_flow_se": float(stratum["K1_se"]),
        "K2_realized": float(KL_mix_E),
        "KL_O_E": KL_O_E,
        "KL_E_mix": KL_E_mix,
        "KL_mix_E": KL_mix_E,
        "KL_mix_O": KL_mix_O,
        "integrated_eldr": integrated_eldr,
        "mc_se": mc_se,
        "T": T,
        "N": N,
        "seed": seed,
        "flow_ckpt_hash_E": doc["flow_hash_E"],
        "flow_ckpt_hash_O": stratum["flow_hash_O"],
        "rl_ckpt_hash_O": stratum["rl_hash_O"],
    }

    _write_hdf5_atomic(output_path, datasets, attrs)
    print(f"saved {output_path}")
    return True


def main():
    """
    CLI dispatcher: parse arguments and dispatch to per_cell or sweep.

    flags:
      --smoke: run single smoke cell (k1_idx=0, beta_idx=0, seed=0); force=True
      --k1-idx K1_IDX: if set with beta-idx and seed, run single cell
      --beta-idx BETA_IDX: if set with k1-idx and seed, run single cell
      --seed SEED: if set with k1-idx and beta-idx, run single cell
      --force: force recomputation (ignore existing HDF5 files)

    behaviors:
      1. load config from ex/semisynth/pendulum/config.yaml
      2. --smoke: load alphas_chosen.yaml; run per_cell(k1_idx=0, beta_idx=0, seed=0, force=True)
      3. single-cell (--k1-idx / --beta-idx / --seed all set): run per_cell once.
      4. default (sweep): iterate k1_idx over len(chosen), beta_idx=0 only (beta fixed 0.5),
         call per_cell(config, k1_idx, 0, seed, force) for each seed,
         track and print summary: processed, skipped, total_cells.
    """

    parser = argparse.ArgumentParser(
        description=(
            "generate trajectory-ELDR data for pendulum via flow policy checkpoints. "
            "requires alphas_chosen.yaml from step0d_stamp_strata.py. "
            "run --smoke to validate a single cell."
        )
    )
    parser.add_argument("--k1-idx", type=int, default=None, help="K1 grid index")
    parser.add_argument("--beta-idx", type=int, default=None, help="beta grid index")
    parser.add_argument("--seed", type=int, default=None, help="seed offset")
    parser.add_argument("--force", action="store_true", help="force recomputation")
    parser.add_argument("--smoke", action="store_true", help="smoke test: 1 cell")
    args = parser.parse_args()

    config_path = "ex/semisynth/pendulum/config.yaml"
    config = _load_config(config_path)

    if args.smoke:
        chosen = load_chosen(config)
        if len(chosen["strata"]) == 0:
            raise ValueError("alphas_chosen.yaml empty or missing. Run step0d_stamp_strata.py first.")
        print(f"smoke: (k1_idx=0, beta_idx=0, seed=0)")
        per_cell(config, 0, 0, 0, force=True)
        print("smoke test complete")

    elif args.k1_idx is not None and args.beta_idx is not None and args.seed is not None:
        per_cell(config, args.k1_idx, args.beta_idx, args.seed, force=args.force)

    else:
        chosen = load_chosen(config)
        seeds_default = config["campaign"]["seeds_default"]

        total_cells = len(chosen["strata"]) * 1
        processed = 0
        skipped = 0

        for k1_idx, beta_idx in product(range(len(chosen["strata"])), range(1)):
            for seed in range(seeds_default):
                result = per_cell(config, k1_idx, beta_idx, seed, force=args.force)
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
