import os
import argparse
import h5py
import yaml
import numpy as np
import torch
from pathlib import Path
from itertools import product
from typing import Optional, Dict, Tuple, Any

from src.utils.gridworld import build_gridworld, reward_to_goal, value_iteration, softmax_policy
from src.utils.occupancy import bellman_occupancy, kl_occupancy, mixture_policy
from ex.utils.prescribed_kls import load_or_build_grid, prescribe_k1
from src.sampling.tabular import (
    sample_occupancy, encode_sa,
    pointwise_discrete_ldr, pointwise_smoothed_ldr,
)
from src.sampling.frozen_flow import FrozenFlow


def _load_config(config_path: str) -> Dict[str, Any]:
    """
    load yaml config from config_path.

    returns dict with all keys described in config.yaml schema.
    raises FileNotFoundError if config_path does not exist.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _set_seed(seed: int) -> None:
    """
    set global numpy and torch random seeds for reproducibility.

    args:
        seed: integer seed value; applied to np.random.seed and torch.manual_seed.

    side effects:
        np.random.seed(seed)
        torch.manual_seed(seed)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def _hdf5_exists(output_path: str) -> bool:
    """
    check if HDF5 file exists (and is a valid file, not a temporary).
    returns True if path exists and is readable.
    """
    return os.path.isfile(output_path) and not output_path.endswith('.tmp')


def _write_hdf5_atomic(
    output_path: str,
    datasets: Dict[str, np.ndarray],
    attrs: Dict[str, Any],
) -> None:
    """
    write HDF5 atomically to avoid partial-file recovery issues.

    writes to {output_path}.tmp, then renames to output_path.

    args:
        output_path: final target path (str).
        datasets: dict of {name: array} to write as h5 datasets.
                  all arrays assumed float32 unless otherwise specified in attrs.
        attrs: dict of {name: value} scalar attributes (float, int, str, etc.).

    process:
        1. create parent directory if needed.
        2. tmp_path = output_path + ".tmp".
        3. with h5py.File(tmp_path, 'w') as f:
             for name, arr in datasets.items():
                 f.create_dataset(name, data=arr, dtype=arr.dtype)
             for name, val in attrs.items():
                 f.attrs[name] = val
        4. os.rename(tmp_path, output_path)  # atomic on POSIX.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"

    with h5py.File(tmp_path, 'w') as f:
        for name, arr in datasets.items():
            f.create_dataset(name, data=arr, dtype=arr.dtype)
        for name, val in attrs.items():
            f.attrs[name] = val

    os.rename(tmp_path, output_path)


def per_cell(
    config: Dict[str, Any],
    k1_idx: int,
    beta_idx: int,
    seed: int,
    force: bool = False,
) -> bool:
    """
    generate data for a single (k1_idx, beta_idx, seed) cell.

    workflow:
        1. set seed
        2. build MDP from config["gridworld"]
        3. construct rewards r_E, r_anti
        4. load/build KL grid
        5. prescribe alpha* from K1; take beta directly from config
        6. compute occupancies d_O, d_E, d_pi^beta
        7. sample (s, a) pairs and encode
        8. compute true LDRs (discrete and smoothed)
        9. compute KL divergences and integrated ELDR
        10. write HDF5 to namespaced path

    args:
        config: loaded yaml config dict
        k1_idx: index into config["kl_targets"]["k1_values"]
        beta_idx: index into config["kl_targets"]["beta_values"]
        seed: seed offset; actual numpy/torch seed = config["seed"] + seed
        force: if False, skip if output HDF5 exists

    returns:
        True if data written successfully; False if cell skipped (exists or infeasible).

    raises:
        ValueError, KeyError on malformed config or missing gridworld data.
    """

    # ===== step 1: set seed =====
    actual_seed = config["seed"] + seed
    _set_seed(actual_seed)

    # ===== step 2-4: load or build KL grid =====
    gw_cfg = config["gridworld"]
    mdp_cfg = {
        "L": gw_cfg["L"],
        "p_slip": gw_cfg["p_slip"],
        "gamma": gw_cfg["gamma"],
        "terminals": gw_cfg["terminals"],
        "expert_goal": gw_cfg["expert_goal"],
        "anti_goal": gw_cfg["anti_goal"],
        "tau": gw_cfg["tau"],
        "G_alpha": config["kl_grid"]["G_alpha"],
        "G_beta": config["kl_grid"]["G_beta"],
        "mu0_kind": gw_cfg.get("mu0_kind", "uniform"),
        "mu0_centers": gw_cfg.get("mu0_centers", []),
        "reward_kind": gw_cfg.get("reward_kind", "sparse"),
        "reward_sigma": float(gw_cfg.get("reward_sigma", 1.0)),
    }

    grid = load_or_build_grid(
        mdp_cfg,
        cache_dir=config["kl_grid"]["cache_dir"],
    )

    mdp = grid["mdp"]
    r_E = grid["r_E"]
    r_anti = grid["r_anti"]

    # ===== step 5: prescribe alpha* from K1; beta is fixed by config =====
    K1 = config["kl_targets"]["k1_values"][k1_idx]
    beta_star = float(config["kl_targets"]["beta_values"][beta_idx])

    res = prescribe_k1(grid["KL1"], grid["alphas"], K1)

    if not res["feasible"]:
        print(f"skipping (k1_idx={k1_idx}, beta_idx={beta_idx}): K1={K1} infeasible: {res['reason']}")
        return False

    alpha_star = res["alpha_star"]
    # beta_star is taken directly from config (fixed mixture weight, not inverted
    # from a prescribed K2). res['realized_K1'] equals K1 by tautology of np.interp;
    # the true occupancy KLs are recomputed below from d_O, d_pi, d_E.

    # ===== step 6: compute occupancies at (alpha*, beta*) =====

    # policy at alpha*
    r_O = (1.0 - alpha_star) * r_E + alpha_star * r_anti
    Q_O = value_iteration(mdp.P, r_O, mdp.gamma)
    pi_O = softmax_policy(Q_O, tau=gw_cfg["tau"])

    # occupancy at alpha*
    d_O = bellman_occupancy(mdp.P, mdp.mu0, pi_O, mdp.gamma)

    # mixture policy at (alpha*, beta*)
    pi_mix = mixture_policy(pi_O, grid["pi_E"], beta_star)

    # occupancy of mixture
    d_pi = bellman_occupancy(mdp.P, mdp.mu0, pi_mix, mdp.gamma)

    # ===== step 7: sample and encode =====
    N = config["num_samples"]

    # cpu generator for determinism
    generator = torch.Generator(device='cpu')
    generator.manual_seed(actual_seed)

    # sample (s, a) pairs
    pstar_s, pstar_a = sample_occupancy(d_pi, N, generator=generator)
    p0_s, p0_a = sample_occupancy(d_O, N, generator=generator)
    p1_s, p1_a = sample_occupancy(grid["d_E"], N, generator=generator)

    # build encoding config
    enc_cfg_base = config["encoding"].copy()
    enc_cfg_base.update({
        "n_states": mdp.n_states,
        "n_actions": mdp.n_actions,
        "L": mdp.L,
    })

    # if flow encoding, instantiate the frozen flow
    if enc_cfg_base["type"] == "flow_pushforward":
        flow_cfg = config["encoding"].get("flow", {})
        enc_cfg_base["flow_module"] = FrozenFlow(
            dim=enc_cfg_base.get("embed_dim", 6),
            n_layers=flow_cfg.get("layers", 4),
            seed=flow_cfg.get("seed", 1729),
        )

    # encode samples
    pstar_x = encode_sa(pstar_s, pstar_a, enc_cfg_base)
    p0_x = encode_sa(p0_s, p0_a, enc_cfg_base)
    p1_x = encode_sa(p1_s, p1_a, enc_cfg_base)

    # ===== step 8: compute pointwise LDRs =====

    # discrete LDR at pstar samples
    true_ldrs_discrete = pointwise_discrete_ldr(pstar_s, pstar_a, d_O, grid["d_E"])

    # smoothed LDR (only for blob/flow encodings; for onehot, use discrete)
    if enc_cfg_base["type"] in ("gaussian_blob", "flow_pushforward"):
        true_ldrs_smoothed = pointwise_smoothed_ldr(pstar_x, enc_cfg_base, d_O, grid["d_E"])
    else:
        true_ldrs_smoothed = true_ldrs_discrete.clone()

    # ===== step 9: compute KL divergences and integrated ELDR =====

    # inverse-direction KLs
    inv_kl_O_E = kl_occupancy(d_O, grid["d_E"])
    kl_pi_E = kl_occupancy(d_pi, grid["d_E"])
    kl_pi_O = kl_occupancy(d_pi, d_O)
    integrated_eldr = kl_pi_E - kl_pi_O

    # honest realized KLs: actual occupancy KL at the inverted (alpha*, beta*),
    # not the np.interp-on-grid value (which equals the prescribed target by tautology).
    realized_K1 = float(kl_occupancy(grid["d_E"], d_O))
    realized_K2 = float(kl_pi_E)

    # ===== step 10: prepare HDF5 output =====

    # determine output path
    encoding_type = enc_cfg_base["type"]
    sigma = enc_cfg_base.get("sigma", float("nan"))

    # sigma directory naming: onehot -> "sigma_na", blob/flow -> "sigma_{sigma:.3f}"
    if encoding_type.startswith("onehot"):
        sigma_dir = "sigma_na"
    else:
        sigma_dir = f"sigma_{sigma:.3f}"

    output_dir = os.path.join(
        config["data_dir"],
        encoding_type,
        sigma_dir,
    )
    output_path = os.path.join(
        output_dir,
        f"kl1_{k1_idx}_beta_{beta_idx}_seed_{seed}.h5"
    )

    # check skip condition
    if _hdf5_exists(output_path) and not force:
        print(f"skipping {output_path} (exists)")
        return False

    # convert tensors to numpy (float32)
    datasets = {
        "pstar_samples": pstar_x.cpu().numpy().astype(np.float32),
        "p0_samples": p0_x.cpu().numpy().astype(np.float32),
        "p1_samples": p1_x.cpu().numpy().astype(np.float32),
        "true_ldrs_discrete": true_ldrs_discrete.cpu().numpy().astype(np.float32),
        "true_ldrs_smoothed": true_ldrs_smoothed.cpu().numpy().astype(np.float32),
    }

    # prepare latent as 2D [N, 2] (s, a pairs)
    datasets["pstar_latent"] = np.stack(
        [pstar_s.cpu().numpy().astype(np.int64), pstar_a.cpu().numpy().astype(np.int64)],
        axis=1
    )
    datasets["p0_latent"] = np.stack(
        [p0_s.cpu().numpy().astype(np.int64), p0_a.cpu().numpy().astype(np.int64)],
        axis=1
    )
    datasets["p1_latent"] = np.stack(
        [p1_s.cpu().numpy().astype(np.int64), p1_a.cpu().numpy().astype(np.int64)],
        axis=1
    )

    # prepare attributes
    attrs = {
        "alpha_star": float(alpha_star),
        "beta": float(beta_star),
        "prescribed_K1": float(K1),
        "realized_K1": float(realized_K1),
        "realized_K2": float(realized_K2),
        "inv_kl_O_E": float(inv_kl_O_E),
        "integrated_eldr": float(integrated_eldr),
        "encoding_type": str(encoding_type),
        "sigma": float(sigma),
        "embed_dim": int(enc_cfg_base.get("embed_dim", 0)),
        "mdp_hash": grid.get("mdp_hash", "unknown"),
        "seed": int(seed),
        "N": int(N),
    }

    # ===== step 11: write HDF5 atomically =====
    _write_hdf5_atomic(output_path, datasets, attrs)
    print(f"saved {output_path}")
    return True


def main():
    """
    dispatch entry point: parse CLI arguments, load config, then either:
    - run single cell (--k1-idx I --beta-idx J --seed K)
    - run all cells x all seeds (default)
    - run smoke test (--smoke)
    """

    parser = argparse.ArgumentParser(
        description="generate data for tabular SMODICE ELDR estimation"
    )
    parser.add_argument("--k1-idx", type=int, default=None, help="K1 grid index (SLURM dispatch)")
    parser.add_argument("--beta-idx", type=int, default=None, help="beta grid index (SLURM dispatch)")
    parser.add_argument("--seed", type=int, default=None, help="seed offset (SLURM dispatch)")
    parser.add_argument("--force", action="store_true", help="force recomputation (ignore existing files)")
    parser.add_argument("--smoke", action="store_true", help="smoke test: 1 cell x 1 seed")
    args = parser.parse_args()

    # load config
    config_path = "ex/synth/occupancy/config.yaml"
    config = _load_config(config_path)

    # dispatch: single cell, all cells, or smoke test
    if args.smoke:
        # smoke test: first cell (k1_idx=0, beta_idx=0, seed=0)
        print("smoke test: (k1_idx=0, beta_idx=0, seed=0)")
        per_cell(config, 0, 0, seed=0, force=True)
        print("smoke test complete")

    elif args.k1_idx is not None and args.beta_idx is not None and args.seed is not None:
        # single cell (SLURM dispatch)
        per_cell(config, args.k1_idx, args.beta_idx, args.seed, force=args.force)

    else:
        # all cells x all seeds, sequential
        k1_targets = config["kl_targets"]["k1_values"]
        beta_targets = config["kl_targets"]["beta_values"]
        seeds_default = config["kl_targets"]["seeds_default"]

        total_cells = len(k1_targets) * len(beta_targets)
        processed = 0
        skipped = 0

        for k1_idx, beta_idx in product(range(len(k1_targets)), range(len(beta_targets))):
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
