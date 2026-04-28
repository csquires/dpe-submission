"""
run a single hpo trial for tsm, ctsm, or vfm on pre-generated mnist eldr data.

loads hdf5 data for eval pairs, creates estimator with hyperparams from json,
fits, predicts ldr, computes mae. saves results json.
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
import yaml

from experiments.mnist_eldr_estimation.hpo_search_spaces import SEARCH_SPACES


def parse_args():
    """parse cli arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=list(SEARCH_SPACES.keys()),
                        help="method name; must be a key of SEARCH_SPACES")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--eval-pairs", type=str, default="0:0,1:0,2:0,3:0")
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def parse_eval_pairs(s):
    """
    split on comma, split each on colon, return list of (alpha_idx, pair_idx).

    Args:
        s: string like "0:0,1:0,2:0"

    Returns:
        list of (alpha_idx, pair_idx) tuples
    """
    pairs = []
    for pair_str in s.split(","):
        alpha_idx, pair_idx = pair_str.split(":")
        pairs.append((int(alpha_idx), int(pair_idx)))
    return pairs


def load_pair_data(data_dir, alpha_idx, pair_idx, device):
    """
    open {data_dir}/alpha_{alpha_idx}_pair_{pair_idx}.h5 and load data.

    Args:
        data_dir: directory containing hdf5 files
        alpha_idx: alpha index
        pair_idx: pair index
        device: torch device

    Returns:
        dict with keys: pstar [N, 14], p0 [N, 14], p1 [N, 14], true_ldrs [N]

    Raises:
        FileNotFoundError: if file missing
    """
    filename = f"{data_dir}/alpha_{alpha_idx}_pair_{pair_idx}.h5"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")

    with h5py.File(filename, "r") as f:
        pstar = torch.from_numpy(f["pstar_samples"][()]).to(device)
        p0 = torch.from_numpy(f["p0_samples"][()]).to(device)
        p1 = torch.from_numpy(f["p1_samples"][()]).to(device)
        true_ldrs = torch.from_numpy(f["true_ldrs"][()]).to(device)

    return {
        "pstar": pstar,
        "p0": p0,
        "p1": p1,
        "true_ldrs": true_ldrs,
    }


def eval_pair(estimator, data, requires_pstar):
    """
    fit estimator on p0/p1, predict on pstar, return mae.

    Args:
        estimator: density ratio estimator
        data: dict with keys pstar, p0, p1, true_ldrs
        requires_pstar: bool indicating whether fit() is called with three tensors

    Returns:
        mae as float
    """
    if requires_pstar:
        estimator.fit(data["p0"], data["p1"], data["pstar"])
    else:
        estimator.fit(data["p0"], data["p1"])
    est = estimator.predict_ldr(data["pstar"])
    mae = torch.mean(torch.abs(est.cpu() - data["true_ldrs"].cpu())).item()
    return mae


def main():
    """
    main entry point.

    parse args -> load configs -> create estimator per pair -> compute mae -> save json
    """
    args = parse_args()

    # load experiment config
    with open("experiments/mnist_eldr_estimation/config.yaml", "r") as f:
        exp_config = yaml.safe_load(f)

    data_dir = exp_config["data_dir"]
    input_dim = exp_config["latent_dim"]
    device = exp_config["device"]
    num_waypoints = exp_config["num_waypoints"]

    # load trial config
    with open(args.config_file, "r") as f:
        trial_config = json.load(f)

    hyperparams = trial_config["hyperparams"]
    trial_id = trial_config["trial_id"]

    # parse eval pairs
    eval_pairs = parse_eval_pairs(args.eval_pairs)

    # run evaluation
    t0 = time.time()
    per_pair_mae = {}

    entry = SEARCH_SPACES[args.method]
    builder_fn = entry["builder"]
    requires_pstar = entry["requires_pstar"]

    for alpha_idx, pair_idx in eval_pairs:
        # load data
        data = load_pair_data(data_dir, alpha_idx, pair_idx, device)

        # create fresh estimator
        estimator = builder_fn(input_dim=input_dim, device=device, num_waypoints=num_waypoints, **hyperparams)

        # compute mae
        mae = eval_pair(estimator, data, requires_pstar)

        print(f"pair ({alpha_idx},{pair_idx}): MAE={mae:.4f}")
        per_pair_mae[f"{alpha_idx}:{pair_idx}"] = mae

    elapsed = time.time() - t0
    mean_mae = np.mean(list(per_pair_mae.values()))

    # save result json
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "method": args.method,
        "trial_id": trial_id,
        "hyperparams": hyperparams,
        "per_pair_mae": per_pair_mae,
        "mean_mae": mean_mae,
        "elapsed_seconds": elapsed,
    }

    output_path = f"{args.output_dir}/trial_{trial_id}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSummary:")
    print(f"  Method: {args.method}")
    print(f"  Trial ID: {trial_id}")
    print(f"  Mean MAE: {mean_mae:.4f}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
