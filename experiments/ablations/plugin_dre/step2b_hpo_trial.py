"""
Run one HPO trial for plugin_dre on a given (method, kl_idx, trial_id).

Fits the method on a fixed subset of plugin instances for the KL setting,
evaluates MAE on the uniform grid, and writes a trial JSON into hpo_results/.
"""

import argparse
import json
import math
import os
import time

import h5py
import numpy as np
import torch
import yaml

from experiments.ablations.plugin_dre.hpo_search_spaces import SEARCH_SPACES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=list(SEARCH_SPACES.keys()))
    parser.add_argument("--kl-idx", type=int, required=True)
    parser.add_argument("--trial-id", type=int, required=True)
    return parser.parse_args()


def select_eval_instances(kl_idx: int, num_instances_per_kl: int, num_eval: int, seed: int) -> list[int]:
    """Choose a fixed subset of instance indices for this KL block."""
    if num_eval >= num_instances_per_kl:
        return list(range(num_instances_per_kl))
    rng = np.random.default_rng(seed + kl_idx * 997)
    return sorted(rng.choice(num_instances_per_kl, size=num_eval, replace=False).tolist())


def run_trial(method: str, kl_idx: int, trial_id: int, config: dict, dataset_file: h5py.File) -> dict:
    device = config["device"]
    num_instances_per_kl = config["num_instances_per_kl"]
    num_eval = config["hpo_num_eval_instances"]
    kl_divergences = config["kl_divergences"]

    config_path = os.path.join(
        config["hpo_configs_dir"], method, f"kl_{kl_idx}", f"trial_{trial_id}.json"
    )
    with open(config_path) as f:
        trial_cfg = json.load(f)
    hyperparams = trial_cfg["hyperparams"]

    local_indices = select_eval_instances(kl_idx, num_instances_per_kl, num_eval, seed=config["seed"])
    row_offset = kl_idx * num_instances_per_kl
    builder = SEARCH_SPACES[method]["builder"]

    mae_per_instance = {}
    t0 = time.perf_counter()

    for local_idx in local_indices:
        row = row_offset + local_idx
        torch.manual_seed(hash((method, kl_idx, trial_id, local_idx)) & 0xFFFFFFFF)

        samples_p0 = torch.from_numpy(dataset_file["samples_p0_arr"][row]).to(device)
        samples_p1 = torch.from_numpy(dataset_file["samples_p1_arr"][row]).to(device)
        grid_points = torch.from_numpy(dataset_file["grid_points_arr"][row]).to(device)
        true_ldrs = torch.from_numpy(dataset_file["true_ldrs_grid_arr"][row])

        estimator = builder(
            input_dim=config["data_dim"],
            device=device,
            config=config,
            **hyperparams,
        )

        try:
            estimator.fit(samples_p0, samples_p1)
            est_ldrs = estimator.predict_ldr(grid_points).cpu()
            mae = torch.mean(torch.abs(est_ldrs - true_ldrs)).item()
        except Exception as e:
            print(f"  instance {local_idx}: failed ({type(e).__name__}: {e})")
            mae = float("nan")

        if not math.isfinite(mae):
            print(f"  instance {local_idx}: non-finite MAE, skipping")
            mae = float("nan")

        mae_per_instance[str(local_idx)] = mae
        print(f"  instance {local_idx}: MAE={mae:.4f}")

    elapsed = time.perf_counter() - t0
    finite_maes = [v for v in mae_per_instance.values() if math.isfinite(v)]
    mean_mae = float(np.mean(finite_maes)) if finite_maes else float("nan")

    return {
        "method": method,
        "kl_idx": kl_idx,
        "kl_value": kl_divergences[kl_idx],
        "trial_id": trial_id,
        "nsamples_train": config["nsamples_train"],
        "hyperparams": hyperparams,
        "mae_per_instance": mae_per_instance,
        "mean_mae": mean_mae,
        "elapsed_seconds": elapsed,
    }


def main():
    args = parse_args()
    config = yaml.safe_load(open("experiments/plugin_dre/config.yaml"))

    print(f"HPO trial: method={args.method}, kl_idx={args.kl_idx}, trial_id={args.trial_id}")
    print(f"  KL={config['kl_divergences'][args.kl_idx]}, nsamples_train={config['nsamples_train']}")

    dataset_path = os.path.join(config["data_dir"], "dataset.h5")
    with h5py.File(dataset_path, "r") as dataset_file:
        result = run_trial(args.method, args.kl_idx, args.trial_id, config, dataset_file)

    out_dir = os.path.join(config["hpo_results_dir"], args.method, f"kl_{args.kl_idx}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"trial_{args.trial_id}.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)

    print(f"\nDone: mean_mae={result['mean_mae']:.4f}, elapsed={result['elapsed_seconds']:.1f}s")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
