"""
Run one HPO trial for a given (method, pstar_idx, trial_id).

Loads the pre-generated hyperparam config from hpo_configs/, fits the method
on a fixed subset of instances using nsamples_pstar_values[pstar_idx] pstar
samples, computes mean MAE, and writes the result to hpo_results/.

Designed to be called as an array task on Babel — see slurm/run_hpo_trial.sh.

Usage:
  python experiments/pstar_sample_complexity/step2b_hpo_trial.py \\
      --method TriangularTSM --pstar-idx 1 --trial-id 5
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

from experiments.ablations.pstar_sample_complexity.hpo_search_spaces import SEARCH_SPACES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=list(SEARCH_SPACES.keys()))
    parser.add_argument("--pstar-idx", type=int, required=True, help="Index into nsamples_pstar_values list")
    parser.add_argument("--trial-id", type=int, required=True)
    return parser.parse_args()


def select_eval_instances(pstar_idx: int, num_instances: int, num_eval: int, seed: int) -> list[int]:
    """
    Pick which instances to evaluate on for this pstar_idx.

    All 30 trials for the same pstar_idx use the same instance subset so that
    trial-to-trial MAE differences reflect hyperparams, not data randomness.
    """
    rng = np.random.default_rng(seed + pstar_idx * 997)
    return sorted(rng.choice(num_instances, size=num_eval, replace=False).tolist())


def run_trial(
    method: str,
    pstar_idx: int,
    trial_id: int,
    config: dict,
    dataset_file: h5py.File,
) -> dict:
    """Fit and evaluate the method on the HPO instance subset, return result dict."""
    device = config["device"]
    num_instances = config["num_instances"]
    nsamples_pstar = config["nsamples_pstar_values"][pstar_idx]
    num_eval = config["hpo_num_eval_instances"]

    # load trial hyperparams
    config_path = os.path.join(
        config["hpo_configs_dir"], method, f"pstar_{pstar_idx}", f"trial_{trial_id}.json"
    )
    with open(config_path) as f:
        trial_cfg = json.load(f)
    hyperparams = trial_cfg["hyperparams"]

    # fixed eval instance indices for this pstar_idx (same across all trials)
    instance_indices = select_eval_instances(pstar_idx, num_instances, num_eval, seed=config["seed"])

    builder = SEARCH_SPACES[method]["builder"]

    mae_per_instance = {}
    t0 = time.perf_counter()

    for instance_idx in instance_indices:
        # seed per (trial, instance) for reproducible training
        torch.manual_seed(hash((method, pstar_idx, trial_id, instance_idx)) & 0xFFFFFFFF)

        samples_p0 = torch.from_numpy(dataset_file["samples_p0_arr"][instance_idx]).to(device)
        samples_p1 = torch.from_numpy(dataset_file["samples_p1_arr"][instance_idx]).to(device)
        # subsample pstar to the target count for this pstar_idx
        samples_pstar = torch.from_numpy(
            dataset_file["samples_pstar_arr"][instance_idx][:nsamples_pstar]
        ).to(device)
        true_ldrs = torch.from_numpy(dataset_file["true_ldrs_arr"][instance_idx])
        samples_test = torch.from_numpy(dataset_file["samples_test_arr"][instance_idx]).to(device)

        estimator = builder(
            input_dim=config["data_dim"],
            device=device,
            config=config,
            **hyperparams,
        )

        try:
            estimator.fit(samples_p0, samples_p1, samples_pstar)
            est_ldrs = estimator.predict_ldr(samples_test).cpu()
            mae = torch.mean(torch.abs(est_ldrs - true_ldrs)).item()
        except Exception as e:
            print(f"  instance {instance_idx}: failed ({type(e).__name__}: {e})")
            mae = float("nan")

        if not math.isfinite(mae):
            print(f"  instance {instance_idx}: non-finite MAE")
            mae = float("nan")

        mae_per_instance[str(instance_idx)] = mae
        print(f"  instance {instance_idx}: MAE={mae:.4f}")

    elapsed = time.perf_counter() - t0

    finite_maes = [v for v in mae_per_instance.values() if math.isfinite(v)]
    mean_mae = float(np.mean(finite_maes)) if finite_maes else float("nan")

    return {
        "method": method,
        "pstar_idx": pstar_idx,
        "nsamples_pstar": nsamples_pstar,
        "trial_id": trial_id,
        "hyperparams": hyperparams,
        "mae_per_instance": mae_per_instance,
        "mean_mae": mean_mae,
        "elapsed_seconds": elapsed,
    }


def main():
    args = parse_args()
    config = yaml.safe_load(open("experiments/pstar_sample_complexity/config.yaml"))

    nsamples_pstar = config["nsamples_pstar_values"][args.pstar_idx]
    print(f"HPO trial: method={args.method}, pstar_idx={args.pstar_idx}, trial_id={args.trial_id}")
    print(f"  nsamples_pstar={nsamples_pstar}")

    dataset_path = os.path.join(config["data_dir"], "dataset.h5")
    with h5py.File(dataset_path, "r") as dataset_file:
        result = run_trial(args.method, args.pstar_idx, args.trial_id, config, dataset_file)

    # atomic write to avoid corrupt JSONs if job is preempted mid-write
    out_dir = os.path.join(config["hpo_results_dir"], args.method, f"pstar_{args.pstar_idx}")
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
