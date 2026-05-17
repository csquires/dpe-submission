"""
Run one HPO trial for a given (method, kl_idx, trial_id).

Loads the pre-generated hyperparam config from hpo_configs/, fits the method
on a fixed subset of instances at the HPO sample size, computes mean MAE,
and writes the result to hpo_results/.

Designed to be called as an array task on Babel — see slurm/run_hpo_trial.sh.

Usage:
  python ex/dre_sample_complexity/step2b_hpo_trial.py \
      --method TSM --kl-idx 0 --trial-id 3
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

from ex.utils.hpo.method_specs import METHOD_SPECS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=list(METHOD_SPECS.keys()))
    parser.add_argument("--kl-idx", type=int, required=True, help="Index into kl_divergences list")
    parser.add_argument("--trial-id", type=int, required=True)
    return parser.parse_args()


def select_eval_instances(kl_idx: int, num_instances_per_kl: int, num_eval: int, seed: int) -> list[int]:
    """
    Pick which instances to evaluate on for this (kl_idx).

    All 30 trials for the same kl_idx use the same instance subset so that
    trial-to-trial MAE differences reflect hyperparams, not data randomness.
    The seed is therefore fixed per kl_idx, not per trial.
    """
    rng = np.random.default_rng(seed + kl_idx * 997)
    return sorted(rng.choice(num_instances_per_kl, size=num_eval, replace=False).tolist())


def run_trial(
    method: str,
    kl_idx: int,
    trial_id: int,
    config: dict,
    dataset_file: h5py.File,
) -> dict:
    """Fit and evaluate the method on the HPO instance subset, return result dict."""
    device = config["device"]
    num_instances_per_kl = config["num_instances_per_kl"]
    hpo_nsamples = config["hpo_nsamples_train"]
    num_eval = config["hpo_num_eval_instances"]
    kl_divergences = config["kl_divergences"]

    # load trial hyperparams
    config_path = os.path.join(
        config["hpo_configs_dir"], method, f"kl_{kl_idx}", f"trial_{trial_id}.json"
    )
    with open(config_path) as f:
        trial_cfg = json.load(f)
    hyperparams = trial_cfg["hyperparams"]

    # fixed eval instance indices for this kl (same across all trials)
    local_indices = select_eval_instances(kl_idx, num_instances_per_kl, num_eval, seed=config["seed"])

    # dataset rows for this kl_idx are contiguous: [kl_idx * N, (kl_idx+1) * N)
    row_offset = kl_idx * num_instances_per_kl

    builder = METHOD_SPECS[method]["builder"]

    mae_per_instance = {}
    t0 = time.perf_counter()

    for local_idx in local_indices:
        row = row_offset + local_idx

        # seed per (trial, instance) for reproducible training
        torch.manual_seed(hash((method, kl_idx, trial_id, local_idx)) & 0xFFFFFFFF)

        # subsample training data to HPO sample size
        samples_p0 = torch.from_numpy(dataset_file["samples_p0_arr"][row][:hpo_nsamples]).to(device)
        samples_p1 = torch.from_numpy(dataset_file["samples_p1_arr"][row][:hpo_nsamples]).to(device)
        samples_pstar = torch.from_numpy(dataset_file["samples_pstar_arr"][row]).to(device)
        true_ldrs = torch.from_numpy(dataset_file["true_ldrs_arr"][row])

        # build fresh estimator from sampled hyperparams
        estimator = builder(
            input_dim=config["data_dim"],
            device=device,
            config=config,
            **hyperparams,
        )

        try:
            estimator.fit(samples_p0, samples_p1)
            est_ldrs = estimator.predict_ldr(samples_pstar).cpu()
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

    # mean over finite values only
    finite_maes = [v for v in mae_per_instance.values() if math.isfinite(v)]
    mean_mae = float(np.mean(finite_maes)) if finite_maes else float("nan")

    return {
        "method": method,
        "kl_idx": kl_idx,
        "kl_value": kl_divergences[kl_idx],
        "trial_id": trial_id,
        "hpo_nsamples_train": hpo_nsamples,
        "hyperparams": hyperparams,
        "mae_per_instance": mae_per_instance,
        "mean_mae": mean_mae,
        "elapsed_seconds": elapsed,
    }


def main():
    args = parse_args()
    config = yaml.safe_load(open("ex/dre_sample_complexity/config.yaml"))

    print(f"HPO trial: method={args.method}, kl_idx={args.kl_idx}, trial_id={args.trial_id}")
    print(f"  KL={config['kl_divergences'][args.kl_idx]}, nsamples_train={config['hpo_nsamples_train']}")

    dataset_path = os.path.join(config["data_dir"], "dataset.h5")
    with h5py.File(dataset_path, "r") as dataset_file:
        result = run_trial(args.method, args.kl_idx, args.trial_id, config, dataset_file)

    # atomic write to avoid corrupt JSONs if job is preempted mid-write
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
