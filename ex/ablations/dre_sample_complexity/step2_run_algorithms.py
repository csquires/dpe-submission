"""
Full evaluation using HPO winner hyperparams.

For each method, loads the per-KL winner hyperparams from hpo_summary/winners.json,
then runs the full evaluation across all sample sizes and instances. Output format
is identical to the pre-HPO version so step3_process_results.py is unchanged.

Usage:
  python experiments/dre_sample_complexity/step2_run_algorithms.py [--method METHOD] [--force]

  --method: run only this method (default: all)
  --force:  overwrite existing results
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm
import yaml

from experiments.utils.hpo.method_specs import METHOD_SPECS as SEARCH_SPACES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None, choices=list(SEARCH_SPACES.keys()),
                        help="Run only this method (default: all)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    return parser.parse_args()


def load_winners(hpo_summary_dir: str) -> dict:
    """Load winners.json; return empty dict with warning if missing."""
    path = os.path.join(hpo_summary_dir, "winners.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Run step2c_pick_winners.py first.")
        return {}
    with open(path) as f:
        return json.load(f)


def build_estimator(method: str, kl_idx: int, winners: dict, config: dict) -> object:
    """Instantiate estimator using HPO winner hyperparams for this (method, kl_idx)."""
    kl_key = f"kl_{kl_idx}"

    if method not in winners or kl_key not in winners[method]:
        raise ValueError(f"No winner found for {method} {kl_key}. Run step2c first.")

    hyperparams = winners[method][kl_key]
    builder = SEARCH_SPACES[method]["builder"]
    return builder(
        input_dim=config["data_dim"],
        device=config["device"],
        config=config,
        **hyperparams,
    )


def main():
    args = parse_args()
    config = yaml.safe_load(open("experiments/dre_sample_complexity/config.yaml"))

    device = config["device"]
    data_dir = config["data_dir"]
    raw_results_dir = config["raw_results_dir"]
    kl_divergences = config["kl_divergences"]
    num_instances_per_kl = config["num_instances_per_kl"]
    nsamples_train_values = config["nsamples_train_values"]
    nsamples_test = config["nsamples_test"]

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    winners = load_winners(config["hpo_summary_dir"])

    os.makedirs(raw_results_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, "dataset.h5")
    results_path = os.path.join(raw_results_dir, "results.h5")

    # check which methods already have results
    existing = set()
    if os.path.exists(results_path):
        with h5py.File(results_path, "r") as f:
            existing = set(f.keys())
        print("Existing results:", list(existing))

    methods_to_run = [args.method] if args.method else list(SEARCH_SPACES.keys())
    nrows = len(kl_divergences) * num_instances_per_kl
    n_nsamples = len(nsamples_train_values)

    with h5py.File(dataset_path, "r") as dataset_file:
        for method in methods_to_run:
            result_key = f"est_ldrs_arr_{method}"
            if result_key in existing and not args.force:
                print(f"Skipping {method} (results exist, use --force to overwrite)")
                continue

            if method not in winners:
                print(f"Skipping {method}: no HPO winners found")
                continue

            print(f"\nRunning {method}...")

            # shape: (nrows, n_nsamples_train, nsamples_test)
            est_ldrs_arr = np.zeros((nrows, n_nsamples, nsamples_test), dtype=np.float32)

            for ntrain_idx, nsamples_train in enumerate(nsamples_train_values):
                print(f"  nsamples_train={nsamples_train}")

                # iterate by kl block so we build estimator once per kl (not per row)
                for kl_idx, kl_value in enumerate(kl_divergences):
                    row_offset = kl_idx * num_instances_per_kl

                    for local_idx in tqdm(range(num_instances_per_kl),
                                          desc=f"    KL={kl_value}"):
                        row = row_offset + local_idx

                        # fresh estimator per (kl_idx, instance, nsamples) to avoid
                        # state leakage across instances after fit()
                        estimator = build_estimator(method, kl_idx, winners, config)

                        samples_p0 = torch.from_numpy(
                            dataset_file["samples_p0_arr"][row][:nsamples_train]
                        ).to(device)
                        samples_p1 = torch.from_numpy(
                            dataset_file["samples_p1_arr"][row][:nsamples_train]
                        ).to(device)
                        samples_pstar = torch.from_numpy(
                            dataset_file["samples_pstar_arr"][row]
                        ).to(device)

                        estimator.fit(samples_p0, samples_p1)
                        est_ldrs = estimator.predict_ldr(samples_pstar).cpu().numpy()
                        est_ldrs_arr[row, ntrain_idx] = est_ldrs

            # write result (same key format as original step2 for step3 compatibility)
            with h5py.File(results_path, "a") as results_file:
                if result_key in results_file:
                    del results_file[result_key]
                results_file.create_dataset(result_key, data=est_ldrs_arr)

            print(f"  saved {result_key} to {results_path}")

    print(f"\nDone. Results in {results_path}")


if __name__ == "__main__":
    main()
