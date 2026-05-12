"""
Full plugin_dre evaluation using HPO winner hyperparameters.

Loads per-KL winners from hpo_summary/winners.json, fits each method on every
plugin instance, predicts the grid LDRs, and writes the usual raw_results.h5
datasets so step3_compute_metrics.py remains compatible.
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
    parser.add_argument("--method", type=str, default=None, choices=list(SEARCH_SPACES.keys()))
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    return parser.parse_args()


def load_winners(hpo_summary_dir: str) -> dict:
    path = os.path.join(hpo_summary_dir, "winners.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Run step2c_pick_winners.py first.")
        return {}
    with open(path) as f:
        return json.load(f)


def build_estimator(method: str, kl_idx: int, winners: dict, config: dict):
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
    config = yaml.safe_load(open("experiments/plugin_dre/config.yaml"))

    device = config["device"]
    data_dir = config["data_dir"]
    results_dir = config["results_dir"]
    kl_divergences = config["kl_divergences"]
    num_instances_per_kl = config["num_instances_per_kl"]

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    winners = load_winners(config["hpo_summary_dir"])
    os.makedirs(results_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, "dataset.h5")
    results_path = os.path.join(results_dir, "raw_results.h5")

    existing = set()
    if os.path.exists(results_path):
        with h5py.File(results_path, "r") as f:
            existing = set(f.keys())
        print("Existing results:", list(existing))

    methods_to_run = [args.method] if args.method else list(SEARCH_SPACES.keys())

    with h5py.File(dataset_path, "r") as dataset_file:
        nrows = dataset_file["kl_divergence_arr"].shape[0]
        num_grid_points = dataset_file["grid_points_arr"].shape[1]

        for method in methods_to_run:
            dataset_name = f"est_ldrs_grid_{method}"
            if dataset_name in existing and not args.force:
                print(f"Skipping {method} (results exist, use --force to overwrite)")
                continue

            if method not in winners:
                print(f"Skipping {method}: no HPO winners found")
                continue

            print(f"\nRunning {method}...")
            est_ldrs_arr = np.zeros((nrows, num_grid_points), dtype=np.float32)

            for kl_idx, kl_value in enumerate(kl_divergences):
                row_offset = kl_idx * num_instances_per_kl
                print(f"  KL={kl_value}")

                for local_idx in tqdm(range(num_instances_per_kl), desc=f"    KL={kl_value}"):
                    row = row_offset + local_idx
                    estimator = build_estimator(method, kl_idx, winners, config)

                    samples_p0 = torch.from_numpy(dataset_file["samples_p0_arr"][row]).to(device)
                    samples_p1 = torch.from_numpy(dataset_file["samples_p1_arr"][row]).to(device)
                    grid_points = torch.from_numpy(dataset_file["grid_points_arr"][row]).to(device)

                    estimator.fit(samples_p0, samples_p1)
                    est_ldrs_arr[row] = estimator.predict_ldr(grid_points).cpu().numpy()

            with h5py.File(results_path, "a") as results_file:
                if dataset_name in results_file:
                    del results_file[dataset_name]
                results_file.create_dataset(dataset_name, data=est_ldrs_arr)

            print(f"  saved {dataset_name} to {results_path}")

    print(f"\nDone. Results in {results_path}")


if __name__ == "__main__":
    main()
