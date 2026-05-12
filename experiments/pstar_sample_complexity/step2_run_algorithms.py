"""
Full evaluation using HPO winner hyperparams.

For each method, loads the per-pstar-idx winner hyperparams from
hpo_summary/winners.json, then runs the full evaluation across all 20 instances
for each nsamples_pstar value. Output format is identical to the pre-HPO version
so step3_process_results.py is unchanged.

Usage:
  python experiments/pstar_sample_complexity/step2_run_algorithms.py [--method METHOD] [--force]

  --method: run only this method (default: all)
  --force:  overwrite existing results
"""

import argparse
import json
import os
import time

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
    path = os.path.join(hpo_summary_dir, "winners.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Run step2c_pick_winners.py first.")
        return {}
    with open(path) as f:
        return json.load(f)


def build_estimator(method: str, pstar_idx: int, winners: dict, config: dict) -> object:
    pstar_key = f"pstar_{pstar_idx}"

    if method not in winners or pstar_key not in winners[method]:
        raise ValueError(f"No winner found for {method} {pstar_key}. Run step2c first.")

    hyperparams = winners[method][pstar_key]
    builder = SEARCH_SPACES[method]["builder"]
    return builder(
        input_dim=config["data_dim"],
        device=config["device"],
        config=config,
        **hyperparams,
    )


def main():
    args = parse_args()
    config = yaml.safe_load(open("experiments/pstar_sample_complexity/config.yaml"))

    device = config["device"]
    data_dir = config["data_dir"]
    raw_results_dir = config["raw_results_dir"]
    nsamples_pstar_values = config["nsamples_pstar_values"]
    num_instances = config["num_instances"]
    nsamples_test = config["nsamples_test"]

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    winners = load_winners(config["hpo_summary_dir"])
    os.makedirs(raw_results_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, "dataset.h5")
    methods_to_run = [args.method] if args.method else list(SEARCH_SPACES.keys())

    with h5py.File(dataset_path, "r") as dataset_file:
        for method in methods_to_run:
            if method not in winners:
                print(f"Skipping {method}: no HPO winners found")
                continue

            print(f"\nRunning {method}...")

            for pstar_idx, nsamples_pstar in enumerate(nsamples_pstar_values):
                pstar_key = f"pstar_{pstar_idx}"
                result_path = os.path.join(
                    raw_results_dir, f"{method}_nsamples_pstar_{nsamples_pstar}.h5"
                )

                if os.path.exists(result_path) and not args.force:
                    print(f"  Skipping {method} pstar_idx={pstar_idx} (results exist, use --force)")
                    continue

                if pstar_key not in winners[method]:
                    print(f"  Skipping {method} pstar_idx={pstar_idx}: no winner")
                    continue

                print(f"  nsamples_pstar={nsamples_pstar}")

                est_ldrs_arr = np.zeros((num_instances, nsamples_test), dtype=np.float32)
                timing_arr = np.zeros(num_instances, dtype=np.float32)
                peak_memory = 0.0

                for instance_idx in tqdm(range(num_instances), desc=f"    instances"):
                    samples_p0 = torch.from_numpy(
                        dataset_file["samples_p0_arr"][instance_idx]
                    ).to(device)
                    samples_p1 = torch.from_numpy(
                        dataset_file["samples_p1_arr"][instance_idx]
                    ).to(device)
                    samples_pstar = torch.from_numpy(
                        dataset_file["samples_pstar_arr"][instance_idx][:nsamples_pstar]
                    ).to(device)
                    samples_test = torch.from_numpy(
                        dataset_file["samples_test_arr"][instance_idx]
                    ).to(device)

                    estimator = build_estimator(method, pstar_idx, winners, config)

                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()

                    t0 = time.perf_counter()
                    try:
                        estimator.fit(samples_p0, samples_p1, samples_pstar)
                        est_ldrs = estimator.predict_ldr(samples_test).cpu().numpy()
                        est_ldrs_arr[instance_idx] = est_ldrs
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"  OOM at instance={instance_idx}")
                            est_ldrs_arr[instance_idx] = np.nan
                            if device == "cuda":
                                torch.cuda.empty_cache()
                        else:
                            raise
                    finally:
                        timing_arr[instance_idx] = time.perf_counter() - t0

                    if device == "cuda":
                        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())

                # atomic write
                tmp_path = result_path + ".tmp"
                with h5py.File(tmp_path, "w") as f:
                    f.create_dataset("est_ldrs_arr", data=est_ldrs_arr)
                    f.create_dataset("timing_arr", data=timing_arr)
                    f.create_dataset("peak_memory", data=np.array(peak_memory, dtype=np.float32))
                os.replace(tmp_path, result_path)
                print(f"  saved {result_path}")

    print(f"\nDone. Results in {raw_results_dir}")


if __name__ == "__main__":
    main()
