"""
step2: run algorithms with hpo winners selection.

for each (method, kl_bucket) pair, evaluate top-3 hpo candidates on full
dataset, select candidate with lowest mae on test_set_idx=0, re-fit, predict
all test sets. replaces hardcoded estimator instantiation with winners.yaml
dispatch.

main flow:
  load config + winners.yaml → validate methods → loop per method:
    allocate output array → loop per row:
      determine kl_idx → retrieve top-3 candidates → evaluate each on full
      data → select winner by mae → re-fit + predict all test_sets → store
      results → write to hdf5.
"""

import argparse
import logging
import os
import sys

import h5py
import numpy as np
import torch
import yaml
from pathlib import Path

from src.utils.io import _load_config
from experiments.model_selection.hpo_search_spaces import SEARCH_SPACES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_winners(winners_path):
    """
    load winners.yaml. return {} if missing, unreadable, or empty.
    log warning but do not error.
    """
    if not os.path.exists(winners_path):
        logger.warning(f"winners.yaml not found at {winners_path}")
        return {}
    try:
        with open(winners_path) as f:
            winners = yaml.safe_load(f)
        if winners is None:
            logger.warning("winners.yaml is empty")
            return {}
        return winners
    except Exception as e:
        logger.warning(f"failed to load winners.yaml: {e}")
        return {}


def validate_winners_list_format(winners):
    """
    assert all winners[method][bucket_idx] are lists, not dicts.
    raise clear migration error on old dict format.
    """
    for method, buckets in winners.items():
        for bucket_idx, entry in buckets.items():
            if not isinstance(entry, list):
                raise ValueError(
                    f"winners.yaml is in old single-winner format (found dict at "
                    f"{method}[{bucket_idx}]); re-run pick_winners with the new top-K logic."
                )


def get_methods_to_run(winners, cli_methods=None):
    """
    determine methods to run: cli whitelist or all winners keys.
    validate each is registered in SEARCH_SPACES.

    return (methods_list, unresolved_methods).
    if any unresolved, caller exits with error.
    """
    if cli_methods:
        # parse comma-separated, strip whitespace
        methods_list = [m.strip() for m in cli_methods.split(",")]
    else:
        methods_list = list(winners.keys())

    unresolved = []
    for method in methods_list:
        if method not in SEARCH_SPACES:
            logger.warning(f"method {method} not registered; skipping")
            continue
        if method not in winners or not winners[method]:
            unresolved.append(method)

    return methods_list, unresolved


def evaluate_candidate(
    method, candidate, dataset_file, idx, device, config
):
    """
    evaluate single candidate on full data (test_set_idx=0).

    return (candidate_trial_id, mean_ae) tuple, or None if error.
    """
    try:
        hyperparams = candidate["hyperparams"]
        trial_id = candidate["trial_id"]

        # build estimator
        builder = SEARCH_SPACES[method]["builder"]
        input_dim = config["num_samples_train"]
        estimator = builder(input_dim=input_dim, **hyperparams)

        # load training data
        samples_p0 = torch.from_numpy(
            dataset_file["samples_p0_arr"][idx]
        ).to(device)  # (nsamples_train, data_dim)
        samples_p1 = torch.from_numpy(
            dataset_file["samples_p1_arr"][idx]
        ).to(device)  # (nsamples_train, data_dim)

        # fit with pstar_train if required
        requires_pstar = SEARCH_SPACES[method].get("requires_pstar", False)
        if requires_pstar:
            pstar_train = torch.from_numpy(
                dataset_file["samples_pstar_train_arr"][idx]
            ).to(device)
            estimator.fit(samples_p0, samples_p1, pstar_train)
        else:
            estimator.fit(samples_p0, samples_p1)

        # predict on test_set_idx=0 (full pstar data)
        pstar_test = torch.from_numpy(
            dataset_file["samples_pstar_arr"][idx, 0]
        ).to(device)  # (nsamples_test, data_dim)
        est_ldrs = estimator.predict_ldr(pstar_test)  # (nsamples_test,)

        # compute mae
        true_ldrs_row = torch.from_numpy(
            dataset_file["true_ldrs_arr"][idx]
        ).to(device)  # (nsamples_test,)
        errors = torch.abs(est_ldrs - true_ldrs_row)
        mean_ae = errors.mean().item()

        return (trial_id, mean_ae)

    except Exception as e:
        logger.error(f"candidate {candidate.get('trial_id')} error: {e}")
        return None


def run_method(method, winners, dataset_file, config, device, force=False):
    """
    run single method: allocate output array, loop per row, evaluate
    top-3 candidates, select winner by mae, re-fit + predict all test_sets.

    return est_ldrs_arr on success, None on error.
    """
    nrows = dataset_file["kl_distance_arr"].shape[0]
    num_instances_per_kl = config["num_instances_per_kl"]
    ntest_sets = config["ntest_sets"]
    nsamples_test = config["num_samples_test"]

    est_ldrs_arr = np.zeros((nrows, ntest_sets, nsamples_test))

    for idx in range(nrows):
        # set deterministic seed per (method, row)
        seed_val = hash((method, idx)) % (2**32)
        torch.manual_seed(seed_val)

        # determine kl bucket
        kl_idx = idx // num_instances_per_kl

        # retrieve top-3 candidates (already sorted by mae, rank 0 is best)
        if kl_idx not in winners[method]:
            logger.warning(
                f"method {method}, row {idx}: kl_idx {kl_idx} not in winners"
            )
            continue

        top_3 = winners[method][kl_idx]
        if not top_3:
            logger.warning(
                f"method {method}, row {idx}: no candidates for kl_idx {kl_idx}"
            )
            continue

        # evaluate each candidate, keep (trial_id, mae) pairs
        results = []
        for candidate in top_3[:3]:  # limit to 3
            result = evaluate_candidate(
                method, candidate, dataset_file, idx, device, config
            )
            if result is not None:
                results.append(result)

        # select winner (minimum mae)
        if not results:
            logger.warning(
                f"method {method}, row {idx}: all top-3 candidates failed"
            )
            continue

        winner_trial_id, _ = min(results, key=lambda x: x[1])

        # find winning candidate by trial_id to get hyperparams
        winner_candidate = next(
            c for c in top_3 if c["trial_id"] == winner_trial_id
        )

        # re-fit on full data
        try:
            hyperparams = winner_candidate["hyperparams"]
            builder = SEARCH_SPACES[method]["builder"]
            input_dim = config["num_samples_train"]
            estimator = builder(input_dim=input_dim, **hyperparams)

            samples_p0 = torch.from_numpy(
                dataset_file["samples_p0_arr"][idx]
            ).to(device)
            samples_p1 = torch.from_numpy(
                dataset_file["samples_p1_arr"][idx]
            ).to(device)

            requires_pstar = SEARCH_SPACES[method].get("requires_pstar", False)
            if requires_pstar:
                pstar_train = torch.from_numpy(
                    dataset_file["samples_pstar_train_arr"][idx]
                ).to(device)
                estimator.fit(samples_p0, samples_p1, pstar_train)
            else:
                estimator.fit(samples_p0, samples_p1)

            # predict all test sets
            for test_set_idx in range(ntest_sets):
                pstar_test_set = torch.from_numpy(
                    dataset_file["samples_pstar_arr"][idx, test_set_idx]
                ).to(device)
                est_ldrs = estimator.predict_ldr(
                    pstar_test_set
                )  # (nsamples_test,)
                est_ldrs_arr[idx, test_set_idx] = (
                    est_ldrs.cpu().numpy().astype(np.float32)
                )  # (ntest_sets, nsamples_test)

        except Exception as e:
            logger.error(f"method {method}, row {idx}: re-fit error: {e}")
            continue

    return est_ldrs_arr


def main():
    parser = argparse.ArgumentParser(
        description="Run algorithms with hpo winners selection"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results in output hdf5",
    )
    parser.add_argument(
        "--methods",
        type=str,
        help="Comma-separated list of methods to run (default: all in winners.yaml)",
    )
    args = parser.parse_args()

    # load config
    config = _load_config("experiments/model_selection/config.yaml")
    required_keys = [
        "data_dir",
        "raw_results_dir",
        "num_instances_per_kl",
        "kl_distances",
        "num_samples_train",
        "num_samples_test",
        "ntest_sets",
        "device",
        "seed",
    ]
    for key in required_keys:
        if key not in config:
            logger.error(f"config missing required key: {key}")
            sys.exit(1)

    # resolve paths
    data_dir = config["data_dir"]
    if data_dir.startswith("${DPE_DATA_ROOT}"):
        dpe_data_root = os.environ.get("DPE_DATA_ROOT")
        if not dpe_data_root:
            logger.error("DPE_DATA_ROOT environment variable not set")
            sys.exit(1)
        data_dir = data_dir.replace("${DPE_DATA_ROOT}", dpe_data_root)
    config["data_dir"] = data_dir

    # set random seeds
    seed = config["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = config["device"]

    # load winners
    winners_path = "experiments/model_selection/winners.yaml"
    winners = load_winners(winners_path)

    # validate list-only format
    try:
        validate_winners_list_format(winners)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # get methods to run
    methods_list, unresolved = get_methods_to_run(winners, args.methods)

    # fail-loud if any registered method lacks winners entry
    if unresolved:
        logger.error(
            f"missing winners.yaml entry for methods: {', '.join(unresolved)}"
        )
        sys.exit(1)

    # open dataset
    dataset_filename = os.path.join(
        config["data_dir"], "dataset_newpstar.h5"
    )
    try:
        dataset_file = h5py.File(dataset_filename, "r")
    except Exception as e:
        logger.error(f"failed to open dataset: {e}")
        sys.exit(1)

    # verify required dataset keys
    required_dataset_keys = [
        "true_ldrs_arr",
        "kl_distance_arr",
        "samples_p0_arr",
        "samples_p1_arr",
        "samples_pstar_arr",
        "samples_pstar_train_arr",
    ]
    for key in required_dataset_keys:
        if key not in dataset_file:
            logger.error(f"dataset missing key: {key}")
            sys.exit(1)

    # create results directory and file
    raw_results_dir = config["raw_results_dir"]
    os.makedirs(raw_results_dir, exist_ok=True)
    results_filename = os.path.join(raw_results_dir, "results.h5")

    # check existing results
    existing_results = set()
    if os.path.exists(results_filename):
        with h5py.File(results_filename, "r") as f:
            existing_results = set(f.keys())

    # run each method
    for method in methods_list:
        if method not in SEARCH_SPACES:
            logger.warning(f"method {method} not registered; skipping")
            continue

        dataset_name = f"est_ldrs_arr_{method}"
        if dataset_name in existing_results and not args.force:
            logger.info(
                f"Skipping {method} (results exist, use --force to overwrite)"
            )
            continue

        # delete if force
        if dataset_name in existing_results and args.force:
            with h5py.File(results_filename, "a") as f:
                del f[dataset_name]

        # run method
        try:
            est_ldrs_arr = run_method(
                method, winners, dataset_file, config, device, args.force
            )

            if est_ldrs_arr is None:
                logger.error(f"method {method} failed")
                continue

            # write results
            with h5py.File(results_filename, "a") as f:
                f.create_dataset(dataset_name, data=est_ldrs_arr)
            logger.info(f"saved {method} to {results_filename}")

        except Exception as e:
            logger.error(f"method {method} runtime error: {e}")
            continue

    dataset_file.close()


if __name__ == "__main__":
    main()
