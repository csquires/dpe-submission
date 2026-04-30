"""
run a single hpo trial for the dbpedia cond-flow experiment.

loads hdf5 data for eval pairs, creates estimator from SEARCH_SPACES (sourced
from dbpedia_eldr_cond_flow), fits, predicts ldr, computes mae. delegates to
run_trial() for seeding, aggregation, and atomic json write. differs from
mnist_eldr_cond_flow only in the config.yaml path and SEARCH_SPACES import.
"""

import argparse
import json
import os

import h5py
import torch
import yaml

from experiments.dbpedia_eldr_cond_flow.hpo_search_spaces import SEARCH_SPACES
from experiments.utils.hpo.trial import run_trial, parse_cells


def parse_args():
    """parse cli arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=list(SEARCH_SPACES.keys()),
                        help="method name; must be a key of SEARCH_SPACES")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--eval-pairs", type=str, default="0:0,1:0,2:0,3:0")
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def load_pair_data(data_dir, alpha_idx, pair_idx, device):
    """open {data_dir}/alpha_{a}_pair_{p}.h5 and load samples + true_ldrs onto device."""
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


def main():
    """parse args -> load configs -> delegate to run_trial via eval_cell closure."""
    args = parse_args()

    # load dbpedia cond-flow experiment config
    with open("experiments/dbpedia_eldr_cond_flow/config.yaml", "r") as f:
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

    # parse eval pairs into cells
    eval_cells = parse_cells(args.eval_pairs)

    # resolve method spec
    entry = SEARCH_SPACES[args.method]
    builder_fn = entry["builder"]
    requires_pstar = entry["requires_pstar"]

    # define eval_cell closure
    def eval_cell(cell):
        """load pair data, build estimator, fit, predict, compute mae for one cell."""
        alpha_idx, pair_idx = cell
        data = load_pair_data(data_dir, alpha_idx, pair_idx, device)

        # fresh estimator per pair
        estimator = builder_fn(input_dim=input_dim, device=device, num_waypoints=num_waypoints, **hyperparams)

        # fit with or without pstar
        if requires_pstar:
            estimator.fit(data["p0"], data["p1"], data["pstar"])
        else:
            estimator.fit(data["p0"], data["p1"])

        # predict and compute mae
        est = estimator.predict_ldr(data["pstar"])
        mae = torch.mean(torch.abs(est.cpu() - data["true_ldrs"].cpu())).item()
        return mae

    # delegate to run_trial for seeding, aggregation, atomic write
    run_trial(
        experiment="dbpedia_eldr_cond_flow",
        method=args.method,
        trial_id=trial_id,
        hyperparams=hyperparams,
        eval_cells=eval_cells,
        eval_cell=eval_cell,
        output_dir=args.output_dir,
        metric_key="per_pair_mae",
    )

    # print summary (run_trial logs via logger; add stdout print for visibility)
    result_path = f"{args.output_dir}/trial_{trial_id}.json"
    with open(result_path, "r") as f:
        result = json.load(f)

    print(f"\nSummary:")
    print(f"  Method: {args.method}")
    print(f"  Trial ID: {trial_id}")
    print(f"  Mean MAE: {result['mean_metric']:.4f}")
    print(f"  Elapsed: {result['elapsed_seconds']:.2f}s")
    print(f"  Saved to: {result_path}")


if __name__ == "__main__":
    main()
