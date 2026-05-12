import argparse
import os
import yaml
import numpy as np
import torch
from pathlib import Path

from ex.utils.run_algorithms import (
    load_winners, load_data, create_estimator, run_method,
)
from ex.utils.hpo.method_specs import METHOD_SPECS as SEARCH_SPACES


def parse_args(args=None):
    """parse cli arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-idx", type=int, required=True, help="Index into alpha values")
    parser.add_argument("--pair-idx", type=int, required=True, help="Index of digit pair within alpha")
    parser.add_argument("--methods", type=str, required=True, help="Comma-separated method names")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    return parser.parse_args(args)


def load_config(config_path):
    """load yaml config from path."""
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_data_paths(config, alpha_idx, pair_idx):
    """construct data and results file paths."""
    data_filename = f"{config['data_dir']}/alpha_{alpha_idx}_pair_{pair_idx}.h5"
    results_filename = f"{config['raw_results_dir']}/alpha_{alpha_idx}_pair_{pair_idx}.h5"
    return data_filename, results_filename


def load_existing_results(results_filename):
    """check which methods already have results saved."""
    import h5py
    existing_results = set()
    if os.path.exists(results_filename):
        with h5py.File(results_filename, "r") as f:
            existing_results = set(f.keys())
    return existing_results


def main():
    """main entry point."""
    args = parse_args()

    # load config and set random seeds
    config = load_config("ex/semisynth/mnist/config.yaml")
    DEVICE = config["device"]
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # load winners from hpo winners.yaml
    winners_path = "ex/semisynth/mnist/winners.yaml"
    winners = load_winners(winners_path)

    # build filenames
    data_filename, results_filename = build_data_paths(config, args.alpha_idx, args.pair_idx)

    # validate input file
    if not os.path.exists(data_filename):
        raise FileNotFoundError(f"Data file not found: {data_filename}")

    # create output directory
    os.makedirs(config["raw_results_dir"], exist_ok=True)

    # load data
    pstar_samples, p0_samples, p1_samples, true_ldrs = load_data(data_filename, DEVICE)

    # load existing results to check what's already computed
    existing_results = load_existing_results(results_filename)
    if existing_results:
        print(f"Existing results found: {sorted(existing_results)}")

    # parse method list and run each
    method_list = [m.strip() for m in args.methods.split(",")]

    # provide input_dim_fn callback: mnist_eldr uses latent_dim (same as mnist_uncond, NOT dbpedia's pca_dim)
    input_dim_fn = lambda c: c['latent_dim']

    for method in method_list:
        run_method(
            method,
            pstar_samples, p0_samples, p1_samples,
            results_filename,
            config,
            DEVICE,
            search_spaces=SEARCH_SPACES,
            alpha_idx=args.alpha_idx,
            force=args.force,
            winners=winners,
            input_dim_fn=input_dim_fn,
        )


if __name__ == "__main__":
    main()
