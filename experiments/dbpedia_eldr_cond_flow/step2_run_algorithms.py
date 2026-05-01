import argparse
import h5py
import numpy as np
import os
import torch
import warnings
import yaml
from pathlib import Path

from src.density_ratio_estimation import BDRE, MDRE
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
from src.density_ratio_estimation.mh_triangular_tdre import MultiHeadTriangularTDRE
from src.models.binary_classification import make_binary_classifier, make_multi_head_binary_classifier
from src.models.multiclass_classification import make_multiclass_classifier

from experiments.dbpedia_eldr_cond_flow.hpo_search_spaces import SEARCH_SPACES


def load_winners(path):
    """
    load winners from yaml file, return empty dict if missing, empty, or malformed.

    returns yaml.safe_load() result if file exists and parses successfully;
    otherwise returns {} and prints a warning. malformed yaml does not crash step2.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return {}

    try:
        with open(path_obj, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"warning: {path} is malformed yaml ({e}); falling back to class defaults")
        return {}

    return data if data is not None else {}


def parse_args(args=None):
    """parse cli arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-idx", type=int, required=True, help="Index into alpha values")
    parser.add_argument("--pair-idx", type=int, required=True, help="Index of digit pair within alpha")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method names; defaults to all in config['algorithms'] when omitted")
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


def load_data(data_filename, device):
    """load samples and true ldrs from hdf5."""
    if not os.path.exists(data_filename):
        raise FileNotFoundError(f"Data file not found: {data_filename}")

    with h5py.File(data_filename, "r") as f:
        pstar_samples = torch.from_numpy(f["pstar_samples"][()]).to(device)
        p0_samples = torch.from_numpy(f["p0_samples"][()]).to(device)
        p1_samples = torch.from_numpy(f["p1_samples"][()]).to(device)
        true_ldrs = torch.from_numpy(f["true_ldrs"][()]).to(device)

    return pstar_samples, p0_samples, p1_samples, true_ldrs


def load_existing_results(results_filename):
    """check which methods already have results saved."""
    existing_results = set()
    if os.path.exists(results_filename):
        with h5py.File(results_filename, "r") as f:
            existing_results = set(f.keys())
    return existing_results


def create_estimator(method, config, device, alpha_idx=0, winners=None):
    """
    instantiate estimator for given method.

    HPO methods (in SEARCH_SPACES) route through the registered builder with
    per-(method, alpha) hyperparams from winners.yaml. missing entries fall
    back to class defaults (with a warning). non-HPO methods stay in if/elif.

    winners.yaml format: {method: {alpha_idx: [{rank0}, {rank1}, {rank2}]}}
    (after F_dbpedia_pick changes to top-3 list). extracts rank-0 entry (best).
    backwards compat: dict format triggers deprecation warning.
    """
    input_dim = config["latent_dim"]
    num_waypoints = config["num_waypoints"]
    winners = winners or {}

    if method in SEARCH_SPACES:
        entry = winners.get(method, {}).get(alpha_idx, {})

        # extract rank-0 from top-3 list format; fallback to dict format with warning
        if isinstance(entry, list):
            hp = entry[0].get("hyperparams", {}) if entry else {}
        elif isinstance(entry, dict):
            warnings.warn(
                f"winners.yaml uses old single-winner dict format; "
                f"re-run pick_winners with top-K logic",
                DeprecationWarning,
                stacklevel=3
            )
            hp = entry.get("hyperparams", {})
        else:
            hp = {}

        if not hp:
            print(
                f"warning: no winners.yaml entry for ({method}, alpha={alpha_idx}); "
                f"falling back to class defaults"
            )
        return SEARCH_SPACES[method]["builder"](
            input_dim=input_dim,
            device=device,
            num_waypoints=num_waypoints,
            **hp,
        )

    elif method == "TriangularMDRE":
        classifier = make_multiclass_classifier(
            name="default",
            input_dim=input_dim,
            num_classes=num_waypoints
        )
        return TriangularMDRE(classifier, device=device)

    elif method == "MultiHeadTriangularTDRE":
        classifier = make_multi_head_binary_classifier(
            input_dim=input_dim,
            num_heads=num_waypoints - 1,
        )
        return MultiHeadTriangularTDRE(
            classifier=classifier,
            num_waypoints=num_waypoints,
            device=device,
        )

    elif method == "BDRE":
        classifier = make_binary_classifier(
            name="default",
            input_dim=input_dim
        )
        return BDRE(classifier, device=device)

    elif method == "MDRE":
        classifier = make_multiclass_classifier(
            name="default",
            input_dim=input_dim,
            num_classes=num_waypoints
        )
        return MDRE(classifier, device=device)

    else:
        raise ValueError(f"Unknown method: {method}")


def run_method(method, pstar_samples, p0_samples, p1_samples, results_filename, config, device, winners, alpha_idx=0, force=False):
    """run single method, save results, return whether method ran."""
    dataset_key = f"est_ldrs_{method}"
    existing_results = load_existing_results(results_filename)

    if dataset_key in existing_results and not force:
        print(f"Skipping {method} (results exist, use --force to overwrite)")
        return False

    print(f"Running {method}...")

    # instantiate and fit estimator
    estimator = create_estimator(method, config, device, alpha_idx=alpha_idx, winners=winners)

    # resolve requires_pstar: SEARCH_SPACES first, else hardcoded list of pstar-needing
    # non-HPO methods. fit dispatches accordingly.
    if method in SEARCH_SPACES:
        requires_pstar = SEARCH_SPACES[method]["requires_pstar"]
    else:
        requires_pstar = method in ["TriangularMDRE", "MultiHeadTriangularTDRE"]

    if requires_pstar:
        estimator.fit(p0_samples, p1_samples, pstar_samples)
    else:
        estimator.fit(p0_samples, p1_samples)

    # predict ldr on pstar_samples
    est_ldrs = estimator.predict_ldr(pstar_samples)
    est_ldrs_np = est_ldrs.detach().cpu().numpy().astype(np.float32)

    # save to hdf5
    with h5py.File(results_filename, "a") as f:
        if dataset_key in f:
            del f[dataset_key]
        f.create_dataset(dataset_key, data=est_ldrs_np)

    print(f"Saved {method} results to {results_filename}")
    return True


def main():
    """main entry point."""
    args = parse_args()

    # load config and set random seeds
    config = load_config("experiments/dbpedia_eldr_cond_flow/config.yaml")
    DEVICE = config["device"]
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # load winners (cond-flow)
    winners = load_winners("experiments/dbpedia_eldr_cond_flow/winners.yaml")

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
    # if --methods not provided, default to config['algorithms']
    if args.methods is None:
        method_list = config["algorithms"]
    else:
        method_list = [m.strip() for m in args.methods.split(",")]

    for method in method_list:
        run_method(
            method,
            pstar_samples, p0_samples, p1_samples,
            results_filename,
            config,
            DEVICE,
            winners,
            alpha_idx=args.alpha_idx,
            force=args.force
        )


if __name__ == "__main__":
    main()
