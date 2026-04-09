import argparse
import h5py
import numpy as np
import os
import torch
import yaml

from src.density_ratio_estimation import BDRE, MDRE, TSM, CTSM
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
from src.density_ratio_estimation.mh_triangular_tdre import MultiHeadTriangularTDRE
from src.density_ratio_estimation.spatial_adapters import make_spatial_velo_denoiser
from src.models.binary_classification import make_binary_classifier, make_multi_head_binary_classifier
from src.models.multiclass_classification import make_multiclass_classifier


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


def create_estimator(method, config, device):
    """instantiate estimator for given method."""
    input_dim = config["latent_dim"]
    num_waypoints = config["num_waypoints"]

    if method == "TriangularMDRE":
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

    elif method == "VFM":
        return make_spatial_velo_denoiser(
            input_dim=input_dim,
            device=device
        )

    elif method == "TSM":
        return TSM(input_dim=input_dim, device=device)

    elif method == "CTSM":
        return CTSM(input_dim=input_dim, device=device)

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


def run_method(method, pstar_samples, p0_samples, p1_samples, results_filename, config, device, force=False):
    """run single method, save results, return whether method ran."""
    dataset_key = f"est_ldrs_{method}"
    existing_results = load_existing_results(results_filename)

    if dataset_key in existing_results and not force:
        print(f"Skipping {method} (results exist, use --force to overwrite)")
        return False

    print(f"Running {method}...")

    # instantiate and fit estimator
    estimator = create_estimator(method, config, device)

    # triangular methods require pstar during fit
    if method in ["TriangularMDRE", "MultiHeadTriangularTDRE"]:
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
    config = load_config("experiments/mnist_eldr_estimation/config.yaml")
    DEVICE = config["device"]
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

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

    for method in method_list:
        run_method(
            method,
            pstar_samples, p0_samples, p1_samples,
            results_filename,
            config,
            DEVICE,
            force=args.force
        )


if __name__ == "__main__":
    main()
