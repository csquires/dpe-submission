import argparse
import h5py
import numpy as np
import os
import torch
from src.density_ratio_estimation import (
    BDRE, MDRE, TSM, CTSM, TriangularMDRE, MultiHeadTriangularTDRE,
    TriangularCTSM, TriangularVFM,
)
from src.waypoints.triangular_continuous import BarycentricCtsm1D, BarycentricVfm1D
from src.models.binary_classification import make_binary_classifier, make_multi_head_binary_classifier
from src.models.multiclass_classification import make_multiclass_classifier
from src.utils.io import _load_config


HPO_PARAMS = {
    "TSM": {
        "n_epochs": 1300,
        "lr": 1.27e-3,
        "batch_size": 128,
        "eps": 5.06e-6,
    },
    "CTSM": {
        "n_epochs": 1030,
        "lr": 1.51e-3,
        "batch_size": 128,
        "sigma": 0.518,
        "eps": 1.82e-3,
    },
    "VFM": {
        "n_epochs": 1057,
        "lr": 7.74e-4,
        "batch_size": 256,
        "k": 40,
        "eps": 1.01e-3,
        "integration_steps": 1373,
    },
}

TRIANGULAR_METHODS = {"TriangularMDRE", "MultiHeadTriangularTDRE", "TriangularCTSM", "TriangularVFM"}
ALL_METHODS = {"BDRE", "MDRE", "TriangularMDRE", "MultiHeadTriangularTDRE", "TSM", "CTSM", "TriangularCTSM", "TriangularVFM"}


def parse_args(args=None):
    """
    parse command-line arguments for step2 on pendulum.

    args:
        args: list of strings (default None uses sys.argv[1:])

    returns: argparse.Namespace with fields:
        - k1_idx: int, index into config["kl_targets"]["k1_values"]
        - k2_idx: int, index into config["kl_targets"]["k2_values"]
        - seed: int, random seed
        - methods: str or None, comma-separated method names (default: run all)
        - force: bool, overwrite existing results (default False)

    required args: --k1-idx, --k2-idx, --seed (no shortcuts; explicit long form).
    optional args: --methods (comma-separated string; default None means all), --force (flag).
    """
    parser = argparse.ArgumentParser(
        description="Fit density-ratio methods on step1 pendulum data."
    )
    parser.add_argument("--k1-idx", type=int, required=True, help="Index into k1_values")
    parser.add_argument("--k2-idx", type=int, required=True, help="Index into k2_values")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method names (default: all in config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results",
    )
    return parser.parse_args(args)


def load_config(config_path: str) -> dict:
    """
    load yaml config from config_path with env-var expansion.

    uses src.utils.io._load_config which applies os.path.expandvars to all strings.

    args:
        config_path: str, path to yaml file

    returns: dict with all config keys (pendulum, q_grid, trajectory, traj_kl_grid,
             kl_targets, num_samples, device, seed, algorithms, num_waypoints,
             data_dir, raw_results_dir, ...)

    raises: FileNotFoundError if config_path does not exist.
    """
    return _load_config(config_path)


def load_data(h5_path: str, device: str) -> tuple:
    """
    load samples from step1 output HDF5.

    input HDF5 contains (per pendulum structure):
        - samples_pstar: shape (5000, 18) float32
        - samples_p0: shape (5000, 18) float32
        - samples_p1: shape (5000, 18) float32

    (log densities log_p_pstar, log_p_p0, log_p_p1 are NOT loaded here;
     step3 reads them separately)

    args:
        h5_path: str, path to data HDF5
        device: str, torch device (e.g. "cpu" or "cuda")

    returns: tuple (pstar_samples, p0_samples, p1_samples) where each is
             float32 Tensor on device.

    raises: FileNotFoundError if h5_path does not exist.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"step1 HDF5 not found: {h5_path}. "
            "run step1_create_data.py first."
        )

    with h5py.File(h5_path, "r") as f:
        pstar_samples = torch.from_numpy(f["samples_pstar"][()]).float().to(device)
        p0_samples = torch.from_numpy(f["samples_p0"][()]).float().to(device)
        p1_samples = torch.from_numpy(f["samples_p1"][()]).float().to(device)

    return pstar_samples, p0_samples, p1_samples


def load_existing_results(h5_path: str) -> set:
    """
    scan output HDF5 for existing est_ldrs_* datasets.

    args:
        h5_path: str, path to results HDF5

    returns: set of method names (without est_ldrs_ prefix) already present.
             returns empty set if file does not exist.
    """
    existing = set()
    if os.path.exists(h5_path):
        with h5py.File(h5_path, "r") as f:
            existing = {k.replace("est_ldrs_", "") for k in f.keys() if k.startswith("est_ldrs_")}
    return existing


def create_estimator(method: str, config: dict, device: str) -> object:
    """
    instantiate DRE estimator for given method.

    dispatches on method name; uses num_waypoints from config for triangular variants.
    applies HPO_PARAMS hyperparameters for TSM, CTSM, VFM.
    other methods (BDRE, MDRE, triangular non-flow) use defaults from their constructors.

    args:
        method: str, one of {"BDRE", "MDRE", "TriangularMDRE", "MultiHeadTriangularTDRE",
                             "TSM", "CTSM", "TriangularCTSM", "TriangularVFM"}
        config: dict, full config dict from yaml (used to read num_waypoints)
        device: torch device

    returns: instantiated estimator with fit() and predict_ldr() methods.

    raises: ValueError if method not recognized.
    """
    input_dim = 18  # fixed for pendulum trajectories
    num_waypoints = config["num_waypoints"]

    if method == "BDRE":
        classifier = make_binary_classifier(name="default", input_dim=input_dim)
        return BDRE(classifier=classifier, device=device)

    elif method == "MDRE":
        classifier = make_multiclass_classifier(
            name="default",
            input_dim=input_dim,
            num_classes=num_waypoints,
        )
        return MDRE(classifier=classifier, device=device)

    elif method == "TriangularMDRE":
        classifier = make_multiclass_classifier(
            name="default",
            input_dim=input_dim,
            num_classes=num_waypoints,
        )
        return TriangularMDRE(classifier=classifier, device=device)

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

    elif method == "TSM":
        hp = HPO_PARAMS["TSM"]
        return TSM(input_dim=input_dim, device=device, **hp)

    elif method == "CTSM":
        hp = HPO_PARAMS["CTSM"]
        path = BarycentricCtsm1D(sigma=hp["sigma"], vertex=0.5, eps=hp["eps"])
        return TriangularCTSM(
            input_dim=input_dim,
            path=path,
            n_epochs=hp["n_epochs"],
            lr=hp["lr"],
            batch_size=hp["batch_size"],
            eps=hp["eps"],
            device=device,
        )

    elif method == "TriangularVFM":
        hp = HPO_PARAMS["VFM"]
        eps = max(hp["eps"], 1e-3)
        path = BarycentricVfm1D(k=hp["k"], vertex=0.5, eps=eps)
        return TriangularVFM(
            input_dim=input_dim,
            path=path,
            n_epochs=hp["n_epochs"],
            lr=hp["lr"],
            batch_size=hp["batch_size"],
            eps=eps,
            integration_steps=hp["integration_steps"],
            device=device,
        )

    else:
        raise ValueError(f"unknown method: {method}")


def run_method(
    method: str,
    samples_pstar,
    samples_p0,
    samples_p1,
    results_filename: str,
    config: dict,
    device: str,
    force: bool = False,
) -> bool:
    """
    fit DRE estimator for a single method; save predictions to HDF5.

    fits the estimator on (p0, p1) samples; predicts LDR at pstar samples;
    writes result under key est_ldrs_{method}. on any exception, writes NaN
    sentinel and returns False.

    per-method error handling: try/except wraps entire fit + predict + save flow;
    prints traceback; saves NaN sentinel; does not crash pipeline.

    args:
        method: str, method name
        samples_pstar: Tensor shape (5000, 18)
        samples_p0: Tensor shape (5000, 18)
        samples_p1: Tensor shape (5000, 18)
        results_filename: str, path to output HDF5
        config: dict, full config
        device: torch device
        force: bool, if True overwrite existing results; if False skip if key exists

    returns: bool, True if success, False if exception occurred.
    """
    dataset_key = f"est_ldrs_{method}"
    existing = load_existing_results(results_filename)

    if method in existing and not force:
        print(f"[{method}] skipping (results exist; use --force to overwrite)")
        return False

    print(f"[{method}] fitting...")

    try:
        estimator = create_estimator(method, config, device)

        # dispatch on fit signature: triangular methods take 3 args, others take 2
        if method in TRIANGULAR_METHODS:
            estimator.fit(samples_p0, samples_p1, samples_pstar)
        else:
            estimator.fit(samples_p0, samples_p1)

        est_ldrs = estimator.predict_ldr(samples_pstar)
        est_ldrs_np = est_ldrs.detach().cpu().numpy().astype(np.float32)

        with h5py.File(results_filename, "a") as f:
            if dataset_key in f:
                del f[dataset_key]
            f.create_dataset(dataset_key, data=est_ldrs_np)

        print(f"[{method}] saved to {results_filename}")
        return True

    except Exception as e:
        print(f"[{method}] failed: {e}")
        with h5py.File(results_filename, "a") as f:
            if dataset_key in f:
                del f[dataset_key]
            f.create_dataset(
                dataset_key,
                data=np.full(samples_pstar.shape[0], np.nan, dtype=np.float32)
            )
        print(f"[{method}] saved NaN sentinel to {results_filename}")
        return False


def main():
    """
    main entry point for step2.

    workflow:
      1. parse CLI arguments
      2. load config from experiments/pendulum_eldr_estimation/config.yaml
      3. set global seeds from config["seed"]
      4. validate input HDF5 exists
      5. create output directory
      6. load samples from input HDF5
      7. parse and validate methods list from CLI (default: all from config)
      8. for each method: call run_method() with per-method error handling
      9. print completion message
    """
    args = parse_args()
    config = _load_config("experiments/pendulum_eldr_estimation/config.yaml")

    device = torch.device(config["device"])

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if args.k1_idx is None or args.k2_idx is None or args.seed is None:
        raise ValueError("--k1-idx, --k2-idx, --seed required")

    k1_idx, k2_idx, seed = args.k1_idx, args.k2_idx, args.seed

    data_subdir = config["data_dir"]
    data_filename = os.path.join(
        data_subdir,
        f"k1_{k1_idx}_k2_{k2_idx}_seed_{seed}.h5"
    )

    results_subdir = config["raw_results_dir"]
    results_filename = os.path.join(
        results_subdir,
        f"k1_{k1_idx}_k2_{k2_idx}_seed_{seed}.h5"
    )

    if not os.path.exists(data_filename):
        raise FileNotFoundError(f"step1 HDF5 not found: {data_filename}")

    os.makedirs(results_subdir, exist_ok=True)

    print(f"loading data from {data_filename}...")
    pstar_samples, p0_samples, p1_samples = load_data(data_filename, device)
    print(f"  pstar shape: {pstar_samples.shape}, p0 shape: {p0_samples.shape}, p1 shape: {p1_samples.shape}")

    all_methods = config["algorithms"]

    if args.methods:
        cli_methods = {m.strip() for m in args.methods.split(",")}
        methods = [m for m in all_methods if m in cli_methods]
    else:
        methods = all_methods

    print(f"running methods: {methods}")

    for method in methods:
        run_method(
            method=method,
            samples_pstar=pstar_samples,
            samples_p0=p0_samples,
            samples_p1=p1_samples,
            results_filename=results_filename,
            config=config,
            device=device,
            force=args.force,
        )

    print(f"step2 complete. results saved to {results_filename}")


if __name__ == "__main__":
    main()
