import argparse
import h5py
import numpy as np
import os
import torch
import yaml
from src.methods import (
    BDRE, MDRE, TSM, CTSM, TriangularMDRE, MultiHeadTriangularTDRE,
    TriangularCTSMV1 as TriangularCTSM,
    TriangularVFMV1 as TriangularVFM,
    TabularPluginDRE, SmoothedTabularPluginDRE,
)
from src.waypoints.path_builders import bary_ctsm, bary_vfm
from src.methods.reg.common._cfgs import OptimCfg
from src.models.binary_classification import make_binary_classifier, make_multi_head_binary_classifier
from src.models.multiclass_classification import make_multiclass_classifier
from src.sampling.frozen_flow import FrozenFlow


SUPPORTED_ENCODINGS = {
    "TabularPluginDRE":          {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    "SmoothedTabularPluginDRE":  {"gaussian_blob", "flow_pushforward"},
    "BDRE":                      {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    "MDRE":                      {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    "TriangularMDRE":            {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    "MultiHeadTriangularTDRE":   {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    "TSM":                       {"gaussian_blob", "flow_pushforward"},
    "CTSM":                      {"gaussian_blob", "flow_pushforward"},
    "TriangularCTSM":            {"gaussian_blob", "flow_pushforward"},
    "TriangularVFM":             {"gaussian_blob", "flow_pushforward"},
}

NEEDS_LATENT = {"SmoothedTabularPluginDRE"}

TRIANGULAR_METHODS = {"TriangularMDRE", "MultiHeadTriangularTDRE", "TriangularCTSM", "TriangularVFM"}

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


def derive_input_dim(encoding_cfg, n_states, n_actions):
    """
    map encoding_cfg to input_dim for DNN methods.

    onehot_joint:        dim = n_states * n_actions   (single one-hot over the |S|*|A| product set)
    onehot_concat:       dim = n_states + n_actions   (two one-hots concatenated)
    gaussian_blob/flow:  dim = encoding_cfg["embed_dim"]
    """
    encoding_type = encoding_cfg["type"]
    if encoding_type == "onehot_joint":
        return n_states * n_actions
    elif encoding_type == "onehot_concat":
        return n_states + n_actions
    elif encoding_type in ("gaussian_blob", "flow_pushforward"):
        return encoding_cfg.get("embed_dim", 6)
    else:
        raise ValueError(f"unknown encoding type: {encoding_type}")


def get_encoding_type(encoding_cfg):
    """extract encoding type string from config dict."""
    return encoding_cfg["type"]


def load_data(data_filename, device):
    """
    load samples and optional latent from step1 HDF5.

    returns:
        (pstar_samples, p0_samples, p1_samples [Tensor], pstar_latent, p0_latent, p1_latent [Tensor])
    all samples are float32, all latent are int64 on the specified device.
    """
    if not os.path.exists(data_filename):
        raise FileNotFoundError(
            f"step1 HDF5 not found: {data_filename}. "
            "run step1_create_data.py first."
        )

    with h5py.File(data_filename, "r") as f:
        pstar_samples = torch.from_numpy(f["pstar_samples"][()]).float().to(device)
        p0_samples = torch.from_numpy(f["p0_samples"][()]).float().to(device)
        p1_samples = torch.from_numpy(f["p1_samples"][()]).float().to(device)
        pstar_latent = torch.from_numpy(f["pstar_latent"][()]).long().to(device)
        p0_latent = torch.from_numpy(f["p0_latent"][()]).long().to(device)
        p1_latent = torch.from_numpy(f["p1_latent"][()]).long().to(device)

    return pstar_samples, p0_samples, p1_samples, pstar_latent, p0_latent, p1_latent


def check_existing_results(results_filename):
    """
    scan output HDF5 for existing est_ldrs_* datasets.

    returns: set of method names that already have results.
    """
    existing = set()
    if os.path.exists(results_filename):
        with h5py.File(results_filename, "r") as f:
            existing = {k.replace("est_ldrs_", "") for k in f.keys() if k.startswith("est_ldrs_")}
    return existing


def create_estimator(method, config, encoding_cfg, n_states, n_actions, device):
    """
    instantiate estimator for given method.

    dispatches on method name; reads input_dim from encoding_cfg and problem size;
    applies HPO_PARAMS for score/flow methods; constructs path objects for
    triangular variants.

    args:
        method: string, one of SUPPORTED_ENCODINGS.keys()
        config: dict, full config from yaml
        encoding_cfg: dict, config["encoding"]
        n_states, n_actions: ints, MDP sizes
        device: torch device

    returns: instantiated estimator with fit()/predict_ldr() interface.
    """
    input_dim = derive_input_dim(encoding_cfg, n_states, n_actions)
    num_waypoints = config["num_waypoints"]

    if method == "TabularPluginDRE":
        encoding_type = get_encoding_type(encoding_cfg)
        decode = "argmax" if encoding_type.startswith("onehot") else "nn"
        return TabularPluginDRE(
            n_states=n_states,
            n_actions=n_actions,
            encoding_cfg=encoding_cfg,
            decode=decode,
            device=device,
        )

    elif method == "SmoothedTabularPluginDRE":
        return SmoothedTabularPluginDRE(
            n_states=n_states,
            n_actions=n_actions,
            encoding_cfg=encoding_cfg,
            device=device,
        )

    elif method == "BDRE":
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
        return TSM(
            input_dim=input_dim, device=device,
            n_epochs=hp["n_epochs"], batch_size=hp["batch_size"],
            optim=OptimCfg(lr=hp["lr"]),
        )

    elif method == "CTSM":
        hp = HPO_PARAMS["CTSM"]
        return CTSM(
            input_dim=input_dim, device=device,
            n_epochs=hp["n_epochs"], batch_size=hp["batch_size"],
            sigma=hp["sigma"],
            optim=OptimCfg(lr=hp["lr"]),
        )

    elif method == "TriangularCTSM":
        hp = HPO_PARAMS["CTSM"]
        path = bary_ctsm(sigma=hp["sigma"], vertex=0.5, eps=max(hp["eps"], 1e-3))
        return TriangularCTSM(
            input_dim=input_dim,
            path=path,
            n_epochs=hp["n_epochs"],
            optim=OptimCfg(lr=hp["lr"]),
            batch_size=hp["batch_size"],
            device=device,
        )

    elif method == "TriangularVFM":
        hp = HPO_PARAMS["VFM"]
        eps = max(hp["eps"], 1e-3)
        path = bary_vfm(k=hp["k"], vertex=0.5, eps=eps)
        return TriangularVFM(
            input_dim=input_dim,
            path=path,
            n_epochs=hp["n_epochs"],
            optim=OptimCfg(lr=hp["lr"]),
            batch_size=hp["batch_size"],
            integration_steps=hp["integration_steps"],
            device=device,
        )

    else:
        raise ValueError(f"unknown method: {method}")


def run_method(
    method,
    pstar_samples,
    p0_samples,
    p1_samples,
    pstar_latent,
    p0_latent,
    p1_latent,
    results_filename,
    config,
    encoding_cfg,
    n_states,
    n_actions,
    device,
    force=False,
):
    """
    fit estimator for method, save est_ldrs to HDF5.

    dispatches to create_estimator; selects fit signature based on method type
    (triangular vs. needs_latent vs. standard); handles fit + predict; writes
    results to HDF5 under key est_ldrs_{method}.

    per-method try/except: log failures and continue; do not crash pipeline.
    """
    dataset_key = f"est_ldrs_{method}"
    existing = check_existing_results(results_filename)

    if method in existing and not force:
        print(f"[{method}] skipping (results exist; use --force to overwrite)")
        return False

    print(f"[{method}] fitting...")

    try:
        estimator = create_estimator(
            method=method,
            config=config,
            encoding_cfg=encoding_cfg,
            n_states=n_states,
            n_actions=n_actions,
            device=device,
        )

        if method in TRIANGULAR_METHODS:
            estimator.fit(p0_samples, p1_samples, pstar_samples)
        elif method in NEEDS_LATENT:
            estimator.fit(
                p0_samples,
                p1_samples,
                latent_p0=p0_latent,
                latent_p1=p1_latent,
            )
        else:
            estimator.fit(p0_samples, p1_samples)

        est_ldrs = estimator.predict_ldr(pstar_samples)
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
                data=np.full(pstar_samples.shape[0], np.nan, dtype=np.float32)
            )
        print(f"[{method}] saved NaN sentinel to {results_filename}")
        return False


def parse_args(args=None):
    """
    parse command-line arguments.

    supports two modes:
      python step2_run_algorithms.py --k1-idx I --k2-idx J --seed K [--methods M1,M2,...] [--force]
      python step2_run_algorithms.py --smoke
    """
    parser = argparse.ArgumentParser(
        description="Fit density-ratio methods on step1 SMODICE data."
    )
    parser.add_argument("--k1-idx", type=int, help="Index into k1_values")
    parser.add_argument("--k2-idx", type=int, help="Index into k2_values")
    parser.add_argument("--seed", type=int, help="Random seed")
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
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test (1 cell, 1 seed, 2 methods)",
    )
    return parser.parse_args(args)


def load_config(config_path):
    """load yaml config from path."""
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    """
    main entry point.

    workflow:
      1. parse CLI and set random seeds
      2. load config from yaml
      3. validate input HDF5 exists
      4. create output directory
      5. load samples and latent indices
      6. filter methods by encoding compatibility
      7. for each method: run_method() with isolated error handling
    """
    args = parse_args()
    config = load_config("ex/synth/occupancy/config.yaml")
    device = torch.device(config["device"])

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if args.smoke:
        k1_vals = config["kl_targets"]["k1_values"]
        k2_vals = config["kl_targets"]["k2_values"]
        k1_idx = k1_vals.index(1.0) if 1.0 in k1_vals else 0
        k2_idx = k2_vals.index(1.0) if 1.0 in k2_vals else 0
        seed = 0
        methods_cli = "SmoothedTabularPluginDRE,TSM"
        print(f"[smoke test] k1_idx={k1_idx}, k2_idx={k2_idx}, seed=0, methods=SmoothedTabularPluginDRE,TSM")
    else:
        if args.k1_idx is None or args.k2_idx is None or args.seed is None:
            parser = argparse.ArgumentParser()
            parser.error("--k1-idx, --k2-idx, --seed required (or use --smoke)")
        k1_idx, k2_idx, seed = args.k1_idx, args.k2_idx, args.seed
        methods_cli = args.methods

    encoding_cfg = dict(config["encoding"])
    encoding_type = get_encoding_type(encoding_cfg)

    def encoding_subdir(base):
        if encoding_type.startswith("onehot"):
            return os.path.join(base, encoding_type, "sigma_na")
        sigma = encoding_cfg["sigma"]
        return os.path.join(base, encoding_type, f"sigma_{sigma:.3f}")

    if encoding_type == "flow_pushforward":
        flow_subcfg = encoding_cfg.get("flow", {})
        encoding_cfg["flow_module"] = FrozenFlow(
            dim=encoding_cfg["embed_dim"],
            n_layers=flow_subcfg.get("layers", 4),
            seed=flow_subcfg.get("seed", config["seed"]),
        )

    data_subdir = encoding_subdir(config["data_dir"])
    data_filename = os.path.join(
        data_subdir,
        f"kl1_{k1_idx}_kl2_{k2_idx}_seed_{seed}.h5",
    )

    results_subdir = encoding_subdir(config["raw_results_dir"])
    results_filename = os.path.join(
        results_subdir,
        f"kl1_{k1_idx}_kl2_{k2_idx}_seed_{seed}.h5",
    )

    if not os.path.exists(data_filename):
        raise FileNotFoundError(
            f"step1 HDF5 not found: {data_filename}\n"
            f"run step1_create_data.py --k1-idx {k1_idx} --k2-idx {k2_idx} --seed {seed}"
        )

    os.makedirs(results_subdir, exist_ok=True)

    print(f"loading data from {data_filename}...")
    pstar_samples, p0_samples, p1_samples, pstar_latent, p0_latent, p1_latent = load_data(
        data_filename, device
    )
    print(f"  pstar shape: {pstar_samples.shape}, p0 shape: {p0_samples.shape}, p1 shape: {p1_samples.shape}")

    # use authoritative MDP shape from gridworld config (latent argmax can undercount
    # if some states are never visited).
    L = config["gridworld"]["L"]
    n_states = L * L
    n_actions = 4
    print(f"  using n_states={n_states}, n_actions={n_actions} (L={L})")

    # enrich encoding_cfg with MDP shape so pointwise_smoothed_ldr / encode_sa can
    # look these up. step1 does the same enrichment locally; we mirror it here.
    encoding_cfg["n_states"] = n_states
    encoding_cfg["n_actions"] = n_actions
    encoding_cfg["L"] = L

    all_methods = config["algorithms"]
    methods = [
        m for m in all_methods
        if encoding_type in SUPPORTED_ENCODINGS.get(m, set())
    ]

    if methods_cli:
        cli_methods = {m.strip() for m in methods_cli.split(",")}
        methods = [m for m in methods if m in cli_methods]

    print(f"running methods: {methods}")

    for method in methods:
        run_method(
            method=method,
            pstar_samples=pstar_samples,
            p0_samples=p0_samples,
            p1_samples=p1_samples,
            pstar_latent=pstar_latent,
            p0_latent=p0_latent,
            p1_latent=p1_latent,
            results_filename=results_filename,
            config=config,
            encoding_cfg=encoding_cfg,
            n_states=n_states,
            n_actions=n_actions,
            device=device,
            force=args.force,
        )

    print(f"step2 complete. results saved to {results_filename}")


if __name__ == "__main__":
    main()
