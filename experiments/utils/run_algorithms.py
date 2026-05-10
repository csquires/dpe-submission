import h5py
import numpy as np
import os
import torch
import warnings
import yaml
from pathlib import Path


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


def load_data(data_filename, device):
    """load samples and true ldrs from hdf5."""
    if not os.path.exists(data_filename):
        raise FileNotFoundError(f"Data file not found: {data_filename}")

    with h5py.File(data_filename, "r") as f:
        pstar_samples = torch.from_numpy(f["pstar_samples"][()]).to(device)  # (n_pstar, input_dim)
        p0_samples = torch.from_numpy(f["p0_samples"][()]).to(device)  # (n_p0, input_dim)
        p1_samples = torch.from_numpy(f["p1_samples"][()]).to(device)  # (n_p1, input_dim)
        true_ldrs = torch.from_numpy(f["true_ldrs"][()]).to(device)  # (n_pstar,)

    return pstar_samples, p0_samples, p1_samples, true_ldrs


def load_existing_results(results_filename):
    """check which methods already have results saved."""
    existing_results = set()
    if os.path.exists(results_filename):
        with h5py.File(results_filename, "r") as f:
            existing_results = set(f.keys())
    return existing_results


def create_estimator(method, config, device, search_spaces, alpha_idx=0, winners=None,
                     input_dim_fn=lambda c: c['latent_dim']):
    """
    instantiate estimator for given method.

    hpo methods (keyed in search_spaces) extract per-alpha hyperparams from winners.
    non-hpo methods use inline classifier construction.

    search_spaces: the per-experiment SEARCH_SPACES dict, passed in by the caller.
                   this module does NOT import SEARCH_SPACES; the caller threads it through.
    input_dim_fn: callback to resolve input dimension from config; defaults to config['latent_dim']
                  but dbpedia uses config.get('pca_dim', config['latent_dim']).
    """
    if winners is None:
        winners = {}

    input_dim = input_dim_fn(config)  # callback to resolve input dimension
    num_waypoints = config["num_waypoints"]

    if method in search_spaces:
        # hpo method: retrieve winner hyperparams, pass to builder
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
        return search_spaces[method]["builder"](
            input_dim=input_dim,
            device=device,
            num_waypoints=num_waypoints,
            **hp,
        )

    elif method == "TriangularMDRE":
        from src.models.multiclass_classification import make_multiclass_classifier
        from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
        classifier = make_multiclass_classifier(
            name="default",
            input_dim=input_dim,
            num_classes=num_waypoints
        )
        return TriangularMDRE(classifier, device=device)

    elif method == "MultiHeadTriangularTDRE":
        from src.models.binary_classification import make_multi_head_binary_classifier
        from src.density_ratio_estimation.mh_triangular_tdre import MultiHeadTriangularTDRE
        classifier = make_multi_head_binary_classifier(
            input_dim=input_dim,
            num_heads=num_waypoints - 1,
        )
        return MultiHeadTriangularTDRE(
            classifier=classifier,
            num_waypoints=num_waypoints,
            device=device,
        )

    elif method == "TriangularFMDRE":
        from src.density_ratio_estimation.triangular_fmdre import TriangularFMDRE
        return TriangularFMDRE(input_dim=input_dim, device=device)

    elif method == "BDRE":
        from src.models.binary_classification import make_binary_classifier
        from src.density_ratio_estimation import BDRE
        classifier = make_binary_classifier(
            name="default",
            input_dim=input_dim
        )
        return BDRE(classifier, device=device)

    elif method == "MDRE":
        from src.models.multiclass_classification import make_multiclass_classifier
        from src.density_ratio_estimation import MDRE
        classifier = make_multiclass_classifier(
            name="default",
            input_dim=input_dim,
            num_classes=num_waypoints
        )
        return MDRE(classifier, device=device)

    else:
        raise ValueError(f"Unknown method: {method}")


def run_method(method, pstar_samples, p0_samples, p1_samples, results_filename, config, device,
               search_spaces, winners=None, alpha_idx=0, force=False,
               input_dim_fn=lambda c: c['latent_dim']):
    """run single method, save results, return whether method ran."""
    if winners is None:
        winners = {}

    dataset_key = f"est_ldrs_{method}"
    existing_results = load_existing_results(results_filename)

    if dataset_key in existing_results and not force:
        print(f"Skipping {method} (results exist, use --force to overwrite)")
        return False

    print(f"Running {method}...")

    # instantiate and fit estimator
    estimator = create_estimator(method, config, device, search_spaces=search_spaces,
                                alpha_idx=alpha_idx, winners=winners, input_dim_fn=input_dim_fn)

    # determine whether method requires pstar
    if method in search_spaces:
        requires_pstar = search_spaces[method]["requires_pstar"]
    else:
        # non-hpo fallback: only triangular methods require pstar (stricter version)
        requires_pstar = method in ["TriangularMDRE", "MultiHeadTriangularTDRE", "TriangularFMDRE"]

    if requires_pstar:
        estimator.fit(p0_samples, p1_samples, pstar_samples)
    else:
        estimator.fit(p0_samples, p1_samples)

    # predict ldr on pstar_samples
    est_ldrs = estimator.predict_ldr(pstar_samples)  # (n_pstar,)
    est_ldrs_np = est_ldrs.detach().cpu().numpy().astype(np.float32)

    # save to hdf5
    with h5py.File(results_filename, "a") as f:
        if dataset_key in f:
            del f[dataset_key]
        f.create_dataset(dataset_key, data=est_ldrs_np)

    print(f"Saved {method} results to {results_filename}")
    return True
