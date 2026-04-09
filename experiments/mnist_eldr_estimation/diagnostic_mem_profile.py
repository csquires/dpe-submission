"""
memory profiler for step2 methods.
tracks gpu memory at each stage of fit() and predict_ldr() for all methods.
usage: python -m experiments.mnist_eldr_estimation.diagnostic_mem_profile --alpha-idx 0 --pair-idx 0
"""

import argparse
import gc
import h5py
import torch
import yaml

from src.density_ratio_estimation import TSM
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
from src.density_ratio_estimation.mh_triangular_tdre import MultiHeadTriangularTDRE
from src.density_ratio_estimation.spatial_adapters import make_spatial_velo_denoiser
from src.models.binary_classification import make_multi_head_binary_classifier
from src.models.multiclass_classification import make_multiclass_classifier


def mb(b):
    return f"{b / 1024**2:.1f}MB"


def mem_report(tag):
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    print(f"  [{tag}] alloc={mb(alloc)}  reserved={mb(reserved)}  peak={mb(peak)}")


def reset_peak():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()


def profile_method(name, estimator, p0, p1, pstar, triangular=False):
    """profile a single method's fit + predict_ldr."""
    print(f"\n{'='*60}")
    print(f"method: {name}")
    print(f"{'='*60}")

    reset_peak()
    mem_report("pre-fit")

    if triangular:
        estimator.fit(p0, p1, pstar)
    else:
        estimator.fit(p0, p1)
    mem_report("post-fit")

    reset_peak()
    with torch.no_grad():
        _ = estimator.predict_ldr(pstar)
    mem_report("post-predict")

    # cleanup
    del estimator
    reset_peak()
    mem_report("post-cleanup")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-idx", type=int, default=0)
    parser.add_argument("--pair-idx", type=int, default=0)
    args = parser.parse_args()

    with open("experiments/mnist_eldr_estimation/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = config["device"]
    input_dim = config["latent_dim"]
    num_waypoints = config["num_waypoints"]

    # load data
    data_file = f"{config['data_dir']}/alpha_{args.alpha_idx}_pair_{args.pair_idx}.h5"
    with h5py.File(data_file, "r") as f:
        pstar = torch.from_numpy(f["pstar_samples"][()]).to(device)
        p0 = torch.from_numpy(f["p0_samples"][()]).to(device)
        p1 = torch.from_numpy(f["p1_samples"][()]).to(device)

    print(f"data loaded: p0={p0.shape}, p1={p1.shape}, pstar={pstar.shape}")
    print(f"data mem: {mb(p0.nbytes + p1.nbytes + pstar.nbytes)}")
    mem_report("data-loaded")

    # --- TriangularMDRE ---
    clf = make_multiclass_classifier(name="default", input_dim=input_dim, num_classes=num_waypoints)
    est = TriangularMDRE(clf, device=device)
    profile_method("TriangularMDRE", est, p0, p1, pstar, triangular=True)

    # --- MultiHeadTriangularTDRE ---
    clf = make_multi_head_binary_classifier(input_dim=input_dim, num_heads=num_waypoints - 1)
    est = MultiHeadTriangularTDRE(classifier=clf, num_waypoints=num_waypoints, device=device)
    profile_method("MultiHeadTriangularTDRE", est, p0, p1, pstar, triangular=True)

    # --- VFM ---
    est = make_spatial_velo_denoiser(input_dim=input_dim, device=device)
    profile_method("VFM", est, p0, p1, pstar, triangular=False)

    # --- TSM ---
    est = TSM(input_dim=input_dim, device=device)
    profile_method("TSM", est, p0, p1, pstar, triangular=False)


if __name__ == "__main__":
    main()
