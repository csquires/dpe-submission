"""Subsample the elbo dataset: 20k samples for p0/p1, 10k for pstar.

Reads:  dataset_d=3,nsamples=50000.h5
Writes: dataset_d=3,n_p0p1=20000,n_pstar=10000.h5

Indices are drawn once per row with a fixed seed so the result is
reproducible.  The paired arrays (theta+y) are always subsampled with
the same indices so samples stay aligned.
"""
import os
import numpy as np
import h5py
import yaml
from tqdm import trange

config = yaml.load(open("ex/synth/elbo/config1.yaml"), Loader=yaml.FullLoader)
DATA_DIR = config["data_dir"]

N_P0P1  = 20_000
N_PSTAR = 10_000
SEED    = 42

src  = os.path.join(DATA_DIR, "dataset_d=3,nsamples=50000.h5")
dst  = os.path.join(DATA_DIR, f"dataset_d=3,n_p0p1={N_P0P1},n_pstar={N_PSTAR}.h5")

print(f"Reading  {src}")
print(f"Writing  {dst}")

rng = np.random.default_rng(SEED)

with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
    nrows    = fin["theta0_samples_arr"].shape[0]
    nsamples = fin["theta0_samples_arr"].shape[1]
    print(f"  rows={nrows}  source nsamples={nsamples}")

    # copy metadata arrays unchanged
    for key in (
        "design_eig_percentage_arr", "alpha_arr",
        "prior_mean_arr", "prior_covariance_arr",
        "mu_q_arr", "Sigma_q_arr",
        "design_arr", "obs_y_arr",
    ):
        fout.create_dataset(key, data=fin[key][:])

    # pre-allocate output arrays
    dim = fin["theta0_samples_arr"].shape[2]
    fout.create_dataset("theta0_samples_arr",    shape=(nrows, N_P0P1,  dim), dtype="float32")
    fout.create_dataset("y0_samples_arr",        shape=(nrows, N_P0P1,  1),   dtype="float32")
    fout.create_dataset("theta1_samples_arr",    shape=(nrows, N_P0P1,  dim), dtype="float32")
    fout.create_dataset("y1_samples_arr",        shape=(nrows, N_P0P1,  1),   dtype="float32")
    fout.create_dataset("theta_star_samples_arr",shape=(nrows, N_PSTAR, dim), dtype="float32")
    fout.create_dataset("y_star_samples_arr",    shape=(nrows, N_PSTAR, 1),   dtype="float32")

    for i in trange(nrows, desc="subsampling rows"):
        idx_p01  = rng.choice(nsamples, size=N_P0P1,  replace=False)
        idx_pstar = rng.choice(nsamples, size=N_PSTAR, replace=False)

        fout["theta0_samples_arr"][i]     = fin["theta0_samples_arr"][i][idx_p01]
        fout["y0_samples_arr"][i]         = fin["y0_samples_arr"][i][idx_p01]
        fout["theta1_samples_arr"][i]     = fin["theta1_samples_arr"][i][idx_p01]
        fout["y1_samples_arr"][i]         = fin["y1_samples_arr"][i][idx_p01]
        fout["theta_star_samples_arr"][i] = fin["theta_star_samples_arr"][i][idx_pstar]
        fout["y_star_samples_arr"][i]     = fin["y_star_samples_arr"][i][idx_pstar]

print("Done.")
