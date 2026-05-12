import argparse
import os

import h5py
import numpy as np
import torch
from tqdm import trange
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='Force re-run of all algorithms, overwriting existing results')
args = parser.parse_args()

from src.models.binary_classification import make_binary_classifier, make_pairwise_binary_classifiers
from src.models.multiclass_classification import make_multiclass_classifier
from src.methods import BDRE, MDRE, TDRE
from src.methods.reg.tsm import TSM
from src.methods.cls.mdre.tri import TriangularMDRE
from src.methods.reg.vfm.spatial_adapters import make_spatial_velo_denoiser
from src.waypoints.triangular_waypoints import TriangularWaypointBuilder1D


config = yaml.load(open('experiments/elbo_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
DEVICE = config['device']
# directories
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
# dataset parameters
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']
# random seed
SEED = config['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)

dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM},nsamples={NSAMPLES}.h5'
results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},nsamples={NSAMPLES}.h5'

existing_results = set()
if os.path.exists(results_filename):
    with h5py.File(results_filename, 'r') as f:
        existing_results = set(f.keys())
        print("Existing results for:", list(f.keys()))

# number of waypoints for TDRE and MDRE (matched to DRE/EIG experiments)
TDRE_WAYPOINTS = 5
MDRE_WAYPOINTS = 15

bdre_classifier = make_binary_classifier(name="default", input_dim=DATA_DIM+1)
bdre = BDRE(bdre_classifier, device=DEVICE)

tdre_classifiers = make_pairwise_binary_classifiers(name="default", num_classes=TDRE_WAYPOINTS, input_dim=DATA_DIM+1)
tdre = TDRE(tdre_classifiers, num_waypoints=TDRE_WAYPOINTS, device=DEVICE)

mdre_classifier = make_multiclass_classifier(name="default", input_dim=DATA_DIM+1, num_classes=MDRE_WAYPOINTS)
mdre = MDRE(mdre_classifier, device=DEVICE)

# instantiate tsm
tsm = TSM(DATA_DIM+1, device=DEVICE)

# instantiate triangular mdre
triangular_mdre_waypoints = 15
triangular_mdre_classifier = make_multiclass_classifier(name="default", input_dim=DATA_DIM+1, num_classes=triangular_mdre_waypoints)
builder = TriangularWaypointBuilder1D(midpoint_oversample=7, gamma_power=3.0)
triangular_mdre = TriangularMDRE(
    triangular_mdre_classifier,
    device=DEVICE,
    waypoint_builder=builder,
)

# instantiate spatial velo denoiser (VFM)
spatial = make_spatial_velo_denoiser(input_dim=DATA_DIM+1, device=DEVICE)

# DRE-based algorithms (use fit/predict pattern)
algorithms = [
    ("BDRE", bdre),
    ("TDRE_5", tdre),
    ("MDRE_15", mdre),
    ("TSM", tsm),
    ("TriangularMDRE", triangular_mdre),
    ("VFM", spatial),
]


_PSTAR_METHODS = {"TriangularMDRE"}


os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
with h5py.File(dataset_filename, 'r') as dataset_file:
    nrows = dataset_file['design_arr'].shape[0]

    for alg_name, alg in algorithms:
        dataset_name = f'est_eldrs_arr_{alg_name}'
        if dataset_name in existing_results and not args.force:
            print(f"Skipping {alg_name} (results exist, use --force to overwrite)")
            continue

        print(f"Running {alg_name}...")
        est_eldrs_arr = np.zeros(nrows)
        for idx in trange(nrows):
            # p_star samples: (theta_star, y_star) from variational posterior
            theta_star = torch.from_numpy(dataset_file['theta_star_samples_arr'][idx]).float().to(DEVICE)  # (NSAMPLES, DATA_DIM)
            y_star = torch.from_numpy(dataset_file['y_star_samples_arr'][idx]).float().to(DEVICE)  # (NSAMPLES, 1)
            samples_pstar = torch.cat([theta_star, y_star], dim=1)  # (NSAMPLES, DATA_DIM+1)

            # p0 samples: prior-induced joint (theta0, y0)
            theta0 = torch.from_numpy(dataset_file['theta0_samples_arr'][idx]).float().to(DEVICE)  # (NSAMPLES, DATA_DIM)
            y0 = torch.from_numpy(dataset_file['y0_samples_arr'][idx]).float().to(DEVICE)  # (NSAMPLES, 1)
            samples_p0 = torch.cat([theta0, y0], dim=1)  # (NSAMPLES, DATA_DIM+1)

            # p1 samples: q(theta) x prior predictive (theta1, y1)
            theta1 = torch.from_numpy(dataset_file['theta1_samples_arr'][idx]).float().to(DEVICE)  # (NSAMPLES, DATA_DIM)
            y1 = torch.from_numpy(dataset_file['y1_samples_arr'][idx]).float().to(DEVICE)  # (NSAMPLES, 1)
            samples_p1 = torch.cat([theta1, y1], dim=1)  # (NSAMPLES, DATA_DIM+1)

            if alg_name in _PSTAR_METHODS:
                alg.fit(samples_p0, samples_p1, samples_pstar)
            else:
                alg.fit(samples_p0, samples_p1)
            est_eldrs_arr[idx] = alg.predict_eldr(samples_pstar).item()

        with h5py.File(results_filename, 'a') as f:
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(dataset_name, data=est_eldrs_arr)
