import os

import h5py
import numpy as np
import torch
from tqdm import trange
import yaml

from src.models.binary_classification import make_binary_classifier, make_pairwise_binary_classifiers
from src.models.multiclass_classification import make_multiclass_classifier
from src.density_ratio_estimation import BDRE, MDRE, TDRE
from src.density_ratio_estimation.spatial_adapters import make_spatial_velo_denoiser
from src.eldr_estimation.direct3_adapter import make_direct3_estimator


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

dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM}.h5'
results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM}.h5'

# number of waypoints for TDRE and MDRE
NUM_WAYPOINTS = 10

bdre_classifier = make_binary_classifier(name="default", input_dim=DATA_DIM+1)
bdre = BDRE(bdre_classifier, device=DEVICE)

tdre_classifiers = make_pairwise_binary_classifiers(name="default", num_classes=NUM_WAYPOINTS, input_dim=DATA_DIM+1)
tdre = TDRE(tdre_classifiers, num_waypoints=NUM_WAYPOINTS, device=DEVICE)

mdre_classifier = make_multiclass_classifier(name="default", input_dim=DATA_DIM+1, num_classes=NUM_WAYPOINTS)
mdre = MDRE(mdre_classifier, device=DEVICE)

# instantiate spatial velo denoiser
spatial = make_spatial_velo_denoiser(input_dim=DATA_DIM+1, device=DEVICE)

# instantiate direct3 estimator
direct3 = make_direct3_estimator(input_dim=DATA_DIM+1, device=DEVICE)

# DRE-based algorithms (use fit/predict pattern)
dre_algorithms = [
    ("BDRE", bdre),
    ("TDRE", tdre),
    ("MDRE", mdre),
    ("Spatial", spatial),
]

# Direct ELDR algorithms (use estimate_eldr directly)
direct_algorithms = [
    ("Direct3", direct3),
]

# Combined list for result tracking
algorithms = dre_algorithms + direct_algorithms


def estimate_eldr_from_dre(dre, samples_pstar, samples_p0, samples_p1):
    """Estimate ELDR using a density ratio estimator (fit/predict pattern)."""
    dre.fit(samples_p0, samples_p1)
    est_ldrs = dre.predict_ldr(samples_pstar)
    return torch.mean(est_ldrs)


os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
with h5py.File(dataset_filename, 'r') as dataset_file:
    nrows = dataset_file['design_arr'].shape[0]

    # pre-allocate result arrays
    est_eldrs_by_alg = {alg_name: np.zeros(nrows) for alg_name, _ in algorithms}

    # Run DRE-based algorithms (fit/predict pattern)
    for alg_name, alg in dre_algorithms:
        print(f"Running {alg_name}...")
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

            # estimate ELDR using fit/predict pattern
            est_eldr = estimate_eldr_from_dre(alg, samples_pstar, samples_p0, samples_p1)
            est_eldrs_by_alg[alg_name][idx] = est_eldr.item() if isinstance(est_eldr, torch.Tensor) else est_eldr

    # Run direct ELDR algorithms (estimate_eldr interface)
    for alg_name, alg in direct_algorithms:
        print(f"Running {alg_name}...")
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

            # estimate ELDR directly
            est_eldr = alg.estimate_eldr(samples_pstar, samples_p0, samples_p1)
            est_eldrs_by_alg[alg_name][idx] = est_eldr if isinstance(est_eldr, (int, float)) else est_eldr.item()

# save results
with h5py.File(results_filename, 'w') as f:
    for alg_name, est_eldrs in est_eldrs_by_alg.items():
        f.create_dataset(f'est_eldrs_arr_{alg_name}', data=est_eldrs)
