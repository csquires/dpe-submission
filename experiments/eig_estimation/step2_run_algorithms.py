import os

import h5py
import numpy as np
import torch
from tqdm import trange
import yaml

from src.models.binary_classification import make_binary_classifier, make_pairwise_binary_classifiers
from src.models.multiclass_classification import make_multiclass_classifier
from src.density_ratio_estimation import BDRE, MDRE, TDRE, TSM
from src.eig_estimation.plugin import EIGPlugin


config = yaml.load(open('experiments/eig_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
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

# instantiate bdre plugin
bdre_classifier = make_binary_classifier(name="default", input_dim=DATA_DIM + 1)
bdre = BDRE(bdre_classifier, device=DEVICE)
bdre_plugin = EIGPlugin(density_ratio_estimator=bdre)

# instantiate tdre plugin (5 waypoints)
tdre_waypoints = 5
tdre_classifiers = make_pairwise_binary_classifiers(
    name="default",
    num_classes=tdre_waypoints,
    input_dim=DATA_DIM + 1,
)
tdre = TDRE(tdre_classifiers, num_waypoints=tdre_waypoints, device=DEVICE)
tdre_plugin = EIGPlugin(density_ratio_estimator=tdre)

# instantiate mdre plugin (15 waypoints)
mdre_waypoints = 15
mdre_classifier = make_multiclass_classifier(
    name="default",
    input_dim=DATA_DIM + 1,
    num_classes=mdre_waypoints,
)
mdre = MDRE(mdre_classifier, device=DEVICE)
mdre_plugin = EIGPlugin(density_ratio_estimator=mdre)

# instantiate tsm plugin
tsm = TSM(DATA_DIM + 1, device=DEVICE)
tsm_plugin = EIGPlugin(density_ratio_estimator=tsm)

algorithms = [
    ("BDRE", bdre_plugin),
    ("TDRE_5", tdre_plugin),
    ("MDRE_15", mdre_plugin),
    ("TSM", tsm_plugin),
]

def compute_true_eig(Sigma_pi: torch.Tensor, xi: torch.Tensor, sigma2: float = 1.0) -> torch.Tensor:
    quad = (xi.T @ Sigma_pi @ xi).squeeze()
    return 0.5 * torch.log1p(quad / sigma2)

os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
with h5py.File(dataset_filename, 'r') as dataset_file:
    nrows = dataset_file['design_arr'].shape[0]

    true_eigs_arr = np.zeros(nrows, dtype=np.float32)
    for idx in trange(nrows):
        design = torch.from_numpy(dataset_file['design_arr'][idx]).to(DEVICE)  # (DATA_DIM, 1)
        Sigma_pi = torch.from_numpy(dataset_file['prior_covariance_arr'][idx]).to(DEVICE)  # (DATA_DIM, DATA_DIM)
        true_eigs_arr[idx] = compute_true_eig(Sigma_pi, design).item()

    with h5py.File(results_filename, 'w') as results_file:
        results_file.create_dataset('true_eigs_arr', data=true_eigs_arr)

        for alg_name, alg in algorithms:
            est_eigs_arr = np.zeros(nrows, dtype=np.float32)
            for idx in trange(nrows):
                theta_samples = torch.from_numpy(dataset_file['theta_samples_arr'][idx]).to(DEVICE)  # (NSAMPLES, DATA_DIM)
                y_samples = torch.from_numpy(dataset_file['y_samples_arr'][idx]).to(DEVICE)  # (NSAMPLES, 1)
                est_eigs_arr[idx] = alg.estimate_eig(theta_samples, y_samples).item()

            results_file.create_dataset(f'est_eigs_arr_{alg_name}', data=est_eigs_arr)
