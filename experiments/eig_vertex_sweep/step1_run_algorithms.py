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

# load config
config = yaml.load(open('experiments/eig_vertex_sweep/config.yaml', 'r'), Loader=yaml.FullLoader)

# extract and bind scalar config values
DEVICE = config['device']
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']
SEED = config['seed']
VERTEX_WAYPOINTS = config['vertex_waypoints']
NUM_WAYPOINTS = config['num_waypoints']
LATENT_DIM = config.get('latent_dim', 10)
MAX_TRAIN_SAMPLES = config.get('max_train_samples', None)
BATCH_SIZE = config.get('batch_size', None)
TRAIN_RATIO = config.get('train_ratio', None)

# set random seeds early (before importing models)
np.random.seed(SEED)
torch.manual_seed(SEED)

# import delayed modules
from src.models.multiclass_classification import make_multiclass_classifier
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
from src.eig_estimation.plugin import EIGPlugin
from src.waypoints.triangular_waypoints import TriangularWaypointBuilder1D


class TriangularMDREEIGAdapter:
    """adapter for TriangularMDRE that uses p0 samples as pstar during fit."""
    def __init__(self, triangular_mdre):
        self.triangular_mdre = triangular_mdre

    def fit(self, samples_p0, samples_p1):
        # use samples_p0 as pstar (joint distribution samples)
        self.triangular_mdre.fit(samples_p0, samples_p1, samples_p0)

    def predict_ldr(self, xs):
        return self.triangular_mdre.predict_ldr(xs)


def compute_true_eig(Sigma_pi: torch.Tensor, xi: torch.Tensor, sigma2: float = 1.0) -> torch.Tensor:
    """
    compute true eig for linear gaussian model.

    inputs:
      Sigma_pi: prior covariance, shape [DATA_DIM, DATA_DIM]
      xi: design vector, shape [DATA_DIM, 1]
      sigma2: observation noise variance (default 1.0)

    returns:
      scalar tensor: 0.5 * log1p(xi^T @ Sigma_pi @ xi / sigma2)
    """
    quad = (xi.T @ Sigma_pi @ xi).squeeze()
    return 0.5 * torch.log1p(quad / sigma2)


# construct filenames
dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM},nsamples={NSAMPLES}.h5'
results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},nsamples={NSAMPLES}.h5'

# check existing results
existing_results = set()
if os.path.exists(results_filename):
    with h5py.File(results_filename, 'r') as f:
        existing_results = set(f.keys())
        print("Existing results for:", list(f.keys()))

# ensure output directory exists
os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

# main context: open dataset in read mode
with h5py.File(dataset_filename, 'r') as dataset_file:
    nrows = dataset_file['design_arr'].shape[0]

    # compute and save true_eigs if not present or --force
    if 'true_eigs_arr' not in existing_results or args.force:
        true_eigs_arr = np.zeros(nrows, dtype=np.float32)
        for idx in trange(nrows):
            design = torch.from_numpy(dataset_file['design_arr'][idx]).to(DEVICE)  # [DATA_DIM, 1]
            Sigma_pi = torch.from_numpy(dataset_file['prior_covariance_arr'][idx]).to(DEVICE)  # [DATA_DIM, DATA_DIM]
            true_eigs_arr[idx] = compute_true_eig(Sigma_pi, design).item()

        with h5py.File(results_filename, 'a') as results_file:
            if 'true_eigs_arr' in results_file:
                del results_file['true_eigs_arr']
            results_file.create_dataset('true_eigs_arr', data=true_eigs_arr)

    # iterate over vertex indices
    for vertex_idx in VERTEX_WAYPOINTS:

        # construct result dataset name
        dataset_name = f'est_eigs_arr_vertex_{vertex_idx}'

        # check skip condition
        if dataset_name in existing_results and not args.force:
            print(f"Skipping vertex {vertex_idx} (results exist, use --force to overwrite)")
            continue
        else:
            print(f'Starting on vertex {vertex_idx}.')

        # reset seeds fresh for this vertex (deterministic classifier init)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # compute vertex parameter (position along waypoint path)
        vertex_v = vertex_idx / (NUM_WAYPOINTS - 1)  # float in [0, 1]

        # create waypoint builder with this vertex
        waypoint_builder = TriangularWaypointBuilder1D(vertex=vertex_v)

        # create fresh classifier for this vertex
        classifier = make_multiclass_classifier(
            name="default",
            input_dim=DATA_DIM + 1,
            num_classes=NUM_WAYPOINTS,
            latent_dim=LATENT_DIM,
            batch_size=BATCH_SIZE,
        )

        # instantiate TriangularMDRE with the classifier
        triangular_mdre = TriangularMDRE(
            classifier,
            waypoint_builder=waypoint_builder,
            device=DEVICE,
            max_train_samples=MAX_TRAIN_SAMPLES,
        )

        # wrap with adapter
        adapter = TriangularMDREEIGAdapter(triangular_mdre)

        # wrap with EIG plugin
        plugin = EIGPlugin(density_ratio_estimator=adapter, train_ratio=TRAIN_RATIO)

        # allocate result array
        est_eigs_arr = np.zeros(nrows, dtype=np.float32)

        # loop over all configurations in dataset
        for idx in trange(nrows):
            theta_samples = torch.from_numpy(dataset_file['theta_samples_arr'][idx]).to(DEVICE)  # [NSAMPLES, DATA_DIM]
            y_samples = torch.from_numpy(dataset_file['y_samples_arr'][idx]).to(DEVICE)  # [NSAMPLES, 1]

            # estimate eig
            result = plugin.estimate_eig(theta_samples, y_samples)
            est_eigs_arr[idx] = result.item() if hasattr(result, 'item') else result

        # save estimated eigs to results file
        with h5py.File(results_filename, 'a') as results_file:
            if dataset_name in results_file:
                del results_file[dataset_name]
            results_file.create_dataset(dataset_name, data=est_eigs_arr)

        # completion message
        print(f"Completed vertex {vertex_idx}.")
