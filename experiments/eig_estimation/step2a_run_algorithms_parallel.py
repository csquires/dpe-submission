import argparse
import json
import math
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import trange
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--task-id', type=int, default=None,
                    help='Task ID. If not provided, uses SLURM_ARRAY_TASK_ID')
parser.add_argument('--num-tasks', type=int, required=True,
                    help='Total number of tasks')
parser.add_argument('--config', type=str, required=True,
                    help='Path to config YAML')
parser.add_argument('--force', action='store_true',
                    help='Force re-run of all algorithms, overwriting existing results')
args = parser.parse_args()

# Get task ID from env or arg
task_id = args.task_id if args.task_id is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

from src.models.binary_classification import make_binary_classifier, make_pairwise_binary_classifiers
from src.models.multiclass_classification import make_multiclass_classifier
from src.density_ratio_estimation import BDRE, MDRE, TDRE
from src.density_ratio_estimation.tsm import TSM
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
from src.eig_estimation.plugin import EIGPlugin
from src.density_ratio_estimation.spatial_adapters import make_spatial_velo_denoiser


class TriangularMDREEIGAdapter:
    """Adapter for TriangularMDRE that uses p0 samples as pstar during fit."""
    def __init__(self, triangular_mdre):
        self.triangular_mdre = triangular_mdre

    def fit(self, samples_p0, samples_p1):
        # Use samples_p0 as pstar (joint distribution samples)
        self.triangular_mdre.fit(samples_p0, samples_p1, samples_p0)

    def predict_ldr(self, xs):
        return self.triangular_mdre.predict_ldr(xs)


config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
DEVICE = config['device']
# directories
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
# dataset parameters
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']
# random seed
SEED = config['seed']
np.random.seed(SEED + task_id)  # Different seed per task for reproducibility
torch.manual_seed(SEED + task_id)

dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM},nsamples={NSAMPLES}.h5'
results_filename = f'{RAW_RESULTS_DIR}/results_task_{task_id}.h5'

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

# instantiate triangular mdre plugin
triangular_mdre_waypoints = 15
triangular_mdre_classifier = make_multiclass_classifier(
    name="default",
    input_dim=DATA_DIM + 1,
    num_classes=triangular_mdre_waypoints,
)
triangular_mdre = TriangularMDRE(
    triangular_mdre_classifier,
    device=DEVICE,
    midpoint_oversample=7,
    gamma_power=3.0,
)
triangular_mdre_adapter = TriangularMDREEIGAdapter(triangular_mdre)
triangular_mdre_plugin = EIGPlugin(density_ratio_estimator=triangular_mdre_adapter)

# instantiate spatial-based EIG plugin (VFM)
spatial_denoiser = make_spatial_velo_denoiser(input_dim=DATA_DIM + 1, device=DEVICE)
spatial_denoiser_plugin = EIGPlugin(density_ratio_estimator=spatial_denoiser)

# instantiate TSM plugin
tsm = TSM(input_dim=DATA_DIM + 1, device=DEVICE)
tsm_plugin = EIGPlugin(density_ratio_estimator=tsm)

algorithms = [
    ("BDRE", bdre_plugin),
    ("TDRE_5", tdre_plugin),
    ("MDRE_15", mdre_plugin),
    ("TriangularMDRE", triangular_mdre_plugin),
    ("TSM", tsm_plugin),
    ("VFM", spatial_denoiser_plugin),
]


def compute_true_eig(Sigma_pi: torch.Tensor, xi: torch.Tensor, sigma2: float = 1.0) -> torch.Tensor:
    quad = (xi.T @ Sigma_pi @ xi).squeeze()
    return 0.5 * torch.log1p(quad / sigma2)


os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

# Open dataset file in read-only mode (safe for concurrent access)
with h5py.File(dataset_filename, 'r') as dataset_file:
    nrows = dataset_file['design_arr'].shape[0]

    # Compute dataset range for this task
    datasets_per_task = math.ceil(nrows / args.num_tasks)
    start_idx = task_id * datasets_per_task
    end_idx = min((task_id + 1) * datasets_per_task, nrows)
    job_nrows = end_idx - start_idx

    if job_nrows <= 0:
        print(f"Task {task_id}: No datasets to process (nrows={nrows}, num_tasks={args.num_tasks})")
        sys.exit(0)

    print(f"Task {task_id}: Processing datasets {start_idx} to {end_idx-1} ({job_nrows} datasets)")

    # Check for existing results in task file
    existing_results = set()
    if os.path.exists(results_filename):
        with h5py.File(results_filename, 'r') as results_file:
            existing_results = set(results_file.keys())
            print(f"Task {task_id}: Existing results for: {list(results_file.keys())}")

    # Compute and save true_eigs if not present or --force
    if 'true_eigs_arr' not in existing_results or args.force:
        true_eigs_arr = np.zeros(job_nrows, dtype=np.float32)
        for local_idx, global_idx in enumerate(trange(start_idx, end_idx, desc=f"Task {task_id} - true_eigs")):
            design = torch.from_numpy(dataset_file['design_arr'][global_idx]).to(DEVICE)
            Sigma_pi = torch.from_numpy(dataset_file['prior_covariance_arr'][global_idx]).to(DEVICE)
            true_eigs_arr[local_idx] = compute_true_eig(Sigma_pi, design).item()

        with h5py.File(results_filename, 'a') as results_file:
            if 'true_eigs_arr' in results_file:
                del results_file['true_eigs_arr']
            results_file.create_dataset('true_eigs_arr', data=true_eigs_arr)

    for alg_name, alg in algorithms:
        dataset_name = f'est_eigs_arr_{alg_name}'
        if dataset_name in existing_results and not args.force:
            print(f"Task {task_id}: Skipping {alg_name} (results exist, use --force to overwrite)")
            continue

        est_eigs_arr = np.zeros(job_nrows, dtype=np.float32)
        for local_idx, global_idx in enumerate(trange(start_idx, end_idx, desc=f"Task {task_id} - {alg_name}")):
            theta_samples = torch.from_numpy(dataset_file['theta_samples_arr'][global_idx]).to(DEVICE)
            y_samples = torch.from_numpy(dataset_file['y_samples_arr'][global_idx]).to(DEVICE)
            result = alg.estimate_eig(theta_samples, y_samples)
            est_eigs_arr[local_idx] = result.item() if hasattr(result, 'item') else result

        with h5py.File(results_filename, 'a') as results_file:
            if dataset_name in results_file:
                del results_file[dataset_name]
            results_file.create_dataset(dataset_name, data=est_eigs_arr)

    # Store metadata for step2b
    alg_names = [name for name, _ in algorithms]
    with h5py.File(results_filename, 'a') as f:
        f.attrs['task_id'] = task_id
        f.attrs['start_idx'] = start_idx
        f.attrs['end_idx'] = end_idx
        f.attrs['num_tasks'] = args.num_tasks
        f.attrs['nrows_total'] = nrows
        f.attrs['algorithm_names'] = json.dumps(alg_names)

print(f"Task {task_id}: Completed. Results saved to {results_filename}")
