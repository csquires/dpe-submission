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
from src.methods import BDRE, MDRE, TDRE, TSM, TriangularTSM
from src.methods.cls.mdre.tri import TriangularMDRE
from src.methods.reg.vfm import VFMOrthros, make_vfm
from ex.utils.eig_ldr import joint_and_shuffled


# methods whose fit signature takes a third positional pstar arg.
# we pass the joint as pstar since eig has no separate pstar distribution.
_PSTAR_METHODS = {"TriangularTSM", "TriangularMDRE"}


def fit_predict_eig(alg, alg_name, theta, y):
    """fit alg on (joint, shuffled-marginals); return predict_eldr(joint)
    (the expected log-density ratio under the joint, i.e. the EIG estimate)."""
    joint, shuffled = joint_and_shuffled(theta, y)
    if alg_name in _PSTAR_METHODS:
        alg.fit(joint, shuffled, joint)
    else:
        alg.fit(joint, shuffled)
    with torch.no_grad():
        return alg.predict_eldr(joint)


config = yaml.load(open('ex/synth/eig/config1.yaml', 'r'), Loader=yaml.FullLoader)
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

# instantiate bdre
bdre_classifier = make_binary_classifier(name="default", input_dim=DATA_DIM + 1)
bdre = BDRE(bdre_classifier, device=DEVICE)

# instantiate tdre (5 waypoints)
tdre_waypoints = 5
tdre_classifiers = make_pairwise_binary_classifiers(
    name="default",
    num_classes=tdre_waypoints,
    input_dim=DATA_DIM + 1,
)
tdre = TDRE(tdre_classifiers, num_waypoints=tdre_waypoints, device=DEVICE)

# instantiate mdre (15 waypoints)
mdre_waypoints = 15
mdre_classifier = make_multiclass_classifier(
    name="default",
    input_dim=DATA_DIM + 1,
    num_classes=mdre_waypoints,
)
mdre = MDRE(mdre_classifier, device=DEVICE)

# instantiate triangular mdre (15 waypoints)
triangular_mdre_classifier = make_multiclass_classifier(
    name="default",
    input_dim=DATA_DIM + 1,
    num_classes=mdre_waypoints,
)
triangular_mdre = TriangularMDRE(triangular_mdre_classifier, device=DEVICE)

# instantiate tsm
tsm = TSM(DATA_DIM + 1, device=DEVICE)

# instantiate triangular tsm
ttsm_cfg = config.get('triangular_tsm', {})
triangular_tsm = TriangularTSM(
    DATA_DIM + 1,
    device=DEVICE,
    vertex=ttsm_cfg.get('vertex', 0.5),
    peak_max=ttsm_cfg.get('peak_max', 1.0),
)

# instantiate spatial velo denoiser (VFM)
spatial_denoiser = make_vfm(input_dim=DATA_DIM + 1, device=DEVICE)

# instantiate spatial velo denoiser orthros (VFMOrthros)
vfm_orthros = VFMOrthros(input_dim=DATA_DIM + 1, device=DEVICE)

algorithms = [
    ("BDRE", bdre),
    ("TDRE", tdre),
    ("MDRE", mdre),
    ("TriangularMDRE", triangular_mdre),
    ("TSM", tsm),
    ("VFM", spatial_denoiser),
    ("VFMOrthros", vfm_orthros),
]

def compute_true_eig(Sigma_pi: torch.Tensor, xi: torch.Tensor, sigma2: float = 1.0) -> torch.Tensor:
    quad = (xi.T @ Sigma_pi @ xi).squeeze()
    return 0.5 * torch.log1p(quad / sigma2)

os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
with h5py.File(dataset_filename, 'r') as dataset_file:
    nrows = dataset_file['design_arr'].shape[0]

    # Compute and save true_eigs if not present or --force
    if 'true_eigs_arr' not in existing_results or args.force:
        true_eigs_arr = np.zeros(nrows, dtype=np.float32)
        for idx in trange(nrows):
            design = torch.from_numpy(dataset_file['design_arr'][idx]).to(DEVICE)  # (DATA_DIM, 1)
            Sigma_pi = torch.from_numpy(dataset_file['prior_covariance_arr'][idx]).to(DEVICE)  # (DATA_DIM, DATA_DIM)
            true_eigs_arr[idx] = compute_true_eig(Sigma_pi, design).item()

        with h5py.File(results_filename, 'a') as results_file:
            if 'true_eigs_arr' in results_file:
                del results_file['true_eigs_arr']
            results_file.create_dataset('true_eigs_arr', data=true_eigs_arr)

    for alg_name, alg in algorithms:
        dataset_name = f'est_eigs_arr_{alg_name}'
        if dataset_name in existing_results and not args.force:
            print(f"Skipping {alg_name} (results exist, use --force to overwrite)")
            continue
        else:
            print(f'Starting on {alg_name}.')

        est_eigs_arr = np.zeros(nrows, dtype=np.float32)
        for idx in trange(nrows):
            theta_samples = torch.from_numpy(dataset_file['theta_samples_arr'][idx]).to(DEVICE)  # (NSAMPLES, DATA_DIM)
            y_samples = torch.from_numpy(dataset_file['y_samples_arr'][idx]).to(DEVICE)  # (NSAMPLES, 1)
            result = fit_predict_eig(alg, alg_name, theta_samples, y_samples)
            est_eigs_arr[idx] = result.item() if hasattr(result, 'item') else result

        with h5py.File(results_filename, 'a') as results_file:
            if dataset_name in results_file:
                del results_file[dataset_name]
            results_file.create_dataset(dataset_name, data=est_eigs_arr)
