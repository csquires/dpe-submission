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
from src.models.binary_classification.gaussian_binary_classifier import make_gaussian_binary_classifier
from src.models.multiclass_classification import make_multiclass_classifier
from src.density_ratio_estimation.bdre import BDRE
from src.density_ratio_estimation.tdre import TDRE
from src.density_ratio_estimation.mdre import MDRE
from src.density_ratio_estimation.tsm import TSM
from src.density_ratio_estimation.triangular_tsm import TriangularTSM
from src.density_ratio_estimation.triangular_tdre import TriangularTDRE
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
from src.density_ratio_estimation.spatial_adapters import make_spatial_velo_denoiser



config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
DEVICE = config['device']
# directories
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
# dataset parameters
DATA_DIM = config['data_dim']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
NTEST_SETS = config['ntest_sets']
# random seed
SEED = config['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)

# dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST},ntestsets={NTEST_SETS}.h5'
dataset_filename = f'{DATA_DIR}/dataset_newpstar.h5'
# results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST},ntestsets={NTEST_SETS}.h5'
results_filename = f'{RAW_RESULTS_DIR}/new_pstar2.h5'

existing_results = set()
if os.path.exists(results_filename):
    with h5py.File(results_filename, 'r') as results_file:
        existing_results = set(results_file.keys())
        print("Existing results for:", results_file.keys())

# instantiate bdre
bdre_classifier = make_binary_classifier(name="default", input_dim=DATA_DIM)
bdre = BDRE(bdre_classifier, device=DEVICE)
# instantiate tdre variants
tdre_waypoints = [5]
tdre_variants = []
for num_waypoints_tdre in tdre_waypoints:
    tdre_classifiers = make_pairwise_binary_classifiers(
        name="default",
        num_classes=num_waypoints_tdre,
        input_dim=DATA_DIM,
    )
    tdre_variants.append((f"TDRE_{num_waypoints_tdre}", TDRE(tdre_classifiers, num_waypoints=num_waypoints_tdre, device=DEVICE)))

# instantiate gaussian-tdre variants
tdre_gauss_variants = []
for num_waypoints_tdre in tdre_waypoints:
    tdre_gauss_classifiers = [make_gaussian_binary_classifier(input_dim=DATA_DIM) for _ in range(num_waypoints_tdre - 1)]
    tdre_gauss_variants.append(
        (f"TDRE_{num_waypoints_tdre}_Gauss", TDRE(tdre_gauss_classifiers, num_waypoints=num_waypoints_tdre, device=DEVICE))
    )

# instantiate mdre variants
mdre_waypoints = [15]
mdre_variants = []
for num_waypoints_mdre in mdre_waypoints:
    mdre_classifier = make_multiclass_classifier(
        name="default",
        input_dim=DATA_DIM,
        num_classes=num_waypoints_mdre,
    )
    mdre_variants.append((f"MDRE_{num_waypoints_mdre}", MDRE(mdre_classifier, device=DEVICE)))

# instantiate tsm
tsm = TSM(DATA_DIM, device=DEVICE)
# instantiate triangular tsm
triangular_tsm = TriangularTSM(DATA_DIM, device=DEVICE)
# instantiate triangular tdre
triangular_tdre_waypoints = 5
triangular_tdre_classifiers = make_pairwise_binary_classifiers(
    name="default",
    num_classes=triangular_tdre_waypoints,
    input_dim=DATA_DIM,
)
triangular_tdre = TriangularTDRE(
    triangular_tdre_classifiers,
    num_waypoints=triangular_tdre_waypoints,
    device=DEVICE,
    midpoint_oversample=7,
    gamma_power=3.0,
)
# instantiate triangular tdre with gaussian classifiers
triangular_tdre_gauss_classifiers = [make_gaussian_binary_classifier(input_dim=DATA_DIM) for _ in range(triangular_tdre_waypoints - 1)]
triangular_tdre_gauss = TriangularTDRE(
    triangular_tdre_gauss_classifiers,
    num_waypoints=triangular_tdre_waypoints,
    device=DEVICE,
    midpoint_oversample=7,
    gamma_power=3.0,
)
# instantiate triangular mdre
triangular_mdre_waypoints = 15
triangular_mdre_classifier = make_multiclass_classifier(
    name="default",
    input_dim=DATA_DIM,
    num_classes=triangular_mdre_waypoints,
)
triangular_mdre = TriangularMDRE(
    triangular_mdre_classifier,
    device=DEVICE,
    midpoint_oversample=7,
    gamma_power=3.0,
)
# instantiate spatial velo denoiser
spatial = make_spatial_velo_denoiser(input_dim=DATA_DIM, device=DEVICE)

algorithms = [
    # ("BDRE", bdre),
    # ("TDRE", tdre),
    # ("MDRE", mdre),
    # ("TriangularTSM", triangular_tsm),
    ("TriangularTDRE", triangular_tdre),
    ("TriangularTDRE_Gauss", triangular_tdre_gauss),
    ("TriangularMDRE", triangular_mdre),
    # ("TSM", tsm),
    *tdre_variants,
    *tdre_gauss_variants,
    *mdre_variants,
    #("Spatial", spatial),
]

os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
with h5py.File(dataset_filename, 'r') as dataset_file:
    nrows = dataset_file['kl_distance_arr'].shape[0]

    for alg_name, alg in algorithms:
        dataset_name = f'est_ldrs_arr_{alg_name}'
        if dataset_name in existing_results and not args.force:
            print(f"Skipping {alg_name} (results exist, use --force to overwrite)")
            continue

        est_ldrs_arr = np.zeros((nrows, NTEST_SETS, NSAMPLES_TEST))
        for idx in trange(nrows):
            samples_p0 = torch.from_numpy(dataset_file['samples_p0_arr'][idx]).to(DEVICE)  # (NSAMPLES_TRAIN, DATA_DIM)
            samples_p1 = torch.from_numpy(dataset_file['samples_p1_arr'][idx]).to(DEVICE)  # (NSAMPLES_TRAIN, DATA_DIM)
            if alg_name in {"TriangularTSM", "TriangularTDRE", "TriangularMDRE"}:
                pstar_train = torch.from_numpy(dataset_file['samples_pstar_train_arr'][idx]).to(DEVICE)
                alg.fit(samples_p0, samples_p1, pstar_train)
            else:
                alg.fit(samples_p0, samples_p1)

            samples_pstar = torch.from_numpy(dataset_file['samples_pstar_arr'][idx]).to(DEVICE)  # (NTEST_SETS, NSAMPLES_TEST, DATA_DIM)
            for test_set_idx in range(NTEST_SETS):
                est_ldrs = alg.predict_ldr(samples_pstar[test_set_idx])  # (NSAMPLES_TEST,)
                est_ldrs_arr[idx, test_set_idx] = est_ldrs.cpu().numpy()

        with h5py.File(results_filename, 'a') as results_file:
            dataset_name = f'est_ldrs_arr_{alg_name}'
            if dataset_name in results_file:
                del results_file[dataset_name]
            results_file.create_dataset(dataset_name, data=est_ldrs_arr)
