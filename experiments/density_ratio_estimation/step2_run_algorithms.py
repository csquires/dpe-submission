import os

import h5py
import numpy as np
import torch
from tqdm import trange
import yaml

from src.density_ratio_estimation.bdre import BDRE
from src.density_ratio_estimation.tdre import TDRE
from src.models.binary_classification.gaussian_binary_classifier import build_gaussian_binary_classifier



config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
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

# bdre = BDRE(DATA_DIM)
tdre = TDRE(DATA_DIM, classifier_builder=build_gaussian_binary_classifier)
num_algs = 1
algorithms = [tdre]

with h5py.File(f'{DATA_DIR}/dataset_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.h5', 'r') as f:
    nrows = f['kl_distance_arr'].shape[0]
    est_ldrs_arr = np.zeros((nrows, num_algs, NTEST_SETS, NSAMPLES_TEST))

    for alg_idx, alg in enumerate(algorithms):
        for idx in trange(nrows):
            samples_p0 = torch.from_numpy(f['samples_p0_arr'][idx])  # (NSAMPLES_TRAIN, DATA_DIM)
            samples_p1 = torch.from_numpy(f['samples_p1_arr'][idx])  # (NSAMPLES_TRAIN, DATA_DIM)
            alg.fit(samples_p0, samples_p1)

            samples_pstar = torch.from_numpy(f['samples_pstar_arr'][idx])  # (NTEST_SETS, NSAMPLES_TEST, DATA_DIM)
            for test_set_idx in range(NTEST_SETS):
                est_ldrs = alg.predict_ldr(samples_pstar[test_set_idx])  # (NSAMPLES_TEST,)
                est_ldrs_arr[idx, alg_idx, test_set_idx] = est_ldrs.numpy()


os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
with h5py.File(f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.h5', 'a') as f:
    f.create_dataset('est_ldrs_arr', data=est_ldrs_arr)
