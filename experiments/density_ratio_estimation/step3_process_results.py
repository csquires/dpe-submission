import os

import h5py
from einops import reduce
import numpy as np
import torch
import yaml


config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)

# directories
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
PROCESSED_RESULTS_DIR = config['processed_results_dir']
# dataset parameters
DATA_DIM = config['data_dim']
KL_DISTANCES = config['kl_distances']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
NTEST_SETS = config['ntest_sets']
NUM_INSTANCES_PER_KL = config['num_instances_per_kl']


with h5py.File(f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.h5', 'r') as f:
    nrows = f['est_ldrs_arr'].shape[0]
    num_algs = f['est_ldrs_arr'].shape[1]
    est_ldrs_arr = f['est_ldrs_arr'][:]  # (nrows, num_algs, NTEST_SETS, NSAMPLES_TEST)

with h5py.File(f'{DATA_DIR}/dataset_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.h5', 'r') as f:
    true_ldrs_arr = f['true_ldrs_arr'][:]  # (nrows, NTEST_SETS, NSAMPLES_TEST)

absolute_errors = np.abs(est_ldrs_arr - true_ldrs_arr[:, np.newaxis, :, :])  # (nrows, num_algs, NTEST_SETS, NSAMPLES_TEST)
maes = reduce(absolute_errors, 'n a t d -> n a t', 'mean')  # (nrows, num_algs, NTEST_SETS)
maes_by_kl = maes.reshape(len(KL_DISTANCES), NUM_INSTANCES_PER_KL, num_algs, NTEST_SETS)

os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)
np.save(f'{PROCESSED_RESULTS_DIR}/maes_by_kl_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.npy', maes_by_kl)