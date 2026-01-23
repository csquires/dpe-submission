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
    nrows = f['est_ldrs_arr_BDRE'].shape[0]
    est_ldrs_arr_bdre = f['est_ldrs_arr_BDRE'][:]  # (nrows, NTEST_SETS, NSAMPLES_TEST)
    # est_ldrs_arr_tdre = f['est_ldrs_arr_TDRE'][:]  # (nrows, NTEST_SETS, NSAMPLES_TEST)

with h5py.File(f'{DATA_DIR}/dataset_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.h5', 'r') as f:
    true_ldrs_arr = f['true_ldrs_arr'][:]  # (nrows, NTEST_SETS, NSAMPLES_TEST)

absolute_errors_bdre = np.abs(est_ldrs_arr_bdre - true_ldrs_arr[: :, :])  # (nrows, NTEST_SETS, NSAMPLES_TEST)
maes_bdre = reduce(absolute_errors_bdre, 'n t d -> n t', 'mean')  # (nrows, NTEST_SETS)
maes_by_kl_bdre = maes_bdre.reshape(len(KL_DISTANCES), NUM_INSTANCES_PER_KL, NTEST_SETS)

# absolute_errors_tdre = np.abs(est_ldrs_arr_tdre - true_ldrs_arr[:, np.newaxis, :, :])  # (nrows, NTEST_SETS, NSAMPLES_TEST)
# maes_tdre = reduce(absolute_errors_tdre, 'n a t d -> n a t', 'mean')  # (nrows, NTEST_SETS)
# maes_by_kl_tdre = maes_tdre.reshape(len(KL_DISTANCES), NUM_INSTANCES_PER_KL, NTEST_SETS)

os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)
with h5py.File(f'{PROCESSED_RESULTS_DIR}/maes_by_kl_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.h5', 'w') as f:
    f.create_dataset('maes_by_kl_bdre', data=maes_by_kl_bdre)
    # f.create_dataset('maes_by_kl_tdre', data=maes_by_kl_tdre)