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

raw_results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.h5'   
dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.h5'
processed_results_filename = f'{PROCESSED_RESULTS_DIR}/maes_by_kl_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.h5'

with h5py.File(raw_results_filename, 'r') as f:
    nrows = f['est_ldrs_arr_BDRE'].shape[0]
    est_ldrs_arr_bdre = f['est_ldrs_arr_BDRE'][:]  # (nrows, NTEST_SETS, NSAMPLES_TEST)
    est_ldrs_arr_tdre = f['est_ldrs_arr_TDRE'][:]  # (nrows, NTEST_SETS, NSAMPLES_TEST)
    est_ldrs_arr_mdre = f['est_ldrs_arr_MDRE'][:]  # (nrows, NTEST_SETS, NSAMPLES_TEST)
    est_ldrs_arr_tsm = f['est_ldrs_arr_TSM'][:]  # (nrows, NTEST_SETS, NSAMPLES_TEST)

with h5py.File(dataset_filename, 'r') as f:
    true_ldrs_arr = f['true_ldrs_arr'][:]  # (nrows, NTEST_SETS, NSAMPLES_TEST)


# BDRE
absolute_errors_bdre = np.abs(est_ldrs_arr_bdre - true_ldrs_arr)  # (nrows, NTEST_SETS, NSAMPLES_TEST)
maes_bdre = reduce(absolute_errors_bdre, 'n t d -> n t', 'mean')  # (nrows, NTEST_SETS)
maes_by_kl_bdre = maes_bdre.reshape(len(KL_DISTANCES), NUM_INSTANCES_PER_KL, NTEST_SETS)

# TDRE
absolute_errors_tdre = np.abs(est_ldrs_arr_tdre - true_ldrs_arr)  # (nrows, NTEST_SETS, NSAMPLES_TEST)
maes_tdre = reduce(absolute_errors_tdre, 'n t d -> n t', 'mean')  # (nrows, NTEST_SETS)
maes_by_kl_tdre = maes_tdre.reshape(len(KL_DISTANCES), NUM_INSTANCES_PER_KL, NTEST_SETS)

# MDRE
absolute_errors_mdre = np.abs(est_ldrs_arr_mdre - true_ldrs_arr)  # (nrows, NTEST_SETS, NSAMPLES_TEST)
maes_mdre = reduce(absolute_errors_mdre, 'n t d -> n t', 'mean')  # (nrows, NTEST_SETS)
maes_by_kl_mdre = maes_mdre.reshape(len(KL_DISTANCES), NUM_INSTANCES_PER_KL, NTEST_SETS)

# TSM
absolute_errors_tsm = np.abs(est_ldrs_arr_tsm - true_ldrs_arr)  # (nrows, NTEST_SETS, NSAMPLES_TEST)
maes_tsm = reduce(absolute_errors_tsm, 'n t d -> n t', 'mean')  # (nrows, NTEST_SETS)
maes_by_kl_tsm = maes_tsm.reshape(len(KL_DISTANCES), NUM_INSTANCES_PER_KL, NTEST_SETS)

# save results
os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)
with h5py.File(processed_results_filename, 'w') as f:
    f.create_dataset('maes_by_kl_bdre', data=maes_by_kl_bdre)
    f.create_dataset('maes_by_kl_tdre', data=maes_by_kl_tdre)
    f.create_dataset('maes_by_kl_mdre', data=maes_by_kl_mdre)
    f.create_dataset('maes_by_kl_tsm', data=maes_by_kl_tsm)