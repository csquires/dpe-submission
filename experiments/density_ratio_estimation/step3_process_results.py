import os

import h5py
from einops import reduce
import numpy as np
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

raw_results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST},ntestsets={NTEST_SETS}.h5'
dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST},ntestsets={NTEST_SETS}.h5'
processed_results_filename = f'{PROCESSED_RESULTS_DIR}/maes_by_kl_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST},ntestsets={NTEST_SETS}.h5'

with h5py.File(raw_results_filename, 'r') as f:
    result_keys = [key for key in f.keys() if key.startswith('est_ldrs_arr_')]
    est_ldrs_by_alg = {key.replace('est_ldrs_arr_', ''): f[key][:] for key in result_keys}

with h5py.File(dataset_filename, 'r') as f:
    true_ldrs_arr = f['true_ldrs_arr'][:]  # (nrows, NTEST_SETS, NSAMPLES_TEST)

maes_by_kl = {}
for alg_name, est_ldrs_arr in est_ldrs_by_alg.items():
    absolute_errors = np.abs(est_ldrs_arr - true_ldrs_arr)  # (nrows, NTEST_SETS, NSAMPLES_TEST)
    maes = reduce(absolute_errors, 'n t d -> n t', 'mean')  # (nrows, NTEST_SETS)
    maes_by_kl[alg_name] = maes.reshape(len(KL_DISTANCES), NUM_INSTANCES_PER_KL, NTEST_SETS)

# save results
os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)
with h5py.File(processed_results_filename, 'w') as f:
    for alg_name, maes_arr in maes_by_kl.items():
        f.create_dataset(f'maes_by_kl_{alg_name}', data=maes_arr)
